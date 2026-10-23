# ============================================================================
#  train.py - Addestramento di un fold, INTERROMPIBILE E RIPRENDIBILE.
#
#  PERCHE' LA RISCRITTURA
#  Il loop precedente aveva due difetti che rendevano impossibile il lavoro
#  sperimentale che questo progetto richiede:
#
#  1. Nessun resume. Una 5-fold CV non entra in una sessione singola, e
#     un'interruzione costava l'intero run. Qui il checkpoint viene scritto
#     dopo OGNI epoca, in modo atomico, con optimizer/scheduler/scaler/RNG:
#     --resume riparte esattamente da dove si era fermato, e --max-hours fa
#     uscire il processo in modo pulito prima di sforare il tetto di sessione.
#
#  2. Selezione del checkpoint sul MASSIMO del Dice di validazione. Con 12-13
#     volumi di validazione quella curva oscilla di +-0.02 (si vede in
#     outputs/train.log: plateau a 0.565 con un picco isolato a 0.586
#     all'epoca 250). Prendere il massimo significa selezionare il rumore, e
#     infatti il "best" 0.586 non e' riproducibile. Qui si seleziona su una
#     MEDIA MOBILE di una metrica composita, e si salva SEMPRE final.pth: con
#     lo scheduler che porta il learning rate a zero, il modello finale e' di
#     norma equivalente al migliore e non e' contaminato dalla selezione. E'
#     la convenzione di nnU-Net.
#
#  USO:
#      python train.py --exp baseline_unet --fold 0 --max-hours 7.5
#      python train.py --exp baseline_unet --fold 0 --resume
# ============================================================================
import os
import sys
import json
import time
import random
import signal
import argparse
import datetime
import subprocess

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW, SGD

from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference
from monai.data import decollate_batch
from monai.transforms import Compose, AsDiscrete
from monai.utils import set_determinism

import config
import splits as splits_mod
from dataset import get_fold_dataloaders
from model import get_model, has_deep_supervision


# ---------------------------------------------------------------------------
#  Stato di interruzione (Ctrl-C non deve far perdere l'epoca in corso)
# ---------------------------------------------------------------------------
_stop_requested = False


def _handle_sigint(signum, frame):
    global _stop_requested
    if _stop_requested:                 # secondo Ctrl-C: uscita immediata
        print("\n[train] Secondo interrupt: uscita immediata.")
        sys.exit(130)
    _stop_requested = True
    print("\n[train] Interrupt ricevuto: completo l'epoca, salvo e esco. "
          "(Ctrl-C di nuovo per uscire subito.)")


# ---------------------------------------------------------------------------
#  Checkpoint
# ---------------------------------------------------------------------------
def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return "n/a"


def save_checkpoint(path: str, **state) -> None:
    """Scrittura ATOMICA: prima su .tmp, poi rename.

    Un checkpoint viene scritto dopo ogni epoca; se il processo morisse a meta'
    scrittura (batteria, OOM, Ctrl-C sfortunato) si troverebbe un file troncato
    al posto dell'unico stato valido. os.replace e' atomico su Windows e su
    POSIX: o c'e' il file vecchio, o quello nuovo, mai un ibrido.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    torch.save(state, tmp)
    os.replace(tmp, path)


def rng_state() -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "torch_cuda": (torch.cuda.get_rng_state_all()
                       if torch.cuda.is_available() else None),
    }


def restore_rng(state: dict) -> None:
    """Ripristina gli stati RNG salvati.

    I .cpu() NON sono ridondanti: il checkpoint viene riletto con
    torch.load(map_location=device), che sposta su CUDA ogni tensore del dict,
    stati RNG compresi. torch.set_rng_state pretende pero' un ByteTensor su
    CPU e fallisce con "RNG state must be a torch.ByteTensor" - un errore che
    si manifesta solo alla PRIMA ripresa, cioe' esattamente quando serve.
    """
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu().to(torch.uint8))
    if state.get("torch_cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(
            [s.cpu().to(torch.uint8) for s in state["torch_cuda"]])


def epoch_seed(exp: str, fold: int, epoch: int) -> int:
    """Seme stabile e riproducibile per (esperimento, fold, epoca).

    NOTA IMPORTANTE SU MONAI: le transform random NON leggono l'RNG globale di
    numpy al momento della chiamata. Ogni Randomizable possiede un proprio
    np.random.RandomState, seminato alla costruzione. Ripristinare lo stato
    globale (restore_rng) quindi NON ripristina la sequenza delle transform.
    La soluzione praticabile e' ri-seminare l'intera catena all'inizio di ogni
    epoca con un seme funzione di (exp, fold, epoca): un run ripreso all'epoca
    N consuma esattamente la stessa sequenza di un run continuo.
    La riproducibilita' che si ottiene e' "a granularita' di epoca", ed e'
    questo che va dichiarato nella relazione - non "bit-identica", che sarebbe
    falso con cudnn.benchmark attivo.
    """
    h = abs(hash((exp, int(fold), int(epoch)))) % (2 ** 31 - 1)
    return int(h)


# ---------------------------------------------------------------------------
#  Loss con deep supervision
# ---------------------------------------------------------------------------
def deep_supervision_loss(outputs, labels, loss_fn):
    """Loss su uscite singole, impilate (DynUNet) o a piu' risoluzioni (SegResNetDS).

    I pesi decadono come 1/2^i e sono normalizzati a somma 1: le uscite
    ausiliarie devono guidare i livelli profondi senza dominare l'obiettivo
    principale, che resta la predizione a piena risoluzione.
    """
    # Caso 1: uscita singola (unet, oppure qualunque rete in eval).
    if torch.is_tensor(outputs) and outputs.ndim == 5:
        return loss_fn(outputs, labels)

    # Caso 2: DynUNet in train -> tensore (B, n_ds+1, C, H, W, D), tutte le
    # uscite gia' riportate alla risoluzione piena.
    if torch.is_tensor(outputs) and outputs.ndim == 6:
        outs = list(torch.unbind(outputs, dim=1))
    else:
        # Caso 3: SegResNetDS -> lista di tensori a risoluzioni DECRESCENTI.
        outs = list(outputs)

    weights = np.array([0.5 ** i for i in range(len(outs))], dtype=np.float64)
    weights /= weights.sum()

    total = 0.0
    for w, o in zip(weights, outs):
        if o.shape[2:] != labels.shape[2:]:
            # Si sottocampiona la LABEL, non si sovracampiona la predizione:
            # ingrandire i logit costerebbe memoria proporzionale al volume
            # pieno per ogni livello, che e' esattamente cio' che la deep
            # supervision a bassa risoluzione vuole evitare.
            lab = F.interpolate(labels.float(), size=o.shape[2:], mode="nearest")
        else:
            lab = labels
        total = total + float(w) * loss_fn(o, lab)
    return total


# ---------------------------------------------------------------------------
#  Validazione rapida (per la SOLA selezione del checkpoint)
# ---------------------------------------------------------------------------
def validate(model, loader, device, use_amp, roi_size, sw_batch, overlap,
             dice_metric, post_pred, post_label):
    """Dice medio + una stima grossolana di detection sullo spazio RICAMPIONATO.

    Deliberatamente NON e' la valutazione ufficiale: quella la fa evaluate.py,
    nello spazio CT originale e con il post-processing di deployment. Qui serve
    solo a decidere quale checkpoint tenere, e per quello un Dice sullo spazio
    ricampionato e' sufficiente e costa una frazione del tempo (niente Invertd,
    niente metriche di superficie).
    """
    import postproc
    from scipy import ndimage

    model.eval()
    dice_metric.reset()
    n_fp = 0
    n_hit = 0
    n_cases = 0

    # Volume di un voxel nello spazio RICAMPIONATO (qui non si applica Invertd).
    vox_mm3 = float(np.prod(config.TARGET_SPACING))

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = sliding_window_inference(
                    inputs=images, roi_size=roi_size, sw_batch_size=sw_batch,
                    predictor=model, overlap=overlap,
                    mode=config.SW_BLEND_MODE)

            outs = [post_pred(o) for o in decollate_batch(logits)]
            labs = [post_label(l) for l in decollate_batch(labels)]
            dice_metric(y_pred=outs, y=labs)

            # Conteggio delle lesioni con LO STESSO filtro sul volume del
            # deployment. Senza il filtro, nelle prime epoche la predizione e'
            # rumore diffuso e ndimage.label trova decine di migliaia di
            # componenti da 1-2 voxel: il punteggio di selezione ne verrebbe
            # dominato (misurato: fp=63700 all'epoca 6) e non misurerebbe piu'
            # nulla di utile.
            prob = torch.softmax(logits.float(), dim=1)[0, 1].cpu().numpy()
            pred, lesions = postproc.probs_to_mask(
                prob, config.POSTPROC, voxel_volume_mm3=vox_mm3)
            gt = (labels[0, 0] > 0).cpu().numpy()
            n_cases += 1
            if lesions:
                lab_arr, n = ndimage.label(pred)
                touching = set(np.unique(lab_arr[gt & (lab_arr > 0)]).tolist())
                touching.discard(0)
                n_fp += n - len(touching)
                n_hit += 1 if touching else 0

    dice = float(dice_metric.aggregate().item())
    return {
        "val_dice": dice,
        "val_hit_rate": n_hit / max(n_cases, 1),
        "val_fp_per_scan": n_fp / max(n_cases, 1),
    }


def selection_score(dice: float, hit: float, fp: float) -> float:
    """Metrica di selezione del checkpoint.

    Non solo Dice: un modello che segmenta bene le lesioni che trova ma ne
    perde meta' e' peggiore, per questo progetto, di uno leggermente meno
    preciso che le trova tutte. Il termine sui falsi positivi ha peso piccolo
    (0.1) perche' i FP si possono ancora filtrare a valle con la soglia e il
    volume minimo, mentre una lesione mai rilevata e' persa definitivamente.
    """
    return 0.5 * dice + 0.5 * (hit - 0.1 * fp)


def moving_average(values: list[float], k: int = 3) -> float:
    v = values[-k:]
    return sum(v) / len(v) if v else float("-inf")


# ---------------------------------------------------------------------------
#  Addestramento di un fold
# ---------------------------------------------------------------------------
def train_one_fold(args) -> dict:
    global _stop_requested
    run_dir = os.path.join(config.OUTPUT_DIR, "runs", args.exp, f"fold{args.fold}")
    os.makedirs(run_dir, exist_ok=True)
    last_path = os.path.join(run_dir, "last.pth")
    best_path = os.path.join(run_dir, "best.pth")
    final_path = os.path.join(run_dir, "final.pth")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    if use_amp:
        # Autotuning dei kernel conv per dimensioni fisse. Non e' compatibile
        # con la riproducibilita' bit-identica, che qui non pretendiamo.
        torch.backends.cudnn.benchmark = True

    # Un seme distinto per fold: fold diversi devono vedere sequenze random
    # diverse, altrimenti si ripete lo stesso ordine di augmentation cinque
    # volte e la varianza fra fold risulta sottostimata.
    set_determinism(seed=config.SEED * 1000 + args.fold)

    print(f"[train] exp={args.exp} fold={args.fold} arch={args.arch} device={device}")
    if device.type == "cuda":
        print(f"[train] GPU: {torch.cuda.get_device_name(0)} "
              f"({torch.cuda.get_device_properties(0).total_memory/2**30:.1f} GiB)")

    # --- Dati ---------------------------------------------------------------
    train_loader, val_loader, train_files, val_files = get_fold_dataloaders(
        fold=args.fold,
        patch_size=tuple(args.patch),
        batch_size=args.batch_size,
        samples_per_volume=args.samples_per_volume,
        num_workers=args.num_workers,
        use_lung_roi=args.lung_roi,
        augment=args.augment,
        cache_rate=args.cache_rate,
        source=args.source,
    )
    print(f"[train] fold {args.fold}: {len(train_files)} train | {len(val_files)} val")

    # --- Modello, loss, ottimizzatore ---------------------------------------
    ds = has_deep_supervision(args.arch) and not args.no_deep_supervision
    model = get_model(device, args.arch, deep_supervision=ds)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"[train] {args.arch}: {n_par/1e6:.1f}M parametri | deep supervision: {ds}")

    # --- Inizializzazione da pesi pre-addestrati ----------------------------
    # Si caricano SOLO i pesi, non lo stato dell'ottimizzatore ne' quello dello
    # scheduler: il fine-tuning ricomincia con il proprio learning rate e il
    # proprio schedule. Ereditare il momento di Adam da un dataset diverso
    # spingerebbe i primi passi nella direzione di LIDC invece di quella di MSD.
    if args.init_from:
        if not os.path.exists(args.init_from):
            raise FileNotFoundError(f"checkpoint di partenza assente: {args.init_from}")
        ck = torch.load(args.init_from, map_location=device, weights_only=False)
        sd = ck.get("model_state_dict", ck)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"[train] inizializzato da {os.path.basename(args.init_from)} "
              f"(epoca {ck.get('epoch', '?')}, exp '{ck.get('exp', '?')}')")
        if missing or unexpected:
            # Non silenzioso: pesi mancanti significano architetture diverse, e
            # un fine-tuning che parte da meta' rete casuale sembrerebbe solo
            # "un po' peggiore" invece di segnalare l'errore.
            print(f"[train] ATTENZIONE: {len(missing)} tensori mancanti, "
                  f"{len(unexpected)} inattesi -> architetture non coincidenti?")

    loss_function = DiceCELoss(to_onehot_y=True, softmax=True)

    if args.optimizer == "sgd":
        # Ricetta nnU-Net: SGD con Nesterov e momentum alto. Il momentum a 0.99
        # e' insolito ma e' deliberato: con batch piccolissimi il gradiente di
        # una singola iterazione e' rumoroso, e la media esponenziale lunga fa
        # da filtro.
        optimizer = SGD(model.parameters(), lr=args.lr, momentum=0.99,
                        nesterov=True, weight_decay=config.WEIGHT_DECAY)
    else:
        optimizer = AdamW(model.parameters(), lr=args.lr,
                          weight_decay=config.WEIGHT_DECAY)

    if args.schedule == "poly":
        # Poly LR di nnU-Net: (1 - e/E)^0.9. Decade piu' lentamente della
        # cosine all'inizio e piu' bruscamente alla fine.
        sched = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda e: (1.0 - e / args.epochs) ** 0.9)
    else:
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs)

    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    dice_metric = DiceMetric(include_background=False, reduction="mean")
    post_pred = Compose([AsDiscrete(argmax=True, to_onehot=config.OUT_CHANNELS)])
    post_label = Compose([AsDiscrete(to_onehot=config.OUT_CHANNELS)])

    # --- Ripresa ------------------------------------------------------------
    start_epoch = 0
    history: list[dict] = []
    best_score = float("-inf")
    best_epoch = -1
    elapsed_prev = 0.0

    if args.resume and os.path.exists(last_path):
        ck = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model_state_dict"])
        optimizer.load_state_dict(ck["optimizer_state_dict"])
        sched.load_state_dict(ck["scheduler_state_dict"])
        scaler.load_state_dict(ck["scaler_state_dict"])
        restore_rng(ck.get("rng"))
        start_epoch = ck["epoch"]
        history = ck.get("history", [])
        best_score = ck.get("best_metric", float("-inf"))
        best_epoch = ck.get("best_epoch", -1)
        elapsed_prev = ck.get("elapsed_seconds", 0.0)
        print(f"[train] RIPRESO da {last_path}: epoca {start_epoch}/{args.epochs}, "
              f"best_score={best_score:.4f} @ {best_epoch}, "
              f"gia' addestrato {elapsed_prev/3600:.2f} h")
    elif args.resume:
        print(f"[train] --resume richiesto ma {last_path} non esiste: parto da zero.")

    if start_epoch >= args.epochs:
        print(f"[train] fold {args.fold} gia' completo ({start_epoch} epoche).")
        write_status(args, run_dir, done=True, epoch=start_epoch,
                     best_score=best_score, best_epoch=best_epoch)
        return {"done": True, "epochs": start_epoch}

    # --- Ciclo --------------------------------------------------------------
    signal.signal(signal.SIGINT, _handle_sigint)
    t_start = time.time()
    deadline = t_start + args.max_hours * 3600 if args.max_hours else None
    epoch_times: list[float] = []
    scores: list[float] = []
    for h in history:
        if h.get("val_score") is not None:
            scores.append(h["val_score"])

    for epoch in range(start_epoch, args.epochs):
        # Ri-semina la catena di transform: vedi epoch_seed().
        seed = epoch_seed(args.exp, args.fold, epoch)
        try:
            train_loader.dataset.transform.set_random_state(seed=seed)
        except AttributeError:
            pass

        t_ep = time.time()
        model.train()
        epoch_loss, step = 0.0, 0

        for batch in train_loader:
            step += 1
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                outputs = model(images)
                loss = deep_supervision_loss(outputs, labels, loss_function)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += float(loss.detach())

        epoch_loss /= max(step, 1)
        sched.step()
        lr_now = sched.get_last_lr()[0]

        # --- Validazione -----------------------------------------------------
        # Saltata all'inizio: il log storico mostra Dice 0.0004 all'epoca 2.
        # Validare li' costa tempo pieno per un'informazione nulla.
        rec = {"epoch": epoch + 1, "train_loss": epoch_loss, "lr": lr_now,
               "val_dice": None, "val_score": None}
        # val_loader None = modello finale su tutti i casi: niente validazione,
        # e quindi nessuna selezione del checkpoint. Si consegna final.pth.
        do_val = (val_loader is not None
                  and (epoch + 1) >= args.val_start
                  and ((epoch + 1) % args.val_every == 0
                       or epoch + 1 == args.epochs))
        if do_val:
            vm = validate(model, val_loader, device, use_amp,
                          tuple(args.patch), config.SW_BATCH_SIZE,
                          args.val_overlap, dice_metric, post_pred, post_label)
            score = selection_score(vm["val_dice"], vm["val_hit_rate"],
                                    vm["val_fp_per_scan"])
            rec.update(vm)
            rec["val_score"] = score
            scores.append(score)

        rec["secs"] = time.time() - t_ep
        epoch_times.append(rec["secs"])
        history.append(rec)

        msg = (f"[epoca {epoch+1}/{args.epochs}] loss {epoch_loss:.4f} | "
               f"lr {lr_now:.2e} | {rec['secs']:.1f}s")
        if do_val:
            msg += (f" | dice {rec['val_dice']:.4f} hit {rec['val_hit_rate']:.2f} "
                    f"fp {rec['val_fp_per_scan']:.2f} score {rec['val_score']:.4f}")
        print(msg, flush=True)

        # --- Salvataggi -------------------------------------------------------
        common = dict(
            exp=args.exp, fold=args.fold, epoch=epoch + 1,
            model_state_dict=model.state_dict(),
            optimizer_state_dict=optimizer.state_dict(),
            scheduler_state_dict=sched.state_dict(),
            scaler_state_dict=scaler.state_dict(),
            history=history, rng=rng_state(),
            elapsed_seconds=elapsed_prev + (time.time() - t_start),
            config=run_config(args),
            torch_version=torch.__version__, git_commit=git_commit(),
        )

        # Selezione su MEDIA MOBILE, non sul valore singolo: e' cio' che evita
        # di eleggere un picco di rumore come "miglior modello".
        if do_val:
            smoothed = moving_average(scores, k=3)
            if smoothed > best_score:
                best_score, best_epoch = smoothed, epoch + 1
                save_checkpoint(best_path, best_metric=best_score,
                                best_epoch=best_epoch, **common)
                print(f"    -> nuovo best (media mobile {best_score:.4f})", flush=True)

        save_checkpoint(last_path, best_metric=best_score, best_epoch=best_epoch,
                        **common)

        # status.json aggiornato a ogni validazione, non solo a fine fold:
        # altrimenti `run_cv.py --status` mostra "da fare" per tutta la durata
        # del run, che e' proprio quando lo si vuole consultare.
        if do_val:
            write_status(args, run_dir, done=False, epoch=epoch + 1,
                         best_score=best_score, best_epoch=best_epoch,
                         last=rec)

        # --- Uscita pulita ----------------------------------------------------
        if _stop_requested:
            print(f"[train] Interrotto all'epoca {epoch+1}. "
                  f"Riprendi con: --exp {args.exp} --fold {args.fold} --resume")
            write_status(args, run_dir, done=False, epoch=epoch + 1,
                         best_score=best_score, best_epoch=best_epoch)
            return {"done": False, "epochs": epoch + 1, "reason": "sigint"}

        if deadline is not None:
            # Stima prudente (1.2x la media recente): meglio fermarsi un'epoca
            # prima che superare il tetto di sessione a meta' di un'epoca.
            eta = 1.2 * (sum(epoch_times[-5:]) / len(epoch_times[-5:]))
            if time.time() + eta > deadline:
                print(f"[train] Tetto di {args.max_hours} h raggiunto all'epoca "
                      f"{epoch+1}/{args.epochs}. Riprendi con --resume.")
                write_status(args, run_dir, done=False, epoch=epoch + 1,
                             best_score=best_score, best_epoch=best_epoch)
                return {"done": False, "epochs": epoch + 1, "reason": "max_hours"}

    # --- Fine fold -----------------------------------------------------------
    save_checkpoint(final_path, best_metric=best_score, best_epoch=best_epoch,
                    exp=args.exp, fold=args.fold, epoch=args.epochs,
                    model_state_dict=model.state_dict(),
                    optimizer_state_dict=optimizer.state_dict(),
                    scheduler_state_dict=sched.state_dict(),
                    scaler_state_dict=scaler.state_dict(),
                    history=history, rng=rng_state(),
                    elapsed_seconds=elapsed_prev + (time.time() - t_start),
                    config=run_config(args), torch_version=torch.__version__,
                    git_commit=git_commit())

    write_history_csv(os.path.join(run_dir, "history.csv"), history)
    write_status(args, run_dir, done=True, epoch=args.epochs,
                 best_score=best_score, best_epoch=best_epoch)

    # history puo' non contenere alcuna validazione: col modello finale
    # (fold -1, addestrato su tutti i casi) non esiste un validation set.
    last_val = [h for h in history if h.get("val_dice") is not None]
    if last_val:
        print(f"\n[train] fold {args.fold} completo. "
              f"Dice finale {last_val[-1]['val_dice']:.4f} | "
              f"best (media mobile) {best_score:.4f} @ epoca {best_epoch}")
        print("[train] final.pth e' il modello primario; best.pth resta per confronto.")
    else:
        print(f"\n[train] MODELLO FINALE completo: {len(train_files)} casi, "
              f"{args.epochs} epoche, nessuna validazione.")
        print("[train] La stima onesta delle sue prestazioni e' quella della "
              "cross-validation con la stessa ricetta, NON un numero misurato")
        print("[train] qui: questo modello ha visto in addestramento tutti i casi.")
    return {"done": True, "epochs": args.epochs}


# ---------------------------------------------------------------------------
#  Stato e log
# ---------------------------------------------------------------------------
def run_config(args) -> dict:
    return {k: (list(v) if isinstance(v, (list, tuple)) else v)
            for k, v in vars(args).items()}


def write_status(args, run_dir, done: bool, epoch: int, best_score: float,
                 best_epoch: int, last: dict | None = None) -> None:
    """status.json per fold, letto da run_cv.py per sapere cosa resta da fare."""
    path = os.path.join(os.path.dirname(run_dir), "status.json")
    data = {}
    if os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)
        except json.JSONDecodeError:
            data = {}
    data[str(args.fold)] = {
        "done": done, "epoch": epoch, "epochs": args.epochs,
        "best_score": best_score, "best_epoch": best_epoch,
        "updated": datetime.datetime.now().replace(microsecond=0).isoformat(),
        # Ultime metriche di validazione: rendono `--status` informativo
        # mentre il run e' ancora in corso.
        "last_val_dice": (last or {}).get("val_dice"),
        "last_train_loss": (last or {}).get("train_loss"),
        "secs_per_epoch": (last or {}).get("secs"),
    }
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
    os.replace(tmp, path)


def write_history_csv(path: str, history: list[dict]) -> None:
    import csv
    fields = ["epoch", "train_loss", "lr", "val_dice", "val_hit_rate",
              "val_fp_per_scan", "val_score", "secs"]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(history)


# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Addestramento di un fold, riprendibile.")
    ap.add_argument("--exp", required=True, help="nome dell'esperimento")
    ap.add_argument("--fold", type=int, required=True,
                    help="0-4 per un fold della CV; -1 per il MODELLO FINALE "
                         "addestrato su tutti i casi (senza validazione)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--max-hours", type=float, default=None,
                    help="esce in modo pulito prima di superare questo tetto")
    ap.add_argument("--epochs", type=int, default=250)
    ap.add_argument("--arch", default="unet",
                    choices=("unet", "dynunet", "segresnetds"))
    ap.add_argument("--no-deep-supervision", action="store_true")
    ap.add_argument("--patch", type=int, nargs=3, default=list(config.PATCH_SIZE))
    ap.add_argument("--batch-size", type=int, default=config.TRAIN_BATCH_SIZE)
    ap.add_argument("--samples-per-volume", type=int, default=config.SAMPLES_PER_VOLUME)
    ap.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    ap.add_argument("--optimizer", default="adamw", choices=("adamw", "sgd"))
    ap.add_argument("--schedule", default="cosine", choices=("cosine", "poly"))
    ap.add_argument("--augment", default="legacy",
                    choices=("legacy", "anatomic", "anatomic_allflip"),
                    help="legacy = augmentation storica (con RandRotate90)")
    ap.add_argument("--lung-roi", action="store_true",
                    help="ritaglia sul bounding box polmonare (preprocess.py)")
    ap.add_argument("--source", default="msd",
                    choices=("msd", "lidc", "nsclc", "lidc+nsclc"),
                    help="msd = MSD Task06 (i 63 casi, con la CV a 5 fold). "
                         "lidc / nsclc / lidc+nsclc = PRETRAINING su dati "
                         "esterni, senza validazione. I due sono "
                         "complementari per dimensione della lesione "
                         "(LIDC mediana 1.8 cm3, NSCLC 78 cm3): 'lidc+nsclc' "
                         "copre tutto lo spettro")
    ap.add_argument("--init-from", default=None, metavar="CKPT",
                    help="parte dai pesi di un checkpoint (solo i pesi): e' il "
                         "fine-tuning dopo un pretraining")
    ap.add_argument("--cache-rate", type=float, default=1.0,
                    help="quota di volumi tenuti in RAM da CacheDataset. "
                         "MISURATO: dopo Spacingd a 1.5 mm un caso LIDC occupa "
                         "~50 MB, quindi 222 casi stanno in ~11 GB e 1.0 va "
                         "bene. Abbassarlo solo se la RAM e' occupata da altro")
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--val-every", type=int, default=10)
    ap.add_argument("--val-start", type=int, default=30,
                    help="prima epoca in cui si valida")
    ap.add_argument("--val-overlap", type=float, default=0.25)
    return ap


def main():
    args = build_parser().parse_args()
    splits_mod.load_splits()           # errore subito se la partizione non esiste
    train_one_fold(args)


if __name__ == "__main__":
    main()
