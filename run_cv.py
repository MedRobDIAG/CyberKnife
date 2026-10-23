# ============================================================================
#  run_cv.py - Esegue la cross-validation completa entro un tetto di sessione.
#
#  Il vincolo operativo e' che nessuna sessione di training superi le ~8 ore.
#  Una 5-fold CV di una configurazione costosa non ci sta, quindi serve un
#  driver che: lanci i fold in ordine, ripartendo sempre con --resume; si
#  fermi quando il budget della sessione e' esaurito; e sappia, al lancio
#  successivo, cosa resta da fare leggendo status.json.
#
#  Il budget viene ripartito fra i fold RIMANENTI, non speso tutto sul primo:
#  cosi' una sessione interrotta lascia tutti i fold a un avanzamento
#  paragonabile invece di lasciarne uno completo e quattro a zero. Se un fold
#  finisce prima, il tempo avanzato torna nel budget dei successivi.
#
#  USO:
#      python run_cv.py --exp baseline_unet --max-hours 7.5 -- --arch unet
#      python run_cv.py --exp baseline_unet --status
# ============================================================================
import os
import sys
import json
import time
import argparse
import subprocess

import config


def status_path(exp: str) -> str:
    return os.path.join(config.OUTPUT_DIR, "runs", exp, "status.json")


def read_status(exp: str) -> dict:
    p = status_path(exp)
    if not os.path.exists(p):
        return {}
    try:
        with open(p, encoding="utf-8") as fh:
            return json.load(fh)
    except json.JSONDecodeError:
        return {}


def show_status(exp: str, folds: list[int]) -> None:
    st = read_status(exp)
    print(f"\n  Stato di '{exp}'")
    print(f"  {'fold':>5} {'stato':>10} {'epoche':>12} {'best':>9} {'@epoca':>7} "
          f"{'val_dice':>9} {'loss':>9}  aggiornato")
    print("  " + "-" * 88)
    for f in folds:
        e = st.get(str(f))
        if not e:
            print(f"  {f:>5} {'da fare':>10} {'-':>12} {'-':>9} {'-':>7} "
                  f"{'-':>9} {'-':>9}")
        else:
            state = "COMPLETO" if e["done"] else "in corso"
            progress = f"{e['epoch']}/{e['epochs']}"
            dice = e.get("last_val_dice")
            dice_s = f"{dice:.4f}" if isinstance(dice, (int, float)) else "-"
            loss = e.get("last_train_loss")
            loss_s = f"{loss:.4f}" if isinstance(loss, (int, float)) else "-"
            print(f"  {f:>5} {state:>10} {progress:>12} "
                  f"{e['best_score']:>9.4f} {e['best_epoch']:>7} "
                  f"{dice_s:>9} {loss_s:>9}  {e.get('updated','')}")
    print()


def main():
    ap = argparse.ArgumentParser(
        description="Cross-validation completa entro un tetto di sessione.")
    ap.add_argument("--exp", required=True)
    ap.add_argument("--folds", default="0-4",
                    help="'0-4' oppure '0,2,3'")
    ap.add_argument("--max-hours", type=float, default=7.5,
                    help="budget TOTALE della sessione, ripartito fra i fold")
    ap.add_argument("--status", action="store_true",
                    help="mostra lo stato e esce")
    ap.add_argument("--watch", type=int, nargs="?", const=30, default=None,
                    metavar="SEC",
                    help="ristampa lo stato ogni SEC secondi (default 30) "
                         "finche' tutti i fold non sono completi. Ctrl-C per uscire")
    ap.add_argument("train_args", nargs="*",
                    help="argomenti passati a train.py (dopo --)")
    args = ap.parse_args()

    if "-" in args.folds and "," not in args.folds:
        a, b = args.folds.split("-")
        folds = list(range(int(a), int(b) + 1))
    else:
        folds = [int(x) for x in args.folds.split(",") if x.strip()]

    if args.watch is not None:
        # Sola lettura di status.json: non interferisce col training in corso,
        # che gira in un altro processo e scrive quel file in modo atomico.
        import datetime as _dt
        try:
            while True:
                st = read_status(args.exp)
                done = all(st.get(str(f), {}).get('done') for f in folds)
                # ANSI: pulisce lo schermo e riporta il cursore in alto, cosi'
                # la tabella si aggiorna sul posto invece di scorrere via.
                print(chr(27) + "[2J" + chr(27) + "[H", end="")
                print(f"  {_dt.datetime.now().strftime('%H:%M:%S')} - "
                      f"aggiornamento ogni {args.watch}s (Ctrl-C per uscire)")
                show_status(args.exp, folds)
                if done:
                    print(f"  Tutti i fold di '{args.exp}' sono completi.\n")
                    return
                time.sleep(args.watch)
        except KeyboardInterrupt:
            print("\n  (interrotto: il training continua nel suo processo)\n")
            return

    if args.status:
        show_status(args.exp, folds)
        return

    st = read_status(args.exp)
    todo = [f for f in folds if not st.get(str(f), {}).get("done")]
    if not todo:
        print(f"[run_cv] '{args.exp}': tutti i fold {folds} sono completi.")
        show_status(args.exp, folds)
        return

    print(f"[run_cv] '{args.exp}': {len(todo)} fold da completare -> {todo}")
    print(f"[run_cv] budget di sessione: {args.max_hours} h")

    t_session = time.time()
    deadline = t_session + args.max_hours * 3600

    for i, fold in enumerate(todo):
        remaining_h = (deadline - time.time()) / 3600.0
        if remaining_h <= 0.05:
            print(f"[run_cv] Budget esaurito prima del fold {fold}. "
                  f"Rilancia lo stesso comando per continuare.")
            break

        # Quota per questo fold = tempo residuo diviso i fold ancora da fare.
        n_left = len(todo) - i
        quota = remaining_h / n_left
        print(f"\n{'='*70}\n[run_cv] fold {fold}  "
              f"(quota {quota:.2f} h, residuo sessione {remaining_h:.2f} h)\n{'='*70}")

        cmd = [sys.executable, "train.py", "--exp", args.exp,
               "--fold", str(fold), "--resume",
               "--max-hours", f"{quota:.4f}"] + list(args.train_args)
        print("[run_cv] " + " ".join(cmd), flush=True)
        rc = subprocess.call(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))
        if rc != 0:
            print(f"[run_cv] train.py e' uscito con codice {rc}. Mi fermo.")
            sys.exit(rc)

    show_status(args.exp, folds)
    st = read_status(args.exp)
    if all(st.get(str(f), {}).get("done") for f in folds):
        print(f"[run_cv] '{args.exp}' COMPLETO. Passo successivo: valutazione "
              f"out-of-fold con evaluate.py")


if __name__ == "__main__":
    main()
