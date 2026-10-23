# ============================================================================
#  uncertainty.py - Quanto la rete e' sicura del tumore che ha trovato.
#
#  IL PROBLEMA
#  -----------
#  Il piano tratta la maschera predetta come se fosse vera. Out-of-fold il
#  Dice e' 0.47 e l'errore di centroide e' bimodale: quando la rete sbaglia,
#  sbaglia di centimetri. Un sistema che pilota un acceleratore deve sapere
#  QUANDO e DOVE non fidarsi, e quel sapere deve arrivare fino al margine del
#  bersaglio e al DVH.
#
#  IL METODO: TTA CON LE TRASFORMAZIONI DELL'ADDESTRAMENTO
#  -------------------------------------------------------
#  Si ripete la predizione N volte su versioni perturbate della stessa CT e
#  si riportano le N mappe nello spazio di partenza. La dispersione misura
#  quanto la risposta dipende da dettagli che non dovrebbero contare (Wang et
#  al., Neurocomputing 2019). Le perturbazioni sono SOLO quelle viste in
#  addestramento (dataset._augment_anatomic): specchiamento destra-sinistra,
#  rotazioni e scale moderate, rumore. Una perturbazione mai vista non
#  misurerebbe l'incertezza, misurerebbe la distribuzione fuori campione.
#
#  PERCHE' NON UN INSIEME DI CHECKPOINT
#  ------------------------------------
#  Il piano iniziale prevedeva best/last/final di ogni fold. Misurato: last e
#  final sono lo STESSO file (epoca 250) e best dista al piu' 40 epoche - in
#  due fold su cinque e' identico. Non e' un insieme, e' un modello copiato.
#  I 5 modelli di fold insieme sarebbero un insieme vero, ma sui 63 casi
#  etichettati 4 su 5 li hanno visti in addestramento: la loro concordanza
#  sarebbe memoria, non confidenza. Per la VALIDAZIONE si usa quindi la TTA
#  del solo modello out-of-fold; per un paziente nuovo la stessa TTA sul
#  modello finale (stessa procedura, pesi diversi).
#
#  COSA SI MISURA
#  --------------
#    p_mean   probabilita' media              -> la maschera, come prima
#    p_std    deviazione standard fra i campioni
#    mi       informazione mutua (BALD): H(p_mean) - media H(p_i). E' la
#             parte di incertezza dovuta al MODELLO, non all'ambiguita'
#             intrinseca del voxel.
#  e per la lesione principale, campione per campione, volume e CENTROIDE:
#  la dispersione del centroide e' l'incertezza dell'isocentro in mm, cioe'
#  proprio il modo in cui il sistema fallisce.
# ============================================================================
import os
import math

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage

from monai.data import Dataset, DataLoader, MetaTensor, decollate_batch
from monai.inferers import SlidingWindowInferer

import config
import postproc
import predict as P

# Derivata da config.UNC_EXP: le mappe sono del modello che si valuta.
OUT_DIR = os.path.join(config.OUTPUT_DIR, "eval", f"{config.UNC_EXP}_unc")


# ---------------------------------------------------------------------------
#  Perturbazioni
# ---------------------------------------------------------------------------
def tta_params(n, seed=0, max_rot_deg=8.0, max_scale=0.08, noise_std=0.01):
    """Elenco di perturbazioni. La prima e' l'identita', la seconda il solo
    specchiamento: cosi' con n=2 si ritrova la TTA classica.

    Ampiezze MINORI di quelle di addestramento (15 gradi, 20%): si vuole
    restare dentro la distribuzione vista, non ai suoi bordi.
    """
    rng = np.random.default_rng(seed)
    out = [dict(flip=False, rot=(0.0, 0.0, 0.0), scale=(1.0, 1.0, 1.0),
                noise=0.0),
           dict(flip=True, rot=(0.0, 0.0, 0.0), scale=(1.0, 1.0, 1.0),
                noise=0.0)]
    for i in range(max(0, n - 2)):
        out.append(dict(
            flip=bool(i % 2),
            rot=tuple(float(v) for v in
                      np.deg2rad(rng.uniform(-max_rot_deg, max_rot_deg, 3))),
            scale=tuple(float(v) for v in
                        1.0 + rng.uniform(-max_scale, max_scale, 3)),
            noise=noise_std,
        ))
    return out[:n]


def _rot(rx, ry, rz):
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _theta(shape, spacing, rot, scale):
    """Matrice per affine_grid, costruita nello spazio FISICO.

    affine_grid lavora in coordinate normalizzate [-1, 1], diverse per ogni
    asse se il volume non e' cubico o i voxel non sono isotropi (qui 1.5 x
    1.5 x 2.0 mm). Una rotazione scritta direttamente in quelle coordinate
    deformerebbe il torace: si passa per i millimetri, E^-1 (R S) E.

    Ordine degli assi: per un tensore (N,C,X,Y,Z) la griglia di grid_sample
    indicizza (Z, Y, X) nelle sue componenti (x, y, z).
    """
    half = np.array(shape, float) * np.array(spacing, float) / 2.0  # X,Y,Z
    M = _rot(*rot) @ np.diag(scale)                                 # in XYZ
    perm = [2, 1, 0]                     # componenti della griglia -> assi
    E = np.diag(half[perm])
    Mg = M[np.ix_(perm, perm)]
    A = np.linalg.inv(E) @ Mg @ E
    return A


def _warp(x, A, mode):
    th = torch.zeros((1, 3, 4), dtype=x.dtype, device=x.device)
    th[0, :, :3] = torch.as_tensor(A, dtype=x.dtype, device=x.device)
    grid = F.affine_grid(th, list(x.shape), align_corners=False)
    return F.grid_sample(x, grid, mode=mode, padding_mode="border",
                         align_corners=False)


# ---------------------------------------------------------------------------
#  Campionamento
# ---------------------------------------------------------------------------
def _binary_entropy(p):
    p = p.clamp(1e-6, 1 - 1e-6)
    return -(p * torch.log(p) + (1 - p) * torch.log(1 - p))


def _primary_stats(p, affine, vox_mm3, cfg):
    """Volume (cm3) e centroide (mm) della lesione principale di un campione."""
    _, lesions = postproc.probs_to_mask(p, cfg, voxel_volume_mm3=vox_mm3,
                                        affine=affine)
    if not lesions:
        return 0.0, np.full(3, np.nan)
    L = lesions[0]
    return L["volume_cm3"], np.asarray(L["com_world"], float)


def to_original(vols, aff_src, shape_dst, aff_dst, chunk=16):
    """Ricampiona mappe (C, X, Y, Z) dallo spazio di rete a quello della CT.

    Sostituisce Invertd per le mappe di incertezza. Misurato: Invertd su 4
    canali a piena risoluzione (512 x 512 x 304) tocca picchi di 23 GB di RAM,
    abbastanza da mandare in swap la macchina. Qui la griglia di destinazione
    si costruisce a fette sulla GPU: ogni voxel CT va in millimetri con la sua
    affine, poi negli indici della rete con l'inversa dell'altra, e si
    interpola. La convenzione (indice intero = centro del voxel) e' quella di
    nibabel e di MONAI, e align_corners=False la rispetta.
    """
    dev = vols.device
    C, sx, sy, sz = vols.shape
    M = torch.as_tensor(np.linalg.inv(aff_src) @ np.asarray(aff_dst),
                        dtype=torch.float32, device=dev)
    n_src = torch.tensor([sx, sy, sz], dtype=torch.float32, device=dev)
    X, Y, Z = shape_dst
    ii, jj = torch.meshgrid(torch.arange(X, device=dev, dtype=torch.float32),
                            torch.arange(Y, device=dev, dtype=torch.float32),
                            indexing="ij")
    out = np.empty((C, X, Y, Z), np.float32)
    src = vols[None].float()
    for k0 in range(0, Z, chunk):
        K = min(chunk, Z - k0)
        kk = torch.arange(k0, k0 + K, device=dev, dtype=torch.float32)
        pts = torch.stack([ii[..., None].expand(X, Y, K),
                           jj[..., None].expand(X, Y, K),
                           kk.expand(X, Y, K)], -1)
        s = pts @ M[:3, :3].T + M[:3, 3]
        u = 2.0 * (s + 0.5) / n_src - 1.0
        grid = u[..., [2, 1, 0]][None]
        r = F.grid_sample(src, grid, mode="bilinear", padding_mode="zeros",
                          align_corners=False)
        out[:, :, :, k0:k0 + K] = r[0].cpu().numpy()
    return out


def sample_case(model, image_path, params, device=None, cfg=None,
                inferer=None, invert="gpu"):
    """N predizioni perturbate di un caso.

    Restituisce le mappe riassuntive nello spazio CT ORIGINALE e, per ogni
    campione, volume e centroide della lesione principale.
    invert: "gpu" (to_original) oppure "monai" (Invertd, molta RAM).
    """
    device = device or torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    cfg = cfg or config.POSTPROC
    pre = P.get_pre_transforms()
    post = P.get_prob_post_transforms(pre)
    ds = Dataset(data=[{"image": image_path}], transform=pre)
    batch = next(iter(DataLoader(ds, batch_size=1, num_workers=0)))
    x0 = batch["image"].to(device)
    inferer = inferer or SlidingWindowInferer(
        roi_size=config.SW_ROI_SIZE, sw_batch_size=config.SW_BATCH_SIZE,
        overlap=config.SW_OVERLAP, mode=config.SW_BLEND_MODE)

    aff_pre = np.asarray(x0.affine[0].cpu().numpy() if x0.affine.ndim == 3
                         else x0.affine.cpu().numpy(), float)
    spacing = np.linalg.norm(aff_pre[:3, :3], axis=0)
    vox_mm3 = float(np.prod(spacing))
    shape = tuple(x0.shape[2:])
    gen = torch.Generator(device=x0.device).manual_seed(0)

    s1 = s2 = sh = None
    vols, cents = [], []
    with torch.no_grad():
        for prm in params:
            x = x0.as_tensor() if hasattr(x0, "as_tensor") else x0
            A = None
            if prm["flip"]:
                x = torch.flip(x, dims=[2])
            if any(abs(r) > 0 for r in prm["rot"]) or \
                    any(abs(s - 1) > 0 for s in prm["scale"]):
                A = _theta(shape, spacing, prm["rot"], prm["scale"])
                x = _warp(x, A, "bilinear")
            if prm["noise"] > 0:
                x = x + prm["noise"] * torch.randn(
                    x.shape, generator=gen, device=x.device, dtype=x.dtype)
            with torch.amp.autocast("cuda", enabled=(x.device.type == "cuda")):
                logits = inferer(inputs=x, network=model)
            p = torch.softmax(logits.float(), dim=1)[:, 1:2]
            if A is not None:
                p = _warp(p, np.linalg.inv(A), "bilinear")
            if prm["flip"]:
                p = torch.flip(p, dims=[2])
            p = p[0, 0]
            s1 = p if s1 is None else s1 + p
            s2 = p * p if s2 is None else s2 + p * p
            h = _binary_entropy(p)
            sh = h if sh is None else sh + h
            v, c = _primary_stats(p.cpu().numpy(), aff_pre, vox_mm3, cfg)
            vols.append(v)
            cents.append(c)

        n = len(params)
        mean = s1 / n
        std = torch.sqrt(torch.clamp(s2 / n - mean * mean, min=0.0))
        mi = torch.clamp(_binary_entropy(mean) - sh / n, min=0.0)
        if invert == "gpu":
            img = nib.load(image_path)
            affine = np.asarray(img.affine, np.float64)
            arr = to_original(torch.stack([mean, std, mi]), aff_pre,
                              tuple(img.shape[:3]), affine)
            arr = np.concatenate([1.0 - arr[:1], arr])
        else:
            stack = torch.stack([1.0 - mean, mean, std, mi])[None]
            pred = MetaTensor(stack, meta=dict(x0.meta))
            pred.applied_operations = x0.applied_operations
            batch["pred"] = pred
            out = [post(it) for it in decollate_batch(batch)][0]["pred"]
            arr = out.cpu().numpy().astype(np.float32)
            affine = np.asarray(out.affine.cpu().numpy(), dtype=np.float64)

    return {
        "p_mean": np.clip(arr[1], 0, 1), "p_std": np.clip(arr[2], 0, 1),
        "mi": np.clip(arr[3], 0, 1), "affine": affine,
        "volumes_cm3": np.asarray(vols), "centroids_mm": np.asarray(cents),
        "n_samples": n,
    }


# ---------------------------------------------------------------------------
#  Persistenza
# ---------------------------------------------------------------------------
def compact_maps(res, floor=P.STORE_FLOOR):
    """Le tre mappe nel solo bounding box dove p o std superano `floor`,
    quantizzate a 8 bit. E' la forma in cui le mappe dei 63 casi sono state
    salvate e VALIDATE (semaforo, calibrazione di tau): usarla anche per i
    progetti non cambia nulla di cio' che e' stato misurato. A piena
    risoluzione le mappe valgono 42 MB compressi per lung_001 (la rete non
    restituisce mai zeri esatti), cosi' 0.26 MB."""
    hot = (res["p_mean"] > floor) | (res["p_std"] > floor)
    out = dict(shape=np.array(res["p_mean"].shape, np.int32),
               affine=np.asarray(res["affine"], np.float64), floor=floor,
               volumes_cm3=np.asarray(res["volumes_cm3"]),
               centroids_mm=np.asarray(res["centroids_mm"]),
               n_samples=int(res["n_samples"]), empty=not hot.any())
    if out["empty"]:
        return out
    sl = ndimage.find_objects(hot.astype(np.uint8))[0]

    def q(a, scale):
        return np.clip(np.rint(a[sl] * scale), 0, 255).astype(np.uint8)
    out.update(offset=np.array([s.start for s in sl], np.int32),
               p=q(res["p_mean"], 255.0), std=q(res["p_std"], 510.0),
               mi=q(res["mi"], 255.0 / math.log(2.0)))
    return out


def expand_maps(c):
    """Inverso di compact_maps."""
    shape = tuple(int(v) for v in c["shape"])
    out = {"affine": np.asarray(c["affine"], np.float64),
           "volumes_cm3": np.asarray(c["volumes_cm3"]),
           "centroids_mm": np.asarray(c["centroids_mm"]),
           "n_samples": int(c["n_samples"])}
    for k in ("p_mean", "p_std", "mi"):
        out[k] = np.zeros(shape, np.float32)
    if not bool(c["empty"]):
        off = [int(v) for v in c["offset"]]
        sl = tuple(slice(o, o + n) for o, n in zip(off, c["p"].shape))
        out["p_mean"][sl] = c["p"] / 255.0
        out["p_std"][sl] = c["std"] / 510.0
        out["mi"][sl] = c["mi"] * math.log(2.0) / 255.0
    return out


def save_maps(path, res, floor=P.STORE_FLOOR):
    """Come predict.save_prob, con tre mappe nello stesso bounding box."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savez_compressed(path, **compact_maps(res, floor))
    return path


def load_maps(path):
    with np.load(path) as z:
        return expand_maps({k: z[k] for k in z.files})


def maps_path(case_id, out_dir=OUT_DIR):
    return os.path.join(out_dir, f"{case_id}.npz")


# ---------------------------------------------------------------------------
#  Segnali per caso
# ---------------------------------------------------------------------------
def case_signals(maps, cfg=None):
    """Segnali di incertezza, calcolabili senza ground truth."""
    cfg = cfg or config.POSTPROC
    p, aff = maps["p_mean"], maps["affine"]
    mask, lesions = postproc.probs_to_mask(p, cfg, affine=aff)
    cents = maps["centroids_mm"]
    vols = maps["volumes_cm3"]
    found = np.isfinite(cents).all(axis=1)
    out = {
        "n_samples": maps["n_samples"],
        "frac_campioni_con_lesione": float(found.mean()) if len(found) else 0.0,
    }
    if lesions:
        L = lesions[0]
        # Stessa etichettatura di postproc.probs_to_mask (sulla soglia, PRIMA
        # del filtro sul volume): rietichettare la maschera filtrata darebbe
        # numeri diversi, e L["label"] indicherebbe un'altra componente.
        lab, _ = ndimage.label(p >= cfg.prob_threshold)
        sel = lab == L["label"]
        ref = np.asarray(L["com_world"])
        # Dispersione dei centroidi dei campioni attorno a quello medio. Un
        # campione senza lesione conta come errore massimo (100 mm): la rete,
        # in quella versione della CT, non ha trovato niente.
        d = np.where(found, np.linalg.norm(
            np.where(found[:, None], cents, 0) - ref, axis=1), 100.0)
        near = (p > 0.2) & (p < 0.8)
        grown = ndimage.binary_dilation(sel, iterations=3)
        out.update({
            "volume_cm3": L["volume_cm3"],
            "prob_mean": L["prob_mean"],
            "std_lesione": float(maps["p_std"][sel].mean()),
            "mi_lesione": float(maps["mi"][sel].mean()),
            "frac_ambigua": float((near & grown).sum() / max(sel.sum(), 1)),
            "centroide_disp_mm": float(np.median(d)),
            "centroide_disp_max_mm": float(d.max()),
            "volume_cv": float(np.std(vols) / max(np.mean(vols), 1e-6)),
        })
    return out


# ---------------------------------------------------------------------------
#  Semaforo
# ---------------------------------------------------------------------------
#: Predizioni perturbate per caso: la stessa quantita' usata in validazione.
N_SAMPLES = 8

#: Oltre questa distanza (mm) fra il centroide medio e quello di ALMENO UNA
#: predizione perturbata, il bersaglio va rivisto: in qualche versione della
#: CT la rete ha scelto un'altra lesione. Scelta annidata sui 5 fold
#: (run_uncertainty.py --analyze): quattro fold su cinque scelgono 50 mm,
#: uno 100 mm. La procedura annidata accetta 43 casi su 63 con isocentro
#: corretto nel 95.3%, contro 53 e 94.3% del solo cancello sul volume: come
#: RIFIUTO il guadagno e' piccolo, per questo il segnale fa scattare il
#: giallo ("rivedi") e non il rosso. Con la soglia fissata a 50 mm, sui 53
#: accettati (stima ottimistica, stessi casi):
#:     sotto soglia (verde)   41 casi, isocentro corretto nel 97.6%
#:     sopra soglia (giallo)  12 casi, di cui 2 sbagliati
#: La dispersione MEDIANA, provata per prima, non serviva: mediana 0.85 mm
#: nei riusciti contro 1.31 nei falliti. Il fallimento e' un salto di pochi
#: campioni verso un altro candidato, e la mediana lo nasconde.
#:
#: RIVALIDATA il 15 ott 2026 sul modello 'finetune_all' (pretrain LIDC+NSCLC).
#: La scelta annidata ora preferisce 10 mm in 4 fold su 5, ma sui 63 casi 10 e
#: 50 mm producono un semaforo IDENTICO: la dispersione massima e' bimodale
#: (mediana 0.83 mm nei riusciti, 70 mm nei falliti, nulla in mezzo), quindi
#: qualunque soglia in quell'intervallo separa allo stesso modo. Si tiene 50.
#: Risultato col modello nuovo: verde 49 casi con isocentro giusto in 49/49,
#: giallo 5 (giusti 3), rosso 9 (giusti 3). Col modello precedente i verdi
#: erano 41 con un errore (lung_070, lesione scelta a 220 mm da quella vera).
DISP_REVIEW_MM = 50.0

#: Sotto questa frazione di predizioni perturbate che trovano ancora una
#: lesione, la rete "vede" il tumore solo in alcune versioni della CT.
MIN_FOUND_FRAC = 0.75


def traffic_light(signals, gate_ok, gate_reason=""):
    """(colore, motivi): 'verde' procedi, 'giallo' rivedi, 'rosso' rifiuta.

    Rosso e giallo non bloccano nulla (config.GATE_BLOCKS_TARGET): dicono
    all'operatore quanto fidarsi, con il motivo scritto.
    """
    reasons = []
    color = "verde"
    if not gate_ok:
        color = "rosso"
        reasons.append(gate_reason or "lesione sotto il volume minimo")
    found = signals.get("frac_campioni_con_lesione", 0.0)
    if found < MIN_FOUND_FRAC:
        color = "rosso"
        reasons.append(f"lesione trovata solo in {100 * found:.0f}% delle "
                       f"predizioni perturbate")
    disp = signals.get("centroide_disp_max_mm")
    if disp is not None and disp > DISP_REVIEW_MM:
        if color == "verde":
            color = "giallo"
        reasons.append(f"in almeno una predizione perturbata il bersaglio "
                       f"salta di {disp:.0f} mm: esiste un candidato "
                       f"alternativo (soglia {DISP_REVIEW_MM:.0f} mm)")
    return color, reasons
