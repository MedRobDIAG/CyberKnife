# ============================================================================
#  preprocess.py - Precalcolo delle ROI polmonari (bounding box) per caso.
#
#  A COSA SERVE
#  Il training e l'inferenza lavorano oggi sull'intero campo di vista della CT:
#  CropForegroundd, che dovrebbe ridurlo, e' un no-op misurato (la soglia "> 0"
#  dopo il rescale HU include anche il lettino, quindi ritaglia due voxel).
#  Restringere la ricerca al torace elimina cupola epatica, spalle, braccia e
#  lettino, cioe' le regioni dove nascono i falsi positivi, e dimezza il numero
#  di finestre di sliding window.
#
#  PERCHE' IL BOUNDING BOX E NON LA MASCHERA
#  Le maschere lung_r/lung_l di organs.py sono maschere d'ARIA (hu < -400
#  dentro il corpo, organs.py:187). Un tumore solido sta intorno a 0-60 HU:
#  nella maschera polmonare e' un BUCO, non un contenuto. Misurato su questi 63
#  casi, solo il ~35% dei voxel tumorali cade dentro la maschera grezza:
#  moltiplicare per essa cancellerebbe i due terzi del bersaglio.
#  Il bounding box invece contiene il tumore per costruzione, perche' e'
#  l'inviluppo del torace: questo script lo VERIFICA caso per caso sul ground
#  truth e riporta il margine effettivamente necessario.
#
#  Il box e' salvato in COORDINATE MONDO (mm), non in indici: dev'essere
#  valido su qualunque griglia (CT originale, griglia ricampionata del
#  training, griglia isotropica della dosimetria), e gli indici non lo sono.
#
#  USO:
#      python preprocess.py                 # calcola su imagesTr + imagesTs
#      python preprocess.py --verify        # verifica il contenimento sul GT
# ============================================================================
import os
import glob
import json
import time
import datetime

import numpy as np
import nibabel as nib
from scipy import ndimage

import config
import organs


ROIS_PATH = os.path.join(config.OUTPUT_DIR, "lung_rois.json")

#: Spacing per la segmentazione degli organi. 3 mm bastano per un bounding box
#: (si cerca un inviluppo, non un contorno) e costano circa meta' di 2 mm.
ROI_SPACING_MM = 3.0

#: Margine aggiunto al box in millimetri. 20 mm coprono le lesioni pleuriche e
#: mediastiniche che sporgono oltre il parenchima aerato.
ROI_MARGIN_MM = 20.0


def _index_bbox(mask):
    """Bounding box (min, max inclusi) di una maschera, in indici."""
    if not mask.any():
        return None
    sl = ndimage.find_objects(mask.astype(np.uint8))[0]
    lo = np.array([s.start for s in sl], dtype=np.float64)
    hi = np.array([s.stop - 1 for s in sl], dtype=np.float64)
    return lo, hi


def _bbox_to_world(lo, hi, affine):
    """Box in indici -> box asse-allineato in coordinate mondo.

    Si trasformano tutti gli 8 VERTICI e si prende min/max: con un'affine
    ruotata o con assi invertiti (frequente: molte CT hanno spacing negativo su
    x/y) trasformare solo lo spigolo minimo e quello massimo darebbe un box
    ribaltato o troppo piccolo.
    """
    A = np.asarray(affine, dtype=np.float64)
    corners = np.array([[lo[0] if b & 1 else hi[0],
                         lo[1] if b & 2 else hi[1],
                         lo[2] if b & 4 else hi[2]] for b in range(8)]).T
    world = (A[:3, :3] @ corners) + A[:3, 3:4]
    return world.min(axis=1), world.max(axis=1)


def lung_roi_world(ct_path: str, spacing_mm: float = ROI_SPACING_MM) -> dict:
    """Bounding box polmonare in coordinate mondo per una CT."""
    img = nib.load(ct_path)
    ct = np.asanyarray(img.dataobj, dtype=np.float32)
    masks, grid_affine, volumes = organs.segment_organs(ct, img.affine, spacing_mm)

    lungs = masks["lung_r"] | masks["lung_l"]
    box = _index_bbox(lungs)
    if box is None:
        return {"ok": False, "reason": "nessun polmone segmentato"}

    lo_w, hi_w = _bbox_to_world(*box, grid_affine)
    return {
        "ok": True,
        "world_min": lo_w.tolist(),
        "world_max": hi_w.tolist(),
        "lung_cm3": float(volumes["lung_r"] + volumes["lung_l"]),
        "roi_spacing_mm": spacing_mm,
    }


def required_margin_mm(label_path: str, roi: dict) -> float:
    """Quanto margine servirebbe perche' il box contenga TUTTO il tumore.

    Valore <= 0 significa che il tumore e' gia' strettamente dentro il box
    (in modulo, di quanto avanza). E' la verifica che rende sicuro l'uso del
    box: un ritaglio che tagliasse via una parte del bersaglio introdurrebbe un
    tetto invalicabile alla sensibilita', e lo farebbe in silenzio.
    """
    img = nib.load(label_path)
    data = np.asanyarray(img.dataobj)
    idx = np.array(np.nonzero(data > 0), dtype=np.float64)
    if idx.size == 0:
        return float("nan")

    A = np.asarray(img.affine, dtype=np.float64)
    world = (A[:3, :3] @ idx) + A[:3, 3:4]
    lo = np.asarray(roi["world_min"], dtype=np.float64)
    hi = np.asarray(roi["world_max"], dtype=np.float64)

    # Sforamento massimo su un qualunque asse, in entrambe le direzioni.
    over_lo = (lo[:, None] - world).max()
    over_hi = (world - hi[:, None]).max()
    return float(max(over_lo, over_hi))


def build(images_dirs=None, verify: bool = True, out_path: str = ROIS_PATH) -> dict:
    images_dirs = images_dirs or [
        config.IMAGES_DIR,
        os.path.join(config.DATA_ROOT, "imagesTs"),
    ]

    paths = []
    for d in images_dirs:
        if os.path.isdir(d):
            paths += [p for p in sorted(glob.glob(os.path.join(d, "*.nii.gz")))
                      if not os.path.basename(p).startswith("._")]
    print(f"[preprocess] {len(paths)} volumi da processare "
          f"(spacing {ROI_SPACING_MM} mm)")

    rois, margins = {}, {}
    t0 = time.time()
    for i, p in enumerate(paths, 1):
        cid = os.path.basename(p).replace(".nii.gz", "")
        try:
            roi = lung_roi_world(p)
        except Exception as exc:
            roi = {"ok": False, "reason": f"{type(exc).__name__}: {exc}"}
        rois[cid] = roi

        note = ""
        if verify and roi.get("ok"):
            lbl = os.path.join(config.LABELS_DIR, f"{cid}.nii.gz")
            if os.path.exists(lbl):
                mg = required_margin_mm(lbl, roi)
                margins[cid] = mg
                roi["gt_margin_mm"] = round(mg, 2)
                note = f"  margine GT: {mg:+7.2f} mm"

        size = ""
        if roi.get("ok"):
            ext = np.asarray(roi["world_max"]) - np.asarray(roi["world_min"])
            size = f"  box {ext[0]:.0f}x{ext[1]:.0f}x{ext[2]:.0f} mm"
        print(f"  [{i:>3}/{len(paths)}] {cid}{size}{note}"
              if roi.get("ok") else
              f"  [{i:>3}/{len(paths)}] {cid}  FALLITO: {roi.get('reason')}")

    payload = {
        "created": datetime.datetime.now().replace(microsecond=0).isoformat(),
        "roi_spacing_mm": ROI_SPACING_MM,
        "margin_mm": ROI_MARGIN_MM,
        "source": "organs.segment_organs -> bbox(lung_r | lung_l)",
        "rois": rois,
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    dt = time.time() - t0
    print(f"\n[preprocess] Scritto {out_path} in {dt/60:.1f} min "
          f"({dt/max(len(paths),1):.1f} s/caso)")

    if margins:
        v = np.array(list(margins.values()))
        worst = max(margins, key=margins.get)
        print(f"\n[preprocess] VERIFICA DI CONTENIMENTO su {len(v)} casi etichettati")
        print(f"  margine necessario: max {v.max():+.2f} mm  (caso {worst})")
        print(f"                      mediana {np.median(v):+.2f} mm")
        n_bad = int((v > ROI_MARGIN_MM).sum())
        if n_bad:
            print(f"  ATTENZIONE: {n_bad} casi eccedono il margine di "
                  f"{ROI_MARGIN_MM} mm -> {[c for c in margins if margins[c] > ROI_MARGIN_MM]}")
        else:
            print(f"  OK: tutti i tumori stanno nel box + {ROI_MARGIN_MM} mm.")
    return payload


def load_rois(path: str = ROIS_PATH) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"ROI non trovate: {path}\nGenerale con: python preprocess.py")
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def roi_slices(case_id: str, shape, affine, rois: dict | None = None,
               margin_mm: float = ROI_MARGIN_MM):
    """Ritaglio da applicare a un volume su una griglia qualunque.

    Restituisce una tupla di slice, oppure None se la ROI non e' disponibile
    (in tal caso il chiamante deve usare il volume intero: meglio piu' lento
    che sbagliato).
    """
    rois = rois or load_rois()
    roi = rois["rois"].get(case_id)
    if not roi or not roi.get("ok"):
        return None

    lo = np.asarray(roi["world_min"], dtype=np.float64) - margin_mm
    hi = np.asarray(roi["world_max"], dtype=np.float64) + margin_mm

    # Gli 8 vertici del box in mondo -> indici di QUESTA griglia, poi min/max:
    # stesso motivo di _bbox_to_world, al contrario.
    corners = np.array([[lo[0] if b & 1 else hi[0],
                         lo[1] if b & 2 else hi[1],
                         lo[2] if b & 4 else hi[2]] for b in range(8)]).T
    inv = np.linalg.inv(np.asarray(affine, dtype=np.float64))
    idx = (inv[:3, :3] @ corners) + inv[:3, 3:4]

    out = []
    for a in range(3):
        start = int(max(np.floor(idx[a].min()), 0))
        stop = int(min(np.ceil(idx[a].max()) + 1, shape[a]))
        if stop <= start:                     # ROI fuori griglia: niente crop
            return None
        out.append(slice(start, stop))
    return tuple(out)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Precalcolo delle ROI polmonari.")
    ap.add_argument("--no-verify", action="store_true",
                    help="salta il controllo di contenimento sul ground truth")
    ap.add_argument("--out", default=ROIS_PATH)
    ap.add_argument("--train-only", action="store_true",
                    help="solo imagesTr (salta i 32 volumi non etichettati)")
    args = ap.parse_args()

    dirs = [config.IMAGES_DIR] if args.train_only else None
    build(images_dirs=dirs, verify=not args.no_verify, out_path=args.out)
