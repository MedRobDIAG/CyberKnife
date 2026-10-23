# ============================================================================
#  splits.py - Definizione UNA VOLTA SOLA della cross-validation a 5 fold.
#
#  PERCHE' ESISTE QUESTO MODULO
#  Lo split precedente (dataset.py) prendeva il primo 20% della lista ordinata
#  come validation: 12 volumi fissi, nessuna cross-validation. Su 12 casi il
#  Dice oscilla di +-0.02 da un'epoca all'altra (si vede in outputs/train.log),
#  cioe' piu' del guadagno atteso da quasi ogni modifica che vorremmo provare.
#  Con quello strumento non e' possibile dire se una modifica ha funzionato.
#
#  Qui la partizione viene calcolata una volta, scritta su disco e poi LETTA
#  VERBATIM da ogni esperimento. Non e' un dettaglio burocratico: se lo split
#  potesse cambiare fra un esperimento e l'altro, i confronti fra ablazioni non
#  sarebbero appaiati e i test statistici sarebbero invalidi.
#
#  STRATIFICAZIONE
#  I volumi tumorali di questo dataset spaziano da 0.74 a ~370 cm3. Il Dice e'
#  fortemente correlato alla dimensione della lesione (una lesione grande perdona
#  un errore di bordo di 2 voxel, una piccola no). Con fold casuali, un fold
#  potrebbe raccogliere per caso le lesioni piu' grandi e risultare molto piu'
#  "facile" degli altri: la varianza fra fold misurerebbe la fortuna del
#  sorteggio invece della varianza del modello.
#
#  Non c'e' scikit-learn nel venv (e non vale la pena aggiungerlo per una
#  funzione): la stratificazione si fa a mano ordinando per volume e assegnando
#  i fold a SERPENTINA (0,1,2,3,4,4,3,2,1,0,...). Su una lista ordinata questo
#  distribuisce l'intero intervallo di dimensioni su ogni fold e compensa il
#  gradiente residuo alternando il verso a ogni passata.
#
#  USO:
#      python splits.py            # (ri)genera outputs/splits_5fold.json
#      python splits.py --check    # verifica un file esistente senza riscriverlo
# ============================================================================
import os
import json
import datetime

import numpy as np
import nibabel as nib

import config
from dataset import build_data_dicts


#: Percorso canonico della partizione. Va COMMITTATO nel repo.
SPLITS_PATH = os.path.join(config.OUTPUT_DIR, "splits_5fold.json")

N_FOLDS = 5


def case_id(path: str) -> str:
    """'.../lung_001.nii.gz' -> 'lung_001'."""
    base = os.path.basename(path)
    for ext in (".nii.gz", ".nii"):
        if base.endswith(ext):
            return base[: -len(ext)]
    return os.path.splitext(base)[0]


def tumor_volume_cm3(label_path: str) -> float:
    """Volume tumorale di un caso, in cm3.

    Si legge la label con nibabel e si moltiplica il conteggio dei voxel non
    nulli per il volume del voxel, ricavato dal determinante della parte lineare
    dell'affine (che e' il volume del parallelepipedo di un voxel in mm3, valido
    anche per affine non diagonali/ruotate).
    """
    img = nib.load(label_path)
    # dataobj + asanyarray evita di materializzare due copie del volume.
    data = np.asanyarray(img.dataobj)
    n_vox = int(np.count_nonzero(data))
    vox_mm3 = abs(np.linalg.det(np.asarray(img.affine)[:3, :3]))
    return n_vox * vox_mm3 / 1000.0


def case_records(images_dir: str = config.IMAGES_DIR,
                 labels_dir: str = config.LABELS_DIR,
                 verbose: bool = True) -> list[dict]:
    """Elenco dei casi etichettati con il rispettivo volume tumorale.

    Riusa build_data_dicts (dataset.py) per l'accoppiamento image/label e il
    filtro dei file '._' di macOS, cosi' esiste una sola definizione di "quali
    file compongono il dataset".
    """
    pairs = build_data_dicts(images_dir, labels_dir)
    records = []
    for i, p in enumerate(pairs, 1):
        cid = case_id(p["image"])
        assert cid == case_id(p["label"]), (
            f"Immagine e label non appaiate: {p['image']} / {p['label']}"
        )
        vol = tumor_volume_cm3(p["label"])
        records.append({"id": cid, "image": p["image"], "label": p["label"],
                        "tumor_cm3": round(vol, 4)})
        if verbose:
            print(f"  [{i:>2}/{len(pairs)}] {cid}  {vol:8.2f} cm3")
    return records


def _serpentine_folds(n_items: int, n_folds: int) -> list[int]:
    """Assegnazione a serpentina: 0,1,..,k-1,k-1,..,1,0,0,1,.. per n_items.

    Applicata a una lista ORDINATA per volume, dà a ogni fold un campione
    dell'intero intervallo di dimensioni. Il verso alternato evita che il fold 0
    riceva sistematicamente l'elemento piu' piccolo di ogni gruppo di 5, che
    introdurrebbe un bias monotono fra i fold.
    """
    out = []
    for i in range(n_items):
        pos = i % n_folds
        # Passate dispari percorse al contrario.
        out.append(pos if (i // n_folds) % 2 == 0 else n_folds - 1 - pos)
    return out


def make_splits(n_folds: int = N_FOLDS, seed: int = config.SEED,
                out_path: str = SPLITS_PATH, verbose: bool = True) -> dict:
    """Calcola la partizione e la scrive su disco. Deterministica."""
    if verbose:
        print("[splits] Lettura dei volumi tumorali...")
    records = case_records(verbose=verbose)

    # Ordinamento per volume; l'id come chiave secondaria rende l'ordine totale
    # e quindi il risultato riproducibile anche a parita' di volume.
    records.sort(key=lambda r: (r["tumor_cm3"], r["id"]))

    fold_of = {}
    for rec, f in zip(records, _serpentine_folds(len(records), n_folds)):
        fold_of[rec["id"]] = f

    all_ids = sorted(fold_of)          # ordine alfabetico, stabile nel JSON
    folds = []
    for f in range(n_folds):
        val = [c for c in all_ids if fold_of[c] == f]
        train = [c for c in all_ids if fold_of[c] != f]
        folds.append({"fold": f, "train": train, "val": val})

    payload = {
        "seed": seed,
        "n_folds": n_folds,
        "stratify": "tumor_cm3",
        "method": "sort by tumor_cm3, serpentine assignment",
        "created": datetime.datetime.now().replace(microsecond=0).isoformat(),
        "n_cases": len(records),
        "folds": folds,
        "case_volumes": {r["id"]: r["tumor_cm3"] for r in sorted(
            records, key=lambda r: r["id"])},
    }

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    if verbose:
        print(f"\n[splits] Scritto {out_path}")
        summarize(payload)
    return payload


def load_splits(path: str = SPLITS_PATH) -> dict:
    """Legge la partizione. Errore esplicito se non esiste: nessun esperimento
    deve poter inventarsi uno split al volo."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Partizione non trovata: {path}\nGenerala con: python splits.py"
        )
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def fold_files(fold: int, splits: dict | None = None,
               images_dir: str = config.IMAGES_DIR,
               labels_dir: str = config.LABELS_DIR
               ) -> tuple[list[dict], list[dict]]:
    """(train_files, val_files) per un fold, nel formato dict di MONAI."""
    splits = splits or load_splits()
    if not 0 <= fold < splits["n_folds"]:
        raise ValueError(f"fold {fold} fuori intervallo 0..{splits['n_folds']-1}")

    entry = splits["folds"][fold]
    assert entry["fold"] == fold, "JSON degli split corrotto: fold fuori ordine"

    def _to_dicts(ids):
        out = []
        for cid in ids:
            img = os.path.join(images_dir, f"{cid}.nii.gz")
            lbl = os.path.join(labels_dir, f"{cid}.nii.gz")
            if not (os.path.exists(img) and os.path.exists(lbl)):
                raise FileNotFoundError(f"Caso '{cid}' assente su disco: {img}")
            out.append({"image": img, "label": lbl, "id": cid})
        return out

    return _to_dicts(entry["train"]), _to_dicts(entry["val"])


def summarize(splits: dict) -> None:
    """Stampa il bilanciamento dei fold: e' la verifica che la stratificazione
    abbia funzionato."""
    vols = splits["case_volumes"]
    print(f"\n  {'fold':>4} {'n_val':>6} {'mediana':>9} {'min':>8} {'max':>9} "
          f"{'somma':>10}")
    print("  " + "-" * 50)
    for entry in splits["folds"]:
        v = np.array([vols[c] for c in entry["val"]])
        print(f"  {entry['fold']:>4} {len(v):>6} {np.median(v):>9.2f} "
              f"{v.min():>8.2f} {v.max():>9.2f} {v.sum():>10.1f}")
    print("  " + "-" * 50)
    all_v = np.array(list(vols.values()))
    print(f"  tutti {len(all_v):>5} {np.median(all_v):>9.2f} {all_v.min():>8.2f} "
          f"{all_v.max():>9.2f} {all_v.sum():>10.1f}   (cm3)")


def check(splits: dict) -> None:
    """Invarianti della partizione. Falliscono rumorosamente, non in silenzio."""
    n_folds = splits["n_folds"]
    all_ids = set(splits["case_volumes"])

    seen = []
    for entry in splits["folds"]:
        val, train = set(entry["val"]), set(entry["train"])
        assert not (val & train), f"fold {entry['fold']}: caso sia in train che in val"
        assert val | train == all_ids, f"fold {entry['fold']}: casi mancanti"
        seen.extend(entry["val"])

    # Ogni caso deve comparire in ESATTAMENTE un fold di validazione: e' cio' che
    # rende l'insieme delle predizioni di validazione un vero out-of-fold sui 63.
    assert len(seen) == len(all_ids), (
        f"{len(seen)} casi di validazione per {len(all_ids)} casi totali")
    assert set(seen) == all_ids, "un caso non compare in nessun fold di validazione"
    assert len(splits["folds"]) == n_folds

    print(f"[splits] OK: {len(all_ids)} casi, {n_folds} fold, "
          f"ogni caso validato esattamente una volta.")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Genera/verifica la CV a 5 fold.")
    ap.add_argument("--check", action="store_true",
                    help="verifica il file esistente senza rigenerarlo")
    ap.add_argument("--folds", type=int, default=N_FOLDS)
    ap.add_argument("--out", default=SPLITS_PATH)
    args = ap.parse_args()

    data = load_splits(args.out) if args.check else make_splits(
        n_folds=args.folds, out_path=args.out)
    if args.check:
        summarize(data)
    check(data)
