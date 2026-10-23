# ============================================================================
#  nsclc.py - NSCLC-Radiomics (DICOM + RTSTRUCT) -> NIfTI: GTV + organi.
#
#  PERCHE' QUESTO DATASET E' DIVERSO DAGLI ALTRI DUE
#  MSD Task06 annota "tumore", LIDC-IDRI annota "nodulo". Questo annota il
#  **GTV**, cioe' la struttura che un oncologo radioterapista disegna PER
#  PIANIFICARE il trattamento: e' letteralmente il bersaglio che il nostro
#  planner deve colpire, non un'approssimazione. 422 pazienti, GTV-1 presente
#  in tutti quelli controllati.
#
#  MA POPOLA IL REGIME OPPOSTO A LIDC. Misurato su 80 pazienti:
#      volume GTV-1: p10 6.0, mediana 78.4, p90 217, max 661 cm3
#  cioe' 15 volte la mediana di MSD (5.2 cm3), e solo il 4% cade nel terzile
#  piccolo dove il modello fallisce. Serve quindi a migliorare la QUALITA' di
#  segmentazione delle lesioni grandi - che hanno isocentro 0.952 ma Dice 0.595,
#  cioe' il modello le trova sempre e le contorna male - non a curare il
#  deficit sulle piccole, per il quale c'e' LIDC. I due dataset sono
#  complementari: LIDC il piccolo, NSCLC il grande, MSD il mezzo.
#
#  GLI ORGANI A RISCHIO SONO UN SECONDO DELIVERABLE
#  Oltre a GTV-1 ci sono Lung-Left, Lung-Right, Spinal-Cord, Esophagus e Heart
#  delineati da un clinico. organs.py oggi li approssima con soglie HU e
#  dichiara esso stesso che il mediastino e' un criterio geometrico, non un
#  contorno clinico. Qui sono veri, e vengono salvati in un volume multi-etichetta
#  separato: servono per la dosimetria agli organi a rischio della Fase 2 e per
#  mostrare un modello 3D del paziente che non sia una soglia di densita'.
#
#  DIFFERENZA TECNICA RISPETTO A lidc.py
#  In LIDC i contorni XML sono in coordinate PIXEL. Qui ContourData e' in
#  coordinate PAZIENTE (mm): va riportato in indici con l'affine inversa prima
#  di riempire il poligono. Scambiare le due convenzioni produce maschere
#  completamente fuori sede.
#
#  USO (ogni passo e' interrompibile e riprendibile):
#    python nsclc.py --index                 # inventario, nessuna scrittura
#    python nsclc.py --convert --workers 6   # riprende da dove si era fermato
#    python nsclc.py --verify
# ============================================================================
import os
import csv
import glob
import json
import argparse
import statistics
import concurrent.futures as cf

import numpy as np
import nibabel as nib

import config


NSCLC_ROOT = os.environ.get("NSCLC_ROOT", os.path.join(
    r"C:\Users\ilgra\Documents\Progetto Medical", "nsclc_radiomics"))

OUT_ROOT = os.path.join(r"C:\Users\ilgra\Documents\Progetto Medical", "NSCLC_nifti")
OUT_IMAGES = os.path.join(OUT_ROOT, "imagesTr")
OUT_LABELS = os.path.join(OUT_ROOT, "labelsTr")
OUT_ORGANS = os.path.join(OUT_ROOT, "organs")
MANIFEST = os.path.join(OUT_ROOT, "manifest.json")

#: Struttura usata come bersaglio. gtv-2/3/4 esistono in alcuni pazienti e
#: corrispondono a localizzazioni aggiuntive (spesso linfonodali): si usa il
#: solo GTV-1, il tumore primario, perche' e' cio' che MSD annota.
TARGET_ROI = "gtv-1"

#: Organi a rischio -> valore nel volume multi-etichetta. L'ordine conta: in
#: caso di sovrapposizione fra contorni vince l'etichetta piu' alta, quindi si
#: mettono per ultimi gli organi piccoli e critici (midollo, esofago), che non
#: devono essere cancellati dal polmone che li circonda.
ORGAN_LABELS = {
    "lung-right": 1,
    "lung-left": 2,
    "heart": 3,
    "esophagus": 4,
    "spinal-cord": 5,
}


# ===========================================================================
#  Inventario
# ===========================================================================
def find_series(patient_dir):
    """(cartella CT, file RTSTRUCT) di un paziente, o (None, None)."""
    import pydicom

    ct_dir, rt_file = None, None
    best_n = 0
    for sd in glob.glob(os.path.join(patient_dir, "*", "*")):
        fs = glob.glob(os.path.join(sd, "*.dcm"))
        if not fs:
            continue
        try:
            d = pydicom.dcmread(fs[0], stop_before_pixels=True, force=True)
        except Exception:
            continue
        mod = str(getattr(d, "Modality", ""))
        if mod == "CT" and len(fs) > best_n:
            # Si tiene la serie CT con piu' slice: alcuni pazienti hanno anche
            # localizer o ricostruzioni parziali.
            ct_dir, best_n = sd, len(fs)
        elif mod == "RTSTRUCT" and rt_file is None:
            rt_file = fs[0]
    return ct_dir, rt_file


def roi_numbers(rt):
    """nome normalizzato -> ROINumber."""
    out = {}
    for roi in getattr(rt, "StructureSetROISequence", []):
        out[str(roi.ROIName).strip().lower()] = roi.ROINumber
    return out


def contours_by_roi(rt, number):
    """[(punti Nx3 in mm, SOPInstanceUID della slice), ...] per una ROI."""
    out = []
    for cs in getattr(rt, "ROIContourSequence", []):
        if cs.ReferencedROINumber != number:
            continue
        for c in getattr(cs, "ContourSequence", []):
            pts = np.array(c.ContourData, dtype=np.float64).reshape(-1, 3)
            if len(pts) < 3:
                continue
            sop = None
            for im in getattr(c, "ContourImageSequence", []):
                sop = str(getattr(im, "ReferencedSOPInstanceUID", "")) or None
                break
            out.append((pts, sop))
    return out


# ===========================================================================
#  Contorni in mm -> maschera
# ===========================================================================
def fill_contours(contours, shape, affine, sops):
    """Poligoni in coordinate PAZIENTE -> maschera booleana.

    ATTENZIONE AL SISTEMA DI RIFERIMENTO - e' il bug che questo codice ha
    avuto e che per poco non e' passato inosservato.
    `ContourData` di RTSTRUCT e' in coordinate DICOM, cioe' **LPS**. L'affine
    costruita da read_dicom_series mappa invece gli indici in **RAS** (NIfTI),
    perche' nega x e y. Applicare l'inversa RAS a punti LPS ribalta due assi
    su tre: su LUNG1-001 produceva indici di riga 683..695 su un volume di 512
    righe, quindi polygon() scartava tutto e la maschera restava vuota.
    Sui casi dove NON finiva fuori griglia era peggio, perche' la maschera
    risultava plausibile: i polmoni sono grossolanamente simmetrici, quindi lo
    specchiamento in x sovrappone il destro alla sede del sinistro e il Dice
    con organs.py restava 0.63. Si converte quindi LPS -> RAS prima di
    invertire.

    Per il SOPInstanceUID della slice si usa il riferimento esatto quando c'e';
    altrimenti si ricade sull'indice z calcolato, arrotondato.
    """
    from skimage.draw import polygon

    inv = np.linalg.inv(np.asarray(affine, dtype=np.float64))
    sop_index = {u: k for k, u in enumerate(sops)}
    mask = np.zeros(shape, dtype=bool)

    for pts, sop in contours:
        ras = pts.copy()
        ras[:, 0] *= -1.0          # LPS -> RAS
        ras[:, 1] *= -1.0
        idx = (inv[:3, :3] @ ras.T) + inv[:3, 3:4]      # (3, N) in indici
        k = sop_index.get(sop)
        if k is None:
            k = int(round(float(np.median(idx[2]))))
        if not (0 <= k < shape[2]):
            continue
        rr, cc = polygon(idx[0], idx[1], shape=shape[:2])
        mask[rr, cc, k] = True
    return mask


# ===========================================================================
#  Conversione di un paziente (eseguita in un processo separato)
# ===========================================================================
def convert_one(patient):
    """Converte un paziente. Restituisce un dict di esito, senza eccezioni.

    Gira in un processo figlio: un'eccezione non gestita qui ucciderebbe il
    worker e farebbe perdere il resto del lotto, quindi si cattura tutto e si
    riporta il motivo.
    """
    import pydicom
    from lidc import read_dicom_series

    cid = "nsclc_" + patient.split("-")[-1]
    img_out = os.path.join(OUT_IMAGES, f"{cid}.nii.gz")
    lbl_out = os.path.join(OUT_LABELS, f"{cid}.nii.gz")
    org_out = os.path.join(OUT_ORGANS, f"{cid}.nii.gz")
    if os.path.exists(img_out) and os.path.exists(lbl_out):
        return {"id": cid, "patient": patient, "status": "gia_fatto"}

    pdir = os.path.join(NSCLC_ROOT, patient)
    try:
        ct_dir, rt_file = find_series(pdir)
        if not ct_dir or not rt_file:
            raise ValueError("CT o RTSTRUCT assenti")

        vol, aff, sops, px, step = read_dicom_series(ct_dir)
        rt = pydicom.dcmread(rt_file, force=True)
        nums = roi_numbers(rt)

        if TARGET_ROI not in nums:
            raise ValueError(f"'{TARGET_ROI}' assente (presenti: {sorted(nums)[:6]})")
        gtv = fill_contours(contours_by_roi(rt, nums[TARGET_ROI]),
                            vol.shape, aff, sops)
        if not gtv.any():
            raise ValueError("GTV vuoto dopo il riempimento dei contorni")

        # Organi a rischio, in un volume multi-etichetta separato.
        organs = np.zeros(vol.shape, dtype=np.uint8)
        found = []
        for name, lab in ORGAN_LABELS.items():
            if name not in nums:
                continue
            m = fill_contours(contours_by_roi(rt, nums[name]),
                              vol.shape, aff, sops)
            if m.any():
                organs[m] = lab
                found.append(name)

        os.makedirs(OUT_IMAGES, exist_ok=True)
        os.makedirs(OUT_LABELS, exist_ok=True)
        os.makedirs(OUT_ORGANS, exist_ok=True)
        nib.save(nib.Nifti1Image(vol.astype(np.int16), aff), img_out)
        nib.save(nib.Nifti1Image(gtv.astype(np.uint8), aff), lbl_out)
        nib.save(nib.Nifti1Image(organs, aff), org_out)

        vox = abs(np.linalg.det(aff[:3, :3]))
        return {
            "id": cid, "patient": patient, "status": "ok",
            "shape": list(vol.shape), "pixel_mm": round(px, 3),
            "slice_mm": round(step, 3),
            "tumor_cm3": round(float(gtv.sum()) * vox / 1000.0, 3),
            "organs": found,
            "organ_cm3": {n: round(float((organs == ORGAN_LABELS[n]).sum())
                                   * vox / 1000.0, 2) for n in found},
        }
    except Exception as exc:
        return {"id": cid, "patient": patient, "status": "errore",
                "reason": f"{type(exc).__name__}: {exc}"}


# ===========================================================================
#  Orchestrazione
# ===========================================================================
def patients():
    return sorted(d for d in os.listdir(NSCLC_ROOT)
                  if os.path.isdir(os.path.join(NSCLC_ROOT, d)))


def load_manifest():
    if os.path.exists(MANIFEST):
        try:
            with open(MANIFEST, encoding="utf-8") as fh:
                return json.load(fh)
        except json.JSONDecodeError:
            pass
    return {"cases": [], "failed": []}


def save_manifest(man):
    """Scrittura ATOMICA del manifest, chiamata a ogni caso completato.

    Prima il manifest si scriveva solo alla fine: un'interruzione a meta'
    lasciava i NIfTI su disco ma nessun indice, e il passo successivo non
    sapeva cosa esisteva. Ora la conversione e' interrompibile in qualunque
    momento senza perdere lavoro.
    """
    os.makedirs(OUT_ROOT, exist_ok=True)
    tmp = MANIFEST + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(man, fh, indent=2)
    os.replace(tmp, MANIFEST)


def convert(workers=6, limit=None):
    pats = patients()
    if limit:
        pats = pats[:limit]
    man = load_manifest()
    done = {c["id"] for c in man["cases"]}
    todo = [p for p in pats
            if ("nsclc_" + p.split("-")[-1]) not in done]
    print(f"[nsclc] {len(pats)} pazienti, {len(done)} gia' fatti, "
          f"{len(todo)} da convertire")
    if not todo:
        return man
    print(f"[nsclc] {workers} processi paralleli. La conversione e' CPU/IO: la "
          f"GPU non ha nulla da fare qui.")
    print(f"[nsclc] interrompibile in qualunque momento (Ctrl-C): rilanciando "
          f"lo stesso comando riprende.")

    n_ok = n_err = 0
    try:
        with cf.ProcessPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(convert_one, p): p for p in todo}
            for i, fut in enumerate(cf.as_completed(futs), 1):
                r = fut.result()
                if r["status"] == "ok":
                    man["cases"].append(r)
                    n_ok += 1
                    print(f"  [{i}/{len(todo)}] {r['id']}: {r['tumor_cm3']:.1f} cm3, "
                          f"organi {len(r['organs'])}", flush=True)
                elif r["status"] == "errore":
                    man["failed"].append(r)
                    n_err += 1
                    print(f"  [{i}/{len(todo)}] {r['id']}: SALTATO ({r['reason']})",
                          flush=True)
                # Manifest aggiornato a ogni caso: vedi save_manifest().
                if r["status"] != "gia_fatto":
                    save_manifest(man)
    except KeyboardInterrupt:
        print("\n[nsclc] interrotto. Il manifest e' salvato: rilanciare lo "
              "stesso comando per riprendere.")
        save_manifest(man)
        return man

    man["target_roi"] = TARGET_ROI
    man["organ_labels"] = ORGAN_LABELS
    man["n_cases"] = len(man["cases"])
    save_manifest(man)
    print(f"\n[nsclc] convertiti {n_ok}, scartati {n_err}. "
          f"Totale in manifest: {len(man['cases'])}")
    if man["cases"]:
        v = sorted(c["tumor_cm3"] for c in man["cases"])
        print(f"[nsclc] GTV: mediana {v[len(v)//2]:.1f} cm3, "
              f"min {v[0]:.2f}, max {v[-1]:.1f}")
    return man


def case_files(min_volume_cm3=0.0, max_volume_cm3=float("inf")):
    """Lista [{'image','label','id'}] per la pipeline di training."""
    man = load_manifest()
    out = []
    for c in man["cases"]:
        if not (min_volume_cm3 <= c["tumor_cm3"] <= max_volume_cm3):
            continue
        img = os.path.join(OUT_IMAGES, f"{c['id']}.nii.gz")
        lbl = os.path.join(OUT_LABELS, f"{c['id']}.nii.gz")
        if os.path.exists(img) and os.path.exists(lbl):
            out.append({"image": img, "label": lbl, "id": c["id"]})
    return out


def verify(limit=10):
    """Controlli ANATOMICI: qui si possono confrontare gli organi delineati
    dal clinico con quelli che organs.py ricava per soglia HU. E' una verifica
    piu' forte di quella di LIDC, perche' esiste un riferimento vero."""
    import organs
    from scipy import ndimage

    man = load_manifest()
    cases = man["cases"][:limit]
    print(f"[verify] {len(cases)} casi\n")
    print(f"{'caso':<14}{'HU polm.':>9}{'GTV cm3':>9}{'polm. veri':>12}"
          f"{'polm. HU':>10}{'scarto':>8}{'GTV nei polm.':>15}")
    print("-" * 79)
    bad = []
    for c in cases:
        img = nib.load(os.path.join(OUT_IMAGES, f"{c['id']}.nii.gz"))
        ct = np.asanyarray(img.dataobj, dtype=np.float32)
        aff = np.asarray(img.affine, dtype=np.float64)
        gtv = np.asanyarray(nib.load(
            os.path.join(OUT_LABELS, f"{c['id']}.nii.gz")).dataobj) > 0
        org = np.asanyarray(nib.load(
            os.path.join(OUT_ORGANS, f"{c['id']}.nii.gz")).dataobj)

        vox = abs(np.linalg.det(aff[:3, :3]))
        lungs_true = np.isin(org, [ORGAN_LABELS["lung-right"],
                                   ORGAN_LABELS["lung-left"]])
        lung_true_cm3 = float(lungs_true.sum()) * vox / 1000.0
        hu_lung = (float(np.median(ct[lungs_true])) if lungs_true.any()
                   else float("nan"))

        masks, gaff, vols = organs.segment_organs(ct, aff, spacing_mm=3.0)
        lung_hu_cm3 = float(vols["lung_r"] + vols["lung_l"])
        err = (abs(lung_hu_cm3 - lung_true_cm3) / lung_true_cm3 * 100
               if lung_true_cm3 > 0 else float("nan"))

        inside = "n/d"
        if gtv.any() and lungs_true.any():
            com = np.array(ndimage.center_of_mass(gtv))
            idx = tuple(int(round(x)) for x in com)
            # Il GTV puo' sporgere dal polmone (invasione della parete o del
            # mediastino): si accetta la vicinanza, non il contenimento.
            d = ndimage.distance_transform_edt(~lungs_true,
                                               sampling=np.abs(np.diag(aff)[:3]))
            inside = f"{float(d[idx]):.0f} mm" if all(
                0 <= i < s for i, s in zip(idx, gtv.shape)) else "fuori griglia"

        flag = ""
        if not (-950 <= hu_lung <= -600):
            flag += " HU"
        if not (1000 <= lung_true_cm3 <= 9000):
            flag += " polmoni"
        if flag:
            bad.append((c["id"], flag.strip()))
        print(f"{c['id']:<14}{hu_lung:>9.0f}{c['tumor_cm3']:>9.1f}"
              f"{lung_true_cm3:>12.0f}{lung_hu_cm3:>10.0f}{err:>7.0f}%"
              f"{inside:>15}{flag}")
    print()
    if bad:
        print(f"[verify] ATTENZIONE su {len(bad)} casi: {bad}")
    else:
        print("[verify] Controlli superati. La colonna 'scarto' misura quanto")
        print("         organs.py (soglie HU) si discosta dai polmoni delineati")
        print("         dal clinico: e' la prima volta che abbiamo un riferimento")
        print("         vero per quell'approssimazione.")
    return bad


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="NSCLC-Radiomics -> NIfTI (GTV + organi a rischio).")
    ap.add_argument("--index", action="store_true")
    ap.add_argument("--convert", action="store_true")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    if args.index:
        import pydicom
        pats = patients()
        print(f"[nsclc] {len(pats)} pazienti in {NSCLC_ROOT}")
        man = load_manifest()
        print(f"[nsclc] gia' convertiti: {len(man['cases'])}, "
              f"scartati: {len(man['failed'])}")
        for p in pats[:5]:
            ct_dir, rt_file = find_series(os.path.join(NSCLC_ROOT, p))
            if rt_file:
                rt = pydicom.dcmread(rt_file, stop_before_pixels=True, force=True)
                print(f"  {p}: CT {len(glob.glob(os.path.join(ct_dir,'*.dcm')))} slice, "
                      f"ROI {sorted(roi_numbers(rt))}")
    elif args.convert:
        convert(workers=args.workers, limit=args.limit)
    elif args.verify:
        verify(limit=args.limit or 10)
    else:
        ap.print_help()
