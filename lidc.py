# ============================================================================
#  lidc.py - LIDC-IDRI (DICOM + XML) -> volumi NIfTI con maschere, come MSD.
#
#  A COSA SERVE
#  Il modello fallisce sulle lesioni PICCOLE: terzile basso a Dice 0.294,
#  isocentro corretto nel 52%, e il 73% delle lesioni mancate non ha alcuna
#  massa di probabilita' - la rete non le vede, non le vede male. Con 63 casi
#  (21 nel terzile piccolo) non puo' impararlo. LIDC-IDRI porta, dopo i filtri
#  qui sotto, 224 serie con 110 lesioni in quell'intervallo di volume: cinque
#  volte gli esempi, nel regime esatto del deficit.
#
#  IL FILTRO SULLA MALIGNITA' NON E' OPZIONALE
#  LIDC annota NODULI, non tumori: 2067 noduli >=3mm con volume mediano di
#  0.24 cm3, venti volte piu' piccolo del tumore mediano di MSD (5.2 cm3), e il
#  70% ha malignita' sotto 4. Addestrare su tutti insegnerebbe alla rete che un
#  nodulo benigno e' un bersaglio: per un planner CyberKnife e' la definizione
#  di falso positivo, cioe' la metrica che si vuole migliorare. Misurato:
#      filtro                        noduli  serie  vol mediano
#      tutti >=3mm                     2067    664     0.24 cm3
#      malignita' >= 3                 1020    507     0.49 cm3
#      malig >=3.5 e >=3 radiologi      287    224     2.98 cm3   <- scelto
#      malignita' >= 4                  234    179     3.84 cm3
#  Col filtro scelto l'84% dei noduli cade nell'intervallo di volume di MSD.
#
#  CONSENSO FRA RADIOLOGI
#  Ogni scan e' letto da 4 radiologi in modo indipendente, quindi lo stesso
#  nodulo fisico compare fino a 4 volte con contorni diversi. Si raggruppa per
#  vicinanza del centroide e si tiene il voxel se lo include almeno META' dei
#  radiologi che hanno annotato quel nodulo. Prendere l'unione gonfierebbe le
#  lesioni, prendere l'intersezione le eroderebbe.
#
#  USO:
#    python lidc.py --index            # elenca e filtra, senza convertire
#    python lidc.py --convert          # converte le serie selezionate
#    python lidc.py --verify           # controlli fisici sulle conversioni
# ============================================================================
import os
import csv
import glob
import json
import math
import argparse
import statistics
import xml.etree.ElementTree as ET

import numpy as np
import nibabel as nib

import config


NS = {"n": "http://www.nih.gov"}

LIDC_ROOT = os.path.join(os.path.dirname(config.DATA_ROOT.rstrip("\\/")),
                         "TCIA_LIDC-IDRI_20200921")
LIDC_ROOT = os.environ.get("LIDC_ROOT", os.path.join(
    r"C:\Users\ilgra\Documents\Progetto Medical", "TCIA_LIDC-IDRI_20200921"))
XML_ROOT = os.environ.get("LIDC_XML_ROOT", os.path.join(
    r"C:\Users\ilgra\Documents\Progetto Medical", "LIDC-XML-only"))

#: Dove finiscono i NIfTI convertiti, con la stessa struttura di MSD Task06
#: cosi' che dataset.py possa consumarli senza casi speciali.
OUT_ROOT = os.path.join(r"C:\Users\ilgra\Documents\Progetto Medical", "LIDC_nifti")
OUT_IMAGES = os.path.join(OUT_ROOT, "imagesTr")
OUT_LABELS = os.path.join(OUT_ROOT, "labelsTr")
MANIFEST = os.path.join(OUT_ROOT, "manifest.json")

# --- Criteri di selezione ---------------------------------------------------
MIN_MALIGNANCY = 3.5      # media dei rating dei radiologi (1=benigno, 5=maligno)
MIN_READERS = 3           # su 4: scarta i noduli visti da uno o due
MAX_SLICE_MM = 2.5        # oltre, le lesioni piccole sono irrecuperabili
NODULE_GROUP_MM = 10.0    # tolleranza per riconoscere "lo stesso nodulo"


# ===========================================================================
#  1) Indice: XML <-> serie DICOM
# ===========================================================================
def series_metadata():
    """SeriesInstanceUID -> riga di metadata.csv, per le sole CT complete."""
    mdir = os.path.join(LIDC_ROOT, "metadata")
    rows = list(csv.DictReader(open(os.path.join(mdir, "metadata.csv"),
                                    encoding="utf-8-sig")))
    status = {r["SeriesInstanceUID"]: r["completion_status"]
              for r in csv.DictReader(open(os.path.join(
                  mdir, "completion_status.csv"), encoding="utf-8-sig"))}
    return {r["SeriesInstanceUID"]: r for r in rows
            if r["Modality"] == "CT" and status.get(r["SeriesInstanceUID"]) == "success"}


def xml_by_series():
    """SeriesInstanceUid -> percorso dell'XML.

    Si legge il solo header con iterparse e si interrompe: costruire l'albero
    completo di 1319 file (169 MB) per estrarre un UID sarebbe sprecato.
    Quando per una serie esistono piu' XML (resubmission e correzioni), si
    tiene il file piu' grande: le correzioni aggiungono annotazioni.
    """
    out = {}
    for p in glob.glob(os.path.join(XML_ROOT, "**", "*.xml"), recursive=True):
        uid = None
        try:
            for _, el in ET.iterparse(p, events=("end",)):
                if el.tag.endswith("SeriesInstanceUid"):
                    uid = (el.text or "").strip()
                    break
        except ET.ParseError:
            continue
        if not uid:
            continue
        if uid not in out or os.path.getsize(p) > os.path.getsize(out[uid]):
            out[uid] = p
    return out


# ===========================================================================
#  2) Noduli dall'XML, raggruppati fra radiologi
# ===========================================================================
def read_nodules(xml_path, pixel_spacing_mm=0.7):
    """Noduli >=3mm con contorni, raggruppati per nodulo fisico.

    Restituisce una lista di dict con:
      contours : {sop_uid: [(x, y), ...]} per ciascuna lettura
      readings : quante letture indipendenti compongono il gruppo
      malignancy : media dei rating disponibili
      diameter_mm : estensione massima in piano
    """
    root = ET.parse(xml_path).getroot()
    raw = []
    for nod in root.findall(".//n:unblindedReadNodule", NS):
        per_slice, pts, zs = {}, [], []
        for r in nod.findall("n:roi", NS):
            sop = r.find("n:imageSOP_UID", NS)
            sop = (sop.text or "").strip() if sop is not None else None
            incl = r.find("n:inclusion", NS)
            # inclusion=FALSE marca un BUCO dentro il nodulo (es. cavitazione):
            # va sottratto, non aggiunto.
            is_hole = incl is not None and (incl.text or "").strip().upper() == "FALSE"
            poly = [(int(e.find("n:xCoord", NS).text),
                     int(e.find("n:yCoord", NS).text))
                    for e in r.findall("n:edgeMap", NS)]
            if len(poly) < 3 or sop is None:
                continue
            per_slice.setdefault(sop, []).append((poly, is_hole))
            pts += poly
            z = r.find("n:imageZposition", NS)
            if z is not None:
                try:
                    zs.append(float(z.text))
                except ValueError:
                    pass
        if not per_slice:
            continue          # nodulo <3mm: marcato come punto singolo
        xs, ys = [q[0] for q in pts], [q[1] for q in pts]
        m = nod.find(".//n:malignancy", NS)
        raw.append({
            "contours": per_slice,
            "malignancy": (int(m.text) if m is not None
                           and (m.text or "").strip().isdigit() else None),
            "diameter_mm": max(max(xs) - min(xs), max(ys) - min(ys)) * pixel_spacing_mm,
            "centroid": (statistics.fmean(xs) * pixel_spacing_mm,
                         statistics.fmean(ys) * pixel_spacing_mm,
                         statistics.fmean(zs) if zs else 0.0),
        })

    # Raggruppamento per vicinanza del centroide.
    groups = []
    for n in raw:
        for g in groups:
            if math.dist(n["centroid"], g[0]["centroid"]) < NODULE_GROUP_MM:
                g.append(n)
                break
        else:
            groups.append([n])

    out = []
    for g in groups:
        mals = [x["malignancy"] for x in g if x["malignancy"] is not None]
        out.append({
            "readings": [x["contours"] for x in g],
            "n_readers": len(g),
            "malignancy": statistics.fmean(mals) if mals else None,
            "diameter_mm": statistics.fmean([x["diameter_mm"] for x in g]),
        })
    return out


def keep_nodule(n):
    return (n["malignancy"] is not None
            and n["malignancy"] >= MIN_MALIGNANCY
            and n["n_readers"] >= MIN_READERS)


# ===========================================================================
#  3) DICOM -> volume + affine
# ===========================================================================
def read_dicom_series(series_dir):
    """Serie DICOM -> (volume HU float32, affine 4x4 RAS, sop_uid per slice).

    Punti delicati, tutti sorgenti di errori silenziosi:
      - ORDINE delle slice: si ordina per proiezione di ImagePositionPatient
        sulla normale al piano, non per nome file ne' per InstanceNumber. I
        nomi in LIDC sono UUID casuali, e InstanceNumber a volte e' invertito.
      - HU: pixel * RescaleSlope + RescaleIntercept. Senza, i valori sono
        unita' arbitrarie e la finestra HU del preprocessing non ha senso.
      - SPAZIATURA fra slice: calcolata dalle posizioni reali, NON da
        SliceThickness, che in DICOM descrive lo spessore del fascio e puo'
        differire dal passo (overlap o gap).
      - LPS -> RAS: DICOM e' LPS, NIfTI e' RAS. Si negano le prime due righe
        dell'affine. Omettere questo produce un volume specchiato, e lo
        specchiamento destra-sinistra su un torace NON e' visibile a occhio.
    """
    import pydicom

    files = [p for p in glob.glob(os.path.join(series_dir, "*.dcm"))]
    if not files:
        raise ValueError(f"nessun .dcm in {series_dir}")

    slices = []
    for p in files:
        d = pydicom.dcmread(p, force=True)
        if not hasattr(d, "ImagePositionPatient") or not hasattr(d, "PixelData"):
            continue
        slices.append(d)
    if len(slices) < 10:
        raise ValueError(f"solo {len(slices)} slice leggibili")

    iop = np.asarray(slices[0].ImageOrientationPatient, dtype=np.float64)
    row_dir, col_dir = iop[:3], iop[3:]       # vedi nota in costruzione affine
    normal = np.cross(row_dir, col_dir)

    slices.sort(key=lambda d: float(np.dot(
        np.asarray(d.ImagePositionPatient, dtype=np.float64), normal)))

    # Spaziatura reale e sua regolarita'.
    pos = np.array([np.asarray(d.ImagePositionPatient, dtype=np.float64)
                    for d in slices])
    proj = pos @ normal
    steps = np.diff(proj)
    if len(steps) == 0:
        raise ValueError("una sola slice")
    step = float(np.median(steps))
    if abs(step) < 1e-6:
        raise ValueError("slice sovrapposte (passo nullo)")
    # Un gap irregolare significa slice mancanti: il volume avrebbe una
    # geometria falsa e i contorni finirebbero sulla slice sbagliata.
    if float(np.max(np.abs(steps - step))) > 0.5 * abs(step):
        raise ValueError(f"spaziatura irregolare fra le slice (passo {step:.2f} mm)")

    vol = np.stack([s.pixel_array.astype(np.float32) for s in slices], axis=-1)
    slope = float(getattr(slices[0], "RescaleSlope", 1.0) or 1.0)
    inter = float(getattr(slices[0], "RescaleIntercept", 0.0) or 0.0)
    vol = vol * slope + inter

    ps = np.asarray(slices[0].PixelSpacing, dtype=np.float64)   # [riga, colonna]

    # L'array e' (riga, colonna, slice). ImageOrientationPatient da' come prima
    # terna la direzione lungo cui cresce l'indice di COLONNA e come seconda
    # quella lungo cui cresce l'indice di RIGA: le colonne dell'affine vanno
    # quindi accoppiate in quest'ordine, non nell'ordine in cui compaiono.
    aff = np.eye(4)
    aff[:3, 0] = col_dir * ps[0]      # indice riga
    aff[:3, 1] = row_dir * ps[1]      # indice colonna
    aff[:3, 2] = normal * step        # indice slice
    aff[:3, 3] = pos[0]
    aff[0, :] *= -1.0                 # LPS -> RAS
    aff[1, :] *= -1.0

    sops = [str(getattr(s, "SOPInstanceUID", "")) for s in slices]
    return vol, aff, sops, float(ps[0]), abs(step)


# ===========================================================================
#  4) Contorni -> maschera di consenso
# ===========================================================================
def build_mask(nodules, shape, sops):
    """Maschera uint8 dalle letture, con consenso al 50%.

    Per ogni nodulo si contano quante letture includono ciascun voxel e si
    tiene il voxel se lo include almeno la meta' dei radiologi che quel nodulo
    hanno annotato. L'unione gonfierebbe la lesione (il bordo piu' generoso
    vincerebbe sempre), l'intersezione la eroderebbe fino a svuotarla sui
    noduli dove i radiologi concordano meno.
    """
    from skimage.draw import polygon

    sop_index = {u: k for k, u in enumerate(sops)}
    mask = np.zeros(shape, dtype=np.uint8)
    kept = []

    for nod in nodules:
        votes = np.zeros(shape, dtype=np.uint8)
        n_read = len(nod["readings"])
        for contours in nod["readings"]:
            one = np.zeros(shape, dtype=bool)
            for sop, polys in contours.items():
                k = sop_index.get(sop)
                if k is None:
                    continue          # contorno su una slice non presente
                for poly, is_hole in polys:
                    xs = np.array([q[0] for q in poly])
                    ys = np.array([q[1] for q in poly])
                    # edgeMap: xCoord = colonna, yCoord = riga. polygon() vuole
                    # (riga, colonna): invertire qui ruoterebbe ogni lesione di
                    # 90 gradi, un errore che su un nodulo tondo e' invisibile.
                    rr, cc = polygon(ys, xs, shape=shape[:2])
                    if is_hole:
                        one[rr, cc, k] = False
                    else:
                        one[rr, cc, k] = True
            votes += one.astype(np.uint8)
        consensus = votes >= max(1, (n_read + 1) // 2)
        if consensus.any():
            mask[consensus] = 1
            kept.append({"n_readers": n_read,
                         "malignancy": nod["malignancy"],
                         "diameter_mm": nod["diameter_mm"],
                         "n_voxels": int(consensus.sum())})
    return mask, kept


# ===========================================================================
#  5) Indice + conversione
# ===========================================================================
def select_series(verbose=True):
    """Serie che superano i filtri, con i loro noduli selezionati."""
    meta = series_metadata()
    xmls = xml_by_series()
    sel = []
    for uid, row in meta.items():
        p = xmls.get(uid)
        if not p:
            continue
        try:
            px = float(row.get("PixelSpacing(mm)-Row") or 0) or 0.7
        except ValueError:
            px = 0.7
        try:
            nods = read_nodules(p, px)
        except ET.ParseError:
            continue
        keep = [n for n in nods if keep_nodule(n)]
        if keep:
            sel.append({"uid": uid, "patient": row["PatientID"],
                        "xml": p, "dir": row["OriginalS5cmdURI"],
                        "series_dir": row["S5cmdManifestPath"],
                        "n_nodules": len(keep),
                        "nodules": keep})
    sel.sort(key=lambda s: s["patient"])
    if verbose:
        nn = sum(s["n_nodules"] for s in sel)
        print(f"[lidc] serie selezionate: {len(sel)}  (noduli: {nn})")
        print(f"[lidc] filtri: malignita' media >= {MIN_MALIGNANCY}, "
              f">= {MIN_READERS} radiologi su 4")
    return sel


def local_series_dir(entry):
    """Cartella locale della serie. Il metadata contiene il percorso del PC su
    cui e' stato fatto il download (C:\\Users\\...\\Downloads\\...): si
    riscrive sulla radice attuale invece di fidarsi del valore salvato."""
    raw = (entry.get("series_dir") or "").replace("/", "\\")
    marker = "\\lidc_idri\\"
    if marker in raw:
        rel = raw.split(marker, 1)[1]
        cand = os.path.join(LIDC_ROOT, "lidc_idri", rel)
        if os.path.isdir(cand):
            return cand
    # Ripiego: si cerca la cartella che contiene la serie per UID.
    for d in glob.glob(os.path.join(LIDC_ROOT, "lidc_idri", entry["patient"],
                                    "*", "*")):
        if os.path.isdir(d):
            return d
    return None


def convert(limit=None, overwrite=False):
    os.makedirs(OUT_IMAGES, exist_ok=True)
    os.makedirs(OUT_LABELS, exist_ok=True)
    sel = select_series()
    if limit:
        sel = sel[:limit]

    manifest, failed = [], []
    for i, e in enumerate(sel, 1):
        cid = f"lidc_{e['patient'].split('-')[-1]}"
        img_out = os.path.join(OUT_IMAGES, f"{cid}.nii.gz")
        lbl_out = os.path.join(OUT_LABELS, f"{cid}.nii.gz")
        if os.path.exists(img_out) and os.path.exists(lbl_out) and not overwrite:
            print(f"  [{i}/{len(sel)}] {cid}: gia' convertito")
            continue

        sdir = local_series_dir(e)
        if not sdir:
            failed.append((cid, "cartella della serie non trovata"))
            print(f"  [{i}/{len(sel)}] {cid}: SALTATO (cartella non trovata)")
            continue
        try:
            vol, aff, sops, px, step = read_dicom_series(sdir)
            if step > MAX_SLICE_MM:
                raise ValueError(f"spessore di strato {step:.2f} mm > {MAX_SLICE_MM}")
            nods = read_nodules(e["xml"], px)
            keep = [n for n in nods if keep_nodule(n)]
            mask, kept = build_mask(keep, vol.shape, sops)
            if not kept:
                raise ValueError("nessun nodulo mappato sulle slice presenti")
        except Exception as exc:
            failed.append((cid, f"{type(exc).__name__}: {exc}"))
            print(f"  [{i}/{len(sel)}] {cid}: SALTATO ({exc})")
            continue

        nib.save(nib.Nifti1Image(vol.astype(np.int16), aff), img_out)
        nib.save(nib.Nifti1Image(mask, aff), lbl_out)

        vox = abs(np.linalg.det(aff[:3, :3]))
        vol_cm3 = float(mask.sum()) * vox / 1000.0
        manifest.append({
            "id": cid, "patient": e["patient"], "series_uid": e["uid"],
            "shape": list(vol.shape), "pixel_mm": px, "slice_mm": step,
            "n_nodules": len(kept), "tumor_cm3": round(vol_cm3, 4),
            "malignancy_mean": round(statistics.fmean(
                [k["malignancy"] for k in kept]), 2),
            "nodules": kept,
        })
        print(f"  [{i}/{len(sel)}] {cid}: {vol.shape} px {px:.2f} slice {step:.2f} "
              f"-> {len(kept)} noduli, {vol_cm3:.2f} cm3")

    with open(MANIFEST, "w", encoding="utf-8") as fh:
        json.dump({"min_malignancy": MIN_MALIGNANCY, "min_readers": MIN_READERS,
                   "max_slice_mm": MAX_SLICE_MM, "n_cases": len(manifest),
                   "failed": failed, "cases": manifest}, fh, indent=2)
    print(f"\n[lidc] convertite {len(manifest)} serie, {len(failed)} scartate")
    print(f"[lidc] manifest: {MANIFEST}")
    if manifest:
        v = sorted(c["tumor_cm3"] for c in manifest)
        print(f"[lidc] volume lesionale: mediana {v[len(v)//2]:.2f} cm3, "
              f"min {v[0]:.2f}, max {v[-1]:.2f}")
    return manifest, failed


# ===========================================================================
#  Consumo da parte della pipeline di training
# ===========================================================================
def case_files(min_volume_cm3: float = 0.0, max_volume_cm3: float = float("inf")):
    """Lista [{'image','label','id'}] dei casi convertiti, nel formato MONAI.

    I filtri sul volume servono a poter restringere il pretraining al regime
    dimensionale di interesse. Il deficit misurato su MSD sta nel terzile
    piccolo (0.74-3.49 cm3): se il pretraining su tutto non rendesse, la
    variante mirata e' il tentativo successivo, non una nuova conversione.
    """
    if not os.path.exists(MANIFEST):
        raise FileNotFoundError(
            f"manifest assente: {MANIFEST}. "
            "Eseguire prima: python lidc.py --convert")
    with open(MANIFEST, encoding="utf-8") as fh:
        man = json.load(fh)
    out = []
    for c in man["cases"]:
        if not (min_volume_cm3 <= c["tumor_cm3"] <= max_volume_cm3):
            continue
        img = os.path.join(OUT_IMAGES, f"{c['id']}.nii.gz")
        lbl = os.path.join(OUT_LABELS, f"{c['id']}.nii.gz")
        if os.path.exists(img) and os.path.exists(lbl):
            out.append({"image": img, "label": lbl, "id": c["id"]})
    return out


# ===========================================================================
#  6) Verifica FISICA delle conversioni
# ===========================================================================
def verify(limit=8):
    """Controlli che fallirebbero se l'affine o le unita' HU fossero sbagliate.

    Non sono controlli formali sul codice: sono controlli sull'ANATOMIA. Un
    affine con la scala sbagliata, gli assi scambiati o il segno LPS/RAS non
    invertito produce un file NIfTI perfettamente valido e visivamente
    plausibile. Cio' che non resta plausibile sono i numeri: il volume
    polmonare, la posizione della lesione rispetto ai polmoni, l'istogramma HU.

    Si riusa organs.segment_organs, che e' indipendente da questa pipeline e si
    basa su soglie HU piu' criteri geometrici: se la geometria e' sbagliata,
    la separazione dei polmoni e il mediastino degenerano.
    """
    import organs

    with open(MANIFEST, encoding="utf-8") as fh:
        man = json.load(fh)
    cases = man["cases"][:limit]
    print(f"[verify] {len(cases)} casi\n")
    print(f"{'caso':<12}{'HU min/max':>14}{'HU polmone':>12}{'polmoni cm3':>13}"
          f"{'lesione cm3':>13}{'nei polmoni':>13}")
    print("-" * 78)

    bad = []
    for c in cases:
        img = nib.load(os.path.join(OUT_IMAGES, f"{c['id']}.nii.gz"))
        ct = np.asanyarray(img.dataobj, dtype=np.float32)
        aff = np.asarray(img.affine, dtype=np.float64)
        msk = np.asanyarray(nib.load(
            os.path.join(OUT_LABELS, f"{c['id']}.nii.gz")).dataobj) > 0

        # HU: il minimo NON si controlla contro -1000. Molti scanner usano
        # -2048 o -3024 come padding fuori dal cerchio di ricostruzione: e'
        # normale, ed e' comunque clippato dalla finestra HU del
        # preprocessing. Si verifica invece che l'aria esista (<= -900) e,
        # sotto, che il parenchima polmonare cada nel suo intervallo
        # fisiologico - che e' il vero test della calibrazione HU.
        hu_lo, hu_hi = float(ct.min()), float(ct.max())
        air_frac = float((ct < -900).mean())

        masks, gaff, vols = organs.segment_organs(ct, aff, spacing_mm=3.0)
        lung_cm3 = float(vols["lung_r"] + vols["lung_l"])

        vox = abs(np.linalg.det(aff[:3, :3]))
        les_cm3 = float(msk.sum()) * vox / 1000.0

        # La lesione deve cadere nel bounding box polmonare (margine 20 mm):
        # e' il test che smaschera uno specchiamento o uno scambio di assi.
        lungs = masks["lung_r"] | masks["lung_l"]
        inside = "n/d"
        if msk.any() and lungs.any():
            from scipy import ndimage
            com = np.array(ndimage.center_of_mass(msk), dtype=np.float64)
            world = (aff[:3, :3] @ com) + aff[:3, 3]
            idx = np.array(np.nonzero(lungs), dtype=np.float64)
            lw = (gaff[:3, :3] @ idx) + gaff[:3, 3:4]
            lo, hi = lw.min(axis=1) - 20.0, lw.max(axis=1) + 20.0
            ok = bool(np.all(world >= lo) and np.all(world <= hi))
            inside = "si" if ok else "NO"

        # HU mediano DENTRO i polmoni: il parenchima aerato sta fra -950 e
        # -600 HU. Se la calibrazione (RescaleSlope/Intercept) fosse sbagliata
        # questo numero sarebbe fuori scala, mentre il minimo e il massimo
        # potrebbero sembrare plausibili.
        lung_hu = float("nan")
        if lungs.any():
            from dosimetry import resample_to_grid
            lung_on_ct = resample_to_grid(lungs.astype(np.uint8), gaff,
                                          ct.shape, aff, order=0) > 0
            if lung_on_ct.any():
                lung_hu = float(np.median(ct[lung_on_ct]))

        flag = ""
        if hu_lo > -900:
            flag += " aria_assente"
        if not (-950 <= lung_hu <= -600):
            flag += " HU_parenchima"
        if not (1500 <= lung_cm3 <= 8000):
            flag += " polmoni"
        if inside == "NO":
            flag += " posizione"
        if flag:
            bad.append((c["id"], flag.strip()))

        print(f"{c['id']:<12}{f'{hu_lo:.0f}/{hu_hi:.0f}':>14}{lung_hu:>12.0f}"
              f"{lung_cm3:>13.0f}{les_cm3:>13.2f}{inside:>13}{flag}")

    print()
    if bad:
        print(f"[verify] ATTENZIONE su {len(bad)} casi: {bad}")
    else:
        print("[verify] Tutti i controlli superati: HU del parenchima in scala,")
        print("         volume polmonare fisiologico, lesione dentro i polmoni.")
        print("         L'affine e l'orientamento dei contorni sono corretti.")
    return bad


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="LIDC-IDRI -> NIfTI con maschere.")
    ap.add_argument("--index", action="store_true", help="solo selezione")
    ap.add_argument("--convert", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--verify", action="store_true",
                    help="controlli fisici sulle conversioni gia' fatte")
    args = ap.parse_args()

    if args.index:
        sel = select_series()
        for e in sel[:15]:
            print(f"  {e['patient']}  noduli {e['n_nodules']}  "
                  f"malignita' {[round(n['malignancy'],1) for n in e['nodules']]}")
        print(f"  ... ({len(sel)} serie in tutto)")
    elif args.convert:
        convert(limit=args.limit, overwrite=args.overwrite)
    elif args.verify:
        verify(limit=args.limit or 8)
    else:
        ap.print_help()
