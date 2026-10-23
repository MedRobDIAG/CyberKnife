# ============================================================================
#  metrics_det.py - Metriche di DETECTION, non solo di sovrapposizione.
#
#  PERCHE' IL DICE DA SOLO NON BASTA QUI
#  Il Dice e' una media pesata sui voxel, e in questo dataset i volumi tumorali
#  vanno da 0.74 a 370 cm3: poche lesioni enormi dominano la media, e un modello
#  che sbaglia sistematicamente le lesioni piccole puo' avere un Dice
#  rispettabile. Ma a valle non c'e' un voxel: c'e' un braccio robotico che deve
#  puntare UN isocentro. Le domande che contano sono:
#     - la lesione e' stata trovata?          -> sensibilita' per lesione
#     - quante volte punta il posto sbagliato? -> falsi positivi per scan
#     - di quanto sbaglia il bersaglio?        -> centroid_dist_mm
#     - il piano ha senso?                     -> isocenter_inside_gt
#  L'ultima e' binaria e brutale: se il centroide predetto cade fuori dal
#  tumore vero, il piano dosimetrico che ne discende irradia tessuto sano.
#  Nessun pacchetto la fornisce, ed e' la metrica piu' rilevante dell'insieme.
#
#  CONVENZIONE DI MATCHING (dichiararla nella relazione)
#  Una componente predetta che sovrappone almeno 1 voxel di una lesione GT e'
#  un TP per quella lesione; una lesione GT senza alcuna predizione sovrapposta
#  e' un FN; una componente predetta che non tocca alcuna lesione GT e' un FP.
#  E' la convenzione CAD standard e non dipende da una soglia di IoU
#  arbitraria; case_metrics riporta anche la variante IoU>=0.1 come controllo
#  di robustezza.
# ============================================================================
import math
import statistics

import numpy as np
from scipy import ndimage

# Le metriche di superficie vengono da MONAI: sono gia' validate e usano la
# stessa definizione della letteratura che citeremo.
from monai.metrics import compute_surface_dice, compute_hausdorff_distance


#: Volume minimo perche' una componente sia una lesione (mm3). Coerente con
#: postproc.PostProcConfig.min_lesion_mm3: la stessa soglia dev'essere usata
#: sul ground truth, altrimenti si conterebbero come FN degli speckle di
#: annotazione da 1-3 voxel che nessun modello puo' o deve trovare.
MIN_LESION_MM3 = 100.0


# ---------------------------------------------------------------------------
#  Utilita' geometriche
# ---------------------------------------------------------------------------
def voxel_volume_mm3(affine) -> float:
    return float(abs(np.linalg.det(np.asarray(affine, dtype=np.float64)[:3, :3])))


def voxel_spacing_mm(affine) -> tuple[float, float, float]:
    """Spacing per asse = norma delle colonne della parte lineare dell'affine."""
    lin = np.asarray(affine, dtype=np.float64)[:3, :3]
    return tuple(float(x) for x in np.linalg.norm(lin, axis=0))


def to_world(affine, com_voxel):
    A = np.asarray(affine, dtype=np.float64)
    return (A[:3, :3] @ np.asarray(com_voxel, dtype=np.float64)) + A[:3, 3]


def label_lesions(mask, vox_mm3: float, min_mm3: float = MIN_LESION_MM3):
    """(etichette, ids, volumi_mm3) delle componenti sopra la soglia di volume."""
    lab, n = ndimage.label(np.asarray(mask) > 0)
    if n == 0:
        return lab, np.empty(0, dtype=int), np.empty(0)
    ids = np.arange(1, n + 1)
    sizes = np.asarray(ndimage.sum(np.ones_like(lab, dtype=np.uint8), lab, ids))
    vols = sizes * vox_mm3
    keep = vols >= min_mm3
    return lab, ids[keep], vols[keep]


def _union_bbox(a, b, margin: int = 8):
    """Bounding box che contiene entrambe le maschere, con margine.

    Le metriche di superficie dipendono solo dalle superfici: ritagliare attorno
    all'unione con un margine non cambia il risultato, ma evita di allocare un
    one-hot a 2 canali su un volume 512x512x288 (600 MB in float32). Il margine
    garantisce che nessuna superficie tocchi il bordo dell'array ritagliato.
    """
    any_ = (np.asarray(a) > 0) | (np.asarray(b) > 0)
    if not any_.any():
        return None
    sl = ndimage.find_objects(any_.astype(np.uint8))[0]
    out = []
    for s, n in zip(sl, any_.shape):
        out.append(slice(max(s.start - margin, 0), min(s.stop + margin, n)))
    return tuple(out)


def _surface_metrics(gt, pred, spacing, tolerance_mm: float):
    """(NSD alla tolleranza data, HD95 in mm). NaN se una maschera e' vuota."""
    import torch

    if not np.any(gt) or not np.any(pred):
        return float("nan"), float("nan")

    box = _union_bbox(gt, pred)
    g = np.asarray(gt, dtype=bool)[box]
    p = np.asarray(pred, dtype=bool)[box]

    # One-hot a 2 canali (background, tumore) come richiesto da MONAI, in
    # float32 sul solo ritaglio.
    def _onehot(m):
        t = torch.from_numpy(np.stack([~m, m]).astype(np.float32))
        return t.unsqueeze(0)                      # (1, 2, X, Y, Z)

    yp, yt = _onehot(p), _onehot(g)
    try:
        nsd = compute_surface_dice(yp, yt, class_thresholds=[float(tolerance_mm)],
                                   include_background=False, spacing=spacing)
        hd95 = compute_hausdorff_distance(yp, yt, include_background=False,
                                          percentile=95, spacing=spacing)
        return float(np.asarray(nsd).ravel()[0]), float(np.asarray(hd95).ravel()[0])
    except Exception:
        # Una metrica di superficie che fallisce non deve interrompere la
        # valutazione di un intero fold.
        return float("nan"), float("nan")


# ---------------------------------------------------------------------------
#  Metriche di un singolo caso
# ---------------------------------------------------------------------------
def case_metrics(gt, pred, affine, min_lesion_mm3: float = MIN_LESION_MM3,
                 nsd_tolerance_mm: float = 2.0, prob=None,
                 case_id: str | None = None, surface: bool = True) -> dict:
    """Tutte le metriche di un caso, calcolate NELLO SPAZIO CT ORIGINALE.

    gt, pred : array 3D binari sulla stessa griglia.
    affine   : affine 4x4 di quella griglia. Da qui vengono i mm.
    prob     : (opzionale) mappa di probabilita' del tumore, per la diagnostica
               max_prob_in_missed_lesion.
    surface  : se False salta NSD e HD95. Servono le distanze punto-superficie,
               che sono la voce di costo dominante: nello sweep di soglia si
               valutano decine di combinazioni per caso e li' interessano solo
               sensibilita', falsi positivi e Dice.
    """
    gt = np.asarray(gt) > 0
    pred = np.asarray(pred) > 0
    vox_mm3 = voxel_volume_mm3(affine)
    spacing = voxel_spacing_mm(affine)

    # --- Sovrapposizione globale --------------------------------------------
    inter = float(np.count_nonzero(gt & pred))
    n_gt_vox, n_pred_vox = float(np.count_nonzero(gt)), float(np.count_nonzero(pred))
    denom = n_gt_vox + n_pred_vox
    # Convenzione: due maschere entrambe vuote sono in perfetto accordo.
    dice = 1.0 if denom == 0 else 2.0 * inter / denom

    vol_gt = n_gt_vox * vox_mm3 / 1000.0
    vol_pred = n_pred_vox * vox_mm3 / 1000.0
    vol_err = float("nan") if vol_gt == 0 else (vol_pred - vol_gt) / vol_gt * 100.0

    # --- Matching per lesione -----------------------------------------------
    gt_lab, gt_ids, gt_vols = label_lesions(gt, vox_mm3, min_lesion_mm3)
    pr_lab, pr_ids, pr_vols = label_lesions(pred, vox_mm3, min_lesion_mm3)

    # Coppie (etichetta_gt, etichetta_pred) che condividono almeno un voxel,
    # ottenute in una sola passata invece che con un doppio ciclo sulle
    # componenti.
    both = (gt_lab > 0) & (pr_lab > 0)
    pairs = set()
    if both.any():
        gv, pv = gt_lab[both], pr_lab[both]
        keep_g, keep_p = set(gt_ids.tolist()), set(pr_ids.tolist())
        for g, p in set(zip(gv.tolist(), pv.tolist())):
            if g in keep_g and p in keep_p:
                pairs.add((g, p))

    matched_gt = {g for g, _ in pairs}
    matched_pr = {p for _, p in pairs}
    tp = len(matched_gt)                       # lesioni GT trovate
    fn = len(gt_ids) - tp
    fp = len(pr_ids) - len(matched_pr)         # predizioni che non toccano nulla
    sens = float("nan") if len(gt_ids) == 0 else tp / len(gt_ids)

    # --- Variante IoU >= 0.1 (controllo di robustezza) ----------------------
    tp_iou = 0
    for g in gt_ids.tolist():
        gm = gt_lab == g
        n_g = float(np.count_nonzero(gm))
        for p in [pp for gg, pp in pairs if gg == g]:
            pm = pr_lab == p
            i = float(np.count_nonzero(gm & pm))
            u = n_g + float(np.count_nonzero(pm)) - i
            if u > 0 and i / u >= 0.1:
                tp_iou += 1
                break
    sens_iou = float("nan") if len(gt_ids) == 0 else tp_iou / len(gt_ids)

    # --- Il bersaglio del planner -------------------------------------------
    # Lesione primaria = la piu' voluminosa, da entrambi i lati. E' la stessa
    # regola che postproc.primary_lesion applica in produzione.
    cdist = float("nan")
    iso_inside = False
    com_pred_world = com_gt_world = None
    if len(gt_ids) and len(pr_ids):
        g_main = int(gt_ids[int(np.argmax(gt_vols))])
        p_main = int(pr_ids[int(np.argmax(pr_vols))])
        com_g = ndimage.center_of_mass(gt_lab == g_main)
        com_p = ndimage.center_of_mass(pr_lab == p_main)
        com_gt_world = to_world(affine, com_g)
        com_pred_world = to_world(affine, com_p)
        cdist = float(np.linalg.norm(com_pred_world - com_gt_world))

        # isocenter_inside_gt: il voxel su cui cadrebbe l'isocentro appartiene
        # al tumore vero? Si arrotonda all'indice piu' vicino, che e' il voxel
        # effettivamente irradiato.
        idx = tuple(int(round(c)) for c in com_p)
        if all(0 <= i < s for i, s in zip(idx, gt.shape)):
            iso_inside = bool(gt[idx])

    # Distanza normalizzata sul raggio della lesione: 5 mm di errore su una
    # lesione di 60 mm e su una di 8 mm non sono lo stesso errore.
    gt_radius = float("nan")
    if len(gt_ids):
        gt_radius = (3.0 * float(gt_vols.max()) / (4.0 * math.pi)) ** (1.0 / 3.0)
    cdist_norm = cdist / gt_radius if gt_radius and not math.isnan(cdist) else float("nan")

    # --- Superficie ----------------------------------------------------------
    if surface:
        nsd, hd95 = _surface_metrics(gt, pred, spacing, nsd_tolerance_mm)
    else:
        nsd = hd95 = float("nan")

    # --- Diagnostica: quanta probabilita' c'era sulle lesioni mancate? -------
    # E' il numero che decide se vale la pena spendere GPU su loss/risoluzione
    # (se la rete non produce massa di probabilita' li') oppure se basta
    # muovere la soglia (se ne produce, ma sotto 0.5).
    missed_probs = []
    if prob is not None and fn > 0:
        pr = np.asarray(prob)
        for g in gt_ids.tolist():
            if g not in matched_gt:
                missed_probs.append(float(pr[gt_lab == g].max()))

    return {
        "case_id": case_id,
        "dice": dice,
        "n_gt_lesions": int(len(gt_ids)),
        "n_pred_lesions": int(len(pr_ids)),
        "tp": int(tp), "fn": int(fn), "fp": int(fp),
        "sensitivity": sens,
        "sensitivity_iou10": sens_iou,
        "fp_per_scan": float(fp),
        "centroid_dist_mm": cdist,
        "centroid_dist_over_gt_radius": cdist_norm,
        "isocenter_inside_gt": iso_inside,
        "nsd_2mm": nsd,
        "hd95_mm": hd95,
        "vol_gt_cm3": vol_gt,
        "vol_pred_cm3": vol_pred,
        "vol_err_pct": vol_err,
        "gt_radius_mm": gt_radius,
        "max_prob_in_missed_lesion": missed_probs,
        "detected": bool(len(pr_ids) > 0),
    }


# ---------------------------------------------------------------------------
#  Aggregazione
# ---------------------------------------------------------------------------
#: Chiavi numeriche riassunte da aggregate().
SUMMARY_KEYS = ("dice", "sensitivity", "fp_per_scan", "centroid_dist_mm",
                "centroid_dist_over_gt_radius", "nsd_2mm", "hd95_mm",
                "vol_err_pct")


def _stats(vals: list[float]) -> dict:
    v = [x for x in vals if x is not None and not (isinstance(x, float) and math.isnan(x))]
    if not v:
        return {"n": 0, "mean": float("nan"), "std": float("nan"),
                "median": float("nan"), "q1": float("nan"), "q3": float("nan")}
    s = sorted(v)
    return {
        "n": len(v),
        "mean": statistics.fmean(v),
        "std": statistics.pstdev(v) if len(v) > 1 else 0.0,
        "median": statistics.median(s),
        "q1": s[len(s) // 4],
        "q3": s[(3 * len(s)) // 4],
    }


def aggregate(rows: list[dict], fold_of: dict | None = None) -> dict:
    """Riassunto pooled sui casi out-of-fold, piu' per-fold e per terzile.

    La stratificazione per terzile di volume tumorale non e' un extra: e' cio'
    che spiega perche' il Dice medio resta intorno a 0.56 anche quando la
    detection funziona. Senza di essa si conclude che "il modello e' mediocre"
    invece di "il modello e' mediocre SULLE LESIONI PICCOLE", che e'
    un'indicazione operativa completamente diversa.
    """
    out = {"n_cases": len(rows), "pooled": {}}
    for k in SUMMARY_KEYS:
        out["pooled"][k] = _stats([r.get(k) for r in rows])

    # Tassi calcolati sui totali, non come media di rapporti: 3 lesioni trovate
    # su 4 in un caso e 0 su 1 in un altro fanno 3/5, non la media di 0.75 e 0.
    tp = sum(r["tp"] for r in rows)
    fn = sum(r["fn"] for r in rows)
    fp = sum(r["fp"] for r in rows)
    out["pooled_detection"] = {
        "lesions_gt": tp + fn,
        "tp": tp, "fn": fn, "fp": fp,
        "sensitivity": (tp / (tp + fn)) if (tp + fn) else float("nan"),
        "fp_per_scan": fp / len(rows) if rows else float("nan"),
        "isocenter_inside_gt_rate": (
            sum(1 for r in rows if r["isocenter_inside_gt"]) / len(rows)
            if rows else float("nan")),
        "cases_with_no_detection": sum(1 for r in rows if not r["detected"]),
    }

    # --- Terzili di dimensione ----------------------------------------------
    sized = sorted([r for r in rows if r["vol_gt_cm3"] > 0],
                   key=lambda r: r["vol_gt_cm3"])
    if sized:
        t = max(len(sized) // 3, 1)
        groups = {"piccole": sized[:t], "medie": sized[t:2 * t], "grandi": sized[2 * t:]}
        out["by_size_tertile"] = {
            name: {
                "n": len(g),
                "vol_range_cm3": [g[0]["vol_gt_cm3"], g[-1]["vol_gt_cm3"]] if g else [],
                "dice": _stats([r["dice"] for r in g]),
                "sensitivity": (
                    sum(r["tp"] for r in g) / max(sum(r["tp"] + r["fn"] for r in g), 1)),
                "isocenter_inside_gt_rate": (
                    sum(1 for r in g if r["isocenter_inside_gt"]) / len(g)) if g else float("nan"),
            }
            for name, g in groups.items()
        }

    # --- Per fold ------------------------------------------------------------
    if fold_of:
        per_fold = {}
        for r in rows:
            per_fold.setdefault(fold_of.get(r["case_id"]), []).append(r)
        out["by_fold"] = {
            str(f): {"n": len(g), "dice": _stats([r["dice"] for r in g])}
            for f, g in sorted(per_fold.items(), key=lambda kv: (kv[0] is None, kv[0]))
        }

    # --- Diagnostica sulle lesioni mancate ----------------------------------
    missed = [p for r in rows for p in r.get("max_prob_in_missed_lesion", [])]
    if missed:
        out["missed_lesion_max_prob"] = {
            **_stats(missed),
            "n_missed": len(missed),
            "frac_below_0.1": sum(1 for p in missed if p < 0.1) / len(missed),
            "frac_0.1_to_0.5": sum(1 for p in missed if 0.1 <= p < 0.5) / len(missed),
        }
    return out


def paired_test(rows_a: list[dict], rows_b: list[dict], key: str = "dice") -> dict:
    """Wilcoxon appaiato sui casi in comune (a = riferimento, b = variante).

    Appaiato e non a due campioni: gli stessi 63 pazienti valutati da due
    modelli. Il test appaiato elimina la variabilita' fra pazienti, che qui e'
    enorme (Dice da ~0 a ~0.9), ed e' l'unico che abbia potenza con n=63.
    """
    from scipy import stats

    ia = {r["case_id"]: r for r in rows_a}
    ib = {r["case_id"]: r for r in rows_b}
    ids = [c for c in ia if c in ib]
    a = [ia[c][key] for c in ids]
    b = [ib[c][key] for c in ids]
    pairs = [(x, y) for x, y in zip(a, b)
             if not (math.isnan(x) or math.isnan(y))]
    if len(pairs) < 6 or all(x == y for x, y in pairs):
        return {"n": len(pairs), "p_value": float("nan"), "note": "campione insufficiente"}

    av = [x for x, _ in pairs]
    bv = [y for _, y in pairs]
    stat, p = stats.wilcoxon(av, bv)
    return {
        "n": len(pairs), "key": key,
        "mean_a": statistics.fmean(av), "mean_b": statistics.fmean(bv),
        "median_delta": statistics.median([y - x for x, y in pairs]),
        "n_better": sum(1 for x, y in pairs if y > x),
        "n_worse": sum(1 for x, y in pairs if y < x),
        "statistic": float(stat), "p_value": float(p),
        "significant_005": bool(p < 0.05),
    }
