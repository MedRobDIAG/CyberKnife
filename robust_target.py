# ============================================================================
#  robust_target.py - Il bersaglio che tiene conto di quanto la rete e' sicura.
#
#  DUE ERRORI, UN SOLO MARGINE
#  ---------------------------
#  Il margine attorno al contorno (CTV -> PTV) esiste per coprire gli errori
#  che non si possono eliminare. Qui ce ne sono due, di natura diversa:
#
#    contorno   dove la rete e' incerta il bordo vero puo' stare oltre quello
#               predetto. E' un errore SISTEMATICO: si ripete identico in ogni
#               seduta, perche' il piano e' sempre lo stesso.
#    moto       l'errore residuo dell'inseguimento: in parte sistematico
#               (deriva), in parte casuale (predizione, rumore).
#
#  Si combinano con la ricetta di van Herk (IJROBP 2000):
#      M = 2.5 * sqrt(Sigma_contorno^2 + Sigma_moto^2) + 0.7 * sigma_moto
#  dove gli errori sistematici si sommano in quadratura.
#
#  IL MARGINE NON E' UNIFORME
#  --------------------------
#  Un margine sferico tratta tutto il bordo allo stesso modo. Ma l'incertezza
#  e' LOCALE: la rete e' sicura dove il tumore confina con l'aria del polmone
#  e incerta dove tocca vasi, pleura o mediastino. La "banda" aggiunge al CTV
#  la sola corona a bassa probabilita' (p_mean >= tau) attaccata alla lesione:
#  il PTV cresce dove la rete esita, non altrove.
#
#  NESSUNA SOGLIA SCELTA SUI DATI DI TEST
#  --------------------------------------
#  tau si sceglie in modo ANNIDATO sui fold (calibrate()): per ogni fold si
#  sceglie sui 4 di training e si misura sul quinto. Il criterio e'
#  geometrico - frazione del tumore VERO contenuta nel PTV - cosi' la scelta
#  non richiede nessun piano di dose. Per un paziente nuovo si usa la mediana
#  delle 5 scelte.
#
#  USO
#     python robust_target.py --calibrate
# ============================================================================
import argparse
import json
import os

import numpy as np
from scipy import ndimage

import config
import postproc
import motion_dose as md

CALIB_PATH = os.path.join(config.OUTPUT_DIR, "eval", f"{config.UNC_EXP}_unc",
                          "tau_calibration.json")
TAU_GRID = (0.50, 0.40, 0.30, 0.20, 0.15, 0.10, 0.05)
BAND_MAX_MM = 10.0
DEFAULT_TAU = 0.20          # usato solo se la calibrazione non esiste
# RISULTATO della calibrazione. La calibrazione e' legata al modello
# (config.UNC_EXP) e va rifatta quando cambia.
#
# Modello 'finetune_all' (pretrain LIDC+NSCLC), 15 ott 2026, 54 casi accettati,
# annidata: il CTV predetto contiene in mediana il 72% del tumore vero. Il
# criterio del 90% e' RAGGIUNTO: la banda annidata ne contiene il 91.4% [IQR
# 75-97%] con un volume 1.6 volte il CTV. tau scelto 0.10 in 3 fold su 5 e
# 0.05 negli altri due; per un paziente nuovo 0.10.
#
# Modello precedente 'aug_anatomic' (53 casi): il CTV conteneva solo il 53%
# del tumore vero, nessuna soglia arrivava al 90% e tutti e cinque i fold
# finivano sull'ultima della griglia, 0.05 (= STORE_FLOOR di predict.py: sotto
# non esiste informazione salvata), con l'82% di copertura e un volume 2.2
# volte il CTV. Il miglioramento del contorno si vede qui direttamente: la
# stessa copertura costa molto meno volume trattato.


# ---------------------------------------------------------------------------
#  Bersagli
# ---------------------------------------------------------------------------
def band_target(ctv, p_mean, affine, tau, max_mm=BAND_MAX_MM):
    """CTV + corona a bassa probabilita' collegata alla lesione.

    Si tengono le componenti di {p >= tau} che toccano il CTV, e se ne
    conserva solo la parte entro `max_mm` dal CTV: una coda di probabilita'
    che scivola lungo un vaso per 3 cm non e' tumore plausibile.
    """
    ctv = np.asarray(ctv) > 0
    if not ctv.any():
        return ctv.copy()
    sp = np.linalg.norm(np.asarray(affine)[:3, :3], axis=0)
    pad = np.ceil(max_mm / sp).astype(int) + 2
    idx = np.argwhere(ctv)
    lo = np.maximum(idx.min(0) - pad, 0)
    hi = np.minimum(idx.max(0) + pad + 1, ctv.shape)
    sl = tuple(slice(a, b) for a, b in zip(lo, hi))
    c = ctv[sl]
    low = (np.asarray(p_mean)[sl] >= tau) | c
    lab, _ = ndimage.label(low)
    keep = np.unique(lab[c])
    keep = keep[keep > 0]
    band = np.isin(lab, keep)
    near = ndimage.distance_transform_edt(~c, sampling=sp) <= max_mm
    sub = c | (band & near)
    sub = ndimage.binary_closing(sub, iterations=1) | c
    out = np.zeros_like(ctv)
    out[sl] = sub
    return out


def contour_sigma(centroids_mm, jump_mm=None):
    """Errore sistematico del contorno per asse: SD dei centroidi TTA.

    Si escludono i campioni senza lesione E quelli il cui centroide dista
    dalla mediana piu' di `jump_mm` (la soglia del semaforo): sono SALTI
    verso un'altra lesione, cioe' errori di rilevamento, non di contorno, e
    se ne occupa il semaforo (giallo). Correzione misurata nella campagna:
    includendoli, una sola predizione su otto che cambia lesione portava la
    deviazione standard a decine di mm e il margine di van Herk a 400 mm
    (lung_059: PTV di 8.5 litri; lung_029: 4.5 litri).
    """
    import uncertainty as U
    jump_mm = U.DISP_REVIEW_MM if jump_mm is None else jump_mm
    c = np.asarray(centroids_mm, float)
    c = c[np.isfinite(c).all(axis=1)]
    if len(c) >= 2:
        med = np.median(c, axis=0)
        c = c[np.linalg.norm(c - med, axis=1) <= jump_mm]
    if len(c) < 2:
        return np.zeros(3)
    return c.std(axis=0)


def combined_margin(sigma_contour, motion_pop):
    """Margine per asse, van Herk con i sistematici in quadratura."""
    Sc = np.asarray(sigma_contour, float)
    Sm = np.asarray(motion_pop["Sigma_mm"], float)
    sm = np.asarray(motion_pop["sigma_mm"], float)
    return 2.5 * np.sqrt(Sc ** 2 + Sm ** 2) + 0.7 * sm


def adaptive_target(ctv, maps, affine, motion_pop, tau=None):
    """PTV adattivo: banda d'incertezza, poi margine combinato."""
    tau = chosen_tau() if tau is None else tau
    band = band_target(ctv, maps["p_mean"], affine, tau)
    m = combined_margin(contour_sigma(maps["centroids_mm"]), motion_pop)
    return md.expand_mask(band, affine, m), {"tau": float(tau),
                                             "margine_mm": m.tolist()}


def fixed_target(ctv, affine, margin_mm=5.0):
    return md.expand_mask(ctv, affine, [margin_mm] * 3)


# ---------------------------------------------------------------------------
#  Calibrazione annidata di tau
# ---------------------------------------------------------------------------
def chosen_tau(path=CALIB_PATH):
    try:
        with open(path, encoding="utf-8") as fh:
            return float(json.load(fh)["tau_nuovo_paziente"])
    except (OSError, KeyError, ValueError):
        # Ripiego VISIBILE, non silenzioso: la calibrazione e' legata al
        # modello (config.UNC_EXP). Cambiando modello senza ricalibrare si
        # userebbe una soglia scelta per un'altra rete senza saperlo.
        print(f"[robust_target] ATTENZIONE: calibrazione assente ({path}). "
              f"Uso tau={DEFAULT_TAU} di ripiego: eseguire "
              f"'python robust_target.py --calibrate' per il modello "
              f"'{config.UNC_EXP}'.")
        return DEFAULT_TAU


def _case_curve(cid, unc_dir):
    import evaluate as E
    import uncertainty as U
    maps = U.load_maps(U.maps_path(cid, unc_dir))
    aff = maps["affine"]
    mask, lesions = postproc.probs_to_mask(maps["p_mean"], config.POSTPROC,
                                           affine=aff)
    if not lesions or not config.GATE.accepts(lesions[0]):
        return None
    ctv, _ = postproc.primary_mask(mask, aff)
    gt_all, _ = E.load_gt(E.case_paths(cid)[1])
    from campaign import matching_gt_lesion
    gt = matching_gt_lesion(gt_all.astype(np.uint8), ctv) > 0
    vv = abs(np.linalg.det(aff[:3, :3])) / 1000.0
    rows = []
    for tau in TAU_GRID:
        b = band_target(ctv, maps["p_mean"], aff, tau)
        rows.append({"tau": tau,
                     "copertura_gt": float((b & gt).sum() / max(gt.sum(), 1)),
                     "volume_cm3": float(b.sum() * vv)})
    return {"case_id": cid, "gt_cm3": float(gt.sum() * vv),
            "ctv_cm3": float(ctv.sum() * vv), "curva": rows}


def calibrate(unc_dir=None, target_cov=0.90):
    """Sceglie tau: la banda PIU' STRETTA la cui copertura geometrica mediana
    del tumore vero raggiunge `target_cov` sui fold di training.

    target_cov e' 0.90 e non 0.95 perche' la banda non e' il margine finale:
    sopra si aggiunge il margine di van Herk.
    """
    import evaluate as E
    import uncertainty as U
    unc_dir = unc_dir or U.OUT_DIR
    cases, fold_of = E.resolve_cases("splits:all")
    curves = []
    for cid in cases:
        if not os.path.exists(U.maps_path(cid, unc_dir)):
            continue
        c = _case_curve(cid, unc_dir)
        if c is not None:
            c["fold"] = fold_of[cid]
            curves.append(c)

    def pick(sub):
        for k, tau in enumerate(TAU_GRID):
            cov = np.median([c["curva"][k]["copertura_gt"] for c in sub])
            if cov >= target_cov:
                return tau
        return TAU_GRID[-1]

    per_fold, test_cov, test_cov_ctv, vol_ratio = {}, [], [], []
    for f in sorted({c["fold"] for c in curves}):
        tr = [c for c in curves if c["fold"] != f]
        te = [c for c in curves if c["fold"] == f]
        tau = pick(tr)
        per_fold[f] = tau
        k = TAU_GRID.index(tau)
        for c in te:
            test_cov.append(c["curva"][k]["copertura_gt"])
            test_cov_ctv.append(c["curva"][0]["copertura_gt"])
            vol_ratio.append(c["curva"][k]["volume_cm3"] / max(c["ctv_cm3"], 1e-6))
    out = {
        "criterio": f"copertura geometrica mediana del GT >= {target_cov}",
        "n_casi_accettati": len(curves),
        "tau_per_fold": {str(k): v for k, v in per_fold.items()},
        "tau_nuovo_paziente": float(np.median(list(per_fold.values()))),
        "copertura_gt_ctv_tau0.5": _summ(test_cov_ctv),
        "copertura_gt_banda_annidata": _summ(test_cov),
        "volume_banda_su_ctv": _summ(vol_ratio),
        "curve_mediane": [
            {"tau": tau,
             "copertura_gt": float(np.median([c["curva"][k]["copertura_gt"]
                                              for c in curves])),
             "volume_su_ctv": float(np.median(
                 [c["curva"][k]["volume_cm3"] / max(c["ctv_cm3"], 1e-6)
                  for c in curves]))}
            for k, tau in enumerate(TAU_GRID)],
    }
    os.makedirs(os.path.dirname(CALIB_PATH), exist_ok=True)
    with open(CALIB_PATH, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps(out, indent=2))
    return out


def _summ(v):
    v = np.asarray(v, float)
    if len(v) == 0:
        return {}
    q1, med, q3 = np.percentile(v, [25, 50, 75])
    return {"mediana": float(med), "q1": float(q1), "q3": float(q3),
            "n": int(len(v))}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--target-cov", type=float, default=0.90)
    a = ap.parse_args()
    if a.calibrate:
        calibrate(target_cov=a.target_cov)
