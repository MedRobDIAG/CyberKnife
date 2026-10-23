# ============================================================================
#  run_uncertainty.py - Mappe di incertezza out-of-fold per i 63 casi, e la
#  domanda che conta: l'incertezza PREDICE l'errore?
#
#  Ogni caso e' campionato dal modello del fold che non l'ha visto (stessa
#  regola di evaluate.py --oof). Riprendibile: i casi gia' salvati si saltano.
#
#  USO
#     python run_uncertainty.py                  campiona i casi mancanti
#     python run_uncertainty.py --analyze        solo analisi
# ============================================================================
import argparse
import csv
import json
import os
import time

import numpy as np

import config
import evaluate as E
import uncertainty as U


def sample_all(n_samples=8, out_dir=U.OUT_DIR, ckpt="final.pth"):
    import torch
    from inference import load_model
    cases, fold_of = E.resolve_cases("splits:all")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    params = U.tta_params(n_samples)
    models = {}
    t0 = time.time()
    todo = [c for c in cases if not os.path.exists(U.maps_path(c, out_dir))]
    print(f"[unc] {len(cases) - len(todo)} casi gia' fatti, {len(todo)} da fare")
    for i, cid in enumerate(todo, 1):
        f = fold_of[cid]
        if f not in models:
            models[f] = load_model(os.path.join(
                config.OUTPUT_DIR, "runs", config.UNC_EXP, f"fold{f}", ckpt),
                device)
        img, _ = E.case_paths(cid)
        r = U.sample_case(models[f], img, params, device)
        U.save_maps(U.maps_path(cid, out_dir), r)
        el = time.time() - t0
        print(f"  [{i}/{len(todo)}] {cid} fold {f}  "
              f"centroidi {np.nanstd(r['centroids_mm'], axis=0).round(1)}  "
              f"({el:.0f} s, ~{el / i * (len(todo) - i):.0f} s alla fine)",
              flush=True)
    with open(os.path.join(out_dir, "config.json"), "w") as fh:
        json.dump({"n_samples": n_samples, "params": params,
                   "oof_exp": config.UNC_EXP, "ckpt": ckpt,
                   "postproc": vars(config.POSTPROC),
                   "git_commit": E.git_commit()}, fh, indent=2)


# ---------------------------------------------------------------------------
#  Analisi
# ---------------------------------------------------------------------------
def auroc(score, positive):
    """Area sotto la curva ROC (Mann-Whitney), con gestione dei pareggi."""
    s = np.asarray(score, float)
    y = np.asarray(positive, bool)
    pos, neg = s[y], s[~y]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    gt = (pos[:, None] > neg[None, :]).sum()
    eq = (pos[:, None] == neg[None, :]).sum()
    return float((gt + 0.5 * eq) / (len(pos) * len(neg)))


def isocenter_inside_gt(maps, cid):
    """Esito del sistema CHE SI USEREBBE: la maschera della TTA media.

    Non si riusa l'esito di aug_anatomic_oof, che e' della predizione
    singola: i due sistemi possono differire proprio sui casi al limite.
    """
    import postproc
    mask, lesions = postproc.probs_to_mask(maps["p_mean"], config.POSTPROC,
                                           affine=maps["affine"])
    if not lesions:
        return False
    gt, _ = E.load_gt(E.case_paths(cid)[1])
    idx = np.round(lesions[0]["com_voxel"]).astype(int)
    idx = np.clip(idx, 0, np.array(gt.shape) - 1)
    return bool(gt[tuple(idx)])


def calibration(case_ids, prob_dir, out_dir, bins=10):
    """Affidabilita' voxel per voxel: la probabilita' dice il vero?

    Si considerano i voxel del bounding box delle mappe (dove p > 0.05): fuori
    la rete e' sicura del fondo e quasi sempre ha ragione, e includerli
    abbasserebbe l'ECE senza dire nulla del tumore.
    """
    edges = np.linspace(0, 1, bins + 1)
    cnt = np.zeros(bins)
    conf = np.zeros(bins)
    acc = np.zeros(bins)
    for cid in case_ids:
        m = U.load_maps(U.maps_path(cid, out_dir))
        gt, _ = E.load_gt(E.case_paths(cid)[1])
        p = m["p_mean"]
        sel = p > 0.05
        if not sel.any():
            continue
        pv = p[sel]
        gv = gt[sel] > 0
        k = np.clip(np.digitize(pv, edges) - 1, 0, bins - 1)
        cnt += np.bincount(k, minlength=bins)
        conf += np.bincount(k, weights=pv, minlength=bins)
        acc += np.bincount(k, weights=gv, minlength=bins)
    ok = cnt > 0
    conf[ok] /= cnt[ok]
    acc[ok] /= cnt[ok]
    ece = float(np.sum(cnt[ok] / cnt.sum() * np.abs(conf[ok] - acc[ok])))
    return {"ece": ece, "bins": edges.tolist(), "count": cnt.tolist(),
            "confidence": conf.tolist(), "accuracy": acc.tolist()}


def analyze(out_dir=U.OUT_DIR):
    cases, fold_of = E.resolve_cases("splits:all")
    cases = [c for c in cases if os.path.exists(U.maps_path(c, out_dir))]
    rows = []
    for cid in cases:
        maps = U.load_maps(U.maps_path(cid, out_dir))
        s = U.case_signals(maps)
        s["case_id"] = cid
        s["fold"] = fold_of[cid]
        s["ok"] = isocenter_inside_gt(maps, cid)
        rows.append(s)

    keys = ["volume_cm3", "prob_mean", "std_lesione", "mi_lesione",
            "frac_ambigua", "centroide_disp_mm", "centroide_disp_max_mm",
            "volume_cv", "frac_campioni_con_lesione"]
    # segno: per ciascun segnale, "alto = fallimento" oppure il contrario
    higher_is_risk = {"volume_cm3": False, "prob_mean": False,
                      "frac_campioni_con_lesione": False}
    fail = np.array([not r["ok"] for r in rows])
    table = {}
    for k in keys:
        v = np.array([r.get(k, np.nan) for r in rows], float)
        # nessuna lesione = rischio massimo
        worst = -np.inf if not higher_is_risk.get(k, True) else np.inf
        v = np.where(np.isfinite(v), v, worst)
        score = v if higher_is_risk.get(k, True) else -v
        table[k] = {
            "auroc_fallimento": auroc(score, fail),
            "mediana_riusciti": float(np.median(v[~fail][np.isfinite(v[~fail])]))
            if (~fail).any() else None,
            "mediana_falliti": float(np.median(v[fail][np.isfinite(v[fail])]))
            if np.isfinite(v[fail]).any() else None,
        }

    # Cancello esteso, validazione ANNIDATA: la soglia sulla dispersione del
    # centroide si sceglie sui fold di training e si applica al fold escluso,
    # in aggiunta al cancello sul volume gia' in uso.
    nested = {k: nested_gate(rows, key=k)
              for k in ("centroide_disp_mm", "centroide_disp_max_mm")}

    cal = calibration(cases, None, out_dir)
    summary = {"n_casi": len(rows), "n_falliti": int(fail.sum()),
               "segnali": table, "cancello_annidato": nested,
               "calibrazione": cal}
    with open(os.path.join(out_dir, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    with open(os.path.join(out_dir, "per_case.csv"), "w", newline="") as fh:
        cols = ["case_id", "fold", "ok"] + keys
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    print(f"\n{len(rows)} casi, {int(fail.sum())} con isocentro fuori dal "
          f"tumore vero\n")
    print(f"{'segnale':28s} {'AUROC':>6s} {'riusciti':>10s} {'falliti':>10s}")
    for k in keys:
        t = table[k]
        f = lambda x: "-" if x is None else f"{x:.3f}"
        print(f"{k:28s} {t['auroc_fallimento']:6.3f} "
              f"{f(t['mediana_riusciti']):>10s} {f(t['mediana_falliti']):>10s}")
    for sig, res in nested.items():
        print(f"\ncancello annidato con {sig}:")
        for k, v in res.items():
            print(f"  {k:26s} {v}")
    print(f"\nECE (voxel del bounding box): {cal['ece']:.3f}")
    return summary


KEY = "centroide_disp_max_mm"


def _gate_eval(rows, vol_thr, disp_thr, key=KEY):
    acc = [r for r in rows
           if r.get("volume_cm3", 0.0) >= vol_thr
           and r.get(key, np.inf) <= disp_thr]
    n_ok = sum(r["ok"] for r in acc)
    return len(acc), n_ok


def nested_gate(rows, min_accept_frac=0.6, key=KEY):
    """Volume (soglia di progetto) + dispersione del centroide (scelta
    annidata). Per ogni fold: si sceglie la soglia di dispersione che
    massimizza l'accuratezza sui fold di training con almeno il 60% di casi
    accettati, e la si applica al fold escluso.

    Due versioni misurate, con lo stesso esito annidato (43/63 accettati,
    95.3% corretti, contro 53/63 e 94.3% del solo volume). La MASSIMA ha
    pero' l'AUROC migliore (0.934) e soglie stabili (50 mm in 4 fold su 5):
    e' quella del semaforo, come segnale di revisione."""
    vol_thr = config.GATE.min_volume_cm3
    folds = sorted({r["fold"] for r in rows})
    grid = [0.5, 1, 1.5, 2, 3, 5, 10, 20, 50, 100, np.inf]
    acc_tot = ok_tot = 0
    chosen = []
    for f in folds:
        tr = [r for r in rows if r["fold"] != f]
        te = [r for r in rows if r["fold"] == f]
        best = None
        for d in grid:
            n, k = _gate_eval(tr, vol_thr, d, key)
            if n < min_accept_frac * len(tr) or n == 0:
                continue
            score = (k / n, n)
            if best is None or score > best[0]:
                best = (score, d)
        d = best[1] if best else np.inf
        chosen.append(d)
        n, k = _gate_eval(te, vol_thr, d, key)
        acc_tot += n
        ok_tot += k
    n_v, k_v = 0, 0
    for f in folds:
        n, k = _gate_eval([r for r in rows if r["fold"] == f], vol_thr, np.inf)
        n_v += n
        k_v += k
    return {
        "segnale": key,
        "solo_volume_accettati": f"{n_v}/{len(rows)}",
        "solo_volume_corretti": round(k_v / max(n_v, 1), 3),
        "con_dispersione_accettati": f"{acc_tot}/{len(rows)}",
        "con_dispersione_corretti": round(ok_tot / max(acc_tot, 1), 3),
        "soglie_scelte_mm": [float(x) for x in chosen],
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--n-samples", type=int, default=8)
    a = ap.parse_args()
    if not a.analyze:
        sample_all(a.n_samples)
    analyze()
