# ============================================================================
#  campaign.py - Campagna di valutazione ad anello chiuso su piu' pazienti.
#
#  LA DOMANDA
#  ----------
#  Un sistema reale ha due errori che la simulazione originale ignorava:
#    1. il CONTORNO e' una predizione (Dice 0.47 out-of-fold), non la verita';
#    2. il TUMORE RESPIRA, e il robot lo insegue con un ritardo e un modello
#       di correlazione imperfetto.
#  Per ogni paziente si pianifica sul contorno PREDETTO e si giudica sul
#  contorno VERO (le etichette del dataset), con il tumore in movimento. E'
#  l'unico modo di misurare cosa riceve davvero la lesione: la copertura del
#  bersaglio pianificato e' sempre ~95% per costruzione, e non dice nulla.
#
#  ONESTA' DEI NUMERI
#  ------------------
#  - Le predizioni sono OUT-OF-FOLD: ogni caso e' predetto dal modello del
#    fold che non l'ha visto (outputs/eval/aug_anatomic_oof).
#  - Solo casi ACCETTATI dal cancello di confidenza: sui rifiutati un piano
#    non si fa, ed e' proprio il comportamento del sistema.
#  - Si riportano mediana [IQR] e percentuale di casi sopra soglia, mai la
#    sola media; il confronto fra scenari e' appaiato (Wilcoxon).
#
#  USO
#     python campaign.py --n 20                 (riprendibile)
#     python campaign.py --cases lung_001 --quick
# ============================================================================
import argparse
import csv
import json
import os
import time

import numpy as np
import nibabel as nib
from scipy import ndimage

import config
import postproc
import predict as P
import organs as og
import motion_model as mm
import motion_dose as md
from dosimetry import compute_plan, apply_affine_fast

OOF_DIR = os.path.join(config.OUTPUT_DIR, "eval", "aug_anatomic_oof", "probs")
# Una cartella per modello (config.UNC_EXP): la campagna del modello precedente
# resta in eval/campaign_aug_anatomic per il confronto.
OUT_DIR = os.path.join(config.OUTPUT_DIR, "eval", f"campaign_{config.UNC_EXP}")


# ---------------------------------------------------------------------------
#  Caricamento di un caso
# ---------------------------------------------------------------------------
def centroid_mm(mask, affine):
    com = np.array(ndimage.center_of_mass(np.asarray(mask) > 0))
    return apply_affine_fast(np.asarray(affine, float),
                             com.reshape(3, 1)).ravel()


def matching_gt_lesion(gt, pred_primary):
    """La lesione VERA corrispondente alla principale predetta.

    Quella con la sovrapposizione maggiore; senza sovrapposizione, quella con
    il centroide piu' vicino. Un caso multi-lesione non va giudicato su una
    lesione che il piano non ha mai inteso trattare.
    """
    lab, n = ndimage.label(np.asarray(gt) > 0)
    if n == 0:
        return np.zeros_like(gt, dtype=np.uint8)
    if n == 1:
        return (lab == 1).astype(np.uint8)
    ov = np.bincount(lab[np.asarray(pred_primary) > 0], minlength=n + 1)[1:]
    if ov.max() > 0:
        k = int(np.argmax(ov)) + 1
    else:
        c = np.array(ndimage.center_of_mass(pred_primary))
        cs = np.array(ndimage.center_of_mass(lab > 0, lab, range(1, n + 1)))
        k = int(np.argmin(np.linalg.norm(cs - c, axis=1))) + 1
    return (lab == k).astype(np.uint8)


def load_case(case_id, prob_dir=OOF_DIR, organ_spacing_mm=2.0, status_cb=None):
    img = nib.load(os.path.join(config.IMAGES_DIR, f"{case_id}.nii.gz"))
    ct = np.asanyarray(img.dataobj, dtype=np.float32)
    affine = img.affine.astype(np.float64)
    gt = (np.asanyarray(nib.load(os.path.join(
        config.LABELS_DIR, f"{case_id}.nii.gz")).dataobj) > 0.5).astype(np.uint8)

    prob, paff = P.load_prob(P.prob_path(prob_dir, case_id))
    if prob.shape != ct.shape or not np.allclose(paff, affine, atol=1e-3):
        raise ValueError(f"{case_id}: probabilita' non allineate alla CT")
    pred_all, lesions = postproc.probs_to_mask(prob, config.POSTPROC,
                                               affine=affine)
    pred, _ = postproc.primary_mask(pred_all, affine)
    primary = postproc.primary_lesion(lesions)

    masks, oaff, _ = og.segment_organs(ct, affine, organ_spacing_mm,
                                       status_cb=status_cb)
    return {
        "case_id": case_id, "ct": ct, "affine": affine,
        "gt_all": gt, "pred": pred, "prob": prob, "lesion": primary,
        "gt": matching_gt_lesion(gt, pred) if pred.any() else gt,
        "organs": masks, "organ_affine": oaff,
        "lungs": masks["lung_r"] | masks["lung_l"],
        "accepted": bool(config.GATE.accepts(primary)),
    }


def make_plan_fn(case, cfg=None, status_cb=None):
    """Pianificatore del caso: compute_plan con organi, SENZA vincoli di sala.

    I vincoli del robot non entrano qui: misurato su lung_001 non cambiano la
    qualita' del piano (copertura 94.93% con e senza, CI 0.858 vs 0.856) e
    costano l'ottimizzazione della sala a ogni piano. In una campagna di
    decine di piani quel tempo non compra informazione.
    """
    m = case["organs"]

    def plan(target):
        target = np.asarray(target, dtype=np.uint8)
        iso = centroid_mm(target, case["affine"])
        return compute_plan(
            case["ct"], target, case["affine"], iso, cfg=cfg,
            bone_mask=m["bones"], bone_affine=case["organ_affine"],
            arms_mask=m["arms"], body_mask=m["body"], status_cb=status_cb)
    return plan


def case_motion(case, seed, cfg=None):
    """Respiro del paziente: ampiezza dalla posizione della lesione."""
    cfg = cfg or mm.MotionConfig()
    c = centroid_mm(case["gt"], case["affine"])
    lung_aff = case["organ_affine"]
    frac = mm.lesion_si_fraction(c, case["lungs"], lung_aff)
    shift = mm.inhale_shift_for_location(frac, cfg)
    cfg = cfg.replace(inhale_shift_mm=shift, seed=seed)
    return cfg, frac


# ---------------------------------------------------------------------------
#  Campagna
# ---------------------------------------------------------------------------
#: Scenari, nell'ordine della tabella finale:
#:   (nome, bersaglio pianificato, moto con cui si valuta)
SCENARIOS = [
    ("ctv_statico", "CTV predetto", "nessuno"),
    ("ctv_fascio_fermo", "CTV predetto", "completo"),
    ("itv_5mm", "ITV del (CTV + 5 mm)", "completo"),
    ("inseguimento", "CTV + margine di moto", "residuo"),
    ("adattivo", "banda + margine combinato", "residuo"),
    ("oracolo", "CTV VERO + margine di moto", "residuo"),
]
PLAN_OF = {"ctv_statico": "ctv", "ctv_fascio_fermo": "ctv",
           "itv_5mm": "itv_5mm", "inseguimento": "inseguimento",
           "adattivo": "adattivo", "oracolo": "oracolo"}
FIELDS = ["case_id", "fold", "gt_cm3", "ctv_cm3", "iso_ok", "si_fraction",
          "shift_si_mm", "scenario", "ptv_cm3", "copertura_pct", "d95_pct",
          "d_min_pct", "paddick_ci", "polmone_dose_media_gy",
          "polmone_v20_pct", "polmone_v5_pct", "n_fasci", "margine_mm",
          "tau", "tempo_s"]


def load_case_unc(case_id, unc_dir=None):
    """Come load_case, ma con la maschera della TTA media (uncertainty.py).

    E' il sistema che la GUI usa con l'analisi di incertezza attiva, ed e'
    quello su cui il semaforo e il PTV adattivo sono stati validati.
    """
    import uncertainty as U
    unc_dir = unc_dir or U.OUT_DIR
    maps = U.load_maps(U.maps_path(case_id, unc_dir))
    img = nib.load(os.path.join(config.IMAGES_DIR, f"{case_id}.nii.gz"))
    ct = np.asanyarray(img.dataobj, dtype=np.float32)
    affine = img.affine.astype(np.float64)
    if maps["p_mean"].shape != ct.shape or not np.allclose(
            maps["affine"], affine, atol=1e-3):
        raise ValueError(f"{case_id}: mappe non allineate alla CT")
    gt = (np.asanyarray(nib.load(os.path.join(
        config.LABELS_DIR, f"{case_id}.nii.gz")).dataobj) > 0.5
          ).astype(np.uint8)
    pred_all, lesions = postproc.probs_to_mask(maps["p_mean"],
                                               config.POSTPROC, affine=affine)
    pred, _ = postproc.primary_mask(pred_all, affine)
    primary = postproc.primary_lesion(lesions)
    return {
        "case_id": case_id, "ct": ct, "affine": affine, "maps": maps,
        "gt_all": gt, "pred": pred, "lesion": primary,
        "gt": matching_gt_lesion(gt, pred) if pred.any() else gt,
        "accepted": bool(config.GATE.accepts(primary)),
    }


def select_cases(n, unc_dir=None):
    """Casi accettati dal cancello, stratificati per volume del tumore vero.

    Il cancello si applica alla lesione principale della TTA media, come fa
    il sistema. Fra gli accettati si prendono n casi equispaziati nella
    classifica per volume: il Dice dipende molto dalla dimensione, e un
    campione casuale potrebbe finire tutto fra le lesioni grandi.
    """
    import evaluate as E
    import uncertainty as U
    unc_dir = unc_dir or U.OUT_DIR
    cases, fold_of = E.resolve_cases("splits:all")
    rows = []
    for cid in cases:
        path = U.maps_path(cid, unc_dir)
        if not os.path.exists(path):
            continue
        maps = U.load_maps(path)
        _, les = postproc.probs_to_mask(maps["p_mean"], config.POSTPROC,
                                        affine=maps["affine"])
        if not les or not config.GATE.accepts(les[0]):
            continue
        gt, gaff = E.load_gt(E.case_paths(cid)[1])
        vol = gt.sum() * abs(np.linalg.det(gaff[:3, :3])) / 1000.0
        rows.append((vol, cid))
    rows.sort()
    if n >= len(rows):
        return [c for _, c in rows], fold_of, len(rows)
    idx = np.linspace(0, len(rows) - 1, n).round().astype(int)
    return [rows[i][1] for i in idx], fold_of, len(rows)


def run_case(cid, fold, status_cb=None, seed=None, only=None):
    """Tutti gli scenari di un caso (o solo quelli in `only`).

    Tutto e' deterministico dato il caso (semi dal numero del caso): rifare
    uno scenario produce lo stesso respiro e lo stesso residuo.
    """
    import robust_target as rt
    t0 = time.time()
    case = load_case_unc(cid)
    masks, oaff, _ = og.segment_organs(case["ct"], case["affine"], 2.0)
    case["organs"], case["organ_affine"] = masks, oaff
    case["lungs"] = masks["lung_r"] | masks["lung_l"]
    aff = case["affine"]
    vv = abs(np.linalg.det(aff[:3, :3])) / 1000.0
    seed = int(cid.split("_")[-1]) if seed is None else seed
    cfg, frac = case_motion(case, seed=seed)

    # Seduta vera e "4DCT" di pianificazione: respiri diversi, perche' l'ITV
    # si disegna prima e la seduta non ripete esattamente quel respiro.
    tr = mm.generate_trace(cfg, 300.0)
    plan_tr = mm.generate_trace(
        cfg.replace(seed=seed + 97, drift_mm_per_min=(0, 0, 0)), 40.0)
    trk = mm.simulate_tracking(tr, cfg.predictor, cfg)
    pop = md.population_residual(cfg, cfg.predictor, n=10,
                                 seed0=1000 + 100 * seed)
    motion_m = pop["margine_mm"]
    ctv, gt = case["pred"], case["gt"]
    adaptive, info = rt.adaptive_target(ctv, case["maps"], aff, pop)
    targets = {
        "ctv": ctv,
        "itv_5mm": md.envelope_mask(rt.fixed_target(ctv, aff, 5.0), aff,
                                    plan_tr.tumor),
        "inseguimento": md.expand_mask(ctv, aff, motion_m),
        "adattivo": adaptive,
        "oracolo": md.expand_mask(gt, aff, motion_m),
    }
    margins = {"ctv": [0.0] * 3, "itv_5mm": [5.0] * 3,
               "inseguimento": motion_m, "adattivo": info["margine_mm"],
               "oracolo": motion_m}
    disp = {"nessuno": np.zeros((1, 3)), "completo": trk.untracked,
            "residuo": trk.residual}

    iso = centroid_mm(ctv, aff)
    gidx = np.round(np.linalg.solve(aff, np.append(iso, 1.0))[:3]).astype(int)
    gidx = np.clip(gidx, 0, np.array(gt.shape) - 1)
    iso_ok = bool(case["gt_all"][tuple(gidx)])

    plan_fn = make_plan_fn(case)
    plans, rows = {}, []
    for name, _, motion in SCENARIOS:
        if only and name not in only:
            continue
        key = PLAN_OF[name]
        if key not in plans:
            if status_cb:
                status_cb(f"{cid}: piano '{key}'")
            plans[key] = plan_fn(targets[key])
        r = md.evaluate(plans[key], gt, aff, disp[motion], case["lungs"],
                        oaff)
        row = {"case_id": cid, "fold": fold,
               "gt_cm3": round(float(gt.sum() * vv), 3),
               "ctv_cm3": round(float(ctv.sum() * vv), 3),
               "iso_ok": iso_ok, "si_fraction": round(frac, 3),
               "shift_si_mm": round(abs(cfg.inhale_shift_mm[2]), 2),
               "scenario": name,
               "ptv_cm3": round(float(np.count_nonzero(targets[key]) * vv), 3),
               "margine_mm": json.dumps([round(float(v), 2)
                                         for v in margins[key]]),
               "tau": info["tau"] if key == "adattivo" else "",
               "tempo_s": round(time.time() - t0, 1)}
        for k, v in r.items():
            if k in FIELDS:
                row[k] = round(v, 4) if isinstance(v, float) else v
        rows.append(row)
    return rows


def _iqr(v):
    q1, med, q3 = np.percentile(np.asarray(v, float), [25, 50, 75])
    return {"mediana": round(float(med), 3), "q1": round(float(q1), 3),
            "q3": round(float(q3), 3)}


def summarize(rows):
    from scipy.stats import wilcoxon
    by = {}
    for r in rows:
        by.setdefault(r["scenario"], {})[r["case_id"]] = r
    if not by:
        return {"n_casi": 0, "scenari": {}, "test_appaiati": {},
                "casi_isocentro_nel_tumore": 0}
    cases = sorted(set.intersection(*[set(v) for v in by.values()]))
    keys = ["copertura_pct", "d95_pct", "ptv_cm3", "paddick_ci",
            "polmone_dose_media_gy", "polmone_v20_pct"]
    out = {"n_casi": len(cases), "scenari": {}}
    for name, _, _ in SCENARIOS:
        if name not in by:
            continue
        d = {k: _iqr([float(by[name][c][k]) for c in cases]) for k in keys}
        cov = np.array([float(by[name][c]["copertura_pct"]) for c in cases])
        d["casi_copertura_ge_95_pct"] = round(float(100 * np.mean(cov >= 95)),
                                              1)
        d["casi_copertura_ge_90_pct"] = round(float(100 * np.mean(cov >= 90)),
                                              1)
        out["scenari"][name] = d
    tests = {}
    for a, b in (("ctv_fascio_fermo", "inseguimento"),
                 ("itv_5mm", "adattivo"), ("inseguimento", "adattivo")):
        if a not in by or b not in by:
            continue
        for k in ("copertura_pct", "ptv_cm3", "polmone_v20_pct"):
            x = np.array([float(by[a][c][k]) for c in cases])
            y = np.array([float(by[b][c][k]) for c in cases])
            try:
                p = float(wilcoxon(x, y).pvalue) if np.any(x != y) else 1.0
            except ValueError:
                p = float("nan")
            tests[f"{a} -> {b}: {k}"] = {
                "mediana_diff": round(float(np.median(y - x)), 3),
                "p_wilcoxon": round(p, 4)}
    out["test_appaiati"] = tests
    out["casi_isocentro_nel_tumore"] = sum(
        str(by["ctv_statico"][c]["iso_ok"]) == "True" for c in cases)
    return out


def print_summary(s):
    print(f"\n{s['n_casi']} casi ({s['casi_isocentro_nel_tumore']} con "
          f"isocentro dentro il tumore vero)\n")
    print(f"{'scenario':18s} {'copertura GT %':>20s} {'>=95%':>6s} "
          f"{'PTV cm3':>18s} {'V20 %':>16s} {'MLD Gy':>16s}")

    def f(d):
        return f"{d['mediana']:.1f} [{d['q1']:.1f}-{d['q3']:.1f}]"
    for name, d in s["scenari"].items():
        print(f"{name:18s} {f(d['copertura_pct']):>20s} "
              f"{d['casi_copertura_ge_95_pct']:5.0f}% "
              f"{f(d['ptv_cm3']):>18s} {f(d['polmone_v20_pct']):>16s} "
              f"{f(d['polmone_dose_media_gy']):>16s}")
    print("\ntest appaiati (Wilcoxon); differenza mediana = secondo - primo")
    for k, v in s["test_appaiati"].items():
        print(f"  {k:50s} {v['mediana_diff']:+9.2f}   p = {v['p_wilcoxon']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--cases", default="")
    ap.add_argument("--summary", action="store_true",
                    help="solo riepilogo dei risultati gia' calcolati")
    ap.add_argument("--only", default="",
                    help="ricalcola SOLO questi scenari sui casi gia' fatti "
                         "(le righe vecchie finiscono in per_case_prima.csv)")
    a = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUT_DIR, "per_case.csv")
    done = []
    if os.path.exists(csv_path):
        with open(csv_path, encoding="utf-8") as fh:
            done = list(csv.DictReader(fh))
    if a.only:
        only = [x.strip() for x in a.only.split(",")]
        old = [r for r in done if r["scenario"] in only]
        with open(os.path.join(OUT_DIR, "per_case_prima.csv"), "w",
                  newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
            w.writeheader()
            w.writerows(old)
        cases = sorted({r["case_id"] for r in done})
        fold_of = {r["case_id"]: r["fold"] for r in done}
        new_rows = []
        for i, cid in enumerate(cases, 1):
            rows = run_case(cid, fold_of[cid], only=only)
            new_rows.extend({k: str(v) for k, v in r.items()} for r in rows)
            print(f"[{i}/{len(cases)}] {cid} " + "  ".join(
                f"{r['scenario']}: copertura {r['copertura_pct']:.0f}%, PTV "
                f"{r['ptv_cm3']:.1f} cm3" for r in rows), flush=True)
        done = [r for r in done if r["scenario"] not in only] + new_rows
        with open(csv_path, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
            w.writeheader()
            w.writerows(done)
    elif not a.summary:
        import evaluate as E
        if a.cases:
            sel = [c.strip() for c in a.cases.split(",") if c.strip()]
            _, fold_of = E.resolve_cases("splits:all")
        else:
            sel, fold_of, n_acc = select_cases(a.n)
            print(f"[campagna] {n_acc} casi accettati dal cancello, "
                  f"selezionati {len(sel)}: {', '.join(sel)}", flush=True)
        have = {r["case_id"] for r in done}
        todo = [c for c in sel if c not in have]
        new = not os.path.exists(csv_path)
        t0 = time.time()
        for i, cid in enumerate(todo, 1):
            rows = run_case(cid, fold_of.get(cid),
                            status_cb=lambda m: print("   ", m, flush=True))
            with open(csv_path, "a", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=FIELDS,
                                   extrasaction="ignore")
                if new:
                    w.writeheader()
                    new = False
                w.writerows(rows)
            done.extend({k: str(v) for k, v in r.items()} for r in rows)
            el = time.time() - t0
            cov = "  ".join(f"{r['scenario']} {r['copertura_pct']:.0f}"
                            for r in rows)
            print(f"[{i}/{len(todo)}] {cid}  copertura GT: {cov}  "
                  f"({el / 60:.0f} min, ~{el / i * (len(todo) - i) / 60:.0f} "
                  f"alla fine)", flush=True)
    s = summarize(done)
    s["configurazione"] = {"postproc": vars(config.POSTPROC),
                           "gate": vars(config.GATE),
                           "moto": mm.MotionConfig().as_dict(),
                           "scenari": SCENARIOS}
    with open(os.path.join(OUT_DIR, "summary.json"), "w",
              encoding="utf-8") as fh:
        json.dump(s, fh, indent=2)
    print_summary(s)


if __name__ == "__main__":
    main()
