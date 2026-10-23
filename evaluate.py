# ============================================================================
#  evaluate.py - Valutazione di un modello con il post-processing DI DEPLOYMENT.
#
#  E' il punto in cui si chiude lo scarto fra "il numero riportato" e "cio' che
#  il sistema fa davvero": tutte le metriche sono calcolate nello spazio CT
#  ORIGINALE, dopo Invertd, passando per postproc.probs_to_mask - la stessa
#  funzione che inference.py usa in produzione. Un miglioramento che non si
#  vede qui non e' un miglioramento.
#
#  DUE RIGHE DI BASELINE, E NON SONO INTERCAMBIABILI
#    (a) outputs/best_model_dice.pth valutato sui SUOI 12 casi held-out.
#        E' l'unico numero onesto ottenibile da quel file: gli altri 51 casi
#        erano nel suo training set, e includerli gonfierebbe il risultato.
#        n=12, intervallo di confidenza larghissimo: va in nota, non in tabella.
#    (b) la stessa ricetta riaddestrata sotto la CV a 5 fold, n=63 out-of-fold.
#        E' il riferimento con cui confrontare ogni ablazione successiva.
#
#  USO:
#    # riga (a): il checkpoint legacy sui suoi 12 casi
#    python evaluate.py --exp legacy_unet --ckpt outputs/best_model_dice.pth \
#           --cases legacy_val12 --out outputs/eval/legacy_unet
#
#    # sweep del punto di lavoro sulle probabilita' gia' salvate (niente GPU)
#    python evaluate.py --exp legacy_unet --out outputs/eval/legacy_unet \
#           --from-probs --sweep
# ============================================================================
import os
import csv
import json
import time
import argparse
import subprocess

import numpy as np
import nibabel as nib
import torch
from scipy import ndimage

import config
import splits as splits_mod
import postproc
import metrics_det as M
import predict as P


#: I 12 casi di validazione del vecchio split (primi 20% della lista ordinata,
#: dataset.py:171-173). Elencati esplicitamente perche' la riga (a) deve restare
#: riproducibile anche dopo che quel codice sara' stato sostituito.
LEGACY_VAL12 = ["lung_001", "lung_003", "lung_004", "lung_005", "lung_006",
                "lung_009", "lung_010", "lung_014", "lung_015", "lung_016",
                "lung_018", "lung_020"]


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return "n/a"


def resolve_cases(spec: str) -> tuple[list[str], dict]:
    """Elenco di case_id da valutare, piu' la mappa caso -> fold.

    spec: 'legacy_val12' | 'splits:all' | 'splits:fold<N>' | 'a,b,c'
    """
    fold_of = {}
    try:
        sp = splits_mod.load_splits()
        for entry in sp["folds"]:
            for c in entry["val"]:
                fold_of[c] = entry["fold"]
    except FileNotFoundError:
        sp = None

    if spec == "legacy_val12":
        return list(LEGACY_VAL12), fold_of
    if spec.startswith("splits:"):
        if sp is None:
            raise FileNotFoundError("Serve outputs/splits_5fold.json: python splits.py")
        what = spec.split(":", 1)[1]
        if what == "all":
            return sorted(fold_of), fold_of
        if what.startswith("fold"):
            f = int(what[4:])
            return list(sp["folds"][f]["val"]), fold_of
        raise ValueError(f"Selettore non riconosciuto: {spec}")
    return [c.strip() for c in spec.split(",") if c.strip()], fold_of


def case_paths(case_id: str) -> tuple[str, str]:
    img = os.path.join(config.IMAGES_DIR, f"{case_id}.nii.gz")
    lbl = os.path.join(config.LABELS_DIR, f"{case_id}.nii.gz")
    return img, lbl


def load_gt(label_path: str):
    img = nib.load(label_path)
    return np.asanyarray(img.dataobj) > 0, np.asarray(img.affine, dtype=np.float64)


# ---------------------------------------------------------------------------
#  Sweep del punto di lavoro
# ---------------------------------------------------------------------------
DEFAULT_THRESHOLDS = (0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90)
DEFAULT_MIN_VOLS = (0.0, 50.0, 100.0, 250.0, 500.0, 1000.0)


def _common_bbox(prob, gt, floor: float, margin: int = 4):
    """Ritaglio comune a predizione e ground truth.

    Fuori da questo box entrambi sono vuoti per ogni soglia >= floor, quindi
    ogni metrica (Dice, componenti connesse, distanze fra centroidi) e'
    identica calcolata sul ritaglio o sul volume intero - ma su un volume da
    512x512x304 = 79 milioni di voxel, ripetere ndimage.label per 54
    combinazioni di soglia e volume costa minuti per caso. Con il ritaglio si
    lavora tipicamente su un decimo dei voxel.

    Il margine evita che una struttura tocchi il bordo dell'array, il che ne
    altererebbe la superficie.
    """
    hot = (prob > floor) | gt
    if not hot.any():
        return None
    sl = ndimage.find_objects(hot.astype(np.uint8))[0]
    return tuple(slice(max(s.start - margin, 0), min(s.stop + margin, n))
                 for s, n in zip(sl, prob.shape))


def sweep(prob_dir: str, case_ids: list[str],
          thresholds=DEFAULT_THRESHOLDS, min_vols=DEFAULT_MIN_VOLS) -> list[dict]:
    """Griglia soglia x volume minimo sulle probabilita' gia' salvate.

    Nessuna GPU: e' aritmetica su array. E' il motivo per cui predict.py salva
    le probabilita' invece delle maschere.
    """
    # Le probabilita' e il ground truth si caricano UNA volta per caso e si
    # riusano su tutta la griglia: ricaricarli per ogni combinazione
    # moltiplicherebbe l'I/O per ~54.
    rows = []
    cache = {}
    floor = min(thresholds)
    for cid in case_ids:
        p_path = P.prob_path(prob_dir, cid)
        if not os.path.exists(p_path):
            continue
        prob, affine = P.load_prob(p_path)
        gt, _ = load_gt(case_paths(cid)[1])
        box = _common_bbox(prob, gt, floor)
        if box is not None:
            prob, gt = prob[box], gt[box]
        cache[cid] = (np.ascontiguousarray(prob), affine, np.ascontiguousarray(gt))
    if cache:
        tot = sum(p.size for p, _, _ in cache.values())
        print(f"  ({len(cache)} casi ritagliati a {tot/1e6:.1f}M voxel totali)")

    for th in thresholds:
        for mv in min_vols:
            cfg = postproc.PostProcConfig(prob_threshold=th, min_lesion_mm3=mv)
            per_case = []
            for cid, (prob, affine, gt) in cache.items():
                mask, _ = postproc.probs_to_mask(prob, cfg, affine=affine)
                per_case.append(M.case_metrics(gt, mask, affine, min_lesion_mm3=mv or M.MIN_LESION_MM3,
                                               case_id=cid, surface=False))
            agg = M.aggregate(per_case)
            d = agg["pooled_detection"]
            rows.append({
                "threshold": th, "min_lesion_mm3": mv,
                "n_cases": len(per_case),
                "dice_mean": agg["pooled"]["dice"]["mean"],
                "sensitivity": d["sensitivity"],
                "fp_per_scan": d["fp_per_scan"],
                "isocenter_inside_gt_rate": d["isocenter_inside_gt_rate"],
                "cases_no_detection": d["cases_with_no_detection"],
                "centroid_dist_mm_median": agg["pooled"]["centroid_dist_mm"]["median"],
            })
            print(f"  th={th:.2f} minvol={mv:>7.0f}  dice={rows[-1]['dice_mean']:.4f} "
                  f"sens={rows[-1]['sensitivity']:.3f} "
                  f"fp/scan={rows[-1]['fp_per_scan']:.2f} "
                  f"iso_in={rows[-1]['isocenter_inside_gt_rate']:.3f}")
    return rows


def best_operating_point(rows: list[dict], max_fp_per_scan: float = 1.0) -> dict | None:
    """Massima sensibilita' con FP/scan sotto il tetto; a pari merito, meno FP.

    Il criterio e' dichiarato PRIMA di guardare i risultati: e' cio' che
    distingue una scelta di progetto da un cherry-picking a posteriori.
    """
    ok = [r for r in rows if r["fp_per_scan"] <= max_fp_per_scan
          and not np.isnan(r["sensitivity"])]
    if not ok:
        return None
    return max(ok, key=lambda r: (r["sensitivity"], -r["fp_per_scan"], r["dice_mean"]))


# ---------------------------------------------------------------------------
#  Programma principale
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Valutazione con post-processing di deployment.")
    ap.add_argument("--exp", required=True, help="nome dell'esperimento (etichetta)")
    ap.add_argument("--ckpt", nargs="*", default=[],
                    help="uno o piu' checkpoint; piu' di uno = ensemble")
    ap.add_argument("--oof", default=None, metavar="EXP",
                    help="valutazione out-of-fold: per ogni caso usa il modello "
                         "del fold che lo escludeva (outputs/runs/EXP/fold*/)")
    ap.add_argument("--ckpt-name", default="final.pth",
                    help="quale checkpoint per fold in modalita' --oof. "
                         "final.pth e' il primario (vedi train.py): best.pth "
                         "e' selezionato sulla validazione e quindi ottimistico")
    ap.add_argument("--cases", default="splits:all",
                    help="legacy_val12 | splits:all | splits:fold<N> | id,id,...")
    ap.add_argument("--out", required=True)
    ap.add_argument("--from-probs", action="store_true",
                    help="riusa SEMPRE le probabilita' salvate, senza rete")
    ap.add_argument("--resume", action="store_true",
                    help="riusa le probabilita' gia' presenti e calcola solo i "
                         "casi mancanti (per riprendere una valutazione interrotta)")
    ap.add_argument("--tta", type=int, default=0,
                    help="numero di assi da specchiare (0=nessuna TTA, 1=solo L-R)")
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--min-lesion-mm3", type=float, default=None)
    ap.add_argument("--overlap", type=float, default=None)
    ap.add_argument("--blend", default="constant", choices=("constant", "gaussian"),
                    help="fusione delle finestre sovrapposte; vedi predict.predict_case")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--max-fp-per-scan", type=float, default=1.0)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    prob_dir = os.path.join(args.out, "probs")
    os.makedirs(prob_dir, exist_ok=True)

    case_ids, fold_of = resolve_cases(args.cases)
    print(f"[eval] esperimento '{args.exp}' | {len(case_ids)} casi | "
          f"selettore '{args.cases}'")

    cfg = postproc.PostProcConfig(
        prob_threshold=(args.threshold if args.threshold is not None
                        else config.POSTPROC.prob_threshold),
        min_lesion_mm3=(args.min_lesion_mm3 if args.min_lesion_mm3 is not None
                        else config.POSTPROC.min_lesion_mm3),
    )
    print(f"[eval] post-processing: soglia={cfg.prob_threshold} "
          f"min_volume={cfg.min_lesion_mm3} mm3")

    # --- Modelli -------------------------------------------------------------
    #
    # In modalita' OOF ogni caso e' predetto SOLO dal modello del fold che lo
    # teneva fuori dal training. E' la condizione che rende i 63 casi una stima
    # onesta: usare per un caso un modello che l'ha visto in training non
    # misura la generalizzazione, misura la memorizzazione.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = []
    models_by_fold: dict[int, list] = {}

    if not args.from_probs:
        from inference import load_model
        if args.oof:
            run_root = os.path.join(config.OUTPUT_DIR, "runs", args.oof)
            for f in sorted({v for v in fold_of.values()}):
                ck = os.path.join(run_root, f"fold{f}", args.ckpt_name)
                if not os.path.exists(ck):
                    ap.error(f"checkpoint mancante per il fold {f}: {ck}")
                models_by_fold[f] = [load_model(ck, device)]
            # In OOF si valutano per definizione tutti i casi out-of-fold.
            case_ids = sorted(fold_of)
            print(f"[eval] OOF su '{args.oof}' ({args.ckpt_name}): "
                  f"{len(models_by_fold)} fold, {len(case_ids)} casi")
        else:
            if not args.ckpt:
                ap.error("servono --ckpt ..., --oof EXP, oppure --from-probs")
            for c in args.ckpt:
                models.append(load_model(c, device))
            print(f"[eval] {len(models)} modello/i su {device}")

    def models_for(case_id: str):
        if not models_by_fold:
            return models
        f = fold_of.get(case_id)
        if f is None or f not in models_by_fold:
            raise KeyError(f"caso '{case_id}' non appartiene a nessun fold OOF")
        return models_by_fold[f]

    # TTA: si specchiano gli assi spaziali del tensore (2,3,4) = (X,Y,Z).
    # Con --tta 1 si usa solo l'asse 2, che dopo l'orientamento RAS e' l'asse
    # destra-sinistra del paziente: l'unico specchiamento anatomicamente
    # plausibile su un torace.
    tta_axes = tuple((2, 3, 4)[:max(args.tta, 0)])

    # --- Predizione + metriche ----------------------------------------------
    rows = []
    t0 = time.time()
    for i, cid in enumerate(case_ids, 1):
        img_path, lbl_path = case_paths(cid)
        if not os.path.exists(lbl_path):
            print(f"  [{i}/{len(case_ids)}] {cid}: nessuna label, saltato")
            continue

        p_path = P.prob_path(prob_dir, cid)
        # --resume riusa le probabilita' gia' salvate. Una valutazione OOF su 63
        # casi dura ~20 minuti e scrive un .npz per caso man mano: se il
        # processo muore a meta' (e' successo: terminato al caso 57 su 63),
        # rifare da zero la parte gia' calcolata e' puro spreco di GPU.
        reuse = args.from_probs or (args.resume and os.path.exists(p_path))
        if reuse:
            if not os.path.exists(p_path):
                print(f"  [{i}/{len(case_ids)}] {cid}: probabilita' assenti, saltato")
                continue
            prob, affine = P.load_prob(p_path)
        else:
            out = P.predict_case(models_for(cid), img_path, device=device,
                                 tta_axes=tta_axes, overlap=args.overlap,
                                 blend_mode=args.blend)
            prob, affine = out["prob"], out["affine"]
            P.save_prob(p_path, prob, affine)

        gt, gt_affine = load_gt(lbl_path)
        if gt.shape != prob.shape:
            raise RuntimeError(
                f"{cid}: forma GT {gt.shape} != predizione {prob.shape}. "
                "Invertd non ha riportato la predizione nello spazio originale.")

        mask, lesions = postproc.probs_to_mask(prob, cfg, affine=affine)
        m = M.case_metrics(gt, mask, gt_affine, min_lesion_mm3=cfg.min_lesion_mm3,
                           prob=prob, case_id=cid)
        m["fold"] = fold_of.get(cid)
        m["n_lesions_kept"] = len(lesions)
        rows.append(m)

        print(f"  [{i:>2}/{len(case_ids)}] {cid}  dice={m['dice']:.3f} "
              f"sens={m['sensitivity']:.2f} fp={m['fp']} "
              f"iso_in={'si' if m['isocenter_inside_gt'] else 'NO'} "
              f"d_cent={m['centroid_dist_mm']:.1f}mm")

    if not rows:
        print("[eval] nessun caso valutato.")
        return

    # --- Aggregazione e salvataggi ------------------------------------------
    agg = M.aggregate(rows, fold_of=fold_of)
    agg["exp"] = args.exp
    agg["oof_exp"] = args.oof
    agg["ckpt_name"] = args.ckpt_name if args.oof else None
    agg["cases_selector"] = args.cases
    agg["postproc"] = {"prob_threshold": cfg.prob_threshold,
                       "min_lesion_mm3": cfg.min_lesion_mm3}
    agg["tta_axes"] = list(tta_axes)
    agg["blend_mode"] = args.blend
    agg["elapsed_s"] = round(time.time() - t0, 1)

    csv_path = os.path.join(args.out, "per_case.csv")
    fields = ["case_id", "fold", "dice", "sensitivity", "sensitivity_iou10",
              "n_gt_lesions", "n_pred_lesions", "n_lesions_kept", "tp", "fn", "fp",
              "centroid_dist_mm", "centroid_dist_over_gt_radius",
              "isocenter_inside_gt", "nsd_2mm", "hd95_mm",
              "vol_gt_cm3", "vol_pred_cm3", "vol_err_pct", "detected"]
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(agg, fh, indent=2, default=float)

    with open(os.path.join(args.out, "config.json"), "w", encoding="utf-8") as fh:
        json.dump({
            "exp": args.exp, "checkpoints": args.ckpt, "cases": args.cases,
            "oof_exp": args.oof, "ckpt_name": args.ckpt_name,
            "target_spacing": list(config.TARGET_SPACING),
            "hu_window": [config.HU_MIN, config.HU_MAX],
            "sw_roi_size": list(config.SW_ROI_SIZE),
            "sw_overlap": args.overlap if args.overlap is not None else config.SW_OVERLAP,
            "tta_axes": list(tta_axes),
            "blend_mode": args.blend,
            "postproc": agg["postproc"],
            "git_commit": git_commit(),
            "torch": torch.__version__,
        }, fh, indent=2)

    report(agg)

    # --- Sweep opzionale -----------------------------------------------------
    if args.sweep:
        print("\n[eval] sweep del punto di lavoro (nessuna GPU)")
        srows = sweep(prob_dir, [r["case_id"] for r in rows])
        with open(os.path.join(args.out, "sweep.csv"), "w", newline="",
                  encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(srows[0]))
            w.writeheader()
            w.writerows(srows)
        best = best_operating_point(srows, args.max_fp_per_scan)
        if best:
            print(f"\n[eval] punto di lavoro migliore a <= {args.max_fp_per_scan} FP/scan:")
            print(f"       soglia={best['threshold']:.2f} "
                  f"min_volume={best['min_lesion_mm3']:.0f} mm3 -> "
                  f"sens={best['sensitivity']:.3f} fp/scan={best['fp_per_scan']:.2f} "
                  f"dice={best['dice_mean']:.4f} "
                  f"iso_in={best['isocenter_inside_gt_rate']:.3f}")
            with open(os.path.join(args.out, "operating_point.json"), "w",
                      encoding="utf-8") as fh:
                json.dump(best, fh, indent=2, default=float)


def report(agg: dict) -> None:
    p, d = agg["pooled"], agg["pooled_detection"]
    print("\n" + "=" * 68)
    print(f"  {agg['exp']}   ({agg['n_cases']} casi, {agg['cases_selector']})")
    print("=" * 68)
    print(f"  Dice                     {p['dice']['mean']:.4f} +- {p['dice']['std']:.4f}"
          f"   mediana {p['dice']['median']:.4f}")
    print(f"  Sensibilita' (lesioni)   {d['sensitivity']:.3f}   "
          f"({d['tp']}/{d['lesions_gt']})")
    print(f"  Falsi positivi / scan    {d['fp_per_scan']:.2f}   (totale {d['fp']})")
    print(f"  Isocentro dentro il GT   {d['isocenter_inside_gt_rate']:.3f}")
    print(f"  Casi senza rilevamento   {d['cases_with_no_detection']}")
    print(f"  Distanza centroide (mm)  mediana {p['centroid_dist_mm']['median']:.2f}"
          f"   media {p['centroid_dist_mm']['mean']:.2f}")
    print(f"  NSD @2mm                 {p['nsd_2mm']['mean']:.4f}")
    print(f"  HD95 (mm)                mediana {p['hd95_mm']['median']:.2f}")

    if "by_size_tertile" in agg:
        print("\n  Per terzile di volume tumorale:")
        print(f"    {'terzile':<10}{'n':>4}{'range cm3':>18}{'dice':>9}{'sens':>8}{'iso_in':>8}")
        for name, g in agg["by_size_tertile"].items():
            rng = g["vol_range_cm3"]
            print(f"    {name:<10}{g['n']:>4}{f'{rng[0]:.2f}-{rng[1]:.2f}':>18}"
                  f"{g['dice']['mean']:>9.4f}{g['sensitivity']:>8.3f}"
                  f"{g['isocenter_inside_gt_rate']:>8.3f}")

    if "missed_lesion_max_prob" in agg:
        mp = agg["missed_lesion_max_prob"]
        print(f"\n  DIAGNOSTICA - probabilita' massima sulle {mp['n_missed']} lesioni mancate:")
        print(f"    mediana {mp['median']:.3f} | sotto 0.1: {mp['frac_below_0.1']:.0%} "
              f"| fra 0.1 e 0.5: {mp['frac_0.1_to_0.5']:.0%}")
        if mp["frac_below_0.1"] > 0.5:
            print("    -> la rete non produce massa di probabilita' li': il collo di")
            print("       bottiglia e' la loss/risoluzione, non la soglia (E6/E7 giustificate).")
        else:
            print("    -> c'e' massa di probabilita' sotto 0.5: il collo di bottiglia e'")
            print("       la SOGLIA. Muoverla costa zero; E6 sarebbe ridondante.")
    print("=" * 68 + "\n")


if __name__ == "__main__":
    main()
