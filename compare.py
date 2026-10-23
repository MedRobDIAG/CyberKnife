# ============================================================================
#  compare.py - Confronto APPAIATO fra due esperimenti valutati out-of-fold.
#
#  PERCHE' APPAIATO
#  I 63 casi hanno difficolta' enormemente diversa: il Dice per caso va da 0 a
#  0.9 e il volume tumorale da 0.74 a 370 cm3. Un test a due campioni su medie
#  di 63 valori con quella dispersione non ha alcuna potenza: la varianza fra
#  PAZIENTI schiaccerebbe qualunque differenza fra MODELLI. Confrontando invece
#  gli stessi pazienti valutati dai due modelli, quella varianza si elide e
#  resta solo cio' che interessa.
#
#  Si usa Wilcoxon dei ranghi con segno e non un t-test appaiato: la
#  distribuzione delle differenze di Dice non e' normale (e' bimodale, con una
#  coda di fallimenti totali a 0), e con n=63 il teorema centrale del limite
#  non salva un t-test su dati cosi' asimmetrici.
#
#  REGOLA DI DECISIONE, dichiarata prima di guardare i risultati:
#  si tiene una modifica se migliora la sensibilita' pooled a FP/scan costante,
#  OPPURE il Dice con p<0.05 appaiato, OPPURE e' neutra ma piu' economica.
#  Per questo progetto conta anche - e soprattutto - isocenter_inside_gt.
#
#  USO:
#    python compare.py --a outputs/eval/baseline_unet_oof \
#                      --b outputs/eval/aug_anatomic_oof
# ============================================================================
import os
import csv
import json
import math
import argparse
import statistics

import metrics_det as M


#: Metriche confrontate caso per caso. 'higher' dice se piu' alto e' meglio.
KEYS = (
    ("dice", True),
    ("centroid_dist_mm", False),
    ("nsd_2mm", True),
    ("hd95_mm", False),
)


def load_rows(eval_dir: str) -> list[dict]:
    """per_case.csv -> righe con i campi numerici convertiti."""
    path = os.path.join(eval_dir, "per_case.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"manca {path}: esegui prima evaluate.py --oof")
    rows = []
    with open(path, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            out = {"case_id": r["case_id"]}
            for k, v in r.items():
                if k == "case_id":
                    continue
                if v in ("", "nan", "None"):
                    out[k] = float("nan")
                elif v in ("True", "False"):
                    out[k] = (v == "True")
                else:
                    try:
                        out[k] = float(v)
                    except ValueError:
                        out[k] = v
            rows.append(out)
    return rows


def summary(eval_dir: str) -> dict:
    with open(os.path.join(eval_dir, "summary.json"), encoding="utf-8") as fh:
        return json.load(fh)


def fmt(x, nd=4):
    return "n/d" if (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}"


def main():
    ap = argparse.ArgumentParser(description="Confronto appaiato fra due esperimenti.")
    ap.add_argument("--a", required=True, help="cartella di riferimento")
    ap.add_argument("--b", required=True, help="cartella della variante")
    ap.add_argument("--name-a", default=None)
    ap.add_argument("--name-b", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    name_a = args.name_a or os.path.basename(args.a.rstrip("/\\"))
    name_b = args.name_b or os.path.basename(args.b.rstrip("/\\"))
    rows_a, rows_b = load_rows(args.a), load_rows(args.b)
    sa, sb = summary(args.a), summary(args.b)

    ids = sorted({r["case_id"] for r in rows_a} & {r["case_id"] for r in rows_b})
    print(f"\n{'='*74}")
    print(f"  {name_a}   ->   {name_b}     ({len(ids)} casi appaiati)")
    print(f"{'='*74}")

    # --- Metriche aggregate (non appaiabili: sono tassi su tutte le lesioni) --
    da, db = sa["pooled_detection"], sb["pooled_detection"]
    print(f"\n  {'metrica':<28}{name_a[:14]:>15}{name_b[:14]:>15}{'delta':>11}")
    print("  " + "-" * 69)
    for lab, key, nd in (("Sensibilita' lesionale", "sensitivity", 3),
                         ("Falsi positivi / scan", "fp_per_scan", 2),
                         ("Isocentro dentro il GT", "isocenter_inside_gt_rate", 3)):
        va, vb = da[key], db[key]
        print(f"  {lab:<28}{fmt(va,nd):>15}{fmt(vb,nd):>15}{vb-va:>+11.{nd}f}")
    va = da["cases_with_no_detection"]; vb = db["cases_with_no_detection"]
    print(f"  {'Casi senza rilevamento':<28}{va:>15}{vb:>15}{vb-va:>+11}")

    # --- Test appaiati -------------------------------------------------------
    print(f"\n  Test appaiati di Wilcoxon (n={len(ids)}):")
    print(f"  {'metrica':<22}{'media A':>10}{'media B':>10}{'mediana d':>11}"
          f"{'B meglio':>10}{'p':>9}")
    print("  " + "-" * 72)
    results = {}
    for key, higher in KEYS:
        t = M.paired_test(rows_a, rows_b, key)
        results[key] = t
        if "p_value" not in t or math.isnan(t.get("p_value", float("nan"))):
            print(f"  {key:<22}{'  campione insufficiente':>50}")
            continue
        star = "*" if t["significant_005"] else " "
        better = t["n_better"] if higher else t["n_worse"]
        print(f"  {key:<22}{t['mean_a']:>10.4f}{t['mean_b']:>10.4f}"
              f"{t['median_delta']:>+11.4f}{better:>7}/{t['n']:<3}"
              f"{t['p_value']:>8.4f}{star}")
    print("  (* = p < 0.05; 'B meglio' conta i casi in cui B batte A"
          " nella direzione giusta)")

    # --- Verdetto secondo la regola dichiarata -------------------------------
    dice = results.get("dice", {})
    # Miglioramento STRETTO, non "non peggiore": un modello identico ad A
    # soddisferebbe "non peggiore" e verrebbe raccomandato, il che rende il
    # verdetto inutile. La tolleranza sui FP e' invece bilaterale, perche' li'
    # si accetta un piccolo peggioramento in cambio di piu' sensibilita'.
    sens_up = db["sensitivity"] > da["sensitivity"]
    fp_ok = db["fp_per_scan"] <= da["fp_per_scan"] + 0.05
    iso_up = db["isocenter_inside_gt_rate"] > da["isocenter_inside_gt_rate"]
    dice_sig = bool(dice.get("significant_005")) and dice.get("mean_b", 0) > dice.get("mean_a", 0)

    print(f"\n  VERDETTO (regola dichiarata a priori):")
    reasons = []
    if sens_up and fp_ok:
        reasons.append("sensibilita' migliore a FP/scan invariati")
    if dice_sig:
        reasons.append(f"Dice migliore con p={dice['p_value']:.4f}")
    if iso_up:
        reasons.append("isocentro dentro il GT piu' spesso")
    if reasons:
        print(f"    TENERE '{name_b}': " + "; ".join(reasons))
    else:
        print(f"    NON tenere '{name_b}': nessun criterio soddisfatto.")
        print(f"    (sens {da['sensitivity']:.3f}->{db['sensitivity']:.3f}, "
              f"fp/scan {da['fp_per_scan']:.2f}->{db['fp_per_scan']:.2f}, "
              f"iso {da['isocenter_inside_gt_rate']:.3f}->"
              f"{db['isocenter_inside_gt_rate']:.3f})")
    print(f"{'='*74}\n")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump({"a": name_a, "b": name_b, "n_paired": len(ids),
                       "detection_a": da, "detection_b": db,
                       "paired_tests": results}, fh, indent=2, default=float)
        print(f"[compare] scritto {args.out}")


if __name__ == "__main__":
    main()
