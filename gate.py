# ============================================================================
#  gate.py - Cancello di confidenza sulla predizione del tumore.
#
#  IL PROBLEMA CHE RISOLVE
#  Misurato sui 63 casi out-of-fold, l'errore del centroide predetto rispetto
#  al tumore vero e' BIMODALE, non continuo:
#      36 casi su 58  entro   5 mm   (mediana 2.5 mm)
#      14 casi su 58  oltre  20 mm   (tipicamente 80-310 mm)
#      quasi nulla in mezzo (p75 = 11 mm, p90 = 155 mm)
#  Cioe': quando il modello funziona e' ottimo, quando fallisce non degrada -
#  indica un'altra regione del torace. Nel 27% dei casi l'isocentro cade FUORI
#  dal tumore, e il sistema lo consegna al planner senza alcun segnale. Un
#  piano dosimetrico costruito su quel bersaglio irradia tessuto sano.
#
#  Nessuna taratura della soglia di probabilita' risolve questo: lo sweep su 54
#  combinazioni porta l'isocentro corretto da 0.730 a 0.746 nel migliore dei
#  casi, cioe' UN caso su 63. La soglia sceglie quanto essere aggressivi, non
#  quando fidarsi.
#
#  L'IDEA
#  I fallimenti sono riconoscibili SENZA ground truth. Mediane, casi riusciti
#  contro falliti:
#      volume della lesione primaria    4.15  vs  0.46 cm3
#      probabilita' media sulla lesione 0.85  vs  0.61
#      rapporto volume 1a / 2a lesione   inf  vs  1.74
#  Il fallimento tipico e' una predizione piccola, poco confidente, con un
#  secondo candidato di volume paragonabile: il modello non sa quale sia il
#  tumore. Un cancello che RIFIUTA questi casi trasforma "consegna un bersaglio
#  sbagliato in silenzio" in "dichiara di non essere affidabile" - che per un
#  sistema che pilota un acceleratore lineare e' una differenza sostanziale.
#
#  VALIDAZIONE ANNIDATA
#  La soglia si sceglie sui 4 fold di training e si applica al quinto, cinque
#  volte. Sceglierla su tutti i 63 casi e poi riportarne l'accuratezza sugli
#  stessi 63 sarebbe la stessa scorciatoia che ha prodotto il "Dice 0.586"
#  storico: un numero scelto per essere alto, non misurato.
#
#  USO:
#    python gate.py --eval-dir outputs/eval/baseline_unet_oof
# ============================================================================
import os
import csv
import json
import argparse
from dataclasses import asdict

import numpy as np

import config
import postproc
import predict as P
import splits as splits_mod
# GateConfig vive in postproc.py (modulo foglia): config.py deve poterla
# importare, e da qui passerebbe per splits/dataset tornando su config.
from postproc import GateConfig


#: Soglie candidate sul volume della lesione primaria (cm3).
VOLUME_GRID = (0.0, 0.2, 0.3, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0)

#: Soglie candidate sulla probabilita' media dentro la lesione primaria.
PROBMEAN_GRID = (0.0, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80)

#: Costo di un caso BUONO rifiutato, in unita' di "bersaglio sbagliato
#: consegnato". 0.25 = far rivedere a mano quattro casi buoni costa quanto
#: consegnare al planner un bersaglio sbagliato. E' l'unico parametro
#: soggettivo del cancello: cambiarlo sposta il punto di lavoro lungo il
#: compromesso, e va dichiarato nella relazione insieme al risultato.
GOOD_REJECTION_COST = 0.25


# ---------------------------------------------------------------------------
#  Raccolta dei segnali
# ---------------------------------------------------------------------------
def case_signals(prob_dir: str, case_id: str, cfg=None) -> dict:
    """Segnali di confidenza di un caso, calcolabili senza ground truth."""
    cfg = cfg or config.POSTPROC
    prob, affine = P.load_prob(P.prob_path(prob_dir, case_id))
    _, lesions = postproc.probs_to_mask(prob, cfg, affine=affine)
    primary = postproc.primary_lesion(lesions)
    if primary is None:
        return {"case_id": case_id, "n_lesions": 0, "volume_cm3": 0.0,
                "prob_mean": 0.0, "prob_max": 0.0, "vol_ratio": 0.0,
                "primary": None}
    return {
        "case_id": case_id,
        "n_lesions": len(lesions),
        "volume_cm3": primary["volume_cm3"],
        "prob_mean": primary["prob_mean"],
        "prob_max": primary["prob_max"],
        # Quanto la primaria domina la seconda. Un rapporto vicino a 1 significa
        # due candidati equivalenti: quale sia il tumore e' una scommessa.
        "vol_ratio": (primary["volume_cm3"] / lesions[1]["volume_cm3"]
                      if len(lesions) > 1 else float("inf")),
        "primary": primary,
    }


def load_outcomes(eval_dir: str) -> dict:
    """case_id -> esito reale (isocentro dentro il tumore) dal per_case.csv."""
    path = os.path.join(eval_dir, "per_case.csv")
    out = {}
    with open(path, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            out[r["case_id"]] = {
                "ok": r["isocenter_inside_gt"] == "True",
                "dice": float(r["dice"]) if r["dice"] not in ("", "nan") else float("nan"),
                "centroid_dist_mm": (float(r["centroid_dist_mm"])
                                     if r["centroid_dist_mm"] not in ("", "nan")
                                     else float("nan")),
                "vol_gt_cm3": float(r["vol_gt_cm3"]),
            }
    return out


# ---------------------------------------------------------------------------
#  Scelta e valutazione
# ---------------------------------------------------------------------------
def score_gate(cases: list[dict], outcomes: dict, gate: GateConfig,
               min_accept_frac: float = 0.6) -> dict:
    """Prestazioni di un cancello su un insieme di casi.

    min_accept_frac impedisce la soluzione degenere: un cancello che rifiuta
    quasi tutto raggiunge una precisione perfetta ed e' inutile. Si richiede
    che accetti almeno il 60% dei casi.
    """
    acc = [c for c in cases if gate.accepts(c["primary"])]
    rej = [c for c in cases if not gate.accepts(c["primary"])]
    n = len(cases)
    if not acc:
        return {"valid": False, "accept_frac": 0.0, "precision": 0.0,
                "n_accepted": 0, "n_rejected": n, "utility": -1.0}

    ok_acc = sum(1 for c in acc if outcomes[c["case_id"]]["ok"])
    bad_rej = sum(1 for c in rej if not outcomes[c["case_id"]]["ok"])
    good_rej = sum(1 for c in rej if outcomes[c["case_id"]]["ok"])
    bad_passed = len(acc) - ok_acc
    precision = ok_acc / len(acc)
    accept_frac = len(acc) / n

    # COSTO ATTESO per caso, in unita' di "un bersaglio sbagliato consegnato".
    #
    # Questa forma ha sostituito una precedente (precision - 0.15*buoni_persi/n)
    # che mescolava un tasso e un conteggio, rendendo il compromesso implicito e
    # difficile da giudicare. Con quella, sul modello anatomic, veniva scelto un
    # cancello che rifiutava 12 casi BUONI per intercettare 2 errori in piu' di
    # una soglia sul volume che ne perdeva 3. Il difetto era nel peso, non nei
    # dati: 0.15 sottostimava il costo di un rifiuto.
    #
    # DICHIARAZIONE: il peso e' stato rivisto DOPO aver osservato quel
    # compromesso. Non e' quindi una scelta cieca, e va detto. Il valore ora e'
    # motivato in modo esplicito e verificabile invece che tarato sull'esito:
    # GOOD_REJECTION_COST = 0.25 significa "far rivedere a mano quattro casi
    # buoni costa quanto consegnare al planner un bersaglio sbagliato".
    utility = -(bad_passed + GOOD_REJECTION_COST * good_rej) / n
    if accept_frac < min_accept_frac:
        utility -= 1.0                      # scarta le soluzioni degeneri

    return {
        "valid": accept_frac >= min_accept_frac,
        "n_accepted": len(acc), "n_rejected": len(rej),
        "accept_frac": accept_frac, "precision": precision,
        "bad_rejected": bad_rej, "good_rejected": good_rej,
        "bad_passed": bad_passed, "utility": utility,
    }


def choose_gate(cases, outcomes, min_accept_frac: float = 0.6) -> GateConfig:
    """Cancello con l'utilita' massima sulla griglia."""
    best, best_u = GateConfig(0.0, 0.0), float("-inf")
    for v in VOLUME_GRID:
        for p in PROBMEAN_GRID:
            g = GateConfig(v, p)
            s = score_gate(cases, outcomes, g, min_accept_frac)
            if s["utility"] > best_u:
                best, best_u = g, s["utility"]
    return best


def nested_evaluation(cases_by_id: dict, outcomes: dict, splits: dict,
                      min_accept_frac: float = 0.6) -> dict:
    """Scelta sui 4 fold di training, misura sul quinto. Cinque volte.

    E' l'unico modo di riportare un numero che non sia stato scelto per essere
    alto: nessun caso contribuisce sia alla scelta della soglia sia alla sua
    valutazione.
    """
    rows, chosen = [], []
    for entry in splits["folds"]:
        val_ids = [c for c in entry["val"] if c in cases_by_id]
        tr_ids = [c for c in entry["train"] if c in cases_by_id]
        gate = choose_gate([cases_by_id[c] for c in tr_ids], outcomes,
                           min_accept_frac)
        chosen.append({"fold": entry["fold"], **asdict(gate)})
        for cid in val_ids:
            c = cases_by_id[cid]
            rows.append({
                "case_id": cid, "fold": entry["fold"],
                "accepted": gate.accepts(c["primary"]),
                "ok": outcomes[cid]["ok"],
                "volume_cm3": c["volume_cm3"], "prob_mean": c["prob_mean"],
                "vol_gt_cm3": outcomes[cid]["vol_gt_cm3"],
                "centroid_dist_mm": outcomes[cid]["centroid_dist_mm"],
                "gate_min_volume_cm3": gate.min_volume_cm3,
                "gate_min_prob_mean": gate.min_prob_mean,
            })

    acc = [r for r in rows if r["accepted"]]
    rej = [r for r in rows if not r["accepted"]]
    dists = [r["centroid_dist_mm"] for r in acc
             if r["centroid_dist_mm"] == r["centroid_dist_mm"]]
    return {
        "n_cases": len(rows),
        "n_accepted": len(acc), "n_rejected": len(rej),
        "accept_frac": len(acc) / len(rows) if rows else 0.0,
        "precision_accepted": (sum(1 for r in acc if r["ok"]) / len(acc)
                               if acc else float("nan")),
        "bad_rejected": sum(1 for r in rej if not r["ok"]),
        "good_rejected": sum(1 for r in rej if r["ok"]),
        "baseline_precision_all": sum(1 for r in rows if r["ok"]) / len(rows),
        "centroid_median_accepted_mm": (float(np.median(dists)) if dists
                                        else float("nan")),
        "centroid_p90_accepted_mm": (float(np.percentile(dists, 90)) if dists
                                     else float("nan")),
        "gates_per_fold": chosen,
        "rows": rows,
    }


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Cancello di confidenza, validato in modo annidato.")
    ap.add_argument("--eval-dir", required=True,
                    help="cartella prodotta da evaluate.py --oof")
    ap.add_argument("--min-accept-frac", type=float, default=0.6)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    out_dir = args.out or args.eval_dir
    prob_dir = os.path.join(args.eval_dir, "probs")
    outcomes = load_outcomes(args.eval_dir)
    splits = splits_mod.load_splits()

    print(f"[gate] segnali di confidenza per {len(outcomes)} casi...")
    cases_by_id = {cid: case_signals(prob_dir, cid) for cid in outcomes}

    res = nested_evaluation(cases_by_id, outcomes, splits, args.min_accept_frac)

    print("\n" + "=" * 68)
    print("  CANCELLO DI CONFIDENZA - validazione annidata (5 fold)")
    print("=" * 68)
    print(f"  Senza cancello: isocentro corretto {res['baseline_precision_all']:.1%} "
          f"su {res['n_cases']} casi")
    print(f"  Con cancello  : accettati {res['n_accepted']}/{res['n_cases']} "
          f"({res['accept_frac']:.1%}), di cui corretti "
          f"{res['precision_accepted']:.1%}")
    print(f"  Rifiutati     : {res['n_rejected']} "
          f"({res['bad_rejected']} davvero sbagliati, "
          f"{res['good_rejected']} buoni persi)")
    print(f"  Centroide sugli accettati: mediana "
          f"{res['centroid_median_accepted_mm']:.2f} mm, "
          f"p90 {res['centroid_p90_accepted_mm']:.2f} mm")
    print("\n  Soglie scelte su ciascun fold di training:")
    for g in res["gates_per_fold"]:
        print(f"    fold {g['fold']}: volume >= {g['min_volume_cm3']:.1f} cm3, "
              f"prob media >= {g['min_prob_mean']:.2f}")
    print("=" * 68 + "\n")

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "gate_nested.json"), "w", encoding="utf-8") as fh:
        json.dump({k: v for k, v in res.items() if k != "rows"}, fh,
                  indent=2, default=float)
    with open(os.path.join(out_dir, "gate_per_case.csv"), "w", newline="",
              encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(res["rows"][0]))
        w.writeheader()
        w.writerows(res["rows"])
    print(f"[gate] scritti gate_nested.json e gate_per_case.csv in {out_dir}")


if __name__ == "__main__":
    main()
