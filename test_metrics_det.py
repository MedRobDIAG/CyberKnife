# ============================================================================
#  test_metrics_det.py - Verifica di metrics_det su maschere sintetiche.
#
#  Le metriche sono lo strumento con cui giudicheremo ogni modifica al modello:
#  se sbagliano, ogni conclusione successiva e' sbagliata e non ce ne
#  accorgeremmo mai, perche' non esiste un riferimento con cui confrontarle.
#  Qui si usano sfere con centro, volume e distanza NOTI per analiticamente,
#  cosi' ogni numero atteso e' calcolabile a mano.
#
#  Si usa un'affine ANISOTROPA (1, 1, 2 mm) di proposito: e' il caso in cui un
#  errore di conversione voxel->mm (usare gli indici al posto dei millimetri, o
#  applicare lo spacing sull'asse sbagliato) produce un risultato plausibile ma
#  falso. Con voxel isotropi quel bug resterebbe invisibile.
#
#  USO:  .venv\Scripts\python.exe test_metrics_det.py
# ============================================================================
import math

import numpy as np

from metrics_det import case_metrics, aggregate, paired_test


SHAPE = (64, 64, 48)
# Affine anisotropa: 1 mm in x/y, 2 mm in z, con una traslazione qualsiasi per
# verificare che le distanze non dipendano dall'origine.
AFFINE = np.array([
    [1.0, 0.0, 0.0, -100.0],
    [0.0, 1.0, 0.0,  -50.0],
    [0.0, 0.0, 2.0,  -30.0],
    [0.0, 0.0, 0.0,    1.0],
])


def sphere(center, radius_vox, shape=SHAPE):
    """Sfera binaria in coordinate VOXEL (isotropa negli indici)."""
    zz, yy, xx = np.ogrid[:shape[0], :shape[1], :shape[2]]
    d2 = ((zz - center[0]) ** 2 + (yy - center[1]) ** 2 + (xx - center[2]) ** 2)
    return (d2 <= radius_vox ** 2).astype(np.uint8)


def approx(a, b, tol=1e-6):
    return abs(a - b) <= tol


failures = []


def check(name, cond, detail=""):
    if cond:
        print(f"  OK   {name}")
    else:
        print(f"  FAIL {name}  {detail}")
        failures.append(name)


print("\n[1] Predizione identica al ground truth")
gt = sphere((32, 32, 24), 6)
m = case_metrics(gt, gt, AFFINE, case_id="identica")
check("dice == 1", approx(m["dice"], 1.0), m["dice"])
check("sensitivity == 1", approx(m["sensitivity"], 1.0), m["sensitivity"])
check("fp == 0", m["fp"] == 0, m["fp"])
check("fn == 0", m["fn"] == 0, m["fn"])
check("centroid_dist_mm == 0", approx(m["centroid_dist_mm"], 0.0, 1e-9),
      m["centroid_dist_mm"])
check("isocenter_inside_gt", m["isocenter_inside_gt"] is True)
check("nsd_2mm == 1", approx(m["nsd_2mm"], 1.0, 1e-6), m["nsd_2mm"])
check("hd95_mm == 0", approx(m["hd95_mm"], 0.0, 1e-6), m["hd95_mm"])
check("vol_err_pct == 0", approx(m["vol_err_pct"], 0.0, 1e-9), m["vol_err_pct"])

print("\n[2] Predizione vuota (il modello non trova nulla)")
m = case_metrics(gt, np.zeros_like(gt), AFFINE, case_id="vuota")
check("dice == 0", approx(m["dice"], 0.0), m["dice"])
check("sensitivity == 0", approx(m["sensitivity"], 0.0), m["sensitivity"])
check("fn == 1", m["fn"] == 1, m["fn"])
check("fp == 0", m["fp"] == 0, m["fp"])
check("detected is False", m["detected"] is False)
check("isocenter_inside_gt is False", m["isocenter_inside_gt"] is False)
check("centroid_dist_mm e' NaN", math.isnan(m["centroid_dist_mm"]))

print("\n[3] Volume del voxel e conversione in mm")
# Sfera r=6 voxel: il volume in voxel e' noto per conteggio; il volume fisico
# deve essere quel conteggio moltiplicato per 1*1*2 = 2 mm3.
n_vox = int(gt.sum())
m = case_metrics(gt, gt, AFFINE)
check("vol_gt_cm3 == n_vox * 2mm3 / 1000",
      approx(m["vol_gt_cm3"], n_vox * 2.0 / 1000.0, 1e-9),
      f"{m['vol_gt_cm3']} vs {n_vox * 2.0 / 1000.0}")

print("\n[4] Traslazione nota lungo l'asse anisotropo (z, 2 mm/voxel)")
# Spostare di 4 voxel lungo l'asse 2 (spacing 2 mm) = 8 mm, NON 4.
pred = sphere((32, 32, 28), 6)
m = case_metrics(gt, pred, AFFINE, case_id="traslata_z")
check("centroid_dist_mm == 8.0", approx(m["centroid_dist_mm"], 8.0, 1e-6),
      m["centroid_dist_mm"])
check("isocenter_inside_gt (8mm < raggio)", m["isocenter_inside_gt"] is True)

print("\n[5] Traslazione lungo un asse isotropo (x, 1 mm/voxel)")
pred = sphere((36, 32, 24), 6)
m = case_metrics(gt, pred, AFFINE, case_id="traslata_x")
check("centroid_dist_mm == 4.0", approx(m["centroid_dist_mm"], 4.0, 1e-6),
      m["centroid_dist_mm"])

print("\n[6] Isocentro FUORI dal tumore (il caso che rovina un piano)")
# Traslazione di 10 voxel in x = 10 mm, oltre il raggio di 6 voxel = 6 mm.
pred = sphere((42, 32, 24), 6)
m = case_metrics(gt, pred, AFFINE, case_id="isocentro_fuori")
check("centroid_dist_mm == 10.0", approx(m["centroid_dist_mm"], 10.0, 1e-6),
      m["centroid_dist_mm"])
check("isocenter_inside_gt is False", m["isocenter_inside_gt"] is False)
check("le sfere si toccano -> tp == 1", m["tp"] == 1, m["tp"])

print("\n[7] Falso positivo lontano dal bersaglio")
pred = gt | sphere((10, 10, 10), 5)
m = case_metrics(gt, pred, AFFINE, case_id="con_fp")
check("tp == 1", m["tp"] == 1, m["tp"])
check("fp == 1", m["fp"] == 1, m["fp"])
check("fp_per_scan == 1", approx(m["fp_per_scan"], 1.0), m["fp_per_scan"])
check("sensitivity == 1 (la lesione vera e' trovata)",
      approx(m["sensitivity"], 1.0), m["sensitivity"])
check("n_pred_lesions == 2", m["n_pred_lesions"] == 2, m["n_pred_lesions"])
check("isocentro sulla lesione VERA (la piu' grande)",
      m["isocenter_inside_gt"] is True)

print("\n[8] Due lesioni GT, il modello ne trova una sola")
gt2 = sphere((20, 20, 16), 6) | sphere((45, 45, 32), 5)
pred = sphere((20, 20, 16), 6)
m = case_metrics(gt2, pred, AFFINE, case_id="una_su_due")
check("n_gt_lesions == 2", m["n_gt_lesions"] == 2, m["n_gt_lesions"])
check("tp == 1", m["tp"] == 1, m["tp"])
check("fn == 1", m["fn"] == 1, m["fn"])
check("sensitivity == 0.5", approx(m["sensitivity"], 0.5), m["sensitivity"])

print("\n[9] Speckle di annotazione sotto la soglia di volume -> ignorato")
# Un singolo voxel = 2 mm3, ben sotto i 100 mm3: non deve diventare un FN,
# altrimenti si penalizzerebbe il modello per non aver trovato del rumore.
gt3 = gt.copy()
gt3[2, 2, 2] = 1
m = case_metrics(gt3, gt, AFFINE, case_id="speckle")
check("n_gt_lesions == 1 (speckle scartato)", m["n_gt_lesions"] == 1,
      m["n_gt_lesions"])
check("fn == 0", m["fn"] == 0, m["fn"])

print("\n[10] Diagnostica: probabilita' massima sulla lesione mancata")
prob = np.zeros(SHAPE, dtype=np.float32)
prob[sphere((45, 45, 32), 5) > 0] = 0.31       # massa presente ma sotto 0.5
m = case_metrics(gt2, sphere((20, 20, 16), 6), AFFINE, prob=prob,
                 case_id="prob_mancata")
check("una lesione mancata registrata",
      len(m["max_prob_in_missed_lesion"]) == 1, m["max_prob_in_missed_lesion"])
check("max prob == 0.31", approx(m["max_prob_in_missed_lesion"][0], 0.31, 1e-6),
      m["max_prob_in_missed_lesion"])

print("\n[11] aggregate() e paired_test()")
rows_a, rows_b = [], []
for i in range(12):
    g = sphere((32, 32, 24), 5 + (i % 4))
    pa = sphere((32 + (i % 3), 32, 24), 5 + (i % 4))     # leggermente peggiore
    rows_a.append(case_metrics(g, pa, AFFINE, case_id=f"c{i}"))
    rows_b.append(case_metrics(g, g, AFFINE, case_id=f"c{i}"))   # perfetta
agg = aggregate(rows_b)
check("aggregate: n_cases == 12", agg["n_cases"] == 12)
check("aggregate: dice medio == 1", approx(agg["pooled"]["dice"]["mean"], 1.0, 1e-9),
      agg["pooled"]["dice"]["mean"])
check("aggregate: sensibilita' pooled == 1",
      approx(agg["pooled_detection"]["sensitivity"], 1.0))
check("aggregate: isocentro dentro nel 100% dei casi",
      approx(agg["pooled_detection"]["isocenter_inside_gt_rate"], 1.0))
check("aggregate: terzili presenti", "by_size_tertile" in agg)

t = paired_test(rows_a, rows_b, "dice")
check("paired_test: n == 12", t["n"] == 12, t)
check("paired_test: b migliore di a", t["mean_b"] > t["mean_a"], t)
check("paired_test: p < 0.05", t["p_value"] < 0.05, t["p_value"])

print("\n" + "=" * 60)
if failures:
    print(f"FALLITI: {len(failures)} -> {failures}")
    raise SystemExit(1)
print("Tutti i controlli superati.")
print("=" * 60)
