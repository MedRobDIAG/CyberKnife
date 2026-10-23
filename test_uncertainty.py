# ============================================================================
#  test_uncertainty.py - Verifica dell'analisi di incertezza e del bersaglio
#  adattivo.
#
#  I controlli piu' importanti sono due:
#    - nessun caso e' campionato da un modello che l'ha visto in training
#      (altrimenti l'incertezza misurata sarebbe memoria, non confidenza);
#    - l'inversione su GPU (to_original) coincide con Invertd di MONAI, che
#      ha prodotto le mappe della validazione.
#
#  USO:  .venv\Scripts\python.exe test_uncertainty.py [--gpu]
#        --gpu ripete il campionamento di lung_001 (~10 s su GPU)
# ============================================================================
import os
import sys

import numpy as np
import torch
from scipy import ndimage

import config
import uncertainty as U
import robust_target as rt

failures = []


def check(name, cond, detail=""):
    if cond:
        print(f"  OK   {name}")
    else:
        print(f"  FAIL {name}  {detail}")
        failures.append(name)


print("\n[1] Perturbazioni")
P = U.tta_params(8)
check("8 perturbazioni", len(P) == 8)
check("la prima e' l'identita'",
      not P[0]["flip"] and P[0]["rot"] == (0, 0, 0) and P[0]["noise"] == 0)
check("la seconda e' il solo specchiamento", P[1]["flip"] and
      P[1]["rot"] == (0, 0, 0))
check("rotazioni entro 8 gradi (training: 15)",
      all(max(abs(np.rad2deg(r)) for r in p["rot"]) <= 8.0 for p in P))
check("stesse perturbazioni a ogni chiamata", U.tta_params(8) == P)

print("\n[2] Deformazione e sua inversa")
A = U._theta((40, 30, 20), (1.5, 1.5, 2.0), (0, 0, 0), (1, 1, 1))
check("identita' -> matrice identica", np.allclose(A, np.eye(3)))
x = torch.zeros((1, 1, 40, 30, 20))
x[0, 0, 15:25, 10:20, 6:14] = 1.0
x = torch.from_numpy(ndimage.gaussian_filter(x.numpy(), 1.5)).float()
A = U._theta((40, 30, 20), (1.5, 1.5, 2.0), np.deg2rad((6, -4, 7)),
             (1.05, 0.95, 1.03))
back = U._warp(U._warp(x, A, "bilinear"), np.linalg.inv(A), "bilinear")
err = float((back - x)[0, 0, 8:32, 6:24, 4:16].abs().max())
check("warp e warp inverso riportano il volume (centro)", err < 0.05, err)
com0 = np.array(ndimage.center_of_mass(x[0, 0].numpy()))
com1 = np.array(ndimage.center_of_mass(back[0, 0].numpy()))
check("il centroide non si sposta (< 0.3 voxel)",
      np.abs(com1 - com0).max() < 0.3, com1 - com0)

print("\n[3] Nessun caso visto dal modello che lo campiona")
import splits as S
sp = S.load_splits()
bad = [c for e in sp["folds"] for c in e["val"] if c in e["train"]]
check("val e train disgiunti in ogni fold", not bad, bad)
seen = {}
for e in sp["folds"]:
    for c in e["val"]:
        seen.setdefault(c, []).append(e["fold"])
check("ogni caso sta nella validazione di UN solo fold",
      all(len(v) == 1 for v in seen.values()),
      {k: v for k, v in seen.items() if len(v) != 1})
check("63 casi coperti", len(seen) == 63, len(seen))

print("\n[4] Segnali e semaforo")
shape = (30, 30, 20)
aff = np.diag([1.0, 1.0, 2.0, 1.0])
p = np.zeros(shape, np.float32)
p[10:20, 10:20, 5:12] = 0.9
same = {"p_mean": p, "p_std": np.zeros_like(p), "mi": np.zeros_like(p),
        "affine": aff, "n_samples": 8, "volumes_cm3": np.full(8, 1.4),
        "centroids_mm": np.tile([14.5, 14.5, 16.0], (8, 1))}
s = U.case_signals(same, config.POSTPROC.replace(min_lesion_mm3=10.0))
check("campioni identici -> deviazione 0", s["std_lesione"] == 0.0)
speck = dict(same, p_mean=p.copy(), p_std=np.zeros_like(p))
speck["p_mean"][1, 1, 1] = 0.9            # scartata dal filtro sul volume
speck["p_std"][10:20, 10:20, 5:12] = 0.2  # incertezza sulla sola lesione
ss = U.case_signals(speck, config.POSTPROC.replace(min_lesion_mm3=10.0))
check("statistiche sulla lesione giusta anche con componenti scartate prima",
      abs(ss["std_lesione"] - 0.2) < 1e-6, ss.get("std_lesione"))
check("campioni identici -> dispersione 0",
      s["centroide_disp_max_mm"] < 1e-9, s["centroide_disp_max_mm"])
jump = dict(same, centroids_mm=np.vstack([same["centroids_mm"][:7],
                                          [[120.0, 14.5, 16.0]]]))
sj = U.case_signals(jump, config.POSTPROC.replace(min_lesion_mm3=10.0))
check("un salto di un campione pesa sulla dispersione massima",
      sj["centroide_disp_max_mm"] > U.DISP_REVIEW_MM)
check("... e non sulla mediana", sj["centroide_disp_mm"] < 20.0,
      sj["centroide_disp_mm"])
check("semaforo verde", U.traffic_light(s, True)[0] == "verde")
check("semaforo giallo con candidato alternativo",
      U.traffic_light(sj, True)[0] == "giallo")
check("semaforo rosso se il cancello rifiuta",
      U.traffic_light(s, False, "volume")[0] == "rosso")
lost = dict(s, frac_campioni_con_lesione=0.5)
check("semaforo rosso se la lesione sparisce nelle perturbazioni",
      U.traffic_light(lost, True)[0] == "rosso")

print("\n[5] Bersaglio adattivo")
ctv = p >= 0.5
halo = p.copy()
halo[8:22, 10:20, 5:12] = np.maximum(halo[8:22, 10:20, 5:12], 0.3)
b_hi = rt.band_target(ctv, halo, aff, tau=0.5)
b_lo = rt.band_target(ctv, halo, aff, tau=0.2)
check("tau alto -> il CTV", np.array_equal(b_hi, ctv))
check("tau basso -> la banda contiene il CTV", np.all(b_lo[ctv]))
check("la banda cresce solo dove la rete esita (asse X)",
      b_lo[8:10].any() and not b_lo[:, 8:10].any())
far = halo.copy()
far[20:30, 14:16, 8:10] = 0.3
b_far = rt.band_target(ctv, far, aff, tau=0.2, max_mm=3.0)
check("una coda lunga viene tagliata a max_mm", not b_far[25:].any())
pop = {"Sigma_mm": [1.0, 0.0, 0.0], "sigma_mm": [0.0, 0.0, 2.0]}
m = rt.combined_margin([0.0, 1.0, 0.0], pop)
check("van Herk con sistematici in quadratura",
      np.allclose(m, [2.5, 2.5, 1.4]), m)
check("un salto verso un'altra lesione non gonfia il margine",
      np.allclose(rt.contour_sigma([[0, 0, 0], [2, 0, 0], [0, 0, 0],
                                    [2, 0, 0], [300, 0, 0]]),
                  [1, 0, 0]),
      rt.contour_sigma([[0, 0, 0], [2, 0, 0], [0, 0, 0], [2, 0, 0],
                        [300, 0, 0]]))
check("dispersione dei centroidi ignora i campioni senza lesione",
      np.allclose(rt.contour_sigma([[0, 0, 0], [2, 0, 0],
                                    [np.nan] * 3]), [1, 0, 0]))

if "--gpu" in sys.argv and torch.cuda.is_available():
    print("\n[6] Inversione su GPU contro Invertd (lung_001)")
    from inference import load_model
    import evaluate as E
    ref = U.load_maps(U.maps_path("lung_001"))
    fold = E.resolve_cases("splits:all")[1]["lung_001"]
    model = load_model(os.path.join(config.OUTPUT_DIR, "runs", "aug_anatomic",
                                    f"fold{fold}", "final.pth"),
                       torch.device("cuda"))
    r = U.sample_case(model, E.case_paths("lung_001")[0], U.tta_params(8))
    check("stessa affine", np.allclose(r["affine"], ref["affine"], atol=1e-4))
    sel = ref["p_mean"] > 0.05
    d = np.abs(r["p_mean"][sel] - ref["p_mean"][sel])
    check("p_mean coincide nel bounding box (p95 < 0.02)",
          np.percentile(d, 95) < 0.02, (np.percentile(d, 95), d.max()))
    n_new = int((r["p_mean"] >= 0.5).sum())
    n_ref = int((ref["p_mean"] >= 0.5).sum())
    check("stesso volume della lesione (+-2%)",
          abs(n_new / max(n_ref, 1) - 1) < 0.02, (n_new, n_ref))

print("\n" + "=" * 60)
if failures:
    print(f"FALLITI: {len(failures)} -> {failures}")
    raise SystemExit(1)
print("Tutti i controlli superati.")
print("=" * 60)
