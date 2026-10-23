# ============================================================================
#  test_project.py - Salvataggio e apertura di un progetto (.ckproj).
#
#  [1] Codifica: ogni tipo di oggetto dello stato torna identico.
#  [2] Con --gui: pipeline reale su lung_001 (inferenza con incertezza,
#      organi, dosimetria, cinematica) in una GUI offscreen, salvataggio,
#      riapertura in una SECONDA finestra, e confronto dei risultati.
#      Qualche minuto.
#
#  USO:  .venv\Scripts\python.exe test_project.py [--gui]
# ============================================================================
import os
import sys
import tempfile

import numpy as np

import kinematics as kin
import motion_model as mm
import project_io
from dosimetry import Beam

failures = []


def check(name, cond, detail=""):
    if cond:
        print(f"  OK   {name}")
    else:
        print(f"  FAIL {name}  {detail}")
        failures.append(name)


def main():
    # Tutto dentro main(), di proposito: su Windows il pool di processi del
    # planner avvia i figli reimportando lo script principale, e senza
    # questa protezione ogni figlio rieseguirebbe l'intero test.
    print("\n[1] Codifica e decodifica")
    import trimesh

    rng = np.random.default_rng(0)
    tr = mm.generate_trace(mm.MotionConfig(seed=4), 30.0)
    res = mm.simulate_tracking(tr, "kalman")
    state = {
        "bool": rng.random((7, 9, 5)) > 0.5,
        "prob": rng.random((120, 100, 100)).astype(np.float32),   # quantizzato
        "u8": (rng.random((120, 100, 100)) > 0.7).astype(np.uint8),
        "dose": (150 * rng.random((50, 40, 30))).astype(np.float32),
        "int_keys": {3: "tre", (1, 2): [1.5, None, float("nan")]},
        "beam": Beam([1, 2, 3], [4, 5, 7], 12.5, weight=0.3, path_cm=4.0,
                     node_id=7, target_id=2, bone_cm=0.4),
        "base": kin.RobotBase(origin_mm=(-900, -800, -100), yaw_deg=12.0),
        "couch": kin.CouchPose(30.0, pivot_mm=(1.0, 2.0, 3.0)),
        "box": kin.Box([0, 0, 0], [10, 20, 30], name="lettino"),
        "cfg": mm.MotionConfig(period_s=3.3, predictor="nlms"),
        "trace": tr, "track": res,
        "mesh": trimesh.creation.icosphere(subdivisions=2),
        "tupla": (1, "a", np.float32(2.5)),
        "stranezza": object(),
    }
    with tempfile.TemporaryDirectory() as d:
        path, omitted = project_io.save_project(os.path.join(d, "p"), state)
        check("estensione aggiunta", path.endswith(project_io.EXTENSION))
        check("oggetto sconosciuto segnalato, non salvato",
              len(omitted) == 1 and "object" in omitted[0], omitted)
        back, meta = project_io.load_project(path)

    check("array booleano identico", np.array_equal(back["bool"], state["bool"]))
    check("maschera uint8 identica, tipo conservato",
          back["u8"].dtype == np.uint8 and
          np.array_equal(back["u8"], state["u8"]))
    check("probabilita' quantizzata entro 1e-5",
          np.abs(back["prob"] - state["prob"]).max() < 1e-5 and
          back["prob"].dtype == np.float32)
    check("dose identica (non quantizzata: valori oltre 1)",
          np.array_equal(back["dose"], state["dose"]))
    check("chiavi non stringa conservate",
          back["int_keys"][3] == "tre" and (1, 2) in back["int_keys"])
    check("NaN conservato", np.isnan(back["int_keys"][(1, 2)][2]))
    b0, b1 = state["beam"], back["beam"]
    check("fascio identico", all(np.allclose(getattr(b0, s), getattr(b1, s))
                                 for s in Beam.__slots__))
    check("base del robot identica",
          np.allclose(back["base"].origin, state["base"].origin) and
          np.allclose(back["base"].R, state["base"].R))
    p = np.array([10.0, -40.0, 300.0])
    check("base del robot: stessa trasformazione",
          np.allclose(back["base"].to_robot(p), state["base"].to_robot(p)))
    check("lettino identico",
          np.allclose(back["couch"].R, state["couch"].R))
    check("ostacolo identico", np.allclose(back["box"].lo, state["box"].lo) and
          back["box"].name == "lettino")
    check("configurazione del respiro identica", back["cfg"] == state["cfg"])
    check("traccia identica", np.array_equal(back["trace"].tumor, tr.tumor) and
          back["trace"].cfg == tr.cfg)
    check("inseguimento identico",
          np.array_equal(back["track"].residual, res.residual))
    check("mesh identica", np.allclose(back["mesh"].vertices,
                                       state["mesh"].vertices, atol=1e-6) and
          np.array_equal(back["mesh"].faces, state["mesh"].faces))
    check("tupla resta tupla", back["tupla"] == (1, "a", 2.5))
    check("oggetto sconosciuto -> None", back["stranezza"] is None)

    if "--gui" in sys.argv:
        print("\n[2] Pipeline reale, salvataggio e riapertura (lung_001)")
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        import time
        from PyQt6.QtWidgets import QApplication
        import config
        app = QApplication.instance() or QApplication(sys.argv)
        import main as M
        from inference_worker import InferenceWorker

        def run_sync(worker, signal_name):
            out = {}
            getattr(worker, signal_name).connect(lambda r: out.setdefault("r", r))
            worker.failed.connect(lambda m: out.setdefault("err", m))
            worker.run()                                # nel thread corrente
            app.processEvents()
            if "err" in out:
                raise RuntimeError(out["err"])
            return out["r"]

        w = M.MainWindow()
        ct = os.path.join(config.IMAGES_DIR, "lung_001.nii.gz")
        t0 = time.time()
        r = run_sync(InferenceWorker(ct, with_uncertainty=True), "resultReady")
        w.tab_perception.on_result(r)
        print(f"   inferenza + incertezza: {time.time() - t0:.0f} s")
        mt = w.tab_motion
        mt.organ_worker.wait()
        app.processEvents()
        mt.combo_tech.setCurrentIndex(mt.combo_tech.findData("tracking"))
        mt._margin_timer.stop()
        mt._start_margin()
        mt.margin_worker.wait()
        app.processEvents()
        check("margine d'inseguimento pronto", mt.margin_ready())
        panel = w.tab_dosimetry.panel
        from viewer3d import MeshBuilderWorker
        mres = run_sync(MeshBuilderWorker(r["ct_data"], r["mask"], r["affine"]),
                        "meshReady")
        panel._on_mesh_ready(mres)
        target, label = panel._planning_target()
        check("bersaglio dalla tecnica", "inseguimento" in label, label)
        from dose_worker import DoseWorker
        panel._target_label, panel._target_mask = label, target
        dw = DoseWorker(r["ct_data"], target, r["affine"], r["com_world"],
                        body_mask=mres["organ_masks"]["body"],
                        bone_mask=mres["organ_masks"]["bones"],
                        bone_affine=mres["organ_affine"],
                        arms_mask=mres["organ_masks"]["arms"],
                        robot_constraints=False)
        dres = run_sync(dw, "doseReady")
        panel._on_dose_ready(dres)
        app.processEvents()
        print(f"   pipeline fino alla dose: {time.time() - t0:.0f} s")
        check("il progetto risulta da salvare", w._dirty)

        with tempfile.TemporaryDirectory() as d:
            w.project_path = os.path.join(d, "prova" + project_io.EXTENSION)
            t1 = time.time()
            ok = w.save_project()
            size = os.path.getsize(w.project_path) / 1e6
            print(f"   salvato in {time.time() - t1:.1f} s, {size:.1f} MB")
            import numpy as _np
            with _np.load(w.project_path) as z:
                big = sorted(((z[k].nbytes, k, z[k].dtype, z[k].shape)
                              for k in z.files if k != "__meta__"),
                             reverse=True)[:5]
            for nb, k, dt, sh in big:
                print(f"      {k:6s} {nb / 1e6:7.1f} MB non compressi  "
                      f"{dt} {sh}")
            check("progetto sotto 15 MB", size < 15.0, size)
            check("salvataggio riuscito", ok and not w._dirty)

            w2 = M.MainWindow()
            from PyQt6.QtWidgets import QFileDialog
            QFileDialog.getOpenFileName = staticmethod(
                lambda *a, **k: (w.project_path, ""))
            t2 = time.time()
            w2.open_project()
            app.processEvents()
            print(f"   riaperto in {time.time() - t2:.1f} s")

        r2 = w2.tab_perception.result
        check("stessa maschera", np.array_equal(r2["mask"], r["mask"]))
        check("stesso semaforo", r2["traffic_light"] == r["traffic_light"],
              (r2["traffic_light"], r["traffic_light"]))
        u1, u2 = r["uncertainty"], r2["uncertainty"]
        check("mappe di incertezza identiche (gia' compatte dal vivo)",
              all(np.array_equal(u2[k], u1[k])
                  for k in ("p_mean", "p_std", "mi")))
        check("stessi centroidi delle predizioni perturbate",
              np.allclose(u2["centroids_mm"], u1["centroids_mm"],
                          equal_nan=True))
        check("stessa tecnica",
              w2.tab_motion.technique() == "tracking"
              and w2.tab_motion.margin_ready())
        p2 = w2.tab_dosimetry.panel
        check("stesse mesh degli organi",
              len(p2.result["organ_meshes"]) == len(mres["organ_meshes"]))
        d2 = p2.dose_result
        check("stessa dose (entro 0.005 punti percentuali)",
              d2 is not None and
              np.abs(d2["dose_pct"] - dres["dose_pct"]).max() <= 0.0051)
        check("stesse metriche del piano",
              d2["metrics"]["coverage_pct"] == dres["metrics"]["coverage_pct"])
        check("stessi fasci", len(d2["beams"]) == len(dres["beams"]) and
              np.allclose(d2["weights"], dres["weights"]))
        check("griglia di densita' ricalcolata identica",
              np.array_equal(d2["red_grid"], dres["red_grid"]))
        check("stessa etichetta del bersaglio", d2["target_label"] == label)
        check("il piano e' arrivato al Tab 4",
              w2.tab_kinematics.plan is not None and
              len(w2.tab_kinematics.plan["beams"]) == len(dres["beams"]))
        check("progetto riaperto non risulta modificato", not w2._dirty)
        w2._dirty = w._dirty = False

    print("\n" + "=" * 60)
    if failures:
        print(f"FALLITI: {len(failures)} -> {failures}")
        raise SystemExit(1)
    print("Tutti i controlli superati.")
    print("=" * 60)


if __name__ == "__main__":
    main()
