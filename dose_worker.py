# ============================================================================
#  dose_worker.py - Incapsulamento Qt del calcolo dosimetrico.
#
#  Sta in un modulo separato da dosimetry.py di proposito: la fisica non deve
#  dipendere da PyQt. Il planner distribuisce il ray-tracing su piu' processi e
#  su Windows ogni processo re-importa la catena dei moduli; se dosimetry
#  importasse Qt, sei processi che non disegnano nulla caricherebbero comunque
#  l'intero toolkit grafico.
# ============================================================================
from PyQt6.QtCore import QThread, pyqtSignal

from dosimetry import compute_plan, DoseMap


class DoseWorker(QThread):
    """Esegue il calcolo dosimetrico fuori dal thread della GUI."""

    progress = pyqtSignal(int)
    status = pyqtSignal(str)
    doseReady = pyqtSignal(dict)
    failed = pyqtSignal(str)

    def __init__(self, ct_data, mask, affine, isocenter_mm, n_beams=40,
                 collimator_mm=20.0, mode="conformal", prescription_pct=None,
                 bone_mask=None, bone_affine=None, arms_mask=None,
                 body_mask=None, cfg=None, robot_constraints=True,
                 couch_poses_max=3, parent=None):
        super().__init__(parent)
        self.robot_constraints = robot_constraints
        self.couch_poses_max = couch_poses_max
        self.ct = ct_data
        self.mask = mask
        self.affine = affine
        self.isocenter = isocenter_mm
        self.n_beams = n_beams
        self.collimator_mm = collimator_mm
        self.mode = mode
        self.prescription_pct = prescription_pct
        self.bone_mask = bone_mask
        self.bone_affine = bone_affine
        self.arms_mask = arms_mask
        self.body_mask = body_mask
        self.cfg = cfg

    def _room_setup(self):
        """Base del robot e angoli di lettino, PRIMA di scegliere i fasci.

        E' il collo di bottiglia che faceva perdere fasci: la dosimetria
        sceglieva i fasci senza sapere da dove il braccio li avrebbe eseguiti,
        e il Tab 3 lo scopriva solo dopo. Misurato su lung_001: 107 fasci
        pianificati, 70 erogabili anche con la sala ottimizzata (61 a lettino
        fermo), cioe' il 39-44% del peso di dose mai consegnato. Con la sala
        decisa prima e passata al planner: 52 fasci, 52 erogati, stessa
        copertura (94.9%) e conformita' (0.858 contro 0.852).

        Restituisce (base, ostacoli, angoli_lettino) oppure None se mancano le
        maschere per ricavare lettino e pavimento.
        """
        import numpy as np
        import kinematics as kin
        if self.body_mask is None or self.bone_affine is None:
            return None
        bounds = kin.bounds_mm(self.body_mask, self.bone_affine)
        if bounds is None:
            return None
        obstacles = kin.treatment_obstacles(bounds)
        ref = kin.RobotBase()
        template = kin.RobotBase(origin_mm=(ref.origin[0],
                                            kin.floor_y_mm(bounds),
                                            ref.origin[2]))
        self.status.emit("Ottimizzazione della sala (base del robot e "
                         "rotazioni del lettino)...")
        setup = kin.optimize_setup(
            obstacles, np.asarray(self.isocenter, dtype=np.float64),
            base_template=template, max_couch_poses=self.couch_poses_max,
            # Senza il braccio del lettino l'ottimizzatore puo' scegliere
            # orientamenti che il meccanismo non raggiunge.
            couch_arm=kin.treatment_couch_arm(bounds),
            status_cb=self.status.emit)
        return setup, obstacles

    def run(self):
        try:
            def on_progress(done, total):
                # Il ray-tracing occupa la fascia 15-85% della barra.
                self.progress.emit(15 + int(70 * done / max(total, 1)))

            self.progress.emit(2)
            # BERSAGLIO: la lesione principale, la stessa che definisce
            # l'isocentro. Qui il filtro non c'era, e il piano della GUI
            # cercava di coprire tutte le lesioni da un unico isocentro: su
            # lung_001 sei noduli per 5.98 cm3, mentre la riga di comando
            # trattava la principale da 2.73. I pesi si disperdevano su
            # bersagli che nessun fascio raggiunge insieme.
            import postproc
            mask_all = self.mask
            self.mask, n_les = postproc.primary_mask(self.mask, self.affine)
            if n_les > 1:
                self.status.emit(
                    f"{n_les} lesioni rilevate: il piano tratta la principale "
                    f"(le altre richiedono una seduta propria).")
            robot = {}
            setup = None
            # I vincoli del robot contano solo per la pianificazione inversa:
            # il piano a isocentro singolo e' un riferimento geometrico fisso.
            if self.mode == "conformal" and self.robot_constraints:
                out = self._room_setup()
                if out is None:
                    # Ripiego VISIBILE: senza maschera del corpo non si sa dove
                    # stanno lettino e pavimento, e il piano risultante non
                    # garantisce l'erogabilita'. Lo si dice, non lo si tace.
                    self.status.emit(
                        "ATTENZIONE: maschera del corpo assente, i fasci "
                        "vengono scelti SENZA i vincoli del robot.")
                else:
                    setup, obstacles = out
                    robot = dict(robot_base=setup["base"],
                                 robot_obstacles=obstacles,
                                 robot_couch_poses=setup["couch_poses"])
            # Collaudo con lo stesso solutore del nodo ROS (n_roll=4), DENTRO
            # la dosimetria: il filtro del planner guarda i fasci uno per uno,
            # la sequenza anche percorsi e manovre del lettino. I fasci che il
            # robot non esegue escono prima della dose finale, cosi' il piano
            # dichiara la dose che verra' erogata. Prima il collaudo avveniva
            # dopo e si limitava ad avvisare: su lung_085 il piano prometteva
            # il 100% del CTV e il robot, senza 7 fasci su 110, ne dava il 64%.
            beam_filter = None
            if setup is not None:
                import kinematics as kin

                def beam_filter(beams):
                    self.progress.emit(88)
                    keep, dg = kin.deliverable_subset(
                        beams, setup["base"], obstacles,
                        couch_poses=setup["couch_poses"], n_roll=4,
                        status_cb=self.status.emit)
                    if not dg.get("verificato", True):
                        self.status.emit(
                            "ATTENZIONE: il collaudo non ha convergito, la "
                            "simulazione potrebbe non erogare tutti i fasci.")
                    return keep
            self.progress.emit(10)
            res = compute_plan(
                self.ct, self.mask, self.affine, self.isocenter,
                n_beams=self.n_beams, collimator_mm=self.collimator_mm,
                mode=self.mode, cfg=self.cfg,
                prescription_pct=self.prescription_pct,
                bone_mask=self.bone_mask, bone_affine=self.bone_affine,
                arms_mask=self.arms_mask, body_mask=self.body_mask,
                beam_filter=beam_filter,
                progress_cb=on_progress, status_cb=self.status.emit,
                **robot,
            )
            self.progress.emit(95)
            # La configurazione di sala fa PARTE del piano: i fasci sono stati
            # scelti per quella. Il Tab 3 la adotta invece di ricalcolarne una.
            res["robot_base"] = robot.get("robot_base")
            res["robot_obstacles"] = robot.get("robot_obstacles")
            res["couch_poses"] = robot.get("robot_couch_poses")
            res["metrics"]["vincoli_robot"] = bool(robot)
            res["metrics"]["lesioni_rilevate"] = int(n_les)
            res["lesioni_rilevate"] = int(n_les)
            res["mask_tutte_le_lesioni"] = mask_all
            if setup is not None:
                res["metrics"]["sala"] = {
                    k: setup[k] for k in ("copertura_direzioni",
                                          "copertura_lettino_fermo",
                                          "direzioni_provate")}

            res["dose_map"] = DoseMap(res["dose_pct"], res["grid_affine"],
                                      self.ct.shape, self.affine)
            self.status.emit("Dosimetria completata")
            self.doseReady.emit(res)
            self.progress.emit(100)

        except Exception as exc:                        # noqa: BLE001
            import traceback
            self.failed.emit(f"{type(exc).__name__}: {exc}\n\n"
                             f"{traceback.format_exc()}")
