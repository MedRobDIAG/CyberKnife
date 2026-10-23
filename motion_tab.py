# ============================================================================
#  motion_tab.py - TAB 2: respiro del paziente e TECNICA di gestione del moto.
#
#  POSIZIONE NELLA PIPELINE
#  ------------------------
#  Sta fra la percezione e la dosimetria perche' la tecnica scelta qui
#  decide due cose a valle:
#    - il BERSAGLIO che la dosimetria (Tab 3) ottimizza: CTV, ITV
#      sull'intera traiettoria, CTV + margine dell'errore d'inseguimento,
#      oppure il PTV adattivo all'incertezza della rete;
#    - il comportamento della simulazione ROS (Tab 4): tumore fermo, tumore
#      che respira con fascio fermo, oppure robot che lo insegue.
#  Pianificare su un ITV e poi inseguire (o viceversa) sarebbe incoerente:
#  per questo la scelta e' una sola e viaggia nel piano.
#
#  Due strumenti per decidere:
#    "Simula seduta"      il robot riesce a seguire il tumore? Tracce del
#                         respiro, errore d'inseguimento, confronto fra i
#                         predittori di latenza. Pochi secondi.
#    "Confronta scenari"  che cosa cambia per la DOSE? Pianifica e valuta il
#                         tumore in movimento con e senza inseguimento. Alcuni
#                         minuti: un piano completo per scenario.
#
#  Per l'ampiezza del respiro serve sapere dove sta la lesione nel polmone:
#  il tab segmenta da se' gli organi (pochi secondi, in background), senza
#  aspettare il modello 3D del Tab 3.
# ============================================================================
import os

import numpy as np

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QPushButton, QLabel,
    QGroupBox, QDoubleSpinBox, QComboBox, QCheckBox, QProgressBar,
    QScrollArea, QSizePolicy, QMessageBox, QTabWidget,
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

import config
import motion_model as mm
import motion_dose as md

BG = "#12141a"
FG = "#c8d0da"
GRID = "#2c3643"
PRED_LABELS = {"none": "nessuno", "linear": "lineare", "kalman": "Kalman",
               "nlms": "nLMS", "gru": "GRU (rete)"}
SCEN_COLORS = {"statico": "#8b95a3", "ctv_senza_inseguimento": "#ff5c5c",
               "itv": "#ffb020", "inseguimento": "#3ddc84",
               "adattivo": "#00e5ff", "gating": "#c678dd"}


#: Tecniche di gestione del moto: (chiave, etichetta, effetto).
TECHNIQUES = [
    ("none", "Nessuna: tumore considerato fermo",
     "Dosimetria sul CTV. In ROS il tumore resta immobile, come prima di "
     "questa fase. Utile come riferimento, non realistico."),
    ("itv", "ITV: margine sull'intera traiettoria, fascio fermo",
     "Dosimetria sull'inviluppo del respiro (CTV + 5 mm, spostato lungo la "
     "traiettoria di una 4DCT simulata). In ROS il tumore respira e il "
     "fascio resta fermo. Copre, ma irradia molto polmone sano."),
    ("gating", "Gating: fascio fermo, spara solo in finestra",
     "Il robot ASPETTA: il fascio resta sulla posizione pianificata e si "
     "accende solo quando la stima del tumore e' entro la finestra. "
     "Dosimetria sul CTV + margine dell'errore residuo a fascio acceso. "
     "Misurato: p95 da 8.6 a 4.3 mm con finestra di 3 mm, ma il fascio e' "
     "acceso ~1/4 del tempo e la seduta si allunga."),
    ("tracking", "Inseguimento (Synchrony)",
     "Dosimetria sul CTV + margine di van Herk dell'errore residuo "
     "d'inseguimento. In ROS il robot insegue il tumore."),
    ("adaptive", "Inseguimento + margine adattivo all'incertezza",
     "Come l'inseguimento, ma il bersaglio si allarga anche dove la rete e' "
     "incerta (analisi di incertezza del Tab 1). E' la tecnica che nella "
     "campagna recupera la copertura del tumore vero."),
]
TECH_LABEL = {k: lab for k, lab, _ in TECHNIQUES}
#: Tecniche che hanno bisogno del margine di popolazione (van Herk).
MARGIN_TECHS = ("gating", "tracking", "adaptive")


def _available_predictors():
    names = ["none", "linear", "kalman", "nlms"]
    try:
        import motion_predictor as mp
        if os.path.exists(mp.WEIGHTS_PATH):
            mp.register()
            names.append("gru")
    except Exception:                        # torch assente o pesi illeggibili
        pass
    return names


# ---------------------------------------------------------------------------
class TrackingWorker(QThread):
    """Una seduta simulata, ripetuta con tutti i predittori."""
    done = pyqtSignal(dict)
    failed = pyqtSignal(str)

    def __init__(self, cfg, duration_s, predictors):
        super().__init__()
        self.cfg, self.duration, self.predictors = cfg, duration_s, predictors

    def run(self):
        try:
            tr = mm.generate_trace(self.cfg, self.duration)
            runs = {}
            h = self.cfg.horizon_samples
            skip = int(20 * self.cfg.fs_hz)
            for name in self.predictors:
                pred = mm.make_predictor(name, self.cfg)
                s_hat = mm.predict_series(pred, tr.surrogate)
                e = mm.prediction_error(tr.surrogate, s_hat, h, skip)
                r = mm.simulate_tracking(tr, name, self.cfg)
                runs[name] = {"result": r, "surrogate_pred": s_hat,
                              "pred_rmse": float(np.sqrt(np.mean(e ** 2))),
                              "stats": mm.error_stats(r.residual)}
            any_r = next(iter(runs.values()))["result"]
            self.done.emit({"trace": tr, "runs": runs,
                            "untracked": mm.error_stats(any_r.untracked)})
        except Exception as exc:
            import traceback
            self.failed.emit(f"{exc}\n\n{traceback.format_exc()}")


class OrganWorker(QThread):
    """Segmentazione degli organi per il respiro (polmoni) e per i piani."""
    done = pyqtSignal(dict)
    failed = pyqtSignal(str)

    def __init__(self, ct, affine):
        super().__init__()
        self.ct, self.affine = ct, affine

    def run(self):
        try:
            import organs as og
            masks, oaff, _ = og.segment_organs(self.ct, self.affine, 2.0)
            self.done.emit({"organ_masks": masks, "organ_affine": oaff})
        except Exception as exc:
            import traceback
            self.failed.emit(f"{exc}\n\n{traceback.format_exc()}")


class MarginWorker(QThread):
    """Margine di popolazione (van Herk) per la configurazione corrente."""
    done = pyqtSignal(str, dict)
    failed = pyqtSignal(str)

    def __init__(self, key, cfg, predictor, gating_mm=None):
        super().__init__()
        self.key, self.cfg, self.predictor = key, cfg, predictor
        self.gating_mm = gating_mm

    def run(self):
        try:
            pop = md.population_residual(self.cfg, self.predictor, n=12,
                                         gating_mm=self.gating_mm)
            self.done.emit(self.key, pop)
        except Exception as exc:
            import traceback
            self.failed.emit(f"{exc}\n\n{traceback.format_exc()}")


class ScenarioWorker(QThread):
    """Pianifica e valuta gli scenari di dose con il tumore che respira."""
    status = pyqtSignal(str)
    done = pyqtSignal(dict)
    failed = pyqtSignal(str)

    def __init__(self, source, organs, cfg, predictor, adaptive=None,
                 pop=None, gating_mm=None):
        super().__init__()
        self.source, self.organs = source, organs
        self.cfg, self.predictor = cfg, predictor
        self.adaptive = adaptive
        self.pop = pop
        self.gating_mm = gating_mm

    def run(self):
        try:
            import postproc
            from dosimetry import compute_plan
            src = self.source
            ctv, _ = postproc.primary_mask(src["mask"], src["affine"])
            m = self.organs["organ_masks"]
            oaff = self.organs["organ_affine"]

            def plan(target):
                target = np.asarray(target, np.uint8)
                from campaign import centroid_mm
                return compute_plan(
                    src["ct_data"], target, src["affine"],
                    centroid_mm(target, src["affine"]),
                    bone_mask=m["bones"], bone_affine=oaff,
                    arms_mask=m["arms"], body_mask=m["body"],
                    status_cb=self.status.emit)

            cfg = self.cfg
            self.status.emit("Respiro della seduta e della 4DCT di "
                             "pianificazione...")
            tr = mm.generate_trace(cfg, 300.0)
            # La 4DCT e' un'acquisizione breve e regolare, fatta prima: niente
            # deriva, un altro seme.
            plan_tr = mm.generate_trace(
                cfg.replace(seed=cfg.seed + 97, drift_mm_per_min=(0, 0, 0)),
                40.0)
            trk = mm.simulate_tracking(tr, self.predictor, cfg)
            pop = self.pop
            if pop is None:
                self.status.emit("Margine di popolazione (van Herk)...")
                pop = md.population_residual(cfg, self.predictor, n=12)
            extra = {}
            if self.gating_mm:
                # Gating: fascio fermo acceso solo in finestra. Il suo margine
                # e i suoi spostamenti sono quelli degli istanti a fascio acceso.
                self.status.emit("Margine del gating...")
                pop_g = md.population_residual(cfg, self.predictor, n=12,
                                               gating_mm=self.gating_mm)
                extra["gating"] = (md.expand_mask(ctv, src["affine"],
                                                  pop_g["margine_mm"]),
                                   trk.gated(self.gating_mm))
            if self.adaptive is not None:
                import robust_target as rt
                extra["adattivo"], info = rt.adaptive_target(
                    ctv, self.adaptive, src["affine"], pop)
            out = md.compare_scenarios(
                src["ct_data"], src["affine"], ctv, tr, trk, plan,
                lungs=m["lung_r"] | m["lung_l"], lung_affine=oaff,
                planning_trace=plan_tr, margin_mm=pop["margine_mm"],
                extra_targets=extra, status_cb=self.status.emit)
            dvh = {}
            for name, p in out["plans"].items():
                dvh[name] = md.dvh_under_motion(
                    p, ctv, src["affine"], out["displacements"][name])
            out.pop("plans")
            out.pop("targets")
            out["dvh"] = dvh
            out["population"] = pop
            self.done.emit(out)
        except Exception as exc:
            import traceback
            self.failed.emit(f"{exc}\n\n{traceback.format_exc()}")


# ---------------------------------------------------------------------------
class MotionTab(QWidget):
    """TAB 2 - Respiro e tecnica di gestione del moto."""

    #: payload per il piano ROS (dict) oppure None se il moto e' ignorato
    configChanged = pyqtSignal(object)
    #: la tecnica o i suoi parametri sono cambiati: il bersaglio va rifatto
    techniqueChanged = pyqtSignal(str)
    #: gli organi segmentati qui, riusabili a valle
    organsReady = pyqtSignal(dict)
    #: un risultato nuovo da salvare nel progetto
    resultChanged = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.source = None
        self.organs = None
        self.worker = None
        self.scen_worker = None
        self.organ_worker = None
        self.margin_worker = None
        self.si_fraction = None
        self.pop = None            # margine di popolazione per _pop_key
        self._pop_key = None
        self._last_tracking = None
        self._last_scen = None
        self.predictors = _available_predictors()
        from PyQt6.QtCore import QTimer
        self._margin_timer = QTimer(self)
        self._margin_timer.setSingleShot(True)
        self._margin_timer.setInterval(900)
        self._margin_timer.timeout.connect(self._start_margin)

        root = QHBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(12)
        root.addWidget(self._build_controls(), 0)
        self.views = QTabWidget()
        self.views.setObjectName("subTabs")
        self.fig_trk, self.can_trk = self._figure()
        self.fig_dose, self.can_dose = self._figure()
        self.views.addTab(self.can_trk, "  Inseguimento  ")
        self.views.addTab(self.can_dose, "  Dose sotto moto  ")
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        self.lbl_metrics = QLabel("")
        self.lbl_metrics.setTextFormat(Qt.TextFormat.RichText)
        self.lbl_metrics.setWordWrap(True)
        self.lbl_metrics.setSizePolicy(QSizePolicy.Policy.Preferred,
                                       QSizePolicy.Policy.Maximum)
        rl.addWidget(self.lbl_metrics)
        rl.addWidget(self.views, 1)
        root.addWidget(right, 1)
        key = self.technique()
        self.lbl_tech.setText(next(e for k, _, e in TECHNIQUES if k == key))
        self.spin_gate.setVisible(key == "gating")
        self.lbl_gate.setVisible(key == "gating")
        self.spin_gate.valueChanged.connect(lambda *_: self._emit_config())
        self._empty(self.fig_trk, self.can_trk,
                    "Premi «Simula seduta» per vedere il robot che insegue "
                    "il respiro.")
        self._empty(self.fig_dose, self.can_dose,
                    "Carica una CT (Tab 1), poi premi «Confronta "
                    "tecniche».\nPianifica e valuta ogni tecnica: alcuni "
                    "minuti.")

    # ------------------------------------------------------------------
    def _figure(self):
        fig = Figure(figsize=(7, 5), facecolor=BG)
        can = FigureCanvas(fig)
        return fig, can

    def _empty(self, fig, can, text):
        fig.clear()
        ax = fig.add_subplot(111, facecolor=BG)
        ax.text(0.5, 0.5, text, ha="center", va="center", color="#7d8590",
                fontsize=11)
        ax.set_axis_off()
        can.draw_idle()

    @staticmethod
    def _style(ax, title):
        ax.set_facecolor(BG)
        ax.set_title(title, color=FG, fontsize=10)
        ax.tick_params(colors="#8b95a3", labelsize=8)
        ax.grid(True, color=GRID, linestyle="--", alpha=0.6)
        for s in ax.spines.values():
            s.set_color(GRID)

    @staticmethod
    def _dspin(lo, hi, val, step, suffix, dec=1):
        s = QDoubleSpinBox()
        s.setRange(lo, hi)
        s.setDecimals(dec)
        s.setSingleStep(step)
        s.setValue(val)
        s.setSuffix(suffix)
        return s

    def _build_controls(self):
        box = QWidget()
        box.setFixedWidth(330)
        outer = QVBoxLayout(box)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        inner = QWidget()
        col = QVBoxLayout(inner)
        col.setContentsMargins(2, 2, 8, 2)
        col.setSpacing(10)

        d = mm.MotionConfig()
        gt = QGroupBox("Tecnica di gestione del moto")
        gtl = QVBoxLayout(gt)
        self.combo_tech = QComboBox()
        for key, lab, _ in TECHNIQUES:
            self.combo_tech.addItem(lab, key)
        self.combo_tech.setCurrentIndex(self.combo_tech.findData("tracking"))
        self._tech_item("adaptive").setEnabled(False)
        self.combo_tech.currentIndexChanged.connect(self._on_technique)
        gtl.addWidget(self.combo_tech)
        row = QHBoxLayout()
        self.lbl_gate = QLabel("Finestra di gating")
        row.addWidget(self.lbl_gate)
        self.spin_gate = self._dspin(1.0, 10.0, 3.0, 0.5, " mm")
        self.spin_gate.setToolTip(
            "Il fascio si accende solo quando la STIMA del tumore e' entro "
            "questa distanza dalla posizione pianificata.\nPiu' stretta = "
            "errore minore ma fascio acceso meno spesso (seduta piu' lunga).\n"
            "Misurato: 3 mm -> p95 4.3 mm, fascio acceso ~24% del tempo.")
        row.addWidget(self.spin_gate)
        gtl.addLayout(row)
        self.lbl_tech = QLabel()
        self.lbl_tech.setWordWrap(True)
        self.lbl_tech.setObjectName("hintLabel")
        gtl.addWidget(self.lbl_tech)
        self.lbl_margin = QLabel("")
        self.lbl_margin.setWordWrap(True)
        gtl.addWidget(self.lbl_margin)
        col.addWidget(gt)

        g = QGroupBox("Respiro del paziente")
        grid = QGridLayout(g)
        self.chk_auto = QCheckBox("Ampiezza dalla posizione della lesione")
        self.chk_auto.setChecked(True)
        self.chk_auto.setToolTip(
            "I tumori vicini al diaframma si muovono di piu' (fino a ~18 mm), "
            "quelli apicali di pochi mm (Seppenwoolde 2002).\nLa posizione si "
            "ricava dalla maschera polmonare, segmentata qui in background.")
        self.chk_auto.toggled.connect(self._sync_auto)
        grid.addWidget(self.chk_auto, 0, 0, 1, 2)
        self.spin_amp = self._dspin(0, 40, abs(d.inhale_shift_mm[2]), 1,
                                    " mm")
        self.spin_amp.setToolTip("Escursione cranio-caudale a fine "
                                 "inspirazione. Gli altri assi scalano in "
                                 "proporzione.")
        grid.addWidget(QLabel("Escursione SI"), 1, 0)
        grid.addWidget(self.spin_amp, 1, 1)
        self.spin_period = self._dspin(2, 10, d.period_s, 0.5, " s")
        grid.addWidget(QLabel("Periodo"), 2, 0)
        grid.addWidget(self.spin_period, 2, 1)
        self.spin_irr = self._dspin(0, 40, 100 * d.period_cv, 2, " %", 0)
        self.spin_irr.setToolTip("Variabilita' di periodo e ampiezza da un "
                                 "ciclo all'altro.")
        grid.addWidget(QLabel("Irregolarita'"), 3, 0)
        grid.addWidget(self.spin_irr, 3, 1)
        self.spin_drift = self._dspin(0, 5, float(np.linalg.norm(
            d.drift_mm_per_min)), 0.1, " mm/min", 2)
        self.spin_drift.setToolTip("Deriva lenta della posizione di riposo.")
        grid.addWidget(QLabel("Deriva"), 4, 0)
        grid.addWidget(self.spin_drift, 4, 1)
        col.addWidget(g)

        g2 = QGroupBox("Sistema di inseguimento")
        g2l = QGridLayout(g2)
        self.combo_pred = QComboBox()
        for n in self.predictors:
            self.combo_pred.addItem(PRED_LABELS[n], n)
        self.combo_pred.setCurrentIndex(self.predictors.index("kalman"))
        self.combo_pred.setToolTip(
            "Compensa i 115 ms di ritardo del sistema predicendo il respiro.\n"
            "Nel nodo ROS (senza torch) la GRU non e' disponibile: li' si usa "
            "Kalman.")
        g2l.addWidget(QLabel("Predittore"), 0, 0)
        g2l.addWidget(self.combo_pred, 0, 1)
        self.spin_kv = self._dspin(10, 180, d.kv_period_s, 10, " s", 0)
        self.spin_kv.setToolTip(
            "Ogni quanto una coppia di radiografie misura il tumore e ritara "
            "il modello di correlazione.\nE' la leva principale: misurato p95 "
            "3.2 mm a 60 s, 2.6 mm a 30 s.")
        g2l.addWidget(QLabel("Radiografie ogni"), 1, 0)
        g2l.addWidget(self.spin_kv, 1, 1)
        self.spin_seed = self._dspin(0, 999, 0, 1, "", 0)
        g2l.addWidget(QLabel("Seme"), 2, 0)
        g2l.addWidget(self.spin_seed, 2, 1)
        col.addWidget(g2)

        self.btn_sim = QPushButton("Simula seduta (5 min)")
        self.btn_sim.setMinimumHeight(36)
        self.btn_sim.clicked.connect(self.on_simulate)
        col.addWidget(self.btn_sim)

        self.chk_adaptive = QCheckBox("Includi il PTV adattivo")
        self.chk_adaptive.setToolTip(
            "Richiede l'analisi di incertezza del Tab 1.")
        self.chk_adaptive.setEnabled(False)
        col.addWidget(self.chk_adaptive)
        self.btn_scen = QPushButton("Confronta tecniche (dose)")
        self.btn_scen.setMinimumHeight(36)
        self.btn_scen.setEnabled(False)
        self.btn_scen.setToolTip(
            "Un piano completo per scenario (~1 min ciascuno):\n"
            "  statico, CTV senza compensazione, ITV a fascio fermo,\n"
            "  inseguimento con margine di van Herk.")
        self.btn_scen.clicked.connect(self.on_scenarios)
        col.addWidget(self.btn_scen)

        self.progress = QProgressBar()
        self.progress.setTextVisible(False)
        self.progress.setRange(0, 1)
        col.addWidget(self.progress)
        self.lbl_status = QLabel(
            "Respiro sintetico (modello di Lujan): il dataset non contiene "
            "4DCT. Ampiezze da letteratura.")
        self.lbl_status.setWordWrap(True)
        self.lbl_status.setObjectName("hintLabel")
        col.addWidget(self.lbl_status)
        col.addStretch(1)

        for w in (self.spin_amp, self.spin_period, self.spin_irr,
                  self.spin_drift, self.spin_kv, self.spin_seed):
            w.valueChanged.connect(lambda *_: self._emit_config())
        self.combo_pred.currentIndexChanged.connect(
            lambda *_: self._emit_config())

        scroll.setWidget(inner)
        outer.addWidget(scroll)
        return box

    # ------------------------------------------------------------------
    def current_config(self):
        d = mm.MotionConfig()
        amp = self.spin_amp.value()
        shift = mm.inhale_shift_for_location(0.5, d)
        k = amp / max(abs(shift[2]), 1e-6)
        dr = np.asarray(d.drift_mm_per_min, float)
        dr = dr / max(np.linalg.norm(dr), 1e-9) * self.spin_drift.value()
        pred = self.combo_pred.currentData()
        return d.replace(
            inhale_shift_mm=tuple(float(v) * k for v in shift),
            period_s=self.spin_period.value(),
            period_cv=self.spin_irr.value() / 100.0,
            amplitude_cv=self.spin_irr.value() / 100.0,
            drift_mm_per_min=tuple(float(v) for v in dr),
            kv_period_s=self.spin_kv.value(),
            predictor=pred,
            seed=int(self.spin_seed.value()))

    def technique(self):
        return self.combo_tech.currentData()

    def ros_config(self):
        """Respiro da scrivere nel piano, o None se la tecnica lo ignora.

        Restituisce {"cfg": MotionConfig, "inseguimento": bool, "tecnica"}.
        Il nodo ROS non ha torch: la GRU vi diventa Kalman, e lo si scrive
        nel piano invece di lasciarlo scoprire.
        """
        tech = self.technique()
        if tech == "none":
            return None
        cfg = self.current_config()
        if cfg.predictor == "gru":
            cfg = cfg.replace(predictor="kalman")
        out = {"cfg": cfg, "inseguimento": tech in ("tracking", "adaptive"),
               "tecnica": tech}
        if tech == "gating":
            out["gating_mm"] = self.spin_gate.value()
        return out

    def _tech_item(self, key):
        return self.combo_tech.model().item(self.combo_tech.findData(key))

    def _emit_config(self):
        gating = self.technique() == "gating"
        self.spin_gate.setVisible(gating)
        self.lbl_gate.setVisible(gating)
        self.configChanged.emit(self.ros_config())
        self.techniqueChanged.emit(self.technique())
        if self.technique() in MARGIN_TECHS:
            self._margin_timer.start()
        self._render_margin()

    def _on_technique(self, *_):
        key = self.technique()
        self.lbl_tech.setText(next(e for k, _, e in TECHNIQUES if k == key))
        self._emit_config()
        self.resultChanged.emit()

    # --- margine d'inseguimento ----------------------------------------
    def _gate(self):
        return self.spin_gate.value() if self.technique() == "gating" else None

    def _margin_key(self):
        # Il margine del gating e' un'altra cosa da quello dell'inseguimento:
        # dipende dalla finestra, e la chiave lo distingue.
        return repr((sorted(self.current_config().as_dict().items()),
                     self._gate()))

    def _start_margin(self):
        if self.technique() not in MARGIN_TECHS:
            return
        key = self._margin_key()
        if key == self._pop_key or (self.margin_worker is not None and
                                    self.margin_worker.isRunning()):
            if self.margin_worker is not None and \
                    self.margin_worker.isRunning():
                self._margin_timer.start()     # riprova a calcolo finito
            return
        cfg = self.current_config()
        self.margin_worker = MarginWorker(key, cfg, cfg.predictor,
                                          self._gate())
        self.margin_worker.done.connect(self._on_margin)
        self.margin_worker.failed.connect(self._on_failed)
        self.margin_worker.start()
        self._render_margin()

    def _on_margin(self, key, pop):
        if key != self._margin_key():
            self._margin_timer.start()         # configurazione cambiata
            return
        self.pop, self._pop_key = pop, key
        self._render_margin()
        self.techniqueChanged.emit(self.technique())
        self.resultChanged.emit()

    def margin_ready(self):
        return self.pop is not None and self._pop_key == self._margin_key()

    def _render_margin(self):
        if self.technique() not in MARGIN_TECHS:
            self.lbl_margin.setText("")
            return
        if self.margin_ready():
            m = self.pop["margine_mm"]
            duty = ""
            if self.pop.get("gating_mm"):
                duty = (f"<br>Fascio acceso {100 * self.pop['fascio_acceso']:.0f}"
                        f"% del tempo: l'erogazione dura circa "
                        f"{1 / max(self.pop['fascio_acceso'], 0.05):.1f} volte.")
            what = ("del gating" if self.pop.get("gating_mm")
                    else "d'inseguimento")
            self.lbl_margin.setText(
                f"<span style='color:#3ddc84'>Margine {what} "
                f"(van Herk, {self.pop['n_sedute']} sedute simulate): "
                f"X {m[0]:.1f} / Y {m[1]:.1f} / Z {m[2]:.1f} mm{duty}</span>")
        else:
            self.lbl_margin.setText(
                "<span style='color:#ffb020'>Calcolo del margine "
                "d'inseguimento in corso (~30 s)...</span>")

    # --- bersaglio per la dosimetria ----------------------------------
    def planning_target(self, ctv, affine, maps=None):
        """Maschera da ottimizzare secondo la tecnica, ed etichetta.

        Solleva RuntimeError con un messaggio leggibile se manca qualcosa
        (margine in calcolo, incertezza assente).
        """
        import robust_target as rt
        tech = self.technique()
        cfg = self.current_config()
        if tech == "none":
            return ctv, "CTV (tecnica: tumore fermo)"
        if tech == "itv":
            plan_tr = mm.generate_trace(
                cfg.replace(seed=cfg.seed + 97, drift_mm_per_min=(0, 0, 0)),
                40.0)
            itv = md.envelope_mask(rt.fixed_target(ctv, affine, 5.0), affine,
                                   plan_tr.tumor)
            p2p = np.ptp(plan_tr.tumor, axis=0)
            return itv, (f"ITV (CTV + 5 mm sulla traiettoria, escursione "
                         f"{p2p[0]:.0f}/{p2p[1]:.0f}/{p2p[2]:.0f} mm)")
        if not self.margin_ready():
            raise RuntimeError(
                "Il margine d'inseguimento e' ancora in calcolo nel Tab 2 "
                "(circa 30 s). Riprova fra poco.")
        m = self.pop["margine_mm"]
        if tech == "gating":
            return (md.expand_mask(ctv, affine, m),
                    f"CTV + margine del gating {m[0]:.1f}/{m[1]:.1f}/"
                    f"{m[2]:.1f} mm (finestra {self.spin_gate.value():.1f} mm)")
        if tech == "tracking":
            return (md.expand_mask(ctv, affine, m),
                    f"CTV + margine d'inseguimento {m[0]:.1f}/{m[1]:.1f}/"
                    f"{m[2]:.1f} mm")
        if maps is None:
            raise RuntimeError(
                "La tecnica adattiva richiede l'analisi di incertezza: "
                "ricarica la CT nel Tab 1 con la casella attiva.")
        ptv, info = rt.adaptive_target(ctv, maps, affine, self.pop)
        mm_ = info["margine_mm"]
        return ptv, (f"PTV adattivo (banda tau {info['tau']:.2f}, margine "
                     f"{mm_[0]:.1f}/{mm_[1]:.1f}/{mm_[2]:.1f} mm)")

    def _sync_auto(self):
        auto = self.chk_auto.isChecked() and self.si_fraction is not None
        self.spin_amp.setEnabled(not auto)
        if auto:
            s = mm.inhale_shift_for_location(self.si_fraction)
            self.spin_amp.setValue(abs(s[2]))

    def set_source_data(self, res):
        self.source = res if res.get("tumor_found") else None
        self.si_fraction = None
        self.organs = None
        self._last_scen = None
        unc = res.get("uncertainty")
        self.chk_adaptive.setEnabled(unc is not None)
        self.chk_adaptive.setChecked(unc is not None)
        self._tech_item("adaptive").setEnabled(unc is not None)
        if unc is None and self.technique() == "adaptive":
            self.combo_tech.setCurrentIndex(
                self.combo_tech.findData("tracking"))
        self._update_location()
        if self.source is not None:
            self.lbl_status.setText("Segmentazione dei polmoni per stimare "
                                    "il respiro...")
            self.organ_worker = OrganWorker(res["ct_data"], res["affine"])
            self.organ_worker.done.connect(self._on_organs_ready)
            self.organ_worker.failed.connect(self._on_failed)
            self.organ_worker.start()

    def _on_organs_ready(self, res):
        self.set_organs(res)
        self.organsReady.emit(res)

    def set_organs(self, res):
        if self.organs is not None and res.get("organ_masks") is \
                self.organs.get("organ_masks"):
            return
        self.organs = res if res.get("organ_masks") else None
        self._update_location()

    def _update_location(self):
        ok = self.source is not None and self.organs is not None
        self.btn_scen.setEnabled(ok)
        if not ok:
            return
        m = self.organs["organ_masks"]
        self.si_fraction = mm.lesion_si_fraction(
            self.source["com_world"], m["lung_r"] | m["lung_l"],
            self.organs["organ_affine"])
        self.lbl_status.setText(
            f"Lesione al {100 * self.si_fraction:.0f}% dell'altezza del "
            f"polmone (0 = diaframma, 100 = apice).")
        self._sync_auto()

    # ------------------------------------------------------------------
    def _busy(self, on):
        self.progress.setRange(0, 0 if on else 1)
        self.btn_sim.setEnabled(not on)
        self.btn_scen.setEnabled(not on and self.source is not None
                                 and self.organs is not None)

    def on_simulate(self):
        self._busy(True)
        self.lbl_status.setText("Simulazione della seduta...")
        self.worker = TrackingWorker(self.current_config(), 300.0,
                                     self.predictors)
        self.worker.done.connect(self._on_tracking)
        self.worker.failed.connect(self._on_failed)
        self.worker.start()

    def on_scenarios(self):
        adaptive = None
        if self.chk_adaptive.isChecked() and self.source is not None:
            adaptive = self.source.get("uncertainty")
        self._busy(True)
        cfg = self.current_config()
        pop = self.pop if (self.margin_ready()
                           and self.technique() != "gating") else None
        self.scen_worker = ScenarioWorker(
            self.source, self.organs, cfg, cfg.predictor, adaptive,
            pop=pop, gating_mm=self.spin_gate.value())
        self.scen_worker.status.connect(self.lbl_status.setText)
        self.scen_worker.done.connect(self._on_scenarios)
        self.scen_worker.failed.connect(self._on_failed)
        self.scen_worker.start()

    def _on_failed(self, msg):
        self._busy(False)
        self._render_margin()
        QMessageBox.critical(self, "Errore nella simulazione del respiro", msg)

    # ------------------------------------------------------------------
    def _on_tracking(self, out):
        self._busy(False)
        self._last_tracking = out
        self.resultChanged.emit()
        tr = out["trace"]
        runs = out["runs"]
        sel = self.combo_pred.currentData()
        r = runs[sel]["result"]
        self.lbl_status.setText("Seduta simulata.")

        fig = self.fig_trk
        fig.clear()
        gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 1], hspace=0.35,
                              wspace=0.25)
        ax = fig.add_subplot(gs[0, :])
        self._style(ax, "Posizione cranio-caudale del tumore: vera e "
                        "stimata dal robot")
        t0 = r.t[r.active][0] if r.active.any() else 0.0
        w = (r.t >= t0) & (r.t <= t0 + 30)
        ax.plot(r.t[w], r.truth[w, 2], color="#ff8a00", lw=1.8, label="tumore")
        est = np.where(r.active, r.estimate[:, 2], np.nan)
        ax.plot(r.t[w], est[w], color="#00e5ff", lw=1.2, ls="--",
                label=f"fascio ({PRED_LABELS[sel]})")
        kv = r.kv_t[(r.kv_t >= t0) & (r.kv_t <= t0 + 30)]
        if len(kv):
            ax.plot(kv, np.interp(kv, r.t, r.truth[:, 2]), "o",
                    color="#ff5c5c", label="radiografia")
        ax.set_xlabel("s", color=FG)
        ax.set_ylabel("Z (mm)", color=FG)
        ax.legend(facecolor=BG, edgecolor=GRID, labelcolor=FG, fontsize=8,
                  loc="upper right")

        ax2 = fig.add_subplot(gs[1, 0])
        self._style(ax2, "Errore 3D nel tempo")
        n3 = np.linalg.norm(r.truth - r.estimate, axis=1)
        n3u = np.linalg.norm(r.truth, axis=1)
        ax2.plot(r.t[r.active], n3u[r.active], color="#ff5c5c", lw=0.6,
                 alpha=0.7, label="fascio fermo")
        ax2.plot(r.t[r.active], n3[r.active], color="#3ddc84", lw=0.6,
                 label="inseguimento")
        for k in r.kv_t:
            ax2.axvline(k, color="#ff5c5c", alpha=0.25, lw=0.8)
        ax2.set_xlabel("s", color=FG)
        ax2.set_ylabel("mm", color=FG)
        ax2.legend(facecolor=BG, edgecolor=GRID, labelcolor=FG, fontsize=7)

        ax3 = fig.add_subplot(gs[1, 1])
        self._style(ax3, "Predittori a confronto (p95 errore 3D)")
        names = list(runs)
        vals = [runs[n]["stats"]["p95_3d"] for n in names]
        cols = ["#00e5ff" if n == sel else "#4a6a72" for n in names]
        ax3.bar([PRED_LABELS[n] for n in names], vals, color=cols)
        ax3.axhline(out["untracked"]["p95_3d"], color="#ff5c5c", ls="--",
                    lw=1, label="fascio fermo")
        ax3.set_ylabel("mm", color=FG)
        ax3.legend(facecolor=BG, edgecolor=GRID, labelcolor=FG, fontsize=7)
        self.can_trk.draw_idle()
        self.views.setCurrentIndex(0)

        rows = "".join(
            f"<tr><td>{PRED_LABELS[n]}</td>"
            f"<td align=right>{runs[n]['pred_rmse']:.2f}</td>"
            f"<td align=right>{runs[n]['stats']['median_3d']:.2f}</td>"
            f"<td align=right>{runs[n]['stats']['p95_3d']:.2f}</td>"
            f"<td align=right>{100 * runs[n]['stats']['within_2mm']:.0f}%</td>"
            "</tr>" for n in names)
        u = out["untracked"]
        kv_err = np.linalg.norm(r.kv_err, axis=1) if len(r.kv_err) else []
        self.lbl_metrics.setText(
            "<table cellspacing=6 style='color:#c8d0da;font-size:11px'>"
            "<tr style='color:#8b95a3'><td>predittore</td><td>RMSE 115 ms</td>"
            "<td>mediana 3D</td><td>p95 3D</td><td>entro 2 mm</td></tr>"
            f"{rows}"
            f"<tr style='color:#ff5c5c'><td>fascio fermo</td><td></td>"
            f"<td align=right>{u['median_3d']:.2f}</td>"
            f"<td align=right>{u['p95_3d']:.2f}</td>"
            f"<td align=right>{100 * u['within_2mm']:.0f}%</td></tr>"
            "</table>"
            f"<span style='color:#8b95a3;font-size:11px'>valori in mm. "
            f"Errore del modello alle radiografie (prima della ritaratura): "
            + (", ".join(f"{v:.1f}" for v in kv_err) or "-") + "</span>")

    def _on_scenarios(self, out):
        self._busy(False)
        self._last_scen = out
        self.resultChanged.emit()
        self.lbl_status.setText("Confronto completato.")
        res = out["results"]
        fig = self.fig_dose
        fig.clear()
        ax = fig.add_subplot(121)
        self._style(ax, "DVH del tumore in movimento")
        for name, (d, v) in out["dvh"].items():
            ax.plot(d, v, color=SCEN_COLORS.get(name, "#ffffff"), lw=1.8,
                    label=md.SCENARIO_LABELS.get(name, name))
        rx = config.PLANNER.prescription_pct
        ax.axvline(rx, color="#ffffff", ls=":", alpha=0.5)
        ax.set_xlim(0, 150)
        ax.set_ylim(0, 102)
        ax.set_xlabel("dose (% scala del piano)", color=FG)
        ax.set_ylabel("volume del tumore (%)", color=FG)
        ax.legend(facecolor=BG, edgecolor=GRID, labelcolor=FG, fontsize=7,
                  loc="lower left")

        ax2 = fig.add_subplot(122)
        self._style(ax2, "Copertura contro polmone irradiato")
        for name, r in res.items():
            ax2.scatter(r.get("polmone_v20_pct", 0), r["copertura_pct"],
                        s=40 + 4 * r["ptv_cm3"],
                        color=SCEN_COLORS.get(name, "#ffffff"))
            ax2.annotate(md.SCENARIO_LABELS.get(name, name),
                         (r.get("polmone_v20_pct", 0), r["copertura_pct"]),
                         color=FG, fontsize=7, xytext=(5, 4),
                         textcoords="offset points")
        ax2.set_xlabel("V20 polmone (%)", color=FG)
        ax2.set_ylabel("copertura del tumore (%)", color=FG)
        fig.tight_layout()
        self.can_dose.draw_idle()
        self.views.setCurrentIndex(1)

        rows = "".join(
            f"<tr style='color:{SCEN_COLORS.get(n, FG)}'>"
            f"<td>{md.SCENARIO_LABELS.get(n, n)}</td>"
            f"<td align=right>{r['ptv_cm3']:.1f}</td>"
            f"<td align=right>{r['copertura_pct']:.1f}</td>"
            f"<td align=right>{r['d95_pct']:.1f}</td>"
            f"<td align=right>{r['paddick_ci']:.2f}</td>"
            f"<td align=right>{r.get('polmone_dose_media_gy', 0):.2f}</td>"
            f"<td align=right>{r.get('polmone_v20_pct', 0):.2f}</td></tr>"
            for n, r in res.items())
        mg = out["margin_mm"]
        self.lbl_metrics.setText(
            "<table cellspacing=6 style='color:#c8d0da;font-size:11px'>"
            "<tr style='color:#8b95a3'><td>scenario</td><td>PTV cm³</td>"
            "<td>copertura %</td><td>D95 %</td><td>CI</td>"
            "<td>MLD Gy</td><td>V20 %</td></tr>"
            f"{rows}</table>"
            f"<span style='color:#8b95a3;font-size:11px'>"
            f"Margine d'inseguimento (van Herk, 12 sedute simulate): "
            f"X {mg[0]:.1f} / Y {mg[1]:.1f} / Z {mg[2]:.1f} mm. "
            f"Dose sotto moto con approssimazione 'static dose cloud'; "
            f"Gy su scala {md.RX_GY:.0f} Gy alla prescrizione.</span>")

    # ------------------------------------------------------------------
    #  Progetto
    # ------------------------------------------------------------------
    _SPINS = ("spin_amp", "spin_period", "spin_irr", "spin_drift", "spin_kv",
              "spin_seed", "spin_gate")

    def export_state(self):
        return {
            "tecnica": self.technique(),
            "auto": self.chk_auto.isChecked(),
            "valori": {n: getattr(self, n).value() for n in self._SPINS},
            "predittore": self.combo_pred.currentData(),
            "adattivo_nel_confronto": self.chk_adaptive.isChecked(),
            "margine": self.pop, "margine_chiave": self._pop_key,
            "inseguimento": self._last_tracking,
            "confronto": self._last_scen,
        }

    def restore_state(self, st):
        if not st:
            return
        widgets = [self.combo_tech, self.chk_auto, self.combo_pred] + \
            [getattr(self, n) for n in self._SPINS]
        for w in widgets:
            w.blockSignals(True)
        try:
            i = self.combo_tech.findData(st.get("tecnica"))
            if i >= 0 and self.combo_tech.model().item(i).isEnabled():  # noqa
                self.combo_tech.setCurrentIndex(i)
            self.chk_auto.setChecked(bool(st.get("auto", True)))
            for n, v in (st.get("valori") or {}).items():
                if hasattr(self, n):
                    getattr(self, n).setValue(v)
            i = self.combo_pred.findData(st.get("predittore"))
            if i >= 0:
                self.combo_pred.setCurrentIndex(i)
        finally:
            for w in widgets:
                w.blockSignals(False)
        self.chk_adaptive.setChecked(bool(st.get("adattivo_nel_confronto")))
        self.spin_amp.setEnabled(not (self.chk_auto.isChecked()
                                      and self.si_fraction is not None))
        if st.get("margine") and st.get("margine_chiave") == \
                self._margin_key():
            self.pop, self._pop_key = st["margine"], st["margine_chiave"]
        key = self.technique()
        self.lbl_tech.setText(next(e for k, _, e in TECHNIQUES if k == key))
        if st.get("inseguimento"):
            self._on_tracking(st["inseguimento"])
        if st.get("confronto"):
            self._on_scenarios(st["confronto"])
        self._emit_config()
