# ============================================================================
#  kinematics_tab.py - TAB 3: cinematica del manipolatore e ponte verso ROS 2.
#
#  E' la scheda che chiude il percorso: il piano calcolato dalla dosimetria
#  (Fase 2) diventa una sequenza di configurazioni di giunto eseguibili dal
#  braccio, e da li' un pacchetto ROS 2 che gira in WSL.
#
#  COSA SI VEDE QUI E PERCHE'
#  --------------------------
#  Un piano dosimetricamente ottimo puo' essere inerogabile: i fasci escono da
#  un ottimizzatore che ragiona solo di dose e non sa nulla dei fine corsa del
#  robot. Questa scheda misura quanto del piano il braccio riesce davvero a
#  erogare, con che margine dai limiti di giunto e con quanto movimento. Sono
#  i numeri che dicono se il piano e' realizzabile, non solo buono sulla carta.
#
#  Il calcolo pesante sta in un QThread: la cinematica inversa su un centinaio
#  di fasci, con i rami di gomito/spalla/polso e la ridondanza, impiega
#  qualche secondo e bloccherebbe l'interfaccia.
# ============================================================================
import json
import os
import subprocess
import sys
import time

import numpy as np

from PyQt6.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QPushButton, QLabel,
    QGroupBox, QDoubleSpinBox, QSpinBox, QProgressBar, QMessageBox,
    QScrollArea, QSizePolicy, QFileDialog,
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

import kinematics as kin
from plan_export import save_plan

HERE = os.path.dirname(os.path.abspath(__file__))
WSL_DISTRO = "Ubuntu-22.04"

#: Il piano che il nodo ROS legge davvero. Il pulsante di avvio ci scrive
#: sopra da solo: dover ricordarsi di esportare prima di lanciare e' un invito
#: a simulare un piano vecchio, ed e' gia' successo.
ROS_PLAN = os.path.join(HERE, "ros2_ws", "src", "cyberknife_ros", "plans",
                        "piano_trattamento.json")
#: Il resoconto che il nodo scrive a fine seduta, accanto al piano.
ROS_REPORT = os.path.join(os.path.dirname(ROS_PLAN),
                          "resoconto_trattamento.json")
ROS_MESHES = os.path.join(HERE, "ros2_ws", "src", "cyberknife_ros", "meshes")


# ---------------------------------------------------------------------------
class KinematicsWorker(QThread):
    """Risolve la cinematica inversa dell'intero piano fuori dal thread GUI."""

    status = pyqtSignal(str)
    progress = pyqtSignal(int)
    solved = pyqtSignal(dict)
    failed = pyqtSignal(str)

    def __init__(self, beams, base, n_roll, obstacles=None, couch_poses=None):
        super().__init__()
        self.beams, self.base, self.n_roll = beams, base, n_roll
        self.obstacles = obstacles or []
        self.couch_poses = couch_poses or None

    def run(self):
        try:
            self.status.emit("Cinematica inversa sui fasci del piano...")
            self.progress.emit(15)
            poses, diag = kin.solve_plan(self.beams, self.base,
                                         n_roll=self.n_roll,
                                         obstacles=self.obstacles,
                                         couch_poses=self.couch_poses)
            self.progress.emit(70)
            if not poses:
                self.failed.emit(
                    "Nessun fascio eseguibile da questa collocazione della "
                    "base: o il braccio non ci arriva, o per arrivarci "
                    "attraversa il lettino. Prova a spostarla: la distanza "
                    "laterale e la quota sono i parametri piu' influenti.")
                return

            self.status.emit("Verifica di chiusura FK/IK...")
            eff = [c.base_in_patient(self.base)
                   for c in (self.couch_poses or [kin.CouchPose()])]
            errs = np.array([
                kin.fk_error_mm(p["q"], p["beam"].source,
                                p["beam"].target - p["beam"].source,
                                eff[p["couch"]])
                for p in poses])
            self.progress.emit(90)

            # Confronto con la sequenza non ordinata: quantifica quanto vale
            # l'euristica del nodo piu' vicino su QUESTO piano.
            _, diag_naive = kin.solve_plan(self.beams, self.base,
                                           n_roll=self.n_roll,
                                           order_by_motion=False,
                                           obstacles=self.obstacles,
                                           couch_poses=self.couch_poses)
            self.progress.emit(100)
            self.solved.emit({
                "poses": poses, "diag": diag,
                "movimento_naive_deg": diag_naive["movimento_totale_deg"],
                "pos_err_mm": float(errs[:, 0].max()),
                "ang_err_deg": float(errs[:, 1].max()),
            })
        except Exception as exc:                     # non silenziare l'errore
            import traceback
            self.failed.emit(f"{exc}\n\n{traceback.format_exc()}")


# ---------------------------------------------------------------------------
class SetupWorker(QThread):
    """Cerca base del robot e angoli di lettino, fuori dal thread GUI."""

    status = pyqtSignal(str)
    solved = pyqtSignal(dict)
    failed = pyqtSignal(str)

    def __init__(self, obstacles, pivot, template, max_poses, couch_arm=None):
        super().__init__()
        self.obstacles, self.pivot = obstacles, pivot
        self.template, self.max_poses = template, max_poses
        self.couch_arm = couch_arm

    def run(self):
        try:
            # Il braccio del lettino va passato: e' il filtro che scarta gli
            # orientamenti che il meccanismo non puo' assumere.
            res = kin.optimize_setup(self.obstacles, self.pivot,
                                     base_template=self.template,
                                     max_couch_poses=self.max_poses,
                                     couch_arm=self.couch_arm,
                                     status_cb=self.status.emit)
            self.solved.emit(res)
        except Exception as exc:
            import traceback
            self.failed.emit(f"{exc}\n\n{traceback.format_exc()}")


# ---------------------------------------------------------------------------
class SceneWorker(QThread):
    """Riesporta anatomia, ostacoli e braccio del lettino per il nodo ROS."""

    status = pyqtSignal(str)
    done = pyqtSignal(str)          # stringa vuota = riuscito

    def __init__(self, ct_path, mask_path, out_dir):
        super().__init__()
        self.ct_path, self.mask_path, self.out_dir = ct_path, mask_path, out_dir

    def run(self):
        try:
            import export_scene
            export_scene.export_scene(self.ct_path, self.mask_path,
                                      self.out_dir, verbose=False)
            self.done.emit("")
        except Exception as exc:                      # noqa: BLE001
            import traceback
            self.done.emit(f"{exc}\n\n{traceback.format_exc()}")


def scene_matches(plan_data, mesh_dir=ROS_MESHES):
    """La scena del pacchetto e' dello stesso paziente del piano?

    Il nodo ROS prende gli ostacoli, il braccio del lettino e le mesh da
    scene.json, e i fasci dal piano: se vengono da due TAC diverse il robot
    evita un lettino che non c'e' e i fasci puntano lontano dal tumore
    disegnato. E' successo: piano di lung_087 con la scena di lung_001, 61
    fasci su 108 non erogati e la dose a 580 mm dal tumore mostrato.
    """
    try:
        scene = json.load(open(os.path.join(mesh_dir, "scene.json"),
                               encoding="utf-8"))
    except (OSError, ValueError):
        return False
    a = (plan_data.get("sorgente") or {}).get("ct")
    b = scene.get("sorgente_ct")
    if not a or not b:
        return False
    return (os.path.normcase(os.path.abspath(os.path.join(HERE, a))) ==
            os.path.normcase(os.path.abspath(os.path.join(HERE, b))))


# ---------------------------------------------------------------------------
class KinematicsTab(QWidget):
    """TAB 4 - Cinematica del manipolatore e simulazione ROS 2."""

    #: la seduta simulata in ROS e' finita con un resoconto: e' l'ultima fase
    #: della pipeline, il momento naturale per salvare il progetto
    sessionFinished = pyqtSignal()
    #: un risultato nuovo da salvare nel progetto
    resultChanged = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.plan = None          # risultato di compute_plan (dal Tab 2)
        self.source = None        # percorsi CT/maschera (dal Tab 1)
        self.obstacles = []       # lettino e paziente (dalla segmentazione)
        self.couch_poses = []     # angoli di lettino scelti dall'ottimizzatore
        self.couch_arm = None     # manipolatore del lettino (dalla segmentazione)
        self.plan_setup = None    # sala per cui la dosimetria ha scelto i fasci
        self.result = None
        self.worker = None
        self.setup_worker = None
        # Respiro da simulare in ROS (dal Tab 4); None = tumore immobile.
        self.motion_cfg = None
        # Simulazione in corso: il processo WSL e l'istante di avvio. Serve a
        # capire QUANDO e' finita e se il resoconto trovato e' di questa seduta
        # o di una precedente.
        self.sim_proc = None
        self._sim_started = 0.0
        self._report_dlg = None
        self.sim_timer = QTimer(self)
        self.sim_timer.setInterval(1500)
        self.sim_timer.timeout.connect(self._poll_sim)

        root = QHBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(12)
        root.addWidget(self._build_controls(), 0)
        root.addWidget(self._build_view(), 1)
        self._set_enabled(False)

    # ------------------------------------------------------------------
    def _build_controls(self):
        box = QWidget()
        box.setFixedWidth(330)
        # La colonna e' lunga: senza scroll i controlli in fondo diventano
        # irraggiungibili sui portatili con schermo basso.
        outer = QVBoxLayout(box)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        inner = QWidget()
        col = QVBoxLayout(inner)
        col.setContentsMargins(2, 2, 8, 2)
        col.setSpacing(10)

        # --- Collocazione del robot ---
        g_base = QGroupBox("Collocazione del robot")
        grid = QGridLayout(g_base)
        self.spin_x = self._spin(-3000, 3000, kin.RobotBase().origin[0], " mm")
        self.spin_y = self._spin(-3000, 3000, kin.RobotBase().origin[1], " mm")
        self.spin_z = self._spin(-3000, 3000, kin.RobotBase().origin[2], " mm")
        self.spin_yaw = self._spin(-180, 180, 0.0, " gradi")
        for r, (lab, w, tip) in enumerate((
                ("X (destra paz.)", self.spin_x,
                 "Positivo verso la destra del paziente."),
                ("Y (verticale)", self.spin_y,
                 "In RAS +Y e' anteriore: per un paziente supino e' la "
                 "verticale della stanza. Valori negativi = verso il pavimento."),
                ("Z (testa-piedi)", self.spin_z,
                 "Positivo verso la testa del paziente."),
                ("Rotazione", self.spin_yaw,
                 "Rotazione attorno alla verticale del robot. Misurata come "
                 "ininfluente: il giunto A1 ha +-185 gradi e la assorbe."))):
            l = QLabel(lab)
            l.setToolTip(tip)
            w.setToolTip(tip)
            grid.addWidget(l, r, 0)
            grid.addWidget(w, r, 1)
        col.addWidget(g_base)

        # --- Parametri di soluzione ---
        g_opt = QGroupBox("Soluzione")
        go = QGridLayout(g_opt)
        self.spin_roll = QSpinBox()
        self.spin_roll.setRange(1, 36)
        self.spin_roll.setValue(4)
        tip = ("Quante rotazioni attorno all'asse del fascio vengono provate.\n"
               "Il collimatore e' circolare, quindi tutte danno lo stesso "
               "fascio ma configurazioni di giunto diverse.\n"
               "Su questo piano il numero non cambia i fasci raggiungibili: "
               "serve come riserva per vincoli piu' stretti.")
        self.spin_roll.setToolTip(tip)
        go.addWidget(QLabel("Rotazioni provate"), 0, 0)
        go.addWidget(self.spin_roll, 0, 1)

        self.spin_couch = QSpinBox()
        self.spin_couch.setRange(1, 6)
        self.spin_couch.setValue(3)
        self.spin_couch.setToolTip(
            "Quanti orientamenti puo' assumere il lettino durante la seduta.\n"
            "Con 1 il lettino resta fermo. Misurato su 353 direzioni: fermo ne "
            "sono eseguibili il 25%, con tre angoli il 40%.\n"
            "Ogni angolo in piu' costa un riposizionamento del paziente.")
        go.addWidget(QLabel("Angoli di lettino"), 1, 0)
        go.addWidget(self.spin_couch, 1, 1)
        col.addWidget(g_opt)

        self.btn_setup = QPushButton("Ottimizza sala (base + lettino)")
        self.btn_setup.clicked.connect(self.on_optimize_setup)
        self.btn_setup.setToolTip(
            "Cerca insieme dove mettere la base del robot e come orientare il "
            "lettino.\nNon sono scelte separabili: la posizione migliore per "
            "la base dipende da come e' orientato il paziente.")
        col.addWidget(self.btn_setup)

        self.lbl_couch = QLabel("Lettino: fermo (nessun angolo scelto).")
        self.lbl_couch.setWordWrap(True)
        self.lbl_couch.setObjectName("hintLabel")
        col.addWidget(self.lbl_couch)

        # --- Azioni ---
        self.btn_solve = QPushButton("Risolvi cinematica")
        self.btn_solve.setMinimumHeight(38)
        self.btn_solve.clicked.connect(self.on_solve)
        col.addWidget(self.btn_solve)

        self.btn_export = QPushButton("Esporta piano per ROS (JSON)")
        self.btn_export.clicked.connect(self.on_export)
        self.btn_export.setToolTip(
            "Scrive il piano dove preferisci. Per la simulazione non serve: "
            "il pulsante di avvio esporta da solo nel pacchetto ROS.")
        col.addWidget(self.btn_export)

        self.btn_launch = QPushButton("Lancia simulazione ROS 2 / RViz")
        self.btn_launch.clicked.connect(self.on_launch)
        self.btn_launch.setToolTip(
            f"Esporta il piano corrente nel pacchetto, compila se serve e "
            f"avvia RViz dentro WSL ({WSL_DISTRO})."
            f"\nNon serve premere «Esporta» prima.")
        col.addWidget(self.btn_launch)

        self.btn_report = QPushButton("Apri resoconto dell'ultima seduta")
        self.btn_report.clicked.connect(self.open_report)
        self.btn_report.setToolTip(
            "Replay del tumore che si riempie di dose e andamento completo "
            "della seduta.\nSi apre da solo quando la simulazione finisce.")
        self.btn_report.setEnabled(os.path.exists(ROS_REPORT))
        col.addWidget(self.btn_report)

        self.progress = QProgressBar()
        self.progress.setTextVisible(False)
        col.addWidget(self.progress)

        self.lbl_status = QLabel("In attesa di un piano dal Tab 3.")
        self.lbl_status.setWordWrap(True)
        col.addWidget(self.lbl_status)
        col.addStretch(1)

        scroll.setWidget(inner)
        outer.addWidget(scroll)
        return box

    @staticmethod
    def _spin(lo, hi, val, suffix):
        s = QDoubleSpinBox()
        s.setRange(lo, hi)
        s.setDecimals(0)
        s.setSingleStep(50.0)
        s.setValue(float(val))
        s.setSuffix(suffix)
        return s

    # ------------------------------------------------------------------
    def _build_view(self):
        box = QWidget()
        lay = QVBoxLayout(box)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)

        self.lbl_metrics = QLabel("")
        self.lbl_metrics.setObjectName("metricsLabel")
        self.lbl_metrics.setTextFormat(Qt.TextFormat.RichText)
        self.lbl_metrics.setWordWrap(True)
        self.lbl_metrics.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.lbl_metrics.setSizePolicy(QSizePolicy.Policy.Preferred,
                                       QSizePolicy.Policy.Maximum)
        lay.addWidget(self.lbl_metrics)

        self.fig = Figure(figsize=(7, 5), facecolor="#12141a")
        self.canvas = FigureCanvas(self.fig)
        lay.addWidget(self.canvas, 1)
        self._draw_empty()
        return box

    def _draw_empty(self):
        self.fig.clear()
        ax = self.fig.add_subplot(111, facecolor="#12141a")
        ax.text(0.5, 0.5,
                "Calcola prima un piano nel Tab 3,\npoi premi "
                "«Risolvi cinematica».",
                ha="center", va="center", color="#7d8590", fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        self.canvas.draw_idle()

    # ------------------------------------------------------------------
    def set_source_data(self, res: dict):
        """Percorsi dell'immagine, per annotare il piano esportato.

        Il Tab 1 chiama il percorso della TAC `image_path`, mentre il piano lo
        cercava sotto `ct_path`: il piano esportato dalla GUI usciva sempre con
        la sorgente vuota, e non c'era modo di accorgersi che la scena del
        pacchetto era di un altro paziente.
        """
        self.source = dict(res)
        self.source.setdefault("ct_path", res.get("image_path"))

    def _save_mask(self):
        """Scrive su disco la maschera del tumore: serve a export_scene.

        La maschera vive solo in memoria (Tab 1); la scena per ROS si ricava
        dai file, come da riga di comando.
        """
        src = self.source or {}
        ct = src.get("ct_path")
        if not ct or src.get("mask") is None or src.get("affine") is None:
            return None
        import nibabel as nib
        name = os.path.basename(ct).split(".")[0]
        path = os.path.join(HERE, "outputs", f"{name}_seg.nii.gz")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        nib.save(nib.Nifti1Image(np.asarray(src["mask"], dtype=np.uint8),
                                 np.asarray(src["affine"])), path)
        src["mask_path"] = path
        return path

    def set_organs(self, res: dict):
        """Ostacoli ricavati dalla segmentazione fatta nel Tab 3.

        Il lettino non e' un valore fisso: il suo piano coincide con il dorso
        del paziente, quindi dipende dall'immagine caricata. Ricavarlo dalla
        maschera del corpo e' anche cio' che tiene d'accordo il vincolo
        dosimetrico e il modello di collisione.
        """
        masks = res.get("organ_masks") or {}
        aff = res.get("organ_affine")
        body = masks.get("body")
        if body is None or aff is None:
            return
        b = kin.bounds_mm(body, aff)
        if b is None:
            return
        self.obstacles = kin.treatment_obstacles(b)
        self.couch_arm = kin.treatment_couch_arm(b)
        # La quota della base non e' una preferenza: il robot appoggia sul
        # pavimento, che sta sotto il lettino, che appoggia sul dorso del
        # paziente. Cambiando immagine cambia, quindi si aggiorna il campo
        # invece di lasciare il valore di un altro caso.
        self.spin_y.setValue(round(kin.floor_y_mm(b)))
        self.lbl_status.setText(
            f"Ostacoli pronti: {', '.join(o.name for o in self.obstacles)}. "
            f"Base portata a quota pavimento (Y = {kin.floor_y_mm(b):.0f} mm).")

    def set_motion(self, cfg):
        """Tecnica e respiro scelti nel Tab 2 (None = moto ignorato)."""
        self.motion_cfg = cfg

    def set_plan(self, res: dict):
        """Riceve il piano calcolato dal Tab 3.

        Se la dosimetria ha scelto i fasci per una configurazione di sala, il
        Tab 3 la ADOTTA: i fasci sono erogabili da quella base e con quelle
        rotazioni del lettino, non da una qualunque. Ricalcolarla qui dopo -
        come si faceva - era la causa dei fasci persi.
        """
        self.plan = res
        n = len(res.get("beams", []))
        self._set_enabled(True)
        base = res.get("robot_base")
        if base is not None:
            self.spin_x.setValue(round(float(base.origin[0])))
            self.spin_y.setValue(round(float(base.origin[1])))
            self.spin_z.setValue(round(float(base.origin[2])))
            self.spin_yaw.setValue(0.0)
            self.couch_poses = list(res.get("couch_poses") or [])
            if res.get("robot_obstacles"):
                self.obstacles = list(res["robot_obstacles"])
            self.plan_setup = self._setup_key()
            self.lbl_couch.setText(
                f"Sala decisa dalla dosimetria: {self._angles_text()}. "
                f"I fasci del piano sono stati scelti per questa sala.")
            self.lbl_status.setText(
                f"Piano ricevuto: {n} fasci, con base e lettino gia' "
                f"ottimizzati. Premi «Risolvi cinematica».")
        else:
            self.plan_setup = None
            self.lbl_status.setText(
                f"Piano ricevuto: {n} fasci, scelti SENZA vincoli del robot: "
                f"una parte potrebbe non essere erogabile.")

    def _setup_key(self):
        """Firma della sala corrente, per accorgersi se e' stata cambiata."""
        return (round(self.spin_x.value()), round(self.spin_y.value()),
                round(self.spin_z.value()), round(self.spin_yaw.value()),
                tuple((round(c.yaw_deg, 1), round(c.pitch_deg, 1))
                      for c in self.couch_poses))

    def _angles_text(self):
        if not self.couch_poses:
            return "lettino fermo"
        return "rotazioni del lettino " + ", ".join(
            f"yaw {c.yaw_deg:.0f}" if abs(c.pitch_deg) < 0.5 else
            f"yaw {c.yaw_deg:.0f} / pitch {c.pitch_deg:.0f}"
            for c in self.couch_poses)

    def _set_enabled(self, on):
        self.btn_solve.setEnabled(on)
        self.btn_export.setEnabled(on)

    # ------------------------------------------------------------------
    def _base(self):
        return kin.RobotBase(
            origin_mm=(self.spin_x.value(), self.spin_y.value(),
                       self.spin_z.value()),
            yaw_deg=self.spin_yaw.value())

    def on_optimize_setup(self):
        """Cerca insieme collocazione della base e angoli di lettino."""
        if not self.obstacles:
            QMessageBox.information(
                self, "Servono gli ostacoli",
                "Calcola prima il modello 3D nel Tab 3: lettino e paziente si "
                "ricavano dalla segmentazione, e senza di essi l'ottimizzazione "
                "non avrebbe alcun vincolo da rispettare.")
            return
        if not self.plan:
            return
        self.btn_setup.setEnabled(False)
        self.progress.setValue(0)
        self.setup_worker = SetupWorker(
            self.obstacles, np.asarray(self.plan["isocenter"], dtype=float),
            self._base(), self.spin_couch.value(), self.couch_arm)
        self.setup_worker.status.connect(self.lbl_status.setText)
        self.setup_worker.solved.connect(self._on_setup)
        self.setup_worker.failed.connect(self._on_failed)
        self.setup_worker.finished.connect(
            lambda: self.btn_setup.setEnabled(True))
        self.setup_worker.start()

    def _on_setup(self, res):
        base, self.couch_poses = res["base"], res["couch_poses"]
        self.spin_x.setValue(round(float(base.origin[0])))
        self.spin_y.setValue(round(float(base.origin[1])))
        self.spin_z.setValue(round(float(base.origin[2])))
        tot, fermo = res["direzioni_provate"], res["copertura_lettino_fermo"]
        self.lbl_couch.setText(
            f"Lettino: {self._angles_text()}. "
            f"Direzioni eseguibili {res['copertura_direzioni']}/{tot} "
            f"contro {fermo}/{tot} a lettino fermo.")
        if self.plan_setup is not None and self._setup_key() != self.plan_setup:
            self.lbl_status.setText(
                "Sala ricalcolata, ma i fasci del piano erano stati scelti per "
                "la sala precedente: per non perderne, ricalcola la dosimetria "
                "nel Tab 3.")
        else:
            self.lbl_status.setText(
                "Sala ottimizzata. Premi «Risolvi cinematica».")

    def on_solve(self):
        if not self.plan:
            return
        self.btn_solve.setEnabled(False)
        self.progress.setValue(0)
        if not self.obstacles:
            # Meglio dirlo che calcolare in silenzio un risultato ottimistico:
            # senza ostacoli il braccio "raggiunge" anche pose che nella sala
            # attraverserebbero il lettino.
            self.lbl_status.setText(
                "Attenzione: nessun ostacolo caricato, le collisioni con il "
                "lettino NON vengono considerate.")
        if self.plan_setup is not None and self._setup_key() != self.plan_setup:
            # Non si blocca: provare un'altra sala e' legittimo. Ma il calo di
            # fasci che ne segue va attribuito alla causa giusta.
            self.lbl_status.setText(
                "Attenzione: base o lettino diversi da quelli per cui la "
                "dosimetria ha scelto i fasci. Alcuni potrebbero non essere "
                "erogabili.")
        self.worker = KinematicsWorker(self.plan["beams"], self._base(),
                                       self.spin_roll.value(), self.obstacles,
                                       self.couch_poses or None)
        self.worker.status.connect(self.lbl_status.setText)
        self.worker.progress.connect(self.progress.setValue)
        self.worker.solved.connect(self._on_solved)
        self.worker.failed.connect(self._on_failed)
        self.worker.finished.connect(lambda: self.btn_solve.setEnabled(True))
        self.worker.start()

    def _on_failed(self, msg):
        self.progress.setValue(0)
        self.lbl_status.setText("Soluzione non riuscita.")
        QMessageBox.critical(self, "Errore nella cinematica", msg)

    def _on_solved(self, res):
        self.result = res
        self._render_metrics(res)
        self._plot(res)
        self.lbl_status.setText("Cinematica risolta.")
        self.resultChanged.emit()

    # ------------------------------------------------------------------
    def _render_metrics(self, res):
        d = res["diag"]
        poses = res["poses"]
        tot, ok = d["fasci_totali"], d["raggiungibili"]
        mg = np.array([p["margine"] for p in poses])
        naive = res["movimento_naive_deg"]
        gain = 100.0 * (1.0 - d["movimento_totale_deg"] / max(naive, 1e-9))

        # Un piano parziale va detto in modo inequivocabile: i fasci non
        # raggiungibili semplicemente non vengono erogati, e la dose calcolata
        # dalla Fase 2 non e' piu' quella che il paziente riceverebbe.
        if ok == tot:
            head = (f"<span style='color:#3fb950'>Tutti i {tot} fasci sono "
                    f"erogabili dal manipolatore.</span>")
        else:
            head = (f"<span style='color:#f85149'><b>Piano parziale: "
                    f"{ok}/{tot} fasci erogabili.</b> I {tot - ok} rimanenti "
                    f"non sono raggiungibili, quindi la dose effettiva sarebbe "
                    f"inferiore a quella pianificata.</span>")

        fr = [p["franco_mm"] for p in poses if p.get("franco_mm") is not None]
        rows = [
            ("Fasci erogabili", f"{ok} / {tot}"),
            ("Margine dai fine corsa",
             f"minimo {mg.min():.3f} &nbsp; mediano {np.median(mg):.3f}"),
            ("Movimento totale dei giunti",
             f"{d['movimento_totale_deg']:.0f}&deg; "
             f"(senza ordinamento {naive:.0f}&deg;, {gain:.0f}% in meno)"),
            ("Chiusura FK/IK",
             f"posizione {res['pos_err_mm']:.1e} mm, "
             f"angolo {res['ang_err_deg']:.1e}&deg;"),
        ]
        if fr:
            rows.insert(2, ("Franco dagli ostacoli",
                            f"minimo {min(fr):.0f} mm &nbsp; "
                            f"mediano {np.median(fr):.0f} mm"))
        if d.get("angoli_lettino", 1) > 1:
            rows.append(("Angoli di lettino",
                         f"{d['angoli_lettino']} "
                         f"({d['riposizionamenti_lettino']} riposizionamenti "
                         f"del paziente) &nbsp; fasci per angolo: "
                         f"{d.get('fasci_per_gruppo')}"))
        if d.get("trasferimenti_con_ritiro"):
            rows.append(("Trasferimenti con ritiro",
                         f"{d['trasferimenti_con_ritiro']} "
                         f"(strada diretta ostruita dal lettino)"))
        if d.get("bloccati_nel_percorso"):
            rows.append(("Bloccati dal percorso",
                         f"{d['bloccati_nel_percorso']} fasci: raggiungibili, "
                         f"ma non collegabili senza attraversare un ostacolo"))
        body = "".join(
            f"<tr><td style='padding:2px 14px 2px 0;color:#8b949e'>{k}</td>"
            f"<td style='color:#e6edf3'>{v}</td></tr>" for k, v in rows)
        self.lbl_metrics.setText(
            f"{head}<table style='margin-top:6px'>{body}</table>")

    def _plot(self, res):
        """Traiettorie dei giunti e margine, lungo la sequenza di trattamento."""
        poses = res["poses"]
        q = np.rad2deg(np.array([p["q"] for p in poses]))
        mg = np.array([p["margine"] for p in poses])
        x = np.arange(len(poses))

        self.fig.clear()
        ax1 = self.fig.add_subplot(211, facecolor="#12141a")
        colors = ["#58a6ff", "#3fb950", "#d29922", "#f85149", "#bc8cff",
                  "#39c5cf"]
        for j in range(6):
            ax1.plot(x, q[:, j], lw=1.3, color=colors[j],
                     label=kin.JOINT_NAMES[j].upper())
            lo, hi = np.rad2deg(kin.JOINT_LIMITS[j])
            # I fine corsa disegnati accanto alla traiettoria: e' il modo
            # diretto di vedere se il braccio lavora al limite.
            ax1.axhline(lo, color=colors[j], lw=0.5, ls=":", alpha=0.35)
            ax1.axhline(hi, color=colors[j], lw=0.5, ls=":", alpha=0.35)
        ax1.set_ylabel("angolo di giunto (gradi)", color="#c9d1d9", fontsize=9)
        ax1.legend(ncol=6, fontsize=7, loc="upper center",
                   facecolor="#12141a", edgecolor="#30363d", labelcolor="#c9d1d9")

        ax2 = self.fig.add_subplot(212, facecolor="#12141a", sharex=ax1)
        ax2.fill_between(x, 0, mg, color="#58a6ff", alpha=0.35)
        ax2.plot(x, mg, lw=1.2, color="#58a6ff")
        ax2.axhline(kin.MARGIN_FLOOR, color="#f85149", lw=1.0, ls="--",
                    label=f"soglia della barriera ({kin.MARGIN_FLOOR})")
        ax2.set_ylim(0, max(0.5, float(mg.max()) * 1.1))
        ax2.set_xlabel("fascio, nell'ordine di erogazione", color="#c9d1d9",
                       fontsize=9)
        ax2.set_ylabel("margine dai limiti", color="#c9d1d9", fontsize=9)
        ax2.legend(fontsize=7, facecolor="#12141a", edgecolor="#30363d",
                   labelcolor="#c9d1d9")

        for ax in (ax1, ax2):
            ax.tick_params(colors="#8b949e", labelsize=8)
            ax.grid(True, color="#30363d", lw=0.4, alpha=0.5)
            for sp in ax.spines.values():
                sp.set_color("#30363d")
        self.fig.tight_layout()
        self.canvas.draw_idle()

    # ------------------------------------------------------------------
    def _write_plan(self, path):
        """Scrive il piano con la configurazione di sala corrente.

        Salva anche base del robot e angoli di lettino: sono parte del piano,
        perche' i fasci sono stati scelti sapendo da dove il braccio li avrebbe
        eseguiti. Un piano senza quella informazione potrebbe essere simulato
        con una geometria diversa da quella per cui e' stato calcolato.
        """
        src = self.source or {}
        return save_plan(self.plan, path,
                         ct_path=src.get("ct_path"),
                         mask_path=src.get("mask_path"),
                         robot_base=self._base(),
                         couch_poses=self.couch_poses or None,
                         motion=self.motion_cfg)

    def on_export(self):
        if not self.plan:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Esporta il piano per ROS 2", ROS_PLAN, "JSON (*.json)")
        if not path:
            return
        try:
            data = self._write_plan(path)
        except Exception as exc:
            QMessageBox.critical(self, "Esportazione non riuscita", str(exc))
            return
        self.lbl_status.setText(
            f"Esportati {len(data['fasci'])} fasci in {os.path.basename(path)}.")

    def on_launch(self):
        """Esporta il piano corrente e avvia la simulazione ROS 2 in WSL."""
        script = os.path.join(HERE, "ros2_ws", "run.sh")
        if not os.path.exists(script):
            QMessageBox.warning(self, "Script mancante",
                                f"Non trovo {script}.")
            return

        # Esportazione AUTOMATICA prima del lancio. Il nodo legge il file dentro
        # il pacchetto: se lo si lascia a un pulsante separato, prima o poi si
        # simula il piano precedente convinti di guardare quello nuovo.
        if self.plan:
            try:
                os.makedirs(os.path.dirname(ROS_PLAN), exist_ok=True)
                self._save_mask()
                data = self._write_plan(ROS_PLAN)
            except Exception as exc:
                QMessageBox.critical(
                    self, "Esportazione non riuscita",
                    f"Non ho potuto scrivere il piano in\n{ROS_PLAN}\n\n{exc}")
                return
            # La SCENA va rigenerata insieme al piano quando e' di un altro
            # paziente: ostacoli, braccio del lettino e mesh vengono da li'.
            # Prima si riesportavano solo piano e URDF, e il nodo simulava i
            # fasci di un paziente dentro la sala di un altro.
            src = self.source or {}
            if not scene_matches(data):
                if not (src.get("ct_path") and src.get("mask_path")):
                    QMessageBox.critical(
                        self, "Scena di un altro paziente",
                        "La scena del pacchetto ROS non e' di questa TAC e "
                        "manca la maschera per rigenerarla: carica la TAC nel "
                        "Tab 1, oppure da terminale\n"
                        "python export_scene.py --ct <tac> --mask <maschera>")
                    return
                self.btn_launch.setEnabled(False)
                self.lbl_status.setText(
                    "Scena del pacchetto di un altro paziente: la rigenero "
                    "(organi, ostacoli, lettino)...")
                self.scene_worker = SceneWorker(src["ct_path"],
                                                src["mask_path"], ROS_MESHES)
                self.scene_worker.done.connect(
                    lambda err, n=len(data["fasci"]): self._after_scene(err, n))
                self.scene_worker.start()
                return
            self._after_scene("", len(data["fasci"]))
        else:
            self.lbl_status.setText(
                "Nessun piano da esportare: si avvia quello gia' presente nel "
                "pacchetto.")
            self._start_sim()

    def _after_scene(self, err, n_beams):
        """Scena pronta: si rigenera l'URDF e si avvia la simulazione."""
        self.btn_launch.setEnabled(True)
        if err:
            QMessageBox.critical(self, "Scena non rigenerata",
                                 f"Il nodo ROS userebbe la scena di un altro "
                                 f"paziente: simulazione non avviata.\n\n{err}")
            return
        # L'URDF va rigenerato INSIEME al piano e DOPO la scena: il giunto che
        # appende il braccio del lettino all'albero ha un'origine che dipende
        # dalla base del robot (piano) e dalla colonna del lettino (scena).
        # Esportando solo il piano, RViz mostrava il lettino ancorato alla
        # sala del piano precedente.
        try:
            import urdf_gen
            pkg = os.path.dirname(os.path.dirname(ROS_PLAN))
            with open(os.path.join(pkg, "urdf", "cyberknife.urdf"), "w",
                      encoding="utf-8") as fh:
                fh.write(urdf_gen.build_urdf(
                    urdf_gen.couch_offset_from_package(pkg)))
        except Exception as exc:
            QMessageBox.critical(
                self, "URDF non aggiornato",
                f"Il piano e' stato esportato ma l'URDF no: il lettino in "
                f"RViz sarebbe nella posizione sbagliata.\n\n{exc}")
            return
        self.lbl_status.setText(
            f"Piano, scena e URDF esportati ({n_beams} fasci), "
            f"simulazione in avvio...")
        self._start_sim()

    def _start_sim(self):
        """Avvia il processo WSL e il controllo di fine seduta."""
        cmd = ["wsl", "-d", WSL_DISTRO, "--", "bash", "ros2_ws/run.sh"]
        try:
            # Nessun wait: la simulazione ha vita propria e l'interfaccia deve
            # restare reattiva. L'eventuale errore si legge nella finestra WSL.
            # Si tiene pero' il processo, per accorgersi di quando finisce e
            # aprire il resoconto.
            self._sim_started = time.time()
            self.sim_proc = subprocess.Popen(cmd, cwd=HERE)
        except (OSError, subprocess.SubprocessError) as exc:
            QMessageBox.critical(
                self, "Avvio non riuscito",
                f"Non sono riuscito ad avviare WSL ({WSL_DISTRO}).\n\n{exc}\n\n"
                f"Da terminale WSL: bash ros2_ws/run.sh")
            return
        # Una seduta alla volta: due simulazioni scriverebbero lo stesso
        # resoconto e aprirebbero RViz due volte.
        self.btn_launch.setEnabled(False)
        self.sim_timer.start()
        self.lbl_status.setText(
            f"Simulazione avviata in WSL ({WSL_DISTRO}). "
            f"La prima compilazione richiede qualche secondo. A fine seduta "
            f"si apre il resoconto.")

    # ------------------------------------------------------------------
    def _poll_sim(self):
        """Controlla se la simulazione e' finita; se si', apre il resoconto."""
        if self.sim_proc is None or self.sim_proc.poll() is None:
            return
        self.sim_timer.stop()
        rc = self.sim_proc.returncode
        self.sim_proc = None
        self.btn_launch.setEnabled(True)
        # Il resoconto vale solo se e' stato scritto DOPO l'avvio: un file di
        # una seduta precedente non va spacciato per quello di questa.
        fresco = (os.path.exists(ROS_REPORT) and
                  os.path.getmtime(ROS_REPORT) >= self._sim_started - 2.0)
        self.btn_report.setEnabled(os.path.exists(ROS_REPORT))
        if fresco:
            self.lbl_status.setText("Seduta terminata: resoconto pronto.")
            self._report_override = None
            self.open_report()
            self.sessionFinished.emit()
        else:
            self.lbl_status.setText(
                f"Simulazione terminata (codice {rc}) senza resoconto: "
                f"probabilmente e' stata chiusa prima dell'ultimo fascio.")

    def _report_paths(self):
        """Resoconto e piano da mostrare: quelli del pacchetto ROS, oppure
        quelli riaperti da un progetto salvato."""
        return getattr(self, "_report_override", None) or (ROS_REPORT,
                                                           ROS_PLAN)

    def open_report(self):
        """Finestra di fine seduta: replay del tumore e andamento."""
        report_path, plan_path = self._report_paths()
        if not os.path.exists(report_path):
            QMessageBox.information(
                self, "Nessun resoconto",
                "Non c'e' ancora un resoconto: lancia una simulazione e "
                "lasciala arrivare all'ultimo fascio.")
            return
        try:
            # Import tardivo: la finestra porta con se' VTK, che non serve
            # finche' non la si apre.
            import report_view
            dlg = report_view.open_report(report_path, plan_path, ROS_MESHES,
                                          parent=self)
        except Exception as exc:                      # noqa: BLE001
            import traceback
            QMessageBox.critical(
                self, "Resoconto non leggibile",
                f"{exc}\n\n{traceback.format_exc()}")
            return
        # Non modale: si puo' continuare a lavorare con il replay aperto.
        self._report_dlg = dlg
        dlg.show()
        dlg.raise_()

    # ------------------------------------------------------------------
    #  Progetto
    # ------------------------------------------------------------------
    def export_state(self):
        """Sala, cinematica risolta e resoconto dell'ultima seduta ROS."""
        import json
        report = plan = None
        rp, pp = self._report_paths()
        if os.path.exists(rp) and os.path.exists(pp):
            with open(rp, encoding="utf-8") as fh:
                report = fh.read()
            with open(pp, encoding="utf-8") as fh:
                plan = fh.read()
            # Il resoconto va salvato solo se e' di QUESTO piano: quello nel
            # pacchetto potrebbe essere di una seduta di un altro paziente.
            try:
                gen = json.loads(report).get("piano_generato")
                if self.plan is None or gen != json.loads(plan).get(
                        "generato"):
                    report = plan = None
            except ValueError:
                report = plan = None
        return {
            "sala": {"x": self.spin_x.value(), "y": self.spin_y.value(),
                     "z": self.spin_z.value(), "yaw": self.spin_yaw.value(),
                     "rotazioni": self.spin_roll.value(),
                     "angoli_lettino": self.spin_couch.value()},
            "lettino": list(self.couch_poses or []),
            "cinematica": self.result,
            "resoconto_json": report if self.result is not None else None,
            "piano_json": plan if self.result is not None else None,
        }

    def restore_state(self, st, work_dir=None):
        """Dopo set_plan: riporta sala, cinematica e resoconto."""
        if not st:
            return
        sala = st.get("sala") or {}
        for key, w in (("x", self.spin_x), ("y", self.spin_y),
                       ("z", self.spin_z), ("yaw", self.spin_yaw)):
            if key in sala:
                w.setValue(float(sala[key]))
        if "rotazioni" in sala:
            self.spin_roll.setValue(int(sala["rotazioni"]))
            self.spin_couch.setValue(int(sala["angoli_lettino"]))
        if st.get("lettino"):
            self.couch_poses = list(st["lettino"])
        if st.get("cinematica"):
            self._on_solved(st["cinematica"])
        if st.get("resoconto_json") and st.get("piano_json") and work_dir:
            # Il resoconto riaperto si scrive in una cartella del progetto,
            # NON nel pacchetto ROS: sovrascrivere il piano del pacchetto
            # cambierebbe la prossima simulazione senza che nessuno lo chieda.
            os.makedirs(work_dir, exist_ok=True)
            rp = os.path.join(work_dir, "resoconto_trattamento.json")
            pp = os.path.join(work_dir, "piano_trattamento.json")
            with open(rp, "w", encoding="utf-8") as fh:
                fh.write(st["resoconto_json"])
            with open(pp, "w", encoding="utf-8") as fh:
                fh.write(st["piano_json"])
            self._report_override = (rp, pp)
            self.btn_report.setEnabled(True)
