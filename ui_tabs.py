# ============================================================================
#  ui_tabs.py - Definizione delle tre schede della Control Station.
#
#    Tab 1  Acquisizione e Percezione  -> OPERATIVO (viewer interattivo)
#    Tab 2  Dosimetria e Simulazione   -> OPERATIVO (3D, sezione, BEV)
#    Tab 3  Cinematica e Controllo ROS2-> OPERATIVO (vedi kinematics_tab.py)
#
#  Nella finestra l'ordine e' 1 Percezione, 2 Respiro e tecnica
#  (motion_tab.py), 3 Dosimetria, 4 Cinematica e ROS: vedi main.py.
# ============================================================================
import os

import numpy as np
from scipy import ndimage

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QFileDialog,
    QProgressBar, QGroupBox, QMessageBox, QSizePolicy, QFrame, QSlider,
    QComboBox, QCheckBox, QTabWidget,
)

import matplotlib
matplotlib.use("QtAgg")   # backend Qt: integra le figure dentro i widget PyQt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.colors import ListedColormap

import config
from inference_worker import InferenceWorker
from viewer3d import Tumor3DPanel
from section_view import BeamSectionView
from bev_view import BeamEyeView
from dvh_view import DvhView


# ---------------------------------------------------------------------------
#  Preset di finestra HU (window/level) come nei visualizzatori clinici.
#  Cambiare finestra non altera i dati: modifica solo il mapping HU -> grigi.
# ---------------------------------------------------------------------------
HU_PRESETS = {
    "Polmone":    (-1000, 400),
    "Mediastino": (-160, 240),
    "Osso":       (-450, 1050),
    "Cervello":   (0, 80),
}

# Colormap a tinta unica per l'overlay del tumore (giallo-arancio acceso).
TUMOR_CMAP = ListedColormap(["#ff8a00"])
# Contorno luminoso: rende la lesione individuabile anche quando e' piccola
# rispetto al campo di vista (una lesione di 17 mm su una CT da 512 px).
EDGE_CMAP = ListedColormap(["#ffe600"])

# Semi-lato (in voxel) della finestra di zoom centrata sul tumore.
ZOOM_HALF_SIZE = 70

# Soglia sotto la quale la dose non viene disegnata: le regioni a dose
# trascurabile restano trasparenti invece di colorare tutta l'immagine.
DOSE_MIN_PCT = 10.0

# Scala della mappa di incertezza (deviazione standard della probabilita').
# Sotto UNC_MIN la rete e' stabile e il voxel resta trasparente; 0.35 e'
# vicino al massimo teorico di una probabilita' che oscilla fra 0 e 1.
UNC_MIN = 0.03
UNC_MAX = 0.35


# ===========================================================================
#  Vista ortogonale interattiva (canvas + slider + rotellina)
# ===========================================================================
class OrthoView(QWidget):
    """Una vista ortogonale scorrevole con overlay del tumore.

    axis indica l'asse lungo cui si scorre nel volume (X, Y, Z):
        axis=2 -> vista Assiale   (piano XY, scorre lungo k)
        axis=1 -> vista Coronale  (piano XZ, scorre lungo j)
        axis=0 -> vista Sagittale (piano YZ, scorre lungo i)

    Ottimizzazione: gli oggetti AxesImage vengono creati UNA SOLA VOLTA e ad
    ogni scorrimento si aggiornano i soli dati con set_data(). Ricreare la
    figura ad ogni slice renderebbe lo scorrimento a scatti.
    """

    AXIS_LABEL = {0: "i", 1: "j", 2: "k"}

    def __init__(self, title: str, axis: int, parent=None):
        super().__init__(parent)
        self.title = title
        self.axis = axis

        self.ct = None            # volume CT completo (X, Y, Z)
        self.mask = None          # maschera tumore, stessa shape
        self.com = None           # centro di massa in voxel
        self.index = 0
        self._im_ct = None
        self._im_mask = None
        self._im_edge = None
        self._im_dose = None
        self._marker = None
        self._show_overlay = True
        self._show_dose = False
        self._dose_map = None
        self._zoomed = False
        self._unc = None
        self._im_unc = None
        self._show_unc = False

        # ---------- Canvas Matplotlib ----------
        self.fig = Figure(figsize=(3.4, 3.4), facecolor="#0b0e12")
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding,
                                  QSizePolicy.Policy.Expanding)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor("#000000")
        self.ax.axis("off")
        self.fig.subplots_adjust(left=0, right=1, top=0.94, bottom=0)

        # Scorrimento con la rotellina del mouse sopra l'immagine.
        self.canvas.mpl_connect("scroll_event", self._on_scroll)

        # ---------- Slider di scorrimento ----------
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setEnabled(False)
        self.slider.valueChanged.connect(self.set_index)

        self.lbl_index = QLabel("--/--")
        self.lbl_index.setObjectName("sliceLabel")
        self.lbl_index.setMinimumWidth(94)
        self.lbl_index.setAlignment(Qt.AlignmentFlag.AlignCenter)

        bar = QHBoxLayout()
        bar.setContentsMargins(0, 0, 0, 0)
        bar.addWidget(self.slider, stretch=1)
        bar.addWidget(self.lbl_index)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)
        layout.addWidget(self.canvas, stretch=1)
        layout.addLayout(bar)

        self._draw_placeholder()

    # ------------------------------------------------------------------
    def _draw_placeholder(self):
        self.ax.clear()
        self.ax.set_facecolor("#000000")
        self.ax.text(0.5, 0.5, "in attesa di scansione", ha="center", va="center",
                     color="#4a535f", fontsize=8, transform=self.ax.transAxes)
        self.ax.set_title(self.title, color="#c8d0da", fontsize=10, pad=4)
        self.ax.axis("off")
        self.canvas.draw_idle()

    # ------------------------------------------------------------------
    def _slice_at(self, volume, idx):
        """Estrae la slice 2D corretta in base all'asse della vista."""
        if self.axis == 0:
            return volume[idx, :, :]
        if self.axis == 1:
            return volume[:, idx, :]
        return volume[:, :, idx]

    @staticmethod
    def _edge_of(mask_slice):
        """Restituisce il solo bordo della lesione come array mascherato.

        Il bordo si ottiene per dilatazione binaria meno la maschera stessa:
        e' un anello di 1 voxel attorno al tumore, disegnato in tinta piena.
        """
        if mask_slice is None or not mask_slice.any():
            return np.ma.masked_all(
                mask_slice.shape if mask_slice is not None else (1, 1))
        dil = ndimage.binary_dilation(mask_slice.astype(bool), iterations=2)
        edge = dil & ~mask_slice.astype(bool)
        return np.ma.masked_where(~edge, edge.astype(np.uint8))

    def _marker_xy(self):
        """Proiezione del centro di massa sul piano della vista corrente."""
        if self.com is None:
            return None
        ci, cj, ck = self.com
        if self.axis == 0:      # sagittale: piano (j, k)
            return (cj, ck)
        if self.axis == 1:      # coronale: piano (i, k)
            return (ci, ck)
        return (ci, cj)         # assiale: piano (i, j)

    # ------------------------------------------------------------------
    def set_volume(self, ct, mask, com_voxel, window):
        """Carica un nuovo volume e inizializza gli artisti Matplotlib."""
        self.ct = ct
        self.mask = mask
        self.com = [int(round(v)) for v in com_voxel] if com_voxel is not None else None

        n_slices = ct.shape[self.axis]
        start = self.com[self.axis] if self.com is not None else n_slices // 2
        start = int(np.clip(start, 0, n_slices - 1))

        # --- Creazione degli artisti (una sola volta) ---
        self.ax.clear()
        self.ax.set_facecolor("#000000")
        lo, hi = window

        ct_slice = self._slice_at(ct, start)
        # imshow direttamente in unita' HU: cambiare finestra = set_clim(),
        # senza rinormalizzare i dati ad ogni cambio preset.
        self._im_ct = self.ax.imshow(ct_slice.T, cmap="gray", origin="lower",
                                     vmin=lo, vmax=hi, interpolation="bilinear")

        mask_slice = self._slice_at(mask, start) if mask is not None else None
        masked = np.ma.masked_where(mask_slice == 0, mask_slice)
        # Overlay tumore: colormap distinta, alpha 0.5 -> resta visibile
        # l'anatomia sottostante.
        self._im_mask = self.ax.imshow(masked.T, cmap=TUMOR_CMAP, origin="lower",
                                       alpha=0.5, vmin=0, vmax=1,
                                       interpolation="nearest")
        # Contorno pieno (alpha 1) sopra l'overlay semitrasparente: "illumina"
        # il bordo della lesione rendendola subito individuabile.
        edge = self._edge_of(mask_slice)
        self._im_edge = self.ax.imshow(edge.T, cmap=EDGE_CMAP, origin="lower",
                                       alpha=1.0, vmin=0, vmax=1,
                                       interpolation="nearest")
        # Mappa di dose (sopra a tutto): creata vuota, popolata quando la
        # dosimetria del Tab 2 e' disponibile.
        empty = np.ma.masked_all(ct_slice.shape)
        self._im_dose = self.ax.imshow(empty.T, cmap="turbo", origin="lower",
                                       alpha=0.45, vmin=DOSE_MIN_PCT, vmax=100.0,
                                       interpolation="bilinear")
        self._im_dose.set_visible(False)
        # Mappa di incertezza (deviazione standard fra le predizioni
        # perturbate): come la dose, creata vuota e popolata a richiesta.
        self._im_unc = self.ax.imshow(empty.T, cmap="magma", origin="lower",
                                      alpha=0.6, vmin=UNC_MIN, vmax=UNC_MAX,
                                      interpolation="bilinear")
        self._im_unc.set_visible(False)

        mk = self._marker_xy()
        self._marker, = self.ax.plot(
            [mk[0]] if mk else [0], [mk[1]] if mk else [0],
            "+", color="#00e5ff", markersize=15, markeredgewidth=1.6,
            visible=False,
        )

        self.ax.axis("off")
        self.slider.setEnabled(True)
        self.slider.setRange(0, n_slices - 1)
        self.slider.blockSignals(True)
        self.slider.setValue(start)
        self.slider.blockSignals(False)
        self.set_index(start)

    # ------------------------------------------------------------------
    def set_index(self, idx: int):
        """Mostra la slice richiesta aggiornando i soli dati degli artisti."""
        if self.ct is None:
            return
        n = self.ct.shape[self.axis]
        idx = int(np.clip(idx, 0, n - 1))
        self.index = idx

        self._im_ct.set_data(self._slice_at(self.ct, idx).T)

        has_tumor = False
        if self.mask is not None:
            ms = self._slice_at(self.mask, idx)
            has_tumor = bool(ms.any())
            masked = np.ma.masked_where(ms == 0, ms)
            self._im_mask.set_data(masked.T)
            self._im_mask.set_visible(self._show_overlay)
            self._im_edge.set_data(self._edge_of(ms).T)
            self._im_edge.set_visible(self._show_overlay and has_tumor)

        # Mappa di dose: interpolata al volo solo per la slice mostrata.
        if self._im_dose is not None:
            if self._show_dose and self._dose_map is not None:
                d = self._dose_map.slice_for(self.axis, idx)
                self._im_dose.set_data(np.ma.masked_less(d, DOSE_MIN_PCT).T)
                self._im_dose.set_visible(True)
            else:
                self._im_dose.set_visible(False)

        if self._im_unc is not None:
            if self._show_unc and self._unc is not None:
                u = self._slice_at(self._unc, idx)
                self._im_unc.set_data(np.ma.masked_less(u, UNC_MIN).T)
                self._im_unc.set_visible(True)
            else:
                self._im_unc.set_visible(False)

        # Il crosshair appare solo sulle slice che contengono la lesione.
        if self._marker is not None:
            self._marker.set_visible(has_tumor and self.com is not None)

        # Titolo: evidenzia quando la slice interseca il tumore.
        if has_tumor:
            self.ax.set_title(f"{self.title}   ● TUMORE",
                              color="#ff8a00", fontsize=10, pad=4)
        else:
            self.ax.set_title(self.title, color="#c8d0da", fontsize=10, pad=4)

        self.lbl_index.setText(f"{self.AXIS_LABEL[self.axis]} = {idx} / {n - 1}")
        if self.slider.value() != idx:
            self.slider.blockSignals(True)
            self.slider.setValue(idx)
            self.slider.blockSignals(False)

        self.canvas.draw_idle()

    # ------------------------------------------------------------------
    def _on_scroll(self, event):
        """Rotellina del mouse: scorre le slice avanti/indietro."""
        if self.ct is None:
            return
        step = 1 if event.button == "up" else -1
        self.set_index(self.index + step)

    # ------------------------------------------------------------------
    def set_window(self, lo, hi):
        """Applica un preset di finestra HU (solo visualizzazione)."""
        if self._im_ct is not None:
            self._im_ct.set_clim(lo, hi)
            self.canvas.draw_idle()

    def set_overlay_visible(self, visible: bool):
        self._show_overlay = visible
        if self._im_mask is not None:
            self._im_mask.set_visible(visible)
            self._im_edge.set_visible(visible and self._im_edge.get_array().any())
            self.canvas.draw_idle()

    def set_dose_map(self, dose_map):
        """Collega la mappa di dose calcolata nel Tab 3."""
        self._dose_map = dose_map
        self.set_index(self.index)

    def set_dose_visible(self, visible: bool):
        self._show_dose = visible
        self.set_index(self.index)

    def set_uncertainty(self, volume):
        """Mappa di incertezza voxel per voxel (stessa griglia della CT)."""
        self._unc = volume
        self.set_index(self.index)

    def set_uncertainty_visible(self, visible: bool):
        self._show_unc = visible
        self.set_index(self.index)

    def set_zoom(self, zoomed: bool):
        """Zoom sulla regione del tumore (o vista intera se disattivato)."""
        self._zoomed = zoomed
        if self.ct is None or self._im_ct is None:
            return
        if zoomed and self.com is not None:
            mk = self._marker_xy()
            h, w = self._im_ct.get_array().shape      # (righe=y, colonne=x)
            self.ax.set_xlim(max(-0.5, mk[0] - ZOOM_HALF_SIZE),
                             min(w - 0.5, mk[0] + ZOOM_HALF_SIZE))
            self.ax.set_ylim(max(-0.5, mk[1] - ZOOM_HALF_SIZE),
                             min(h - 0.5, mk[1] + ZOOM_HALF_SIZE))
        else:
            self.ax.autoscale(enable=True)
            self.ax.relim()
            self.ax.autoscale_view()
        self.canvas.draw_idle()

    def goto_tumor(self):
        """Porta la vista sulla slice del centro di massa."""
        if self.com is not None:
            self.set_index(self.com[self.axis])


# ===========================================================================
#  TAB 1 - ACQUISIZIONE E PERCEZIONE (operativo)
# ===========================================================================
class PerceptionTab(QWidget):
    """Carica una CT, esegue l'inferenza in background e mostra il target."""

    # Emesso a inferenza completata: consente agli altri tab (es. il
    # visualizzatore 3D) di riutilizzare volume, maschera e coordinate.
    resultAvailable = pyqtSignal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.worker = None
        self.result = None
        self._build_ui()

    # ------------------------------------------------------------------
    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        # ---------- Barra comandi ----------
        cmd_bar = QHBoxLayout()
        self.btn_load = QPushButton("  Carica Scansione CT (.nii.gz)")
        self.btn_load.setObjectName("primaryButton")
        self.btn_load.setMinimumHeight(38)
        self.btn_load.clicked.connect(self.on_load_clicked)
        cmd_bar.addWidget(self.btn_load)

        self.chk_unc_run = QCheckBox("Analisi di incertezza")
        self.chk_unc_run.setChecked(True)
        self.chk_unc_run.setToolTip(
            "Ripete la predizione su 8 versioni perturbate della CT (le stesse\n"
            "trasformazioni viste in addestramento) e misura quanto la\n"
            "risposta cambia. ~10 s in piu' su GPU.\n"
            "Alimenta il semaforo, la mappa di incertezza e il PTV adattivo.")
        cmd_bar.addWidget(self.chk_unc_run)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setMinimumHeight(38)
        cmd_bar.addWidget(self.progress, stretch=1)
        root.addLayout(cmd_bar)

        # ---------- Barra strumenti del viewer ----------
        tools = QHBoxLayout()
        tools.setSpacing(8)

        tools.addWidget(QLabel("Finestra HU:"))
        self.combo_window = QComboBox()
        self.combo_window.addItems(HU_PRESETS.keys())
        self.combo_window.setMinimumWidth(130)
        self.combo_window.currentTextChanged.connect(self.on_window_changed)
        tools.addWidget(self.combo_window)

        self.chk_overlay = QCheckBox("Evidenzia tumore")
        self.chk_overlay.setChecked(True)
        self.chk_overlay.toggled.connect(self.on_overlay_toggled)
        tools.addWidget(self.chk_overlay)

        self.chk_dose = QCheckBox("Mappa isodose")
        self.chk_dose.setChecked(True)
        self.chk_dose.setEnabled(False)     # attiva dopo il calcolo nel Tab 2
        self.chk_dose.toggled.connect(self.on_dose_toggled)
        tools.addWidget(self.chk_dose)

        self.chk_unc = QCheckBox("Mappa incertezza")
        self.chk_unc.setChecked(False)
        self.chk_unc.setEnabled(False)
        self.chk_unc.setToolTip(
            "Deviazione standard della probabilita' fra le predizioni "
            "perturbate:\nchiaro = la rete cambia idea, scuro/assente = "
            "risposta stabile.")
        self.chk_unc.toggled.connect(self.on_unc_toggled)
        tools.addWidget(self.chk_unc)

        self.chk_zoom = QCheckBox("Zoom sul tumore")
        self.chk_zoom.setChecked(False)
        self.chk_zoom.setEnabled(False)
        self.chk_zoom.toggled.connect(self.on_zoom_toggled)
        tools.addWidget(self.chk_zoom)

        self.btn_goto = QPushButton("Vai al tumore")
        self.btn_goto.setEnabled(False)
        self.btn_goto.clicked.connect(self.on_goto_tumor)
        tools.addWidget(self.btn_goto)

        tools.addStretch(1)
        self.lbl_hint = QLabel("Scorri le slice con la rotellina del mouse "
                               "sopra le viste, o con gli slider.")
        self.lbl_hint.setObjectName("hintLabel")
        tools.addWidget(self.lbl_hint)
        root.addLayout(tools)

        # ---------- Corpo: viste + pannello dati ----------
        body = QHBoxLayout()
        body.setSpacing(10)

        views_box = QGroupBox("Viste Ortogonali")
        views_layout = QHBoxLayout(views_box)
        views_layout.setSpacing(6)
        # axis: 2=assiale (scorre k), 1=coronale (scorre j), 0=sagittale (scorre i)
        self.view_axial = OrthoView("Assiale", axis=2)
        self.view_coronal = OrthoView("Coronale", axis=1)
        self.view_sagittal = OrthoView("Sagittale", axis=0)
        self.views = (self.view_axial, self.view_coronal, self.view_sagittal)
        for v in self.views:
            views_layout.addWidget(v)
        body.addWidget(views_box, stretch=3)

        # --- Pannello dati laterale ---
        info_box = QGroupBox("Dati Target")
        info_layout = QVBoxLayout(info_box)
        info_box.setMinimumWidth(320)
        info_box.setMaximumWidth(400)

        self.lbl_info = QLabel()
        self.lbl_info.setObjectName("infoPanel")
        self.lbl_info.setFont(QFont("Consolas", 10))
        self.lbl_info.setAlignment(
            Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft
        )
        self.lbl_info.setWordWrap(True)
        self.lbl_info.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        info_layout.addWidget(self.lbl_info, stretch=1)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setObjectName("separator")
        info_layout.addWidget(sep)

        self.btn_export = QPushButton("Esporta Target (.txt)")
        self.btn_export.setEnabled(False)
        self.btn_export.clicked.connect(self.on_export_clicked)
        info_layout.addWidget(self.btn_export)

        body.addWidget(info_box, stretch=1)
        root.addLayout(body, stretch=1)

        self._render_info_idle()

    # ------------------------------------------------------------------
    #  Pannello informativo
    # ------------------------------------------------------------------
    def _render_info_idle(self):
        self.lbl_info.setText(
            "<div style='color:#8b95a3;'>"
            "<b style='color:#c8d0da;'>STATO</b><br>"
            "In attesa di una scansione CT.<br><br>"
            "Premere <i>Carica Scansione CT</i> per avviare<br>"
            "la pipeline di percezione."
            "</div>"
        )

    def _render_info_running(self, message: str):
        self.lbl_info.setText(
            "<div style='color:#8b95a3;'>"
            "<b style='color:#c8d0da;'>STATO</b><br>"
            f"<span style='color:#ffb020;'>{message}</span><br><br>"
            "Elaborazione in corso, la GUI resta reattiva."
            "</div>"
        )

    def _render_info_result(self, res: dict):
        if not res.get("tumor_found"):
            self.lbl_info.setText(
                "<div style='color:#8b95a3;'>"
                "<b style='color:#c8d0da;'>STATO</b><br>"
                "<span style='color:#ff5c5c;'>Nessun tumore rilevato</span><br><br>"
                "La rete non ha identificato lesioni in questo volume.<br>"
                "Le viste restano navigabili."
                "</div>"
            )
            return

        cv = res["com_voxel"]
        cw = res["com_world"]
        self.lbl_info.setText(
            self._light_html(res) +
            "<div style='color:#8b95a3; line-height:150%;'>"
            "<b style='color:#c8d0da;'>COORDINATE REALI TARGET</b><br>"
            "<span style='color:#6f7a88; font-size:11px;'>spazio fisico paziente (mm)</span><br>"
            f"<span style='color:#00e5ff; font-size:15px;'><b>"
            f"X = {cw[0]:+9.2f} mm<br>"
            f"Y = {cw[1]:+9.2f} mm<br>"
            f"Z = {cw[2]:+9.2f} mm</b></span><br><br>"

            "<b style='color:#c8d0da;'>COORDINATE VOXEL</b><br>"
            f"i = {cv[0]:8.2f}<br>"
            f"j = {cv[1]:8.2f}<br>"
            f"k = {cv[2]:8.2f}<br><br>"

            "<b style='color:#c8d0da;'>MORFOLOGIA LESIONE</b><br>"
            f"Voxel tumore  : {res['n_tumor_voxels']:,}<br>"
            f"Volume        : {res['volume_cm3']:.2f} cm&sup3;<br>"
            # Due conteggi distinti: 'rilevate' sono le componenti grezze sopra
            # soglia (misura del rumore), 'lesioni' quelle sopra la soglia di
            # volume, cioe' quelle su cui il planner lavora davvero. Mostrarne
            # una sola nascondeva quanto materiale veniva scartato.
            f"Componenti    : {res['n_components']} rilevate / "
            f"{res.get('n_lesions_kept', '?')} lesioni<br><br>"

            "<b style='color:#c8d0da;'>STATO</b><br>"
            # Un bersaglio inaffidabile non va mostrato come un successo: nel
            # 27% dei casi l'isocentro cade fuori dal tumore, e il piano che ne
            # discende irradia tessuto sano. Il valore resta visibile (serve a
            # studiare i fallimenti), ma marcato.
            + ("<span style='color:#3ddc84;'><b>Inferenza completata</b></span><br>"
               if res.get("target_reliable", True) else
               "<span style='color:#ffb020;'><b>Bersaglio POCO AFFIDABILE</b></span><br>"
               f"<span style='color:#ffb020; font-size:11px;'>"
               f"{res.get('gate_reason', '')}<br>"
               f"Verificare manualmente prima di pianificare.</span><br>") +
            f"<span style='font-size:11px;'>Device: {res['device'].upper()}<br>"
            f"File: {os.path.basename(res['image_path'])}</span>"
            "</div>"
        )

    @staticmethod
    def _light_html(res):
        """Semaforo del bersaglio, in cima al pannello."""
        light = res.get("traffic_light")
        if not light:
            return ""
        color, reasons = light
        hexc = {"verde": "#3ddc84", "giallo": "#ffb020",
                "rosso": "#ff5c5c"}[color]
        text = {"verde": "BERSAGLIO AFFIDABILE",
                "giallo": "BERSAGLIO DA RIVEDERE",
                "rosso": "BERSAGLIO NON AFFIDABILE"}[color]
        sig = res.get("uncertainty_signals") or {}
        extra = ""
        if "centroide_disp_mm" in sig:
            extra = (f"<span style='color:#8b95a3; font-size:11px;'>"
                     f"dispersione isocentro {sig['centroide_disp_mm']:.1f} mm"
                     f" · volume &plusmn;{100 * sig['volume_cv']:.0f}%"
                     f" · {sig['n_samples']} predizioni</span><br>")
        why = "".join(f"<span style='color:{hexc}; font-size:11px;'>"
                      f"- {r}</span><br>" for r in reasons)
        return ("<div style='margin-bottom:8px;'>"
                f"<span style='color:{hexc}; font-size:22px;'>&#9679;</span> "
                f"<b style='color:{hexc};'>{text}</b><br>{why}{extra}</div>")

    # ------------------------------------------------------------------
    #  Slot: caricamento file e avvio worker
    # ------------------------------------------------------------------
    def on_load_clicked(self):
        start_dir = config.DATA_ROOT if os.path.isdir(config.DATA_ROOT) else ""
        path, _ = QFileDialog.getOpenFileName(
            self, "Seleziona una scansione CT", start_dir,
            "Immagini NIfTI (*.nii.gz *.nii);;Tutti i file (*)",
        )
        if not path:
            return

        self.btn_load.setEnabled(False)
        self.btn_export.setEnabled(False)
        self.btn_goto.setEnabled(False)
        self.progress.setValue(0)

        self.worker = InferenceWorker(
            path, with_uncertainty=self.chk_unc_run.isChecked())
        self.worker.progress.connect(self.progress.setValue)
        self.worker.status.connect(self.on_status)
        self.worker.resultReady.connect(self.on_result)
        self.worker.failed.connect(self.on_failed)
        self.worker.finished.connect(lambda: self.btn_load.setEnabled(True))
        self.worker.start()

    # ------------------------------------------------------------------
    #  Slot collegati ai segnali del worker
    # ------------------------------------------------------------------
    def on_status(self, message: str):
        self._render_info_running(message)
        win = self.window()
        if hasattr(win, "statusBar"):
            win.statusBar().showMessage(message)

    def on_result(self, res: dict):
        self.result = res
        self._render_info_result(res)

        ct = res["ct_data"]
        mask = res["mask"]
        com = res.get("com_voxel") if res.get("tumor_found") else None
        window = HU_PRESETS[self.combo_window.currentText()]

        # Carica il volume nelle tre viste: da qui sono navigabili.
        for v in self.views:
            v.set_volume(ct, mask, com, window)
            v.set_overlay_visible(self.chk_overlay.isChecked())
            # Una dose calcolata su un volume precedente non ha alcun senso
            # sulla nuova anatomia: va scartata insieme al volume vecchio.
            v.set_dose_map(None)
            unc = res.get("uncertainty")
            v.set_uncertainty(unc["p_std"] if unc is not None else None)
            v.set_uncertainty_visible(self.chk_unc.isChecked())
        self.chk_dose.setEnabled(False)
        self.chk_unc.setEnabled(res.get("uncertainty") is not None)

        found = res.get("tumor_found", False)
        self.btn_goto.setEnabled(found)
        self.btn_export.setEnabled(found)
        self.chk_zoom.setEnabled(found)
        if self.chk_zoom.isChecked():
            self.on_zoom_toggled(True)

        # Rende i dati disponibili al Tab 2 (ricostruzione 3D).
        self.resultAvailable.emit(res)

    def on_failed(self, message: str):
        self.progress.setValue(0)
        self._render_info_idle()
        QMessageBox.critical(self, "Errore durante l'inferenza", message)

    # ------------------------------------------------------------------
    #  Progetto
    # ------------------------------------------------------------------
    def export_state(self):
        """Risultato dell'inferenza, SENZA il volume CT (si ricarica)."""
        if not self.result:
            return None
        import project_io
        st = {k: v for k, v in self.result.items() if k != "ct_data"}
        if st.get("uncertainty") is not None:
            import uncertainty as U
            st["uncertainty"] = {"compatte": U.compact_maps(
                st["uncertainty"])}
        st["tac"] = project_io.ct_fingerprint(self.result["image_path"])
        st["ui"] = {"finestra": self.combo_window.currentText(),
                    "incertezza": self.chk_unc_run.isChecked()}
        return st

    def restore_state(self, st):
        """Ricarica la TAC dal suo percorso e ripristina il risultato.

        Restituisce un avviso (stringa vuota se tutto torna). Se la TAC non
        e' piu' dov'era, o non ha la stessa geometria, si solleva: le
        maschere salvate non avrebbero alcun senso su un altro volume.
        """
        import nibabel as nib
        st = dict(st)
        ui = st.pop("ui", {}) or {}
        tac = st.pop("tac", {}) or {}
        path = st["image_path"]
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"La TAC del progetto non si trova piu' in\n{path}\n"
                f"Il progetto salva il percorso, non il volume.")
        img = nib.load(path)
        ct = np.asanyarray(img.dataobj, dtype=np.float32)
        aff = img.affine.astype(np.float64)
        if ct.shape != np.asarray(st["mask"]).shape or \
                not np.allclose(aff, st["affine"], atol=1e-4):
            raise ValueError("La TAC nel percorso salvato ha una geometria "
                             "diversa da quella del progetto.")
        warn = ""
        try:
            now = os.stat(path)
            if tac.get("byte") not in (None, now.st_size) or \
                    abs(float(tac.get("modificato", now.st_mtime))
                        - now.st_mtime) > 1.0:
                warn = ("La TAC risulta modificata dopo il salvataggio del "
                        "progetto: i risultati potrebbero non corrispondere.")
        except OSError:
            pass
        st["ct_data"] = ct
        unc = st.get("uncertainty")
        if isinstance(unc, dict) and "compatte" in unc:
            import uncertainty as U
            st["uncertainty"] = U.expand_maps(unc["compatte"])
        if ui.get("finestra") in HU_PRESETS:
            self.combo_window.setCurrentText(ui["finestra"])
        self.chk_unc_run.setChecked(bool(ui.get("incertezza", True)))
        self.progress.setValue(100)
        self.on_result(st)
        return warn

    # ------------------------------------------------------------------
    #  Slot degli strumenti del viewer
    # ------------------------------------------------------------------
    def on_window_changed(self, name: str):
        lo, hi = HU_PRESETS[name]
        for v in self.views:
            v.set_window(lo, hi)

    def on_overlay_toggled(self, checked: bool):
        for v in self.views:
            v.set_overlay_visible(checked)

    def on_zoom_toggled(self, checked: bool):
        for v in self.views:
            v.set_zoom(checked)

    def set_dose_map(self, dose_map):
        """Slot collegato al Tab 3: riceve la mappa di dose e la mostra."""
        for v in self.views:
            v.set_dose_map(dose_map)
            v.set_dose_visible(self.chk_dose.isChecked())
        self.chk_dose.setEnabled(True)

    def on_dose_toggled(self, checked: bool):
        for v in self.views:
            v.set_dose_visible(checked)

    def on_unc_toggled(self, checked: bool):
        for v in self.views:
            v.set_uncertainty_visible(checked)

    def on_goto_tumor(self):
        for v in self.views:
            v.goto_tumor()

    # ------------------------------------------------------------------
    def on_export_clicked(self):
        """Salva le coordinate del target: input per la fase di path-planning."""
        if not self.result or not self.result.get("tumor_found"):
            return
        default = os.path.join(
            config.OUTPUT_DIR,
            os.path.basename(self.result["image_path"]).split(".")[0] + "_target.txt",
        )
        path, _ = QFileDialog.getSaveFileName(
            self, "Esporta coordinate target", default, "File di testo (*.txt)")
        if not path:
            return

        cw, cv = self.result["com_world"], self.result["com_voxel"]
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("# CyberKnife Control Center - Target Fase 1\n")
            fh.write(f"# Sorgente: {self.result['image_path']}\n")
            fh.write("# Coordinate world in mm (spazio fisico paziente, RAS)\n")
            fh.write(f"world_x_mm={cw[0]:.4f}\n")
            fh.write(f"world_y_mm={cw[1]:.4f}\n")
            fh.write(f"world_z_mm={cw[2]:.4f}\n")
            fh.write(f"voxel_i={cv[0]:.4f}\n")
            fh.write(f"voxel_j={cv[1]:.4f}\n")
            fh.write(f"voxel_k={cv[2]:.4f}\n")
            fh.write(f"volume_cm3={self.result['volume_cm3']:.4f}\n")
            fh.write(f"tumor_voxels={self.result['n_tumor_voxels']}\n")
        QMessageBox.information(self, "Esportazione completata",
                                f"Target salvato in:\n{path}")


# ===========================================================================
#  Scheletro comune per le schede non ancora implementate
# ===========================================================================
class DosimetryTab(QWidget):
    """TAB 2 - Dosimetria: modello 3D e analisi dei fasci in sezione.

    Tre viste, dalla scala piu' ampia alla piu' stretta:
      - "Modello 3D" : l'insieme (organi, tumore, fasci, superficie isodose);
      - "Sezione"    : un piano a piena risoluzione, per vedere dove passano i
                       fasci e che cosa attraversano;
      - "Vista dal fascio" : un singolo fascio alla volta, guardato dalla sua
                       sorgente, per verificarne inquadratura e ostacoli.
    Il 3D da' il colpo d'occhio ma con decine di fasci sovrapposti diventa
    illeggibile; le altre due servono al giudizio di dettaglio.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.source = None

        self.views = QTabWidget()
        self.views.setObjectName("subTabs")
        self.panel = Tumor3DPanel()
        self.section = BeamSectionView()
        self.bev = BeamEyeView()
        self.dvh = DvhView()
        self.views.addTab(self.panel, "  Modello 3D  ")
        self.views.addTab(self.section, "  Sezione (dove passano)  ")
        self.views.addTab(self.bev, "  Vista dal fascio (BEV)  ")
        self.views.addTab(self.dvh, "  Istogramma Dose-Volume (DVH)  ")
        layout.addWidget(self.views)

        # I risultati prodotti nella vista 3D alimentano le altre due.
        self.panel.organsAvailable.connect(self._on_organs)
        self.panel.planAvailable.connect(self._on_plan)

    def export_state(self):
        return self.panel.export_state()

    def restore_state(self, st):
        self.panel.restore_state(st)

    def set_source_data(self, res: dict):
        """Inoltra i dati prodotti dall'inferenza del Tab 1 alle viste."""
        self.source = res
        self.panel.set_source_data(res)
        self.section.set_patient(res["ct_data"], res["affine"], res["mask"],
                                 res.get("com_voxel"))

    def _on_organs(self, res: dict):
        self.section.set_organs(res.get("organ_masks"), res.get("organ_affine"))

    def _on_plan(self, res: dict):
        self.section.set_plan(res.get("dose_map"), res.get("beams"))
        # La BEV riusa la griglia di densita' gia' costruita per la dosimetria:
        # e' esattamente la grandezza che serve a simulare la radiografia.
        organ_res = self.panel.result or {}
        self.bev.set_data(
            res.get("red_grid"), res.get("grid_affine"), res.get("beams"),
            self.source["mask"], self.source["affine"],
            bone_mask=organ_res.get("organ_masks", {}).get("bones"),
            bone_affine=organ_res.get("organ_affine"),
        )
        self.dvh.set_data(res, self.source, organ_res)


# Il Tab 3 vive in kinematics_tab.py: la cinematica e il ponte verso ROS non
# condividono nulla con le viste di imaging e tenerli qui allungherebbe questo
# file senza motivo. Si ri-esporta il nome perche' main.py lo importa da qui.
from kinematics_tab import KinematicsTab   # noqa: E402,F401
