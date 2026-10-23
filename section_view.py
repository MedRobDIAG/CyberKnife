# ============================================================================
#  section_view.py - Vista in sezione per l'analisi dei fasci.
#
#  Il rendering 3D mostra l'insieme, ma con decine di fasci sovrapposti e
#  organi semitrasparenti diventa illeggibile. Questa vista fa l'opposto: UN
#  SOLO piano, a piena risoluzione CT, con sopra tutto cio' che serve a
#  giudicare un fascio:
#
#     - anatomia CT nella finestra HU scelta;
#     - CONTORNI degli organi (non superfici piene): non nascondono l'immagine;
#     - fasci proiettati con la LARGHEZZA REALE del cono, cosi' si vede se il
#       bordo del fascio intercetta una costa;
#     - colore del fascio secondo l'osso attraversato;
#     - curve di isodose.
#
#  Il criterio di selezione dei fasci e' il punto chiave. Tutti i fasci
#  convergono sul bersaglio, quindi "passa vicino al piano" e' vero per tutti e
#  non seleziona nulla. Si misura invece QUANTA PARTE del percorso di ciascun
#  fascio giace nella fetta di spessore scelto: i fasci che stanno nel piano
#  hanno frazione alta, quelli che lo bucano perpendicolarmente frazione bassa.
# ============================================================================
import numpy as np
from scipy import ndimage

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QCheckBox, QSlider,
    QGroupBox, QSizePolicy, QPushButton,
)

import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Polygon
import matplotlib.lines as mlines

import config
import organs as og
from dosimetry import apply_affine_fast, DoseMap, SAD

# Piani: nome -> (asse CT tagliato, indice in-piano X, indice in-piano Y)
PLANES = {
    "Assiale":   (2, 0, 1),
    "Coronale":  (1, 0, 2),
    "Sagittale": (0, 1, 2),
}

HU_PRESETS = {
    "Polmone":    (-1000, 400),
    "Mediastino": (-160, 240),
    "Osso":       (-450, 1050),
}

# Livelli di isodose disegnati (% della prescrizione) e relativi colori.
ISODOSE_LEVELS = [30.0, 50.0, 80.0, 100.0]
ISODOSE_COLORS = ["#4a90d9", "#3ddc84", "#ffb020", "#ff3b30"]

# Oltre questo tratto di osso il fascio viene segnalato in rosso.
BONE_WARN_CM = 0.5

# Sottocampionamento usato per estrarre i contorni degli organi: a meta'
# risoluzione i contorni sono identici a vedersi, ma si calcolano 4 volte piu'
# in fretta, il che tiene fluido lo scorrimento delle slice.
CONTOUR_STEP = 2


class BeamSectionView(QWidget):
    """Sezione singola con anatomia, organi, fasci e isodosi."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.ct = None
        self.ct_affine = None
        self.tumor = None
        self.organ_maps = {}      # chiave -> DoseMap (riusata come ricampionatore)
        self.body_mask = None     # maschera corpo, per trovare l'ingresso cutaneo
        self.body_affine = None
        self.dose_map = None
        self.beams = []
        self.com_voxel = None
        self._entries = {}        # id(beam) -> punto di ingresso sulla pelle
        self.index = 0
        self._build_ui()

    # ------------------------------------------------------------------
    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        # ---------- Barra: piano, slice, finestra ----------
        bar = QHBoxLayout()
        bar.setSpacing(8)
        bar.addWidget(QLabel("Piano:"))
        self.combo_plane = QComboBox()
        self.combo_plane.addItems(PLANES.keys())
        self.combo_plane.currentTextChanged.connect(self._on_plane_changed)
        bar.addWidget(self.combo_plane)

        bar.addWidget(QLabel("Finestra:"))
        self.combo_window = QComboBox()
        self.combo_window.addItems(HU_PRESETS.keys())
        self.combo_window.currentTextChanged.connect(lambda _: self.redraw())
        bar.addWidget(self.combo_window)

        self.btn_goto = QPushButton("Vai al tumore")
        self.btn_goto.clicked.connect(self.goto_tumor)
        self.btn_goto.setEnabled(False)
        bar.addWidget(self.btn_goto)

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setEnabled(False)
        self.slider.valueChanged.connect(self.set_index)
        bar.addWidget(self.slider, stretch=1)

        self.lbl_index = QLabel("--/--")
        self.lbl_index.setObjectName("sliceLabel")
        self.lbl_index.setMinimumWidth(96)
        self.lbl_index.setAlignment(Qt.AlignmentFlag.AlignCenter)
        bar.addWidget(self.lbl_index)
        root.addLayout(bar)

        # ---------- Barra: livelli visualizzati ----------
        tools = QHBoxLayout()
        tools.setSpacing(10)
        self.chk_organs = QCheckBox("Contorni organi")
        self.chk_beams = QCheckBox("Fasci")
        self.chk_cone = QCheckBox("Larghezza cono")
        self.chk_dose = QCheckBox("Isodosi")
        self.chk_entry = QCheckBox("Punti d'ingresso")
        for c in (self.chk_organs, self.chk_beams, self.chk_cone,
                  self.chk_dose, self.chk_entry):
            c.setChecked(True)
            c.toggled.connect(lambda _: self.redraw())
            tools.addWidget(c)

        tools.addWidget(QLabel("Spessore fetta:"))
        self.slider_slab = QSlider(Qt.Orientation.Horizontal)
        self.slider_slab.setRange(5, 80)
        # 12 mm e' abbastanza stretto da isolare i fasci quasi complanari:
        # con fette larghe rientrano quasi tutti e la selezione perde utilita'.
        self.slider_slab.setValue(12)
        self.slider_slab.setMaximumWidth(130)
        self.slider_slab.valueChanged.connect(self._on_slab_changed)
        tools.addWidget(self.slider_slab)
        self.lbl_slab = QLabel("±12 mm")
        self.lbl_slab.setObjectName("sliceLabel")
        self.lbl_slab.setMinimumWidth(66)
        tools.addWidget(self.lbl_slab)
        tools.addStretch(1)
        root.addLayout(tools)

        # ---------- Canvas + pannello ----------
        body = QHBoxLayout()
        body.setSpacing(8)

        box = QGroupBox("Sezione")
        vb = QVBoxLayout(box)
        self.fig = Figure(figsize=(6.5, 6.5), facecolor="#0b0e12")
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding,
                                  QSizePolicy.Policy.Expanding)
        self.ax = self.fig.add_subplot(111)
        self.fig.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.02)
        self.canvas.mpl_connect("scroll_event", self._on_scroll)
        vb.addWidget(self.canvas)
        body.addWidget(box, stretch=3)

        info = QGroupBox("Fasci in questa sezione")
        ib = QVBoxLayout(info)
        info.setMinimumWidth(270)
        info.setMaximumWidth(330)
        self.lbl_info = QLabel()
        self.lbl_info.setObjectName("infoPanel")
        self.lbl_info.setAlignment(Qt.AlignmentFlag.AlignTop |
                                   Qt.AlignmentFlag.AlignLeft)
        self.lbl_info.setWordWrap(True)
        ib.addWidget(self.lbl_info, stretch=1)
        body.addWidget(info, stretch=1)
        root.addLayout(body, stretch=1)

        self._placeholder()

    # ------------------------------------------------------------------
    def _placeholder(self, msg="Eseguire l'inferenza nel Tab 1."):
        self.ax.clear()
        self.ax.set_facecolor("#000000")
        self.ax.text(0.5, 0.5, msg, ha="center", va="center", color="#5c6672",
                     fontsize=10, transform=self.ax.transAxes)
        self.ax.axis("off")
        self.canvas.draw_idle()
        self.lbl_info.setText("<div style='color:#8b95a3;'>Nessun piano "
                              "calcolato.</div>")

    # ==================================================================
    #  Ingresso dei dati
    # ==================================================================
    def set_patient(self, ct, ct_affine, tumor_mask, com_voxel=None):
        self.ct = ct
        self.ct_affine = np.asarray(ct_affine, dtype=np.float64)
        self.tumor = tumor_mask
        self.com_voxel = (None if com_voxel is None
                          else [int(round(v)) for v in com_voxel])
        # Un nuovo paziente invalida piano e organi precedenti.
        self.organ_maps = {}
        self.dose_map = None
        self.beams = []
        self._entries = {}
        self.btn_goto.setEnabled(self.com_voxel is not None)
        self._reset_slider()
        self.redraw()

    def set_organs(self, masks, organ_affine):
        """Registra le maschere: DoseMap viene riusata come ricampionatore."""
        self.organ_maps = {}
        self._entries = {}
        if not masks or self.ct is None:
            return
        # Il corpo non si disegna come contorno (coincide col profilo gia'
        # visibile), ma serve a individuare l'ingresso cutaneo dei fasci.
        self.body_mask = masks.get("body")
        self.body_affine = np.asarray(organ_affine, dtype=np.float64)
        for key, m in masks.items():
            if key == "body" or not m.any():
                continue
            self.organ_maps[key] = DoseMap(m.astype(np.float32), organ_affine,
                                           self.ct.shape, self.ct_affine)
        self.redraw()

    def set_plan(self, dose_map, beams):
        self.dose_map = dose_map
        self.beams = list(beams or [])
        self._entries = {}
        self.redraw()

    # ==================================================================
    #  Navigazione
    # ==================================================================
    def _axes(self):
        return PLANES[self.combo_plane.currentText()]

    def _reset_slider(self):
        if self.ct is None:
            return
        ax, _, _ = self._axes()
        n = self.ct.shape[ax]
        start = (self.com_voxel[ax] if self.com_voxel is not None else n // 2)
        self.slider.setEnabled(True)
        self.slider.blockSignals(True)
        self.slider.setRange(0, n - 1)
        self.slider.setValue(int(np.clip(start, 0, n - 1)))
        self.slider.blockSignals(False)
        self.index = int(np.clip(start, 0, n - 1))

    def _on_plane_changed(self, _):
        self._reset_slider()
        self.redraw()

    def _on_slab_changed(self, v):
        self.lbl_slab.setText(f"±{v} mm")
        self.redraw()

    def set_index(self, idx):
        if self.ct is None:
            return
        ax, _, _ = self._axes()
        self.index = int(np.clip(idx, 0, self.ct.shape[ax] - 1))
        if self.slider.value() != self.index:
            self.slider.blockSignals(True)
            self.slider.setValue(self.index)
            self.slider.blockSignals(False)
        self.redraw()

    def _on_scroll(self, event):
        if self.ct is not None:
            self.set_index(self.index + (1 if event.button == "up" else -1))

    def goto_tumor(self):
        if self.com_voxel is not None:
            ax, _, _ = self._axes()
            self.set_index(self.com_voxel[ax])

    # ==================================================================
    #  Geometria del piano
    # ==================================================================
    def _plane_distance(self, points_world):
        """Distanza con segno (mm) di punti mondo dal piano della slice.

        Il piano e' l'insieme dei punti con indice CT costante lungo l'asse
        tagliato. Ricavarlo dall'affine inversa lo rende corretto per qualsiasi
        orientamento del volume, non solo per affini diagonali.
        """
        ax, _, _ = self._axes()
        inv = np.linalg.inv(self.ct_affine)
        n = inv[ax, :3]
        c = inv[ax, 3]
        norm = np.linalg.norm(n)
        return ((n @ np.asarray(points_world).T) + c - self.index) / max(norm, 1e-9)

    def _to_display(self, points_world):
        """Da coordinate mondo ai due indici in-piano usati dall'immagine."""
        _, ix, iy = self._axes()
        idx = apply_affine_fast(np.linalg.inv(self.ct_affine),
                                np.asarray(points_world).T)
        return idx[ix], idx[iy]

    def _slice_of(self, volume):
        ax, _, _ = self._axes()
        if ax == 0:
            return volume[self.index, :, :]
        if ax == 1:
            return volume[:, self.index, :]
        return volume[:, :, self.index]

    def _inplane_mm(self):
        """Millimetri per unita' di indice lungo i due assi in-piano."""
        _, ix, iy = self._axes()
        lin = self.ct_affine[:3, :3]
        return np.linalg.norm(lin[:, ix]), np.linalg.norm(lin[:, iy])

    # ==================================================================
    #  Selezione dei fasci pertinenti alla sezione
    # ==================================================================
    def _beam_entry(self, beam):
        """Punto in cui il fascio entra nel corpo (cache per fascio)."""
        key = id(beam)
        if key in self._entries:
            return self._entries[key]
        d = beam.target - beam.source
        dist = float(np.linalg.norm(d))
        t = np.linspace(0.0, dist, 400)
        pts = beam.source[None, :] + (d / dist)[None, :] * t[:, None]

        if self.body_mask is not None:
            # Si usa la MASCHERA DEL CORPO, non una soglia HU: il lettino di
            # trattamento supera la soglia dei tessuti e verrebbe scambiato per
            # la pelle, collocando l'ingresso fuori dal paziente.
            idx = apply_affine_fast(np.linalg.inv(self.body_affine), pts.T)
            val = ndimage.map_coordinates(self.body_mask.astype(np.float32),
                                          idx, order=0, mode="constant", cval=0.0)
            inside = np.flatnonzero(val > 0.5)
        else:
            idx = apply_affine_fast(np.linalg.inv(self.ct_affine), pts.T)
            hu = ndimage.map_coordinates(self.ct, idx, order=0, mode="constant",
                                         cval=-1000.0)
            inside = np.flatnonzero(hu > -320)

        entry = pts[inside[0]] if len(inside) else beam.source
        self._entries[key] = entry
        return entry

    def _beams_in_slab(self, tol_mm):
        """Fasci il cui percorso giace in buona parte nella fetta corrente.

        Restituisce [(beam, frazione_nella_fetta, ingresso), ...]. Si campiona
        il tratto ingresso->bersaglio: e' quello che attraversa il paziente,
        l'unico che conta per giudicare cosa il fascio incontra.
        """
        out = []
        for b in self.beams:
            entry = self._beam_entry(b)
            seg = np.linspace(0.0, 1.0, 60)[:, None]
            pts = entry[None, :] + (b.target - entry)[None, :] * seg
            frac = float(np.mean(np.abs(self._plane_distance(pts)) <= tol_mm))
            # Soglia sulla frazione, non sulla sola vicinanza: tutti i fasci
            # convergono sul bersaglio e quindi passano tutti vicino al piano
            # quando il piano taglia il tumore. Discrimina solo QUANTO del
            # tragitto giace nella fetta.
            if frac > 0.25:
                out.append((b, frac, entry))
        return out

    # ==================================================================
    #  Disegno
    # ==================================================================
    def redraw(self):
        if self.ct is None:
            self._placeholder()
            return

        self.ax.clear()
        self.ax.set_facecolor("#000000")

        # --- 1) Anatomia CT ---
        lo, hi = HU_PRESETS[self.combo_window.currentText()]
        ct_s = self._slice_of(self.ct)
        self.ax.imshow(ct_s.T, cmap="gray", origin="lower", vmin=lo, vmax=hi,
                       interpolation="bilinear")
        # Proporzioni fisiche corrette: sui piani coronale e sagittale i due
        # assi hanno spaziature diverse e senza correzione l'anatomia apparirebbe
        # schiacciata.
        mx, my = self._inplane_mm()
        self.ax.set_aspect(my / mx)

        # --- 2) Contorni degli organi ---
        if self.chk_organs.isChecked():
            ax_i, _, _ = self._axes()
            for key, name, color, *_ in og.ORGAN_STYLE:
                dm = self.organ_maps.get(key)
                if dm is None:
                    continue
                sl = dm.slice_for(ax_i, self.index, step=CONTOUR_STEP)
                if sl.max() < 0.5:
                    continue
                self.ax.contour(
                    np.arange(sl.shape[0]) * CONTOUR_STEP,
                    np.arange(sl.shape[1]) * CONTOUR_STEP,
                    sl.T, levels=[0.5], colors=[color], linewidths=1.0,
                    alpha=0.85)

        # --- 3) Tumore (contorno spesso: e' il bersaglio) ---
        if self.tumor is not None:
            tm = self._slice_of(self.tumor)
            if tm.any():
                self.ax.contour(tm.T.astype(float), levels=[0.5],
                                colors=["#ff8a00"], linewidths=2.4)

        # --- 4) Isodosi ---
        if self.chk_dose.isChecked() and self.dose_map is not None:
            ax_i, _, _ = self._axes()
            ds = self.dose_map.slice_for(ax_i, self.index, step=CONTOUR_STEP)
            if ds.max() >= ISODOSE_LEVELS[0]:
                gx = np.arange(ds.shape[0]) * CONTOUR_STEP
                gy = np.arange(ds.shape[1]) * CONTOUR_STEP
                cs = self.ax.contour(gx, gy, ds.T, levels=ISODOSE_LEVELS,
                                     colors=ISODOSE_COLORS, linewidths=1.2,
                                     alpha=0.9)
                # Si etichetta solo la prescrizione: su un bersaglio di pochi
                # millimetri quattro etichette si sovrappongono e coprono
                # proprio la zona che interessa leggere.
                presc = [l for l in cs.levels if abs(l - 80.0) < 1e-6]
                if presc:
                    self.ax.clabel(cs, levels=presc, fmt="%d%%", fontsize=7,
                                   colors="white")

        # --- 5) Fasci ---
        n_shown = n_bone = 0
        if self.chk_beams.isChecked() and self.beams:
            n_shown, n_bone = self._draw_beams()

        title = f"{self.combo_plane.currentText()} - slice {self.index}"
        self.ax.set_title(title, color="#c8d0da", fontsize=10, pad=6)
        self.ax.axis("off")
        ax_i, _, _ = self._axes()
        self.lbl_index.setText(f"{self.index} / {self.ct.shape[ax_i] - 1}")
        self.canvas.draw_idle()
        self._update_info(n_shown, n_bone)

    # ------------------------------------------------------------------
    def _draw_beams(self):
        tol = float(self.slider_slab.value())
        sel = self._beams_in_slab(tol)
        if not sel:
            return 0, 0

        wmax = max((b.weight for b, _, _ in sel), default=1.0) or 1.0
        mx, my = self._inplane_mm()
        mm_per_px = 0.5 * (mx + my)
        show_cone = self.chk_cone.isChecked()
        n_bone = 0

        for beam, frac, entry in sel:
            rel_w = beam.weight / wmax
            crosses_bone = beam.bone_cm > BONE_WARN_CM
            n_bone += int(crosses_bone)
            color = "#ff5c5c" if crosses_bone else "#00e5ff"
            # L'opacita' combina il peso del fascio e quanto giace nel piano:
            # un fascio importante e complanare risalta, uno marginale sfuma.
            alpha = float(np.clip(0.20 + 0.65 * rel_w * frac, 0.12, 0.9))

            x0, y0 = self._to_display([entry])
            x1, y1 = self._to_display([beam.target])
            x0, y0, x1, y1 = float(x0[0]), float(y0[0]), float(x1[0]), float(y1[0])

            if show_cone:
                # Il fascio e' un cono divergente: la semiapertura cresce con la
                # distanza dalla sorgente. Disegnarne la larghezza reale e'
                # l'unico modo per vedere se il BORDO del fascio intercetta una
                # costa, cosa che la sola linea d'asse non mostrerebbe.
                d_entry = float(np.linalg.norm(entry - beam.source))
                hw_entry = (beam.cone_mm / 2.0) * d_entry / SAD / mm_per_px
                hw_target = (beam.cone_mm / 2.0) / mm_per_px
                dx, dy = x1 - x0, y1 - y0
                L = float(np.hypot(dx, dy))
                if L > 1e-6:
                    nx, ny = -dy / L, dx / L
                    poly = np.array([
                        [x0 + nx * hw_entry, y0 + ny * hw_entry],
                        [x1 + nx * hw_target, y1 + ny * hw_target],
                        [x1 - nx * hw_target, y1 - ny * hw_target],
                        [x0 - nx * hw_entry, y0 - ny * hw_entry],
                    ])
                    self.ax.add_patch(Polygon(poly, closed=True, facecolor=color,
                                              edgecolor="none",
                                              alpha=alpha * 0.30, zorder=3))

            self.ax.plot([x0, x1], [y0, y1], color=color,
                         linewidth=0.5 + 1.8 * rel_w, alpha=alpha, zorder=4)
            if self.chk_entry.isChecked():
                self.ax.plot([x0], [y0], "o", color=color, markersize=2.5 + 3 * rel_w,
                             alpha=min(alpha + 0.2, 1.0), zorder=5)

        handles = [
            mlines.Line2D([], [], color="#00e5ff", lw=2,
                          label="fascio libero"),
            mlines.Line2D([], [], color="#ff5c5c", lw=2,
                          label=f"attraversa osso (>{BONE_WARN_CM:g} cm)"),
        ]
        leg = self.ax.legend(handles=handles, loc="lower right", fontsize=7,
                             facecolor="#171d25", edgecolor="#232a34",
                             framealpha=0.85)
        for t in leg.get_texts():
            t.set_color("#c8d0da")
        return len(sel), n_bone

    # ------------------------------------------------------------------
    def _update_info(self, n_shown, n_bone):
        if not self.beams:
            self.lbl_info.setText(
                "<div style='color:#8b95a3;'>"
                "<b style='color:#c8d0da;'>NESSUN PIANO</b><br>"
                "Calcolare la dosimetria nella vista 3D per analizzare i "
                "fasci in sezione.</div>")
            return

        tot = len(self.beams)
        bone_all = sum(1 for b in self.beams if b.bone_cm > BONE_WARN_CM)
        w = np.array([b.weight for b in self.beams], dtype=float)
        bone_cm = np.array([b.bone_cm for b in self.beams], dtype=float)
        bone_avg = float(np.average(bone_cm, weights=np.maximum(w, 1e-9)))
        cones = sorted({b.cone_mm for b in self.beams})

        self.lbl_info.setText(
            "<div style='color:#8b95a3; line-height:150%;'>"
            "<b style='color:#c8d0da;'>PIANO COMPLETO</b><br>"
            f"Fasci totali    : {tot}<br>"
            f"Coni            : {' / '.join(f'{c:g}' for c in cones)} mm<br>"
            f"<span style='color:{'#ff5c5c' if bone_all else '#3ddc84'};'>"
            f"Che toccano osso: {bone_all}</span><br>"
            f"Osso medio      : {bone_avg:.2f} cm<br><br>"

            "<b style='color:#c8d0da;'>IN QUESTA SEZIONE</b><br>"
            f"<span style='color:#00e5ff;'>Fasci visibili  : {n_shown}</span><br>"
            f"<span style='color:{'#ff5c5c' if n_bone else '#8b95a3'};'>"
            f"Con osso        : {n_bone}</span><br><br>"

            "<span style='color:#6f7a88; font-size:10px;'>"
            "Sono mostrati i fasci il cui percorso giace in buona parte nella "
            "fetta selezionata. Allargare lo spessore per vederne di piu', "
            "restringerlo per isolare quelli complanari.<br><br>"
            "La banda colorata e' la larghezza reale del cono: se sfiora una "
            "costa, quel fascio irradia la parete toracica."
            "</span></div>")
