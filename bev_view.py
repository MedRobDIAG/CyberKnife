# ============================================================================
#  bev_view.py - Beam's Eye View: la scena vista DALLA SORGENTE del fascio.
#
#  E' la verifica fascio-per-fascio che si fa in clinica. La sezione mostra
#  dove passano i fasci nel paziente; la BEV mostra che cosa VEDE un singolo
#  fascio guardando lungo il proprio asse:
#
#     - DRR (Digitally Reconstructed Radiograph): una radiografia simulata,
#       ottenuta integrando l'attenuazione lungo i raggi che partono dalla
#       sorgente. E' la stessa immagine che produrrebbe una lastra posta
#       dietro il paziente su quella direzione;
#     - la SAGOMA del bersaglio proiettata in prospettiva dalla sorgente;
#     - l'APERTURA del collimatore, cioe' cio' che il fascio irradia davvero;
#     - la proiezione delle OSSA, per vedere se l'apertura cade su una costa.
#
#  Le due domande a cui risponde, entrambe quantificate:
#     "il fascio inquadra tutto il bersaglio?"  -> copertura dell'apertura
#     "e per arrivarci passa dall'osso?"        -> occupazione ossea
#
#  Sistema di riferimento: millimetri sul piano dell'isocentro. In questo
#  frame l'apertura ha esattamente raggio cono/2, e la sagoma del bersaglio ha
#  la dimensione con cui il fascio la "vede".
# ============================================================================
import numpy as np
from scipy import ndimage

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QCheckBox,
    QGroupBox, QSizePolicy, QPushButton,
)

import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Circle

from dosimetry import apply_affine_fast

# Risoluzione del DRR: 160x160 raggi x 200 campioni costano ~0.2 s, quindi la
# vista resta reattiva al cambio di fascio.
BEV_PIXELS = 160
BEV_SAMPLES = 200
# Campo visivo minimo (semiampiezza in mm sul piano dell'isocentro): serve a
# mostrare il contesto attorno all'apertura, non solo il bersaglio.
BEV_MIN_FOV = 45.0
# Spessore osseo sotto il quale non si segnala nulla: l'interpolazione ai bordi
# delle ossa produce valori residui su quasi tutti i raggi, e senza soglia
# l'intero campo apparirebbe occupato da osso.
BONE_MIN_MM = 3.0


def beam_frame(beam):
    """Terna ortonormale del fascio: (asse, trasverso 1, trasverso 2)."""
    d = beam.direction / max(np.linalg.norm(beam.direction), 1e-9)
    tmp = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(tmp, d))) > 0.9:
        tmp = np.array([1.0, 0.0, 0.0])
    e1 = np.cross(d, tmp)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(d, e1)
    return d, e1, e2


def project_to_bev(points_world, beam):
    """Proiezione prospettica di punti mondo sul piano dell'isocentro.

    Un punto P visto dalla sorgente S si proietta dove la retta S->P incontra
    il piano dell'isocentro. Con w = distanza di P lungo l'asse e L = distanza
    sorgente-bersaglio, il fattore di magnificazione e' L/w: cio' che sta piu'
    vicino alla sorgente appare piu' grande, esattamente come su una lastra.

    Restituisce (u_mm, v_mm, w_mm); w<=0 indica punti dietro la sorgente.
    """
    d, e1, e2 = beam_frame(beam)
    L = float(np.linalg.norm(beam.target - beam.source))
    rel = np.asarray(points_world, dtype=np.float64) - beam.source[None, :]
    w = rel @ d
    safe = np.where(np.abs(w) < 1e-6, 1e-6, w)
    mag = L / safe
    return (rel @ e1) * mag, (rel @ e2) * mag, w


def raycast(volume, volume_affine, beam, fov_mm, n_pix=BEV_PIXELS,
            n_samples=BEV_SAMPLES, unit_mm=False, t_span=(-320.0, 220.0)):
    """Integra un volume lungo i raggi che partono dalla sorgente.

    E' l'operazione alla base sia della radiografia simulata sia delle sagome:
    per ogni pixel del rivelatore si somma il volume lungo il raggio.

    unit_mm=True restituisce il risultato in millimetri attraversati (usato per
    le maschere: "quanti mm di osso incontra questo raggio"), altrimenti in
    centimetri equivalenti d'acqua (usato per il DRR).

    Il ray-casting e' l'unico modo corretto di ottenere una sagoma: proiettare
    i centri dei voxel in un istogramma produrrebbe una nuvola di punti isolati
    con buchi, perche' i voxel proiettati non tassellano il piano.
    """
    d, e1, e2 = beam_frame(beam)
    L = float(np.linalg.norm(beam.target - beam.source))

    ax = np.linspace(-fov_mm, fov_mm, n_pix)
    U, V = np.meshgrid(ax, ax, indexing="ij")
    plane = (beam.target[None, :]
             + U.ravel()[:, None] * e1[None, :]
             + V.ravel()[:, None] * e2[None, :])
    dirs = plane - beam.source[None, :]
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)

    # Intervallo di integrazione, espresso rispetto all'isocentro: per il DRR
    # copre tutto il paziente, per le maschere si ferma al bersaglio.
    t = np.linspace(max(L + t_span[0], 1.0), L + t_span[1], n_samples)
    step = float(t[1] - t[0])

    pts = beam.source[None, None, :] + dirs[:, None, :] * t[None, :, None]
    idx = apply_affine_fast(np.linalg.inv(np.asarray(volume_affine, np.float64)),
                            pts.reshape(-1, 3).T)
    val = ndimage.map_coordinates(volume, idx, order=1, mode="constant",
                                  cval=0.0).reshape(n_pix * n_pix, n_samples)
    scale = step if unit_mm else step / 10.0
    return (val.sum(axis=1) * scale).reshape(n_pix, n_pix)


def mask_thickness(mask, mask_affine, beam, fov_mm, n_pix=BEV_PIXELS,
                   entrance_only=False):
    """Millimetri di maschera attraversati da ciascun raggio.

    entrance_only limita il conteggio al tratto sorgente->bersaglio. Per l'osso
    e' la scelta corretta: un raggio attraversa la gabbia toracica due volte,
    ma solo le coste PRIMA del bersaglio attenuano il fascio e ricevono la dose
    d'ingresso. Contare anche quelle in uscita saturerebbe l'immagine e non
    corrisponderebbe alla metrica di percorso osseo usata dall'ottimizzatore.
    """
    if mask is None or not np.any(mask):
        return None
    span = (-320.0, 0.0) if entrance_only else (-320.0, 220.0)
    return raycast(np.asarray(mask, dtype=np.float32), mask_affine, beam,
                   fov_mm, n_pix, n_samples=180, unit_mm=True, t_span=span)


class BeamEyeView(QWidget):
    """Vista dal fascio, con selettore del fascio e metriche di inquadratura."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.red_grid = None
        self.grid_affine = None
        self.beams = []
        self.tumor = None
        self.tumor_affine = None
        self.bones = None
        self.bones_affine = None
        self._cache = {}
        self._build_ui()

    # ------------------------------------------------------------------
    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        bar = QHBoxLayout()
        bar.setSpacing(8)
        bar.addWidget(QLabel("Fascio:"))
        self.combo_beam = QComboBox()
        self.combo_beam.setMinimumWidth(330)
        self.combo_beam.currentIndexChanged.connect(lambda _: self.redraw())
        bar.addWidget(self.combo_beam)

        self.btn_prev = QPushButton("◀")
        self.btn_next = QPushButton("▶")
        for b, step in ((self.btn_prev, -1), (self.btn_next, +1)):
            b.setMaximumWidth(38)
            b.clicked.connect(lambda _, s=step: self._step_beam(s))
            bar.addWidget(b)

        self.chk_drr = QCheckBox("Radiografia (DRR)")
        self.chk_tumor = QCheckBox("Bersaglio")
        self.chk_bone = QCheckBox("Ossa")
        self.chk_ap = QCheckBox("Apertura collimatore")
        for c in (self.chk_drr, self.chk_tumor, self.chk_bone, self.chk_ap):
            c.setChecked(True)
            c.toggled.connect(lambda _: self.redraw())
            bar.addWidget(c)
        bar.addStretch(1)
        root.addLayout(bar)

        body = QHBoxLayout()
        body.setSpacing(8)
        box = QGroupBox("Vista dal fascio (piano dell'isocentro, mm)")
        vb = QVBoxLayout(box)
        self.fig = Figure(figsize=(6, 6), facecolor="#0b0e12")
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding,
                                  QSizePolicy.Policy.Expanding)
        self.ax = self.fig.add_subplot(111)
        self.fig.subplots_adjust(left=0.1, right=0.98, top=0.94, bottom=0.09)
        self.canvas.mpl_connect("scroll_event", self._on_scroll)
        vb.addWidget(self.canvas)
        body.addWidget(box, stretch=3)

        info = QGroupBox("Analisi del fascio")
        ib = QVBoxLayout(info)
        info.setMinimumWidth(280)
        info.setMaximumWidth(340)
        self.lbl_info = QLabel()
        self.lbl_info.setObjectName("infoPanel")
        self.lbl_info.setAlignment(Qt.AlignmentFlag.AlignTop |
                                   Qt.AlignmentFlag.AlignLeft)
        self.lbl_info.setWordWrap(True)
        ib.addWidget(self.lbl_info, stretch=1)
        body.addWidget(info, stretch=1)
        root.addLayout(body, stretch=1)

        self._placeholder()

    def _placeholder(self):
        self.ax.clear()
        self.ax.set_facecolor("#000000")
        self.ax.text(0.5, 0.5, "Calcolare la dosimetria\nper analizzare i fasci.",
                     ha="center", va="center", color="#5c6672", fontsize=10,
                     transform=self.ax.transAxes)
        self.ax.axis("off")
        self.canvas.draw_idle()
        self.lbl_info.setText("<div style='color:#8b95a3;'>Nessun fascio.</div>")

    # ------------------------------------------------------------------
    def set_data(self, red_grid, grid_affine, beams, tumor_mask, tumor_affine,
                 bone_mask=None, bone_affine=None):
        self.red_grid = red_grid
        self.grid_affine = np.asarray(grid_affine, dtype=np.float64)
        self.tumor = tumor_mask
        self.tumor_affine = np.asarray(tumor_affine, dtype=np.float64)
        self.bones = bone_mask
        self.bones_affine = (None if bone_affine is None
                             else np.asarray(bone_affine, dtype=np.float64))
        self._cache = {}

        # I fasci si elencano dal piu' pesante: sono quelli che determinano il
        # piano e che conviene ispezionare per primi.
        self.beams = sorted(beams or [], key=lambda b: -b.weight)
        self.combo_beam.blockSignals(True)
        self.combo_beam.clear()
        wmax = max((b.weight for b in self.beams), default=1.0) or 1.0
        for i, b in enumerate(self.beams):
            flag = " ⚠ osso" if b.bone_cm > 0.5 else ""
            self.combo_beam.addItem(
                f"#{i + 1}  cono {b.cone_mm:g} mm  peso {100 * b.weight / wmax:.0f}%"
                f"  osso {b.bone_cm:.1f} cm{flag}")
        self.combo_beam.blockSignals(False)
        if self.beams:
            self.combo_beam.setCurrentIndex(0)
        self.redraw()

    def _step_beam(self, s):
        n = self.combo_beam.count()
        if n:
            self.combo_beam.setCurrentIndex((self.combo_beam.currentIndex() + s) % n)

    def _on_scroll(self, event):
        self._step_beam(1 if event.button == "up" else -1)

    # ------------------------------------------------------------------
    def _fov(self, beam):
        return max(BEV_MIN_FOV, beam.cone_mm * 2.2)

    def _layers(self, beam, i):
        """Calcola (e memorizza) DRR e sagome del fascio selezionato."""
        if i in self._cache:
            return self._cache[i]
        fov = self._fov(beam)
        drr = raycast(self.red_grid, self.grid_affine, beam, fov)
        # Spessori in mm: la sagoma del bersaglio e' dove se ne attraversa
        # almeno un po', mentre per l'osso lo spessore stesso e' l'informazione
        # utile (2 mm di costa non equivalgono a 10).
        tum_mm = mask_thickness(self.tumor, self.tumor_affine, beam, fov)
        bone_mm = (None if self.bones is None else
                   mask_thickness(self.bones, self.bones_affine, beam, fov,
                                  entrance_only=True))
        self._cache[i] = (fov, drr, tum_mm, bone_mm)
        return self._cache[i]

    # ------------------------------------------------------------------
    def redraw(self):
        i = self.combo_beam.currentIndex()
        if self.red_grid is None or not self.beams or i < 0:
            self._placeholder()
            return
        beam = self.beams[i]
        fov, drr, tum_mm, bone_mm = self._layers(beam, i)
        ext = [-fov, fov, -fov, fov]
        # Un raggio "vede" il bersaglio se ne attraversa almeno 1 mm; la soglia
        # elimina i raggi che ne sfiorano appena il bordo.
        tum = None if tum_mm is None else (tum_mm > 1.0)

        self.ax.clear()
        self.ax.set_facecolor("#000000")

        if self.chk_drr.isChecked():
            # Percentili invece di min/max: pochi voxel molto densi (osso
            # corticale) schiaccerebbero tutto il resto in un grigio piatto.
            lo, hi = np.percentile(drr, [1, 99])
            self.ax.imshow(drr.T, origin="lower", extent=ext, cmap="bone",
                           vmin=lo, vmax=max(hi, lo + 1e-6),
                           interpolation="bilinear")
        else:
            self.ax.imshow(np.zeros_like(drr).T, origin="lower", extent=ext,
                           cmap="gray", vmin=0, vmax=1)

        grid = np.linspace(-fov, fov, drr.shape[0])
        if self.chk_bone.isChecked() and bone_mm is not None and bone_mm.max() > BONE_MIN_MM:
            # Overlay a tinta unica con TRASPARENZA proporzionale allo spessore.
            # Una colormap classica colorerebbe anche i valori minimi: poiche'
            # quasi ogni raggio sfiora un po' di osso (effetto di volume
            # parziale sui bordi), il campo risulterebbe tutto rosso e il DRR
            # sottostante invisibile. Sotto BONE_MIN_MM l'overlay e' del tutto
            # trasparente, e sopra cresce fino alla costa presa in pieno.
            t = np.clip((bone_mm - BONE_MIN_MM) / 12.0, 0.0, 1.0)
            rgba = np.zeros(bone_mm.shape + (4,), dtype=np.float32)
            rgba[..., 0] = 1.0
            rgba[..., 1] = 0.30
            rgba[..., 2] = 0.30
            rgba[..., 3] = 0.72 * t
            self.ax.imshow(np.transpose(rgba, (1, 0, 2)), origin="lower",
                           extent=ext, interpolation="bilinear")
            self.ax.contour(grid, grid, bone_mm.T, levels=[8.0],
                            colors=["#ff2d2d"], linewidths=0.9, alpha=0.85)

        if self.chk_tumor.isChecked() and tum is not None and tum.any():
            self.ax.contourf(grid, grid, tum.T.astype(float), levels=[0.5, 1.5],
                             colors=["#ff8a00"], alpha=0.35)
            self.ax.contour(grid, grid, tum.T.astype(float), levels=[0.5],
                            colors=["#ff8a00"], linewidths=2.0)

        if self.chk_ap.isChecked():
            self.ax.add_patch(Circle((0, 0), beam.cone_mm / 2.0, fill=False,
                                     edgecolor="#00e5ff", linewidth=1.8,
                                     linestyle="--"))
        self.ax.plot(0, 0, "+", color="#00e5ff", markersize=10, markeredgewidth=1.2)

        self.ax.set_xlim(-fov, fov)
        self.ax.set_ylim(-fov, fov)
        self.ax.set_aspect("equal")
        self.ax.set_xlabel("mm", color="#8b95a3", fontsize=8)
        self.ax.set_ylabel("mm", color="#8b95a3", fontsize=8)
        self.ax.tick_params(colors="#5c6672", labelsize=7)
        for sp in self.ax.spines.values():
            sp.set_color("#232a34")
        self.ax.set_title(f"Fascio #{i + 1} di {len(self.beams)}",
                          color="#c8d0da", fontsize=10, pad=6)
        self.canvas.draw_idle()
        self._update_info(beam, i, fov, tum, bone_mm)

    # ------------------------------------------------------------------
    def _update_info(self, beam, i, fov, tum, bone_mm):
        n = tum.shape[0] if tum is not None else BEV_PIXELS
        g = np.linspace(-fov, fov, n)
        X, Y = np.meshgrid(g, g, indexing="ij")
        aperture = (X ** 2 + Y ** 2) <= (beam.cone_mm / 2.0) ** 2

        # Quanta parte del bersaglio, come lo vede QUESTO fascio, cade dentro
        # l'apertura: sotto il 100% quel fascio da solo non copre la lesione.
        cov = (100.0 * float((tum & aperture).sum()) / max(int(tum.sum()), 1)
               if tum is not None and tum.any() else 0.0)
        # Sull'osso si riportano DUE numeri: quanta parte dell'apertura lo
        # incontra, e quanto e' spesso in media. Una costa sfiorata di striscio
        # e una presa in pieno hanno conseguenze molto diverse.
        if bone_mm is not None:
            hit = bone_mm > BONE_MIN_MM
            bone_frac = 100.0 * float((hit & aperture).sum()) / max(int(aperture.sum()), 1)
            bone_thick = float(bone_mm[aperture].mean())
        else:
            bone_frac, bone_thick = 0.0, 0.0

        wmax = max((b.weight for b in self.beams), default=1.0) or 1.0

        def col(v, good, ok_):
            return "#3ddc84" if v >= good else ("#ffb020" if v >= ok_ else "#ff5c5c")

        self.lbl_info.setText(
            "<div style='color:#8b95a3; line-height:150%;'>"
            f"<b style='color:#c8d0da;'>FASCIO #{i + 1}</b><br>"
            f"Collimatore : {beam.cone_mm:g} mm<br>"
            f"Peso        : {100 * beam.weight / wmax:.0f}% del massimo<br>"
            f"Percorso    : {beam.path_cm:.1f} cm (radiologici)<br>"
            f"<span style='color:{'#ff5c5c' if beam.bone_cm > 0.5 else '#3ddc84'};'>"
            f"Osso su asse: {beam.bone_cm:.2f} cm</span><br><br>"

            "<b style='color:#c8d0da;'>INQUADRATURA</b><br>"
            f"<span style='color:{col(cov, 99, 90)};'><b>"
            f"Bersaglio nell'apertura: {cov:.0f}%</b></span><br>"
            "<span style='color:#6f7a88; font-size:11px;'>quota della sagoma del "
            "tumore che questo fascio irradia</span><br><br>"

            "<b style='color:#c8d0da;'>OSTACOLI</b><br>"
            f"<span style='color:{col(100 - bone_frac, 85, 60)};'>"
            f"Apertura su osso : {bone_frac:.0f}%</span><br>"
            f"Spessore medio   : {bone_thick:.1f} mm<br>"
            "<span style='color:#6f7a88; font-size:11px;'>quota dell'apertura che "
            "incontra scheletro, e di quanto</span><br><br>"

            "<span style='color:#6f7a88; font-size:10px;'>"
            "Sfondo: radiografia ricostruita (DRR) dal punto di vista della "
            "sorgente. Cerchio tratteggiato: apertura del collimatore. "
            "Arancione: bersaglio. Rosso: scheletro, tanto piu' acceso quanto "
            "piu' spesso.<br><br>"
            "Rotellina o frecce per scorrere i fasci."
            "</span></div>")
