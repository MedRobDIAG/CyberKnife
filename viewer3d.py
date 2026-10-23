# ============================================================================
#  viewer3d.py - Visualizzatore 3D: tumore collocato dentro il busto del paziente.
#
#  Contiene:
#    - MeshBuilderWorker : QThread che ricostruisce le mesh (busto + tumore)
#                          senza bloccare la GUI.
#    - Mesh3DView        : widget PyQt con canvas Matplotlib 3D ruotabile.
#
#  Logica di ricostruzione:
#    BUSTO  -> soglia HU (tutto cio' che e' piu' denso dell'aria) + riempimento
#              cavita' + componente principale -> Marching Cubes -> decimazione.
#    TUMORE -> Marching Cubes sulla maschera predetta dalla rete.
#    Entrambe le mesh hanno i vertici in MILLIMETRI (spazio fisico del paziente),
#    quindi risultano automaticamente allineate fra loro e col centro di massa.
# ============================================================================
import os

import numpy as np
import nibabel as nib
from nibabel.affines import apply_affine
from scipy import ndimage
from skimage import measure
import trimesh

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QComboBox,
    QCheckBox, QSlider, QGroupBox, QSizePolicy, QFileDialog, QMessageBox,
    QProgressBar, QSpinBox, QScrollArea, QFrame,
)

import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection

import config
from tumor_3d import build_mesh as build_tumor_mesh
import dosimetry as dz
from dose_worker import DoseWorker
import organs as og
import organ_mesh as omsh

# ---------------------------------------------------------------------------
#  Backend della vista 3D
#
#  Si preferisce PyVista/VTK, che rasterizza su GPU: sulla scena reale del
#  progetto passa da ~7 fps a oltre 100. Se VTK o OpenGL non sono disponibili
#  si ricade sulla vista Matplotlib, piu' lenta ma priva di dipendenze native,
#  cosi' l'applicazione resta utilizzabile ovunque.
# ---------------------------------------------------------------------------
try:
    from mesh_view_gpu import Mesh3DViewGPU
    GPU_3D_AVAILABLE = True
except Exception:                                    # noqa: BLE001
    Mesh3DViewGPU = None
    GPU_3D_AVAILABLE = False

# Soglia HU che separa il corpo dall'aria circostante (-1000 HU = aria).
BODY_HU_THRESHOLD = -320
# Sottocampionamento del volume per la mesh del busto: la superficie esterna
# non richiede il dettaglio pieno e cosi' la ricostruzione resta istantanea.
BODY_STRIDE = (4, 4, 3)
# Numero di facce target: matplotlib 3D diventa lento oltre ~10k poligoni.
BODY_TARGET_FACES = 6000
TUMOR_TARGET_FACES = 3000

# Estremi dello zoom nella vista 3D, come frazione dell'inquadratura iniziale.
# 0.04 corrisponde a un campo di ~15 mm su un torace di 350: abbastanza per
# esaminare la lesione, non tanto da perdere ogni riferimento anatomico.
ZOOM_MIN, ZOOM_MAX = 0.04, 1.0

# Punti di vista predefiniti (elevazione, azimut).
VIEW_PRESETS = {
    "Isometrica": (18, -60),
    "Anteriore":  (0, -90),
    "Laterale":   (0, 0),
    "Superiore":  (89, -90),
}


# ===========================================================================
#  Worker: costruzione delle mesh in background
# ===========================================================================
class MeshBuilderWorker(QThread):
    """Ricostruisce la mesh del busto e quella del tumore."""

    progress = pyqtSignal(int)
    status = pyqtSignal(str)
    meshReady = pyqtSignal(dict)
    failed = pyqtSignal(str)

    def __init__(self, ct_data, mask, affine, parent=None):
        super().__init__(parent)
        self.ct = ct_data
        self.mask = mask
        self.affine = affine

    # ------------------------------------------------------------------
    def run(self):
        try:
            # ---------- 1) Segmentazione degli organi ----------
            # Soglie in unita' Hounsfield + morfologia (nessuna rete neurale):
            # le densita' di aria, tessuto molle e osso sono cosi' distinte da
            # rendere sufficiente una soglia.
            self.progress.emit(5)
            masks, organ_affine, volumes = og.segment_organs(
                self.ct, self.affine, status_cb=self.status.emit)
            self.progress.emit(35)

            def mesh_progress(done, total):
                self.progress.emit(35 + int(50 * done / max(total, 1)))

            # La TAC va passata: la ricostruzione si appoggia ai valori
            # Hounsfield, non alla sola maschera. Il budget di facce dipende
            # dal motore grafico: la vista GPU regge centomila triangoli,
            # quella di ripiego in matplotlib rallenta molto prima.
            organ_meshes = og.build_organ_meshes(
                masks, organ_affine, ct_data=self.ct, ct_affine=self.affine,
                faces_scale=1.0 if GPU_3D_AVAILABLE else 0.45,
                progress_cb=mesh_progress, status_cb=self.status.emit)
            self.progress.emit(85)

            # QUALITA' MISURATA della ricostruzione. Non e' un vezzo: la
            # fedelta' in millimetri e la chiusura delle superfici sono cio'
            # che distingue un modello utilizzabile da un'immagine
            # verosimile, e il pannello le mostra invece di lasciarle nei log.
            self.status.emit("Controllo qualita' della ricostruzione...")
            quality = {}
            for o in organ_meshes:
                try:
                    rec = omsh.RECIPES.get(o["key"])
                    hu = None
                    if rec and rec["source"] != "mask":
                        hu, _, _ = omsh.organ_grid(
                            self.ct, self.affine, masks[o["key"]],
                            organ_affine, omsh.MESH_SPACING_MM)
                    ref = omsh.reference_mask(o["riferimento"], hu, rec) \
                        if rec else o["riferimento"]
                    quality[o["key"]] = omsh.mesh_report(
                        o["mesh"], ref, o["rif_affine"], n_samples=15000)
                except Exception:      # una metrica non deve fermare il 3D
                    continue

            # Il corpo resta accessibile a parte: e' il riferimento per
            # l'inquadratura della scena e per l'esportazione.
            body_mesh = next((o["mesh"] for o in organ_meshes
                              if o["key"] == "body"), None)

            # ---------- 2) Mesh del tumore ----------
            tumor_mesh = None
            tumor_volume_cm3 = None
            n_signif = 0
            if self.mask is not None and self.mask.any():
                self.status.emit("Ricostruzione mesh del tumore...")
                # La maschera della rete puo' avere piu' componenti (misurato
                # su lung_001: 22, di cui 6 sopra i 100 mm3). Il piano tratta
                # la MAGGIORE - e' quella che definisce l'isocentro - quindi il
                # modello 3D mostra la stessa, e il pannello dice quante ce ne
                # sono: un volume di 5.98 cm3 accanto a un piano da 2.73
                # significa che si sta guardando un'altra cosa.
                import postproc
                mask_u8, n_signif = postproc.primary_mask(self.mask > 0.5,
                                                          self.affine)
                mask_t = mask_u8 > 0
                # Stessa catena degli organi: la maschera del tumore non ha un
                # appoggio nella densita' (viene dalla rete, non da una
                # soglia), quindi si usa la sua distanza firmata, che da' una
                # superficie continua senza i gradini del voxel.
                tumor_mesh = omsh.build_mesh(
                    mask_t, None, self.affine,
                    dict(omsh.RECIPES["tumor"], faces=TUMOR_TARGET_FACES))
                if tumor_mesh is None:
                    tumor_mesh = build_tumor_mesh(mask_t, self.affine,
                                                  smoothing_iters=10)
                voxel_vol = abs(np.linalg.det(self.affine[:3, :3]))
                tumor_volume_cm3 = float(mask_t.sum()) * voxel_vol / 1000.0
            self.progress.emit(95)

            self.status.emit("Modello 3D pronto")
            self.meshReady.emit({
                "body_mesh": body_mesh,
                "tumor_mesh": tumor_mesh,
                "tumor_volume_cm3": tumor_volume_cm3,
                "organ_meshes": organ_meshes,
                "organ_quality": quality,
                "tumor_components": n_signif,
                "organ_masks": masks,
                "organ_affine": organ_affine,
                "organ_volumes": volumes,
            })
            self.progress.emit(100)

        except Exception as exc:                       # noqa: BLE001
            import traceback
            self.failed.emit(f"{type(exc).__name__}: {exc}\n\n{traceback.format_exc()}")


# ===========================================================================
#  Widget: canvas 3D interattivo
# ===========================================================================
class Mesh3DViewMPL(QWidget):
    """Ricaduta software: canvas Matplotlib 3D (nessuna GPU).

    La rotazione avviene trascinando col mouse (funzionalita' nativa degli assi
    3D di Matplotlib).
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.body_mesh = None
        self.tumor_mesh = None
        self.com_world = None
        self._body_coll = None
        self._tumor_coll = None
        self._beam_coll = None
        self._isodose_coll = None
        self.organ_colls = {}      # chiave organo -> Poly3DCollection
        self._base_center = None   # inquadratura di riferimento per lo zoom
        self._base_half = None
        self._zoom = 1.0

        self.fig = Figure(figsize=(6, 5), facecolor="#0b0e12")
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding,
                                  QSizePolicy.Policy.Expanding)
        self.ax = self.fig.add_subplot(111, projection="3d")
        # computed_zorder=False: consente di imporre l'ordine di disegno, cosi'
        # il tumore viene tracciato PRIMA del busto semitrasparente e resta
        # visibile per trasparenza al suo interno.
        self.ax.computed_zorder = False
        self._style_axes()
        self.canvas.mpl_connect("scroll_event", self._on_scroll)
        self.canvas.mpl_connect("button_press_event", self._on_double_click)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas)
        self._draw_placeholder()

    # ------------------------------------------------------------------
    def _style_axes(self):
        self.ax.set_facecolor("#0b0e12")
        for axis in (self.ax.xaxis, self.ax.yaxis, self.ax.zaxis):
            axis.set_pane_color((0.05, 0.06, 0.08, 1.0))
            axis.label.set_color("#8b95a3")
            axis._axinfo["grid"]["color"] = (0.16, 0.18, 0.21, 1.0)
        self.ax.tick_params(colors="#5c6672", labelsize=7)
        self.ax.set_xlabel("X (mm)", fontsize=8)
        self.ax.set_ylabel("Y (mm)", fontsize=8)
        self.ax.set_zlabel("Z (mm)", fontsize=8)

    def _draw_placeholder(self):
        self.ax.clear()
        self.ax.set_axis_off()
        self.ax.text2D(0.5, 0.5,
                       "Nessun modello 3D generato.\n"
                       "Eseguire prima l'inferenza nel Tab 1,\n"
                       "poi premere \"Genera Modello 3D\".",
                       ha="center", va="center", color="#5c6672", fontsize=10,
                       transform=self.ax.transAxes)
        self.canvas.draw_idle()

    # ------------------------------------------------------------------
    def set_meshes(self, body_mesh, tumor_mesh, com_world=None,
                   organ_meshes=None):
        """Disegna organi, busto e tumore nello stesso sistema di coordinate.

        L'ordine di disegno (zorder) va dal piu' interno al piu' esterno e le
        opacita' crescono verso l'interno: cosi' le strutture profonde restano
        visibili attraverso quelle che le contengono. Il tumore, che e' il
        bersaglio, e' l'unico completamente opaco.
        """
        self.body_mesh = body_mesh
        self.tumor_mesh = tumor_mesh
        self.com_world = com_world
        self.organ_colls = {}

        self.ax.clear()
        self.ax.set_axis_on()
        self.ax.computed_zorder = False
        self._style_axes()
        # ax.clear() ha rimosso gli artisti: i riferimenti vanno azzerati per
        # non conservare puntatori a oggetti non piu' presenti negli assi.
        self._beam_coll = None
        self._isodose_coll = None

        # --- Tumore (zorder basso: disegnato per primo) ---
        if tumor_mesh is not None and len(tumor_mesh.faces):
            self._tumor_coll = Poly3DCollection(
                tumor_mesh.vertices[tumor_mesh.faces],
                facecolor="#ff8a00", edgecolor="#803f00", linewidths=0.08,
                alpha=1.0, zorder=1,
            )
            self.ax.add_collection3d(self._tumor_coll)

            if com_world is not None:
                self.ax.scatter(*com_world, color="#00e5ff", s=90, marker="+",
                                linewidths=2.0, zorder=2, depthshade=False,
                                label="Centro di massa (target)")

        # --- Organi, dal piu' interno al piu' esterno ---
        self._body_coll = None
        for org in (organ_meshes or []):
            mesh = org["mesh"]
            if mesh is None or not len(mesh.faces):
                continue
            coll = Poly3DCollection(
                mesh.vertices[mesh.faces], facecolor=org["color"],
                edgecolor="none", alpha=org["alpha"], zorder=org["zorder"],
            )
            self.ax.add_collection3d(coll)
            self.organ_colls[org["key"]] = coll
            if org["key"] == "body":
                self._body_coll = coll

        # Compatibilita': se non sono stati passati organi, si disegna il solo
        # profilo corporeo come prima.
        if self._body_coll is None and body_mesh is not None and len(body_mesh.faces):
            self._body_coll = Poly3DCollection(
                body_mesh.vertices[body_mesh.faces],
                facecolor="#7fb6d6", edgecolor="none", alpha=0.12, zorder=8)
            self.ax.add_collection3d(self._body_coll)
            self.organ_colls["body"] = self._body_coll

        # --- Inquadratura: proporzioni reali basate sul busto ---
        ref = body_mesh if body_mesh is not None else tumor_mesh
        if ref is not None:
            lo, hi = ref.bounds
            # Inquadratura di partenza, memorizzata come riferimento dello zoom:
            # ogni livello di ingrandimento e' una frazione di questa, cosi' un
            # doppio clic riporta esattamente alla vista iniziale.
            self._base_center = (np.asarray(hi) + np.asarray(lo)) / 2.0
            self._base_half = (np.asarray(hi) - np.asarray(lo)) / 2.0
            self._zoom = 1.0
            try:
                self.ax.set_box_aspect(hi - lo)   # scala 1:1:1 -> niente distorsioni
            except Exception:
                pass
            self._apply_zoom()

        self.set_view("Isometrica")
        if self.ax.get_legend_handles_labels()[0]:
            leg = self.ax.legend(loc="upper right", fontsize=8,
                                 facecolor="#171d25", edgecolor="#232a34")
            for txt in leg.get_texts():
                txt.set_color("#c8d0da")
        self.canvas.draw_idle()

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    #  Sovrapposizioni dosimetriche
    # ------------------------------------------------------------------
    def set_beams(self, beams, visible=True):
        """Disegna i fasci, ciascuno verso il PROPRIO bersaglio.

        Lo spessore e l'opacita' della linea sono proporzionali al peso del
        fascio: si vede a colpo d'occhio quali direzioni portano piu' dose.
        """
        if self._beam_coll is not None:
            self._beam_coll.remove()
            self._beam_coll = None
        if not beams:
            self.canvas.draw_idle()
            return

        weights = np.array([b.weight for b in beams], dtype=np.float64)
        wmax = max(weights.max(), 1e-12)
        rel = weights / wmax

        segs, widths, alphas = [], [], []
        for b, r in zip(beams, rel):
            d = b.target - b.source
            n = np.linalg.norm(d)
            if n < 1e-6:
                continue
            # Si disegna solo l'ultimo tratto: il segmento completo dal nodo
            # (a 800 mm) uscirebbe dall'inquadratura del busto.
            segs.append([b.target - (d / n) * min(n, 260.0), b.target])
            widths.append(0.35 + 2.4 * r)
            alphas.append(0.12 + 0.55 * r)

        self._beam_coll = Line3DCollection(
            segs, colors="#00e5ff", linewidths=widths, zorder=2)
        self._beam_coll.set_alpha(alphas)
        self.ax.add_collection3d(self._beam_coll)
        self._beam_coll.set_visible(visible)
        self.canvas.draw_idle()

    def set_isodose(self, dose_pct, grid_affine, level=50.0, visible=True):
        """Superficie isodose: guscio che racchiude la regione sopra soglia.

        Si estrae con Marching Cubes sulla griglia di dose e si convertono i
        vertici in millimetri, cosi' la superficie e' automaticamente allineata
        a tumore e busto.
        """
        if self._isodose_coll is not None:
            self._isodose_coll.remove()
            self._isodose_coll = None

        if dose_pct is None or not (dose_pct >= level).any():
            self.canvas.draw_idle()
            return
        try:
            verts, faces, _, _ = measure.marching_cubes(
                dose_pct.astype(np.float32), level=float(level))
        except (ValueError, RuntimeError):
            self.canvas.draw_idle()
            return

        verts_mm = apply_affine(np.asarray(grid_affine, dtype=np.float64), verts)
        mesh = trimesh.Trimesh(vertices=verts_mm, faces=faces, process=True)
        if len(mesh.faces) > 4000:
            try:
                mesh = mesh.simplify_quadric_decimation(face_count=4000)
            except Exception:
                pass

        self._isodose_coll = Poly3DCollection(
            mesh.vertices[mesh.faces], facecolor="#3ddc84", edgecolor="none",
            alpha=0.22, zorder=2)
        self.ax.add_collection3d(self._isodose_coll)
        self._isodose_coll.set_visible(visible)
        self.canvas.draw_idle()

    def set_beams_visible(self, visible):
        if self._beam_coll is not None:
            self._beam_coll.set_visible(visible)
            self.canvas.draw_idle()

    def set_isodose_visible(self, visible):
        if self._isodose_coll is not None:
            self._isodose_coll.set_visible(visible)
            self.canvas.draw_idle()

    # ------------------------------------------------------------------
    #  Zoom
    # ------------------------------------------------------------------
    def _on_scroll(self, event):
        """Rotellina del mouse: ingrandisce e rimpicciolisce la scena."""
        if self._base_half is None:
            return
        step = 1.0 / 1.15 if event.button == "up" else 1.15
        self._zoom = float(np.clip(self._zoom * step, ZOOM_MIN, ZOOM_MAX))
        self._apply_zoom()

    def _on_double_click(self, event):
        """Doppio clic: ritorno all'inquadratura iniziale."""
        if event.dblclick and self._base_half is not None:
            self._zoom = 1.0
            self._apply_zoom()

    def _apply_zoom(self):
        """Ricalcola i limiti degli assi al livello di zoom corrente.

        Il centro non resta fisso: mentre si stringe, l'inquadratura scivola
        verso il bersaglio. Zoomando su un busto intero il centro geometrico e'
        a meta' del torace, quindi uno zoom centrato lascerebbe la lesione fuori
        campo proprio quando la si vuole guardare da vicino.
        """
        if self._base_half is None:
            return
        c0 = self._base_center
        focus = self.com_world if self.com_world is not None else c0
        focus = np.asarray(focus, dtype=np.float64)

        center = c0 + (focus - c0) * (1.0 - self._zoom)
        half = np.maximum(self._base_half * self._zoom, 1.0)

        self.ax.set_xlim(center[0] - half[0], center[0] + half[0])
        self.ax.set_ylim(center[1] - half[1], center[1] + half[1])
        self.ax.set_zlim(center[2] - half[2], center[2] + half[2])
        self.canvas.draw_idle()

    # ------------------------------------------------------------------
    def set_organ_visible(self, key, visible):
        """Accende o spegne un singolo organo."""
        coll = self.organ_colls.get(key)
        if coll is not None:
            coll.set_visible(visible)
            self.canvas.draw_idle()

    def set_body_visible(self, visible):
        if self._body_coll is not None:
            self._body_coll.set_visible(visible)
            self.canvas.draw_idle()

    def set_tumor_visible(self, visible):
        if self._tumor_coll is not None:
            self._tumor_coll.set_visible(visible)
            self.canvas.draw_idle()

    def set_body_alpha(self, alpha):
        if self._body_coll is not None:
            self._body_coll.set_alpha(alpha)
            # Ripristina il colore: set_alpha da solo puo' azzerare la facecolor.
            self._body_coll.set_facecolor("#7fb6d6")
            self.canvas.draw_idle()

    def set_view(self, name):
        elev, azim = VIEW_PRESETS.get(name, VIEW_PRESETS["Isometrica"])
        self.ax.view_init(elev=elev, azim=azim)
        self.canvas.draw_idle()


#: Vista effettivamente usata dall'applicazione.
Mesh3DView = Mesh3DViewGPU if GPU_3D_AVAILABLE else Mesh3DViewMPL


# ===========================================================================
#  Pannello completo (viewer + controlli) da inserire nel Tab 2
# ===========================================================================
class Tumor3DPanel(QWidget):
    """Pannello con il visualizzatore 3D e i relativi controlli."""

    # Emesso a dosimetria completata: alimenta la mappa isodose del Tab 1.
    doseAvailable = pyqtSignal(object)
    # Segmentazione degli organi pronta (maschere + affine della loro griglia).
    organsAvailable = pyqtSignal(dict)
    # Piano completo: mappa di dose e fasci, per l'analisi in sezione.
    planAvailable = pyqtSignal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.source = None          # dati provenienti dal Tab 1
        self.worker = None
        self.dose_worker = None
        self.result = None
        self.dose_result = None
        self._build_ui()

    # ------------------------------------------------------------------
    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        # ---------- Barra comandi ----------
        bar = QHBoxLayout()
        self.btn_build = QPushButton("  Genera Modello 3D")
        self.btn_build.setObjectName("primaryButton")
        self.btn_build.setMinimumHeight(36)
        self.btn_build.setEnabled(False)
        self.btn_build.clicked.connect(self.on_build_clicked)
        bar.addWidget(self.btn_build)

        self.btn_export = QPushButton("Esporta STL")
        self.btn_export.setEnabled(False)
        self.btn_export.clicked.connect(self.on_export_clicked)
        bar.addWidget(self.btn_export)

        self.btn_dose = QPushButton("Calcola Dosimetria")
        self.btn_dose.setEnabled(False)
        self.btn_dose.clicked.connect(self.on_dose_clicked)
        bar.addWidget(self.btn_dose)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setMinimumHeight(36)
        bar.addWidget(self.progress, stretch=1)
        root.addLayout(bar)

        # ---------- Strumenti di visualizzazione ----------
        tools = QHBoxLayout()
        tools.setSpacing(10)

        tools.addWidget(QLabel("Vista:"))
        self.combo_view = QComboBox()
        self.combo_view.addItems(VIEW_PRESETS.keys())
        self.combo_view.currentTextChanged.connect(
            lambda n: self.viewer.set_view(n))
        tools.addWidget(self.combo_view)

        self.chk_tumor = QCheckBox("Tumore")
        self.chk_tumor.setChecked(True)
        self.chk_tumor.setStyleSheet("QCheckBox::indicator:checked "
                                     "{background:#ff8a00; border-color:#ff8a00;}")
        self.chk_tumor.toggled.connect(lambda v: self.viewer.set_tumor_visible(v))
        tools.addWidget(self.chk_tumor)

        # Una casella per organo, con l'indicatore del colore usato nel 3D:
        # la legenda coincide cosi' con il comando.
        self.organ_checks = {}
        for key, name, color, *_ in og.ORGAN_STYLE:
            chk = QCheckBox(name.split(" (")[0])
            chk.setChecked(True)
            chk.setToolTip(name)
            chk.setStyleSheet(
                f"QCheckBox::indicator:checked {{background:{color};"
                f" border-color:{color};}}")
            chk.toggled.connect(
                lambda v, k=key: self.viewer.set_organ_visible(k, v))
            tools.addWidget(chk)
            self.organ_checks[key] = chk
        # Il corpo ha anche il suo slider di trasparenza, gia' presente.
        self.chk_body = self.organ_checks["body"]

        tools.addWidget(QLabel("Trasparenza busto:"))
        self.slider_alpha = QSlider(Qt.Orientation.Horizontal)
        self.slider_alpha.setRange(2, 60)      # 0.02 - 0.60
        self.slider_alpha.setValue(12)
        self.slider_alpha.setMaximumWidth(150)
        self.slider_alpha.valueChanged.connect(
            lambda v: self.viewer.set_body_alpha(v / 100.0))
        tools.addWidget(self.slider_alpha)

        tools.addStretch(1)
        self.lbl_hint = QLabel("Trascina per ruotare · rotella per zoomare · tasto centrale per spostare")
        self.lbl_hint.setObjectName("hintLabel")
        tools.addWidget(self.lbl_hint)
        root.addLayout(tools)

        # ---------- Parametri e visualizzazione della dosimetria ----------
        dose_tools = QHBoxLayout()
        dose_tools.setSpacing(10)

        dose_tools.addWidget(QLabel("Nodi:"))
        self.spin_beams = QSpinBox()
        self.spin_beams.setRange(4, 160)
        self.spin_beams.setValue(40)
        self.spin_beams.setSingleStep(4)
        self.spin_beams.setToolTip("Nodi del robot usati in modalita' "
                                   "isocentro singolo")
        dose_tools.addWidget(self.spin_beams)

        dose_tools.addWidget(QLabel("Budget fasci:"))
        self.spin_budget = QSpinBox()
        self.spin_budget.setRange(6, 300)
        self.spin_budget.setValue(config.PLANNER.beam_budget)
        self.spin_budget.setSingleStep(10)
        self.spin_budget.setToolTip("Numero massimo di fasci del piano "
                                    "conformale (sparsificazione)")
        dose_tools.addWidget(self.spin_budget)

        dose_tools.addWidget(QLabel("Piano:"))
        self.combo_mode = QComboBox()
        self.combo_mode.addItem("Conformale (coni variabili)", "conformal")
        self.combo_mode.addItem("Isocentro singolo (sferico)", "single")
        self.combo_mode.setMinimumWidth(190)
        self.combo_mode.currentIndexChanged.connect(self._sync_mode_controls)
        dose_tools.addWidget(self.combo_mode)

        dose_tools.addWidget(QLabel("Bersaglio:"))
        self.combo_target = QComboBox()
        self.combo_target.addItem("Secondo la tecnica (Tab 2)", "technique")
        self.combo_target.addItem("CTV predetto", "ctv")
        self.combo_target.addItem("CTV + 5 mm", "fixed")
        self.combo_target.addItem("PTV adattivo, solo incertezza", "adaptive")
        self.combo_target.setToolTip(
            "Su cosa si ottimizza la dose. Il DVH giudica sempre il CTV.\n"
            "  Secondo la tecnica: il bersaglio che richiede la gestione del\n"
            "  moto scelta nel Tab 2 (CTV, ITV, CTV + margine d'inseguimento,\n"
            "  PTV adattivo). E' la scelta coerente con la simulazione ROS.\n"
            "  Le altre voci ignorano il respiro: servono per confronto.\n"
            "  CTV + 5 mm: margine uniforme.\n"
            "  PTV adattivo, solo incertezza: il CTV si allarga dove la rete\n"
            "  e' incerta, senza margine di moto.")
        dose_tools.addWidget(self.combo_target)

        self.lbl_cone = QLabel("Collimatore:")
        dose_tools.addWidget(self.lbl_cone)
        self.combo_cone = QComboBox()
        for c in dz.CONE_SIZES:
            self.combo_cone.addItem(f"{c:g} mm", c)
        self.combo_cone.setCurrentIndex(list(dz.CONE_SIZES).index(20.0))
        # In modalita' conformale il cono e' scelto per fascio: il selettore
        # serve solo al piano di riferimento a isocentro singolo.
        self.combo_cone.setEnabled(False)
        self.lbl_cone.setEnabled(False)
        dose_tools.addWidget(self.combo_cone)

        self.chk_beams = QCheckBox("Mostra fasci")
        self.chk_beams.setChecked(True)
        self.chk_beams.toggled.connect(lambda v: self.viewer.set_beams_visible(v))
        dose_tools.addWidget(self.chk_beams)

        self.chk_isodose = QCheckBox("Superficie isodose")
        self.chk_isodose.setChecked(True)
        self.chk_isodose.toggled.connect(
            lambda v: self.viewer.set_isodose_visible(v))
        dose_tools.addWidget(self.chk_isodose)

        dose_tools.addWidget(QLabel("Livello:"))
        self.slider_iso = QSlider(Qt.Orientation.Horizontal)
        self.slider_iso.setRange(20, 95)
        # 80% e' la linea di prescrizione tipica in radiochirurgia.
        self.slider_iso.setValue(80)
        self.slider_iso.setMaximumWidth(140)
        # sliderReleased invece di valueChanged: ogni aggiornamento richiede un
        # Marching Cubes, troppo pesante per essere rifatto ad ogni pixel.
        self.slider_iso.sliderReleased.connect(self.on_isodose_level_changed)
        dose_tools.addWidget(self.slider_iso)

        self.lbl_iso = QLabel("80%")
        self.lbl_iso.setObjectName("sliceLabel")
        self.lbl_iso.setMinimumWidth(42)
        self.slider_iso.valueChanged.connect(
            lambda v: self.lbl_iso.setText(f"{v}%"))
        dose_tools.addWidget(self.lbl_iso)

        dose_tools.addStretch(1)
        root.addLayout(dose_tools)

        # ---------- Viewer + pannello dati ----------
        body = QHBoxLayout()
        body.setSpacing(10)

        view_box = QGroupBox("Modello 3D - Tumore nel busto del paziente")
        vb = QVBoxLayout(view_box)
        self.viewer = Mesh3DView()
        vb.addWidget(self.viewer)
        body.addWidget(view_box, stretch=3)

        info_box = QGroupBox("Geometria")
        ib = QVBoxLayout(info_box)
        info_box.setMinimumWidth(280)
        info_box.setMaximumWidth(340)
        self.lbl_info = QLabel()
        self.lbl_info.setObjectName("infoPanel")
        self.lbl_info.setAlignment(Qt.AlignmentFlag.AlignTop |
                                   Qt.AlignmentFlag.AlignLeft)
        self.lbl_info.setWordWrap(True)
        self.lbl_info.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse)

        # Il pannello contiene ormai volumi degli organi, diagnostica dei cinque
        # stadi del planner e dose a ciascun organo: molto piu' testo di quanto
        # entri in colonna. Senza area scorrevole il contenuto in fondo
        # verrebbe semplicemente troncato, senza alcun indizio che ci sia.
        self.info_scroll = QScrollArea()
        self.info_scroll.setWidget(self.lbl_info)
        self.info_scroll.setWidgetResizable(True)
        self.info_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.info_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.info_scroll.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        ib.addWidget(self.info_scroll, stretch=1)

        self.lbl_wip = QLabel(
            "Work in Progress: analisi dosimetrica e visualizzazione dei fasci "
            "in fase di sviluppo.")
        self.lbl_wip.setObjectName("wipNote")
        self.lbl_wip.setWordWrap(True)
        self.lbl_wip.setAlignment(Qt.AlignmentFlag.AlignCenter)
        ib.addWidget(self.lbl_wip)

        body.addWidget(info_box, stretch=1)
        root.addLayout(body, stretch=1)

        self._render_info_idle()

    # ------------------------------------------------------------------
    def _sync_mode_controls(self):
        """Il cono fisso ha senso solo nel piano a isocentro singolo."""
        single = self.combo_mode.currentData() == "single"
        self.combo_cone.setEnabled(single)
        self.lbl_cone.setEnabled(single)

    def _render_info_idle(self):
        self.lbl_info.setText(
            "<div style='color:#8b95a3;'>"
            "<b style='color:#c8d0da;'>STATO</b><br>"
            "In attesa dei dati dal Tab 1.<br><br>"
            "Caricare una CT ed eseguire l'inferenza,<br>"
            "poi premere <i>Genera Modello 3D</i>."
            "</div>")

    def _render_info_result(self, res):
        tm = res.get("tumor_mesh")
        bm = res.get("body_mesh")
        rows = ["<div style='color:#8b95a3; line-height:150%;'>"]
        if tm is not None:
            dims = tm.bounds[1] - tm.bounds[0]
            rows.append(
                "<b style='color:#c8d0da;'>TUMORE</b><br>"
                f"<span style='color:#ff8a00;'><b>Volume: "
                f"{res['tumor_volume_cm3']:.2f} cm&sup3;</b></span><br>"
                + (f"<span style='color:#6f7a88; font-size:10px;'>"
                   f"{res['tumor_components']} componenti sopra "
                   f"{config.POSTPROC.min_lesion_mm3:.0f} mm&sup3;: mostrata e "
                   f"trattata la maggiore</span><br>"
                   if res.get("tumor_components", 1) > 1 else "") +
                f"Facce mesh : {len(tm.faces):,}<br>"
                f"Chiusa     : {'si' if tm.is_watertight else 'no'}<br>"
                f"Ingombro   : {dims[0]:.1f} x {dims[1]:.1f} x {dims[2]:.1f} mm<br><br>")
        if self.source and self.source.get("com_world") is not None:
            cw = self.source["com_world"]
            rows.append(
                "<b style='color:#c8d0da;'>TARGET (mm)</b><br>"
                f"<span style='color:#00e5ff;'>"
                f"X = {cw[0]:+.2f}<br>Y = {cw[1]:+.2f}<br>Z = {cw[2]:+.2f}"
                "</span><br><br>")
        vols = res.get("organ_volumes") or {}
        if vols:
            rows.append("<b style='color:#c8d0da;'>ORGANI SEGMENTATI</b><br>")
            for key, name, color, *_ in og.ORGAN_STYLE:
                if key in vols and vols[key] > 0:
                    rows.append(
                        f"<span style='color:{color};'>&#9632;</span> "
                        f"{name.split(' (')[0]:<18} {vols[key]:7.0f} cm&sup3;<br>")
            rows.append("<span style='color:#6f7a88; font-size:10px;'>"
                        "Segmentazione per soglia HU e morfologia. Il mediastino "
                        "e' delimitato geometricamente (tessuto molle fra i "
                        "polmoni): la densita' da sola non separa cuore, vasi ed "
                        "esofago.</span><br>")
            # Avviso VISIBILE quando il mediastino ha inglobato tessuto
            # sotto-diaframmatico: da quella maschera esce anche la dose al
            # mediastino, e un volume doppio la rende fuorviante.
            if vols.get("mediastinum", 0.0) > og.MEDIASTINUM_MAX_ML:
                rows.append(
                    f"<span style='color:#ff8c42; font-size:10px;'>"
                    f"Su questo paziente il mediastino misura "
                    f"{vols['mediastinum']:.0f} cm&sup3;, oltre i "
                    f"{og.MEDIASTINUM_MAX_ML:.0f} plausibili: alla quota del "
                    f"diaframma cuore e fegato sono tessuto contiguo della "
                    f"stessa densita'. La dose al mediastino va letta con "
                    f"cautela.</span><br>")
            rows.append("<br>")
        qual = res.get("organ_quality") or {}
        if qual:
            rows.append("<b style='color:#c8d0da;'>QUALITA' DELLA "
                        "RICOSTRUZIONE</b><br>")
            for key, name, color, *_ in og.ORGAN_STYLE:
                q = qual.get(key)
                if not q:
                    continue
                # Verde sotto il millimetro: e' meno della meta' del voxel di
                # ricostruzione, quindi la superficie descrive la CT e non la
                # griglia su cui e' stata campionata.
                col = "#3ddc84" if q["media_mm"] < 1.0 else (
                    "#ffb020" if q["media_mm"] < 1.5 else "#ff5c5c")
                rows.append(
                    f"<span style='color:{color};'>&#9632;</span> "
                    f"{name.split(' (')[0]:<14} "
                    f"<span style='color:{col};'>{q['media_mm']:.2f} mm</span> "
                    f"<span style='color:#6f7a88;'>{q['facce'] // 1000}k facce"
                    f"{'' if q['chiusa'] else ', aperta'}</span><br>")
            rows.append("<span style='color:#6f7a88; font-size:10px;'>"
                        "Scostamento medio fra la superficie ricostruita e "
                        "quella definita dai valori Hounsfield. Ricostruzione a "
                        f"{omsh.MESH_SPACING_MM} mm, "
                        "segmentazione a 2 mm.</span><br><br>")
        if bm is not None:
            dims = bm.bounds[1] - bm.bounds[0]
            rows.append(
                "<b style='color:#c8d0da;'>BUSTO</b><br>"
                f"Ingombro   : {dims[0]:.0f} x {dims[1]:.0f} x {dims[2]:.0f} mm<br>")
        rows.append("</div>")
        self.lbl_info.setText("".join(rows))

    # ------------------------------------------------------------------
    #  API usata dalla MainWindow per passare i dati dal Tab 1
    # ------------------------------------------------------------------
    def set_source_data(self, res: dict):
        """Riceve il risultato dell'inferenza e abilita la generazione 3D."""
        self.source = res
        # Il PTV adattivo esiste solo se il Tab 1 ha misurato l'incertezza.
        item = self.combo_target.model().item(3)
        has_unc = res.get("uncertainty") is not None
        item.setEnabled(has_unc)
        if not has_unc and self.combo_target.currentData() == "adaptive":
            self.combo_target.setCurrentIndex(0)
        # Un piano calcolato su un'altra CT non vale piu'.
        self.dose_result = None
        self.btn_build.setEnabled(True)
        self.lbl_info.setText(
            "<div style='color:#8b95a3;'>"
            "<b style='color:#c8d0da;'>STATO</b><br>"
            f"Dati ricevuti: {os.path.basename(res['image_path'])}<br><br>"
            "Premere <i>Genera Modello 3D</i> per ricostruire<br>"
            "busto e tumore."
            "</div>")

    # ------------------------------------------------------------------
    def on_build_clicked(self):
        if not self.source:
            return
        self.btn_build.setEnabled(False)
        self.btn_export.setEnabled(False)
        self.progress.setValue(0)

        self.worker = MeshBuilderWorker(
            self.source["ct_data"], self.source["mask"], self.source["affine"])
        self.worker.progress.connect(self.progress.setValue)
        self.worker.status.connect(self._on_status)
        self.worker.meshReady.connect(self._on_mesh_ready)
        self.worker.failed.connect(self._on_failed)
        self.worker.finished.connect(lambda: self.btn_build.setEnabled(True))
        self.worker.start()

    def _on_status(self, msg):
        win = self.window()
        if hasattr(win, "statusBar"):
            win.statusBar().showMessage(msg)

    def _on_mesh_ready(self, res):
        self.result = res
        com = self.source.get("com_world") if self.source else None
        self.viewer.set_meshes(res["body_mesh"], res["tumor_mesh"], com,
                               organ_meshes=res.get("organ_meshes"))
        self.viewer.set_body_alpha(self.slider_alpha.value() / 100.0)
        # Riapplica lo stato delle caselle: la scena e' stata ricostruita.
        for key, chk in self.organ_checks.items():
            self.viewer.set_organ_visible(key, chk.isChecked())
        self._render_info_result(res)
        has_tumor = res.get("tumor_mesh") is not None
        self.btn_export.setEnabled(has_tumor)
        self.btn_dose.setEnabled(has_tumor)

        self.organsAvailable.emit(res)

        # Preseleziona il cono piu' piccolo che copre il bersaglio col margine
        # standard: e' la scelta che farebbe un pianificatore.
        if has_tumor and self.source is not None:
            cone = dz.suggest_collimator(self.source["mask"],
                                         self.source["affine"],
                                         self.source["com_world"])
            idx = self.combo_cone.findData(cone)
            if idx >= 0:
                self.combo_cone.setCurrentIndex(idx)

    def _on_failed(self, msg):
        self.progress.setValue(0)
        QMessageBox.critical(self, "Errore nella costruzione del modello 3D", msg)

    # ------------------------------------------------------------------
    #  Dosimetria
    # ------------------------------------------------------------------
    def set_target_provider(self, fn):
        """fn(ctv, affine, maps) -> (maschera, etichetta): la tecnica di
        gestione del moto del Tab 2. Puo' sollevare RuntimeError."""
        self.target_provider = fn

    def _planning_target(self):
        """Maschera su cui ottimizzare, secondo il selettore del bersaglio."""
        import postproc
        import robust_target as rt
        kind = self.combo_target.currentData()
        ctv, _ = postproc.primary_mask(self.source["mask"],
                                       self.source["affine"])
        if kind == "technique":
            fn = getattr(self, "target_provider", None)
            if fn is None:
                return self.source["mask"], "CTV predetto"
            return fn(ctv, self.source["affine"],
                      self.source.get("uncertainty"))
        if kind == "fixed":
            return (rt.fixed_target(ctv, self.source["affine"], 5.0),
                    "CTV + 5 mm")
        if kind == "adaptive" and self.source.get("uncertainty") is not None:
            # Solo l'errore del contorno: il moto ha il suo margine nel Tab 4.
            no_motion = {"Sigma_mm": [0.0] * 3, "sigma_mm": [0.0] * 3}
            ptv, info = rt.adaptive_target(ctv, self.source["uncertainty"],
                                           self.source["affine"], no_motion)
            m = info["margine_mm"]
            return ptv, (f"PTV adattivo (tau {info['tau']:.2f}, margine "
                         f"{m[0]:.1f}/{m[1]:.1f}/{m[2]:.1f} mm)")
        return self.source["mask"], "CTV predetto"

    def on_dose_clicked(self):
        if not self.source or self.source.get("com_world") is None:
            return
        try:
            target, label = self._planning_target()
        except RuntimeError as exc:
            QMessageBox.information(self, "Bersaglio non ancora pronto",
                                    str(exc))
            return
        self.btn_dose.setEnabled(False)
        self.progress.setValue(0)
        self._target_label = label
        self._target_mask = target
        if target is not self.source["mask"]:
            vv = abs(np.linalg.det(self.source["affine"][:3, :3])) / 1000.0
            self._on_status(f"Bersaglio: {label}, "
                            f"{np.count_nonzero(target) * vv:.2f} cm3")

        self.dose_worker = DoseWorker(
            self.source["ct_data"], target, self.source["affine"],
            self.source["com_world"],
            n_beams=self.spin_beams.value(),
            collimator_mm=float(self.combo_cone.currentData()),
            mode=self.combo_mode.currentData(),
            prescription_pct=float(self.slider_iso.value()),
            # La maschera ossea viene dalla segmentazione del modello 3D:
            # serve a penalizzare i fasci che tagliano le coste.
            bone_mask=(self.result or {}).get("organ_masks", {}).get("bones"),
            bone_affine=(self.result or {}).get("organ_affine"),
            # Corpo e braccia servono ai vincoli di fattibilita' dei nodi:
            # quota del lettino e attraversamento degli arti.
            arms_mask=(self.result or {}).get("organ_masks", {}).get("arms"),
            body_mask=(self.result or {}).get("organ_masks", {}).get("body"),
            # Il budget di fasci e' l'unico parametro del planner regolabile
            # dall'interfaccia: tutti gli altri stanno in config.PlannerConfig.
            cfg=config.PLANNER.replace(beam_budget=self.spin_budget.value()),
            # La sala (base del robot + rotazioni del lettino) si decide PRIMA
            # di scegliere i fasci, dentro il worker: altrimenti il planner
            # sceglie fasci che il braccio non puo' eseguire.
            robot_constraints=True,
        )
        self.dose_worker.progress.connect(self.progress.setValue)
        self.dose_worker.status.connect(self._on_status)
        self.dose_worker.doseReady.connect(self._on_dose_ready)
        self.dose_worker.failed.connect(self._on_dose_failed)
        self.dose_worker.finished.connect(
            lambda: self.btn_dose.setEnabled(True))
        self.dose_worker.start()

    def _on_dose_ready(self, res):
        res["target_label"] = getattr(self, "_target_label", "CTV predetto")
        res["target_mask"] = getattr(self, "_target_mask", None)
        # Il tumore (CTV) sulla griglia di dose: la seduta simulata giudica la
        # dose su di lui, non sul bersaglio con i margini.
        if self.source is not None and res.get("ctv_grid") is None:
            import postproc
            ctv, _ = postproc.primary_mask(self.source["mask"],
                                           self.source["affine"])
            res["ctv_grid"] = dz.resample_to_grid(
                ctv.astype(np.float32), self.source["affine"],
                res["dose_pct"].shape, res["grid_affine"], order=0)
            rx = res["metrics"].get("prescription_pct", 80.0)
            sel = res["ctv_grid"] > 0.5
            if sel.any():
                res["metrics"]["copertura_ctv_pct"] = float(
                    100.0 * np.mean(res["dose_pct"][sel] >= rx))
        self.dose_result = res
        level = float(self.slider_iso.value())
        self.viewer.set_beams(res["beams"], visible=self.chk_beams.isChecked())
        self.viewer.set_isodose(res["dose_pct"], res["grid_affine"], level,
                                visible=self.chk_isodose.isChecked())
        self._render_dose_metrics(res, level)
        # Passa la mappa al Tab 1 per la sovrapposizione sulle slice CT.
        self.doseAvailable.emit(res["dose_map"])
        # E il piano completo alla vista in sezione.
        self.planAvailable.emit(res)

    def _on_dose_failed(self, msg):
        self.progress.setValue(0)
        QMessageBox.critical(self, "Errore nel calcolo dosimetrico", msg)

    # ------------------------------------------------------------------
    #  Progetto
    # ------------------------------------------------------------------
    #: Grandezze del piano che si ricalcolano all'apertura invece di salvarle:
    #: la griglia di densita' pesa decine di MB e si rifa' in un secondo.
    _DERIVED = ("red_grid", "dose_map", "tumor_grid", "mask_tutte_le_lesioni",
                "ctv_grid")

    def export_state(self):
        dose = None
        if self.dose_result:
            dose = {k: v for k, v in self.dose_result.items()
                    if k not in self._DERIVED}
            # Dose in centesimi di punto percentuale su 16 bit: errore
            # massimo 0.005 punti, invisibile su DVH e isodose. In float32 le
            # cifre meno significative sono rumore e non si comprimono:
            # misurato su lung_001, la sola dose era meta' del progetto.
            # Le metriche del piano restano quelle originali (salvate a parte).
            d = np.asarray(dose.pop("dose_pct"), np.float32)
            dose["dose_centesimi"] = np.round(
                np.clip(d, 0.0, 655.0) * 100.0).astype(np.uint16)
        return {
            "modello": self.result,
            "piano": dose,
            "ui": {"budget": self.spin_budget.value(),
                   "nodi": self.spin_beams.value(),
                   "modalita": self.combo_mode.currentData(),
                   "bersaglio": self.combo_target.currentData(),
                   "cono": self.combo_cone.currentData(),
                   "livello": self.slider_iso.value()},
        }

    def restore_state(self, st):
        """Riporta modello 3D e piano senza ricalcolarli."""
        if not st:
            return
        ui = st.get("ui") or {}
        if "budget" in ui:
            self.spin_budget.setValue(int(ui["budget"]))
            self.spin_beams.setValue(int(ui["nodi"]))
            for combo, key in ((self.combo_mode, "modalita"),
                               (self.combo_target, "bersaglio"),
                               (self.combo_cone, "cono")):
                i = combo.findData(ui.get(key))
                if i >= 0:
                    combo.setCurrentIndex(i)
            self.slider_iso.setValue(int(ui["livello"]))
        if st.get("modello"):
            self._on_mesh_ready(st["modello"])
        res = st.get("piano")
        if res and self.source is not None:
            import postproc
            if "dose_centesimi" in res:
                res["dose_pct"] = (res.pop("dose_centesimi").astype(np.float32)
                                   / 100.0)
            ct, aff = self.source["ct_data"], self.source["affine"]
            red, gaff = dz.build_red_grid(ct, aff, res["metrics"]["spacing_mm"])
            if not np.allclose(gaff, res["grid_affine"], atol=1e-4):
                raise ValueError("la griglia di calcolo non coincide con "
                                 "quella del piano salvato")
            res["red_grid"] = red
            target = res.get("target_mask")
            if target is None:
                target, _ = postproc.primary_mask(self.source["mask"], aff)
            res["tumor_grid"] = dz.resample_to_grid(
                np.asarray(target, np.float32), aff, red.shape, gaff, order=0)
            res["dose_map"] = dz.DoseMap(res["dose_pct"], gaff, ct.shape, aff)
            res["mask_tutte_le_lesioni"] = self.source["mask"]
            self._target_label = res.get("target_label", "CTV predetto")
            self._target_mask = res.get("target_mask")
            self._on_dose_ready(res)

    def on_isodose_level_changed(self):
        """Ricalcola la sola superficie isodose (nessun ricalcolo di dose)."""
        if not self.dose_result:
            return
        level = float(self.slider_iso.value())
        res = self.dose_result
        self.viewer.set_isodose(res["dose_pct"], res["grid_affine"], level,
                                visible=self.chk_isodose.isChecked())
        # La conformita' dipende dall'isodose considerata: va ricalcolata al
        # nuovo livello, altrimenti il pannello mostrerebbe l'indice di un'altra
        # superficie rispetto a quella disegnata.
        vv = abs(np.linalg.det(np.asarray(res["grid_affine"])[:3, :3]))
        res["metrics"].update(
            dz.conformity_metrics(res["dose_pct"], res["tumor_grid"], level, vv))
        self._render_dose_metrics(res, level)

    def _render_dose_metrics(self, res, level):
        """Pannello con gli indicatori di qualita' del piano."""
        m = res["metrics"]
        tum = res["tumor_grid"] > 0.5
        cov = float(((res["dose_pct"] >= level) & tum).sum()) / max(int(tum.sum()), 1)

        def color(v, good, ok_):
            return "#3ddc84" if v >= good else ("#ffb020" if v >= ok_ else "#ff5c5c")

        cones = m.get("cones_mm", [])
        cones_txt = " / ".join(f"{c:g}" for c in cones) + " mm" if cones else "-"
        conformal = m.get("mode") == "conformal"
        ci = m.get("conformity_index", 0.0)

        self.lbl_info.setText(
            "<div style='color:#8b95a3; line-height:150%;'>"
            "<b style='color:#c8d0da;'>PIANO DI TRATTAMENTO</b><br>"
            f"Modalita'    : {'conformale' if conformal else 'isocentro singolo'}<br>"
            f"Fasci attivi : {m['n_beams']}"
            + (f" / {m.get('candidati', '?')} candidati<br>" if conformal else "<br>")
            + (f"Punti di mira: {m.get('n_targets', 1)}<br>" if conformal else "")
            + f"Coni usati   : {cones_txt}<br>"
            + (f"Rapporto pesi: {m.get('weight_ratio', 1):.0f}x<br>" if conformal else "")
            + "<br>"

            "<b style='color:#c8d0da;'>CONFORMITA'</b><br>"
            f"<span style='color:{color(ci * 100, 75, 55)};'><b>"
            f"Paddick CI = {ci:.3f}</b></span><br>"
            "<span style='color:#6f7a88; font-size:11px;'>1.0 = volume irradiato "
            "coincidente col tumore</span><br>"
            f"Volume irradiato : {m.get('piv_cm3', 0):.2f} cm&sup3;<br>"
            f"Volume tumore    : {m.get('tv_cm3', 0):.2f} cm&sup3;<br>"
            f"<span style='color:{color(100 - m.get('spill_cm3', 0) * 40, 75, 40)};'>"
            f"Spill su sano    : {m.get('spill_cm3', 0):.2f} cm&sup3;</span><br><br>"

            "<b style='color:#c8d0da;'>COPERTURA</b><br>"
            f"<span style='color:{color(cov * 100, 95, 85)};'>"
            f"Isodose {level:.0f}% : {cov * 100:.1f}% del bersaglio</span><br>"
            f"Dose media  : {m['d_mean_pct']:.0f}%<br>"
            f"Omogeneita' : {m.get('homogeneity_index', 0):.2f}"
            "<span style='color:#6f7a88; font-size:11px;'> (max/prescr.)</span>"
            "<br><br>"

            + self._planner_html(m) +

            "<b style='color:#c8d0da;'>PERCORSO NEI TESSUTI</b><br>"
            f"Profondita' media : {m.get('path_mean_cm', 0):.1f} cm<br>"
            f"Massima           : {m.get('path_max_cm', 0):.1f} cm<br>"
            f"<span style='color:{color(100 - m.get('bone_mean_cm', 0) * 60, 70, 40)};'>"
            f"Osso attraversato : {m.get('bone_mean_cm', 0):.2f} cm</span>"
            "<span style='color:#6f7a88; font-size:11px;'> (media pesata)</span>"
            "<br><br>"

            + self._organ_dose_html(res) +

            "<span style='color:#6f7a88; font-size:10px;'>Dosi in percentuale "
            "della prescrizione. Il piano e' normalizzato perche' l'isodose di "
            "prescrizione copra il bersaglio.</span>"
            "</div>")

    @staticmethod
    def _delivery_html(m):
        """Esito del collaudo con la sequenza del robot (dose_worker)."""
        n_ok, n = m.get("fasci_erogabili"), m.get("n_beams")
        if n_ok is None or not n:
            return ""
        color = "#3ddc84" if n_ok == n else "#ff5c5c"
        note = ("" if n_ok == n else
                "<br><span style='color:#ff5c5c; font-size:11px;'>la "
                "simulazione ROS non erogera' tutti i fasci: la copertura "
                "finale sara' inferiore a quella pianificata</span>")
        return (f"<span style='color:{color};'>Erogabili in sequenza : "
                f"{n_ok} / {n}</span>{note}<br>")

    def _planner_html(self, m):
        """Diagnostica del planner: zone escluse e resa di ciascuno stadio."""
        if "nodi_ammessi" not in m:
            return ""
        gen = m.get("nodi_generati", 0)
        if m.get("vincoli_robot"):
            sala = m.get("sala") or {}
            robot_html = (
                "<b style='color:#c8d0da;'>ROBOT</b><br>"
                f"<span style='color:#3ddc84;'>Candidati erogabili : "
                f"{m.get('candidati_erogabili', '-')} / "
                f"{m.get('candidati', '-')}</span><br>"
                f"Rotazioni lettino   : {m.get('angoli_lettino_usati', '-')}<br>"
                + self._delivery_html(m) +
                f"<span style='color:#6f7a88; font-size:10px;'>i fasci sono "
                f"scelti solo fra quelli che il braccio esegue; direzioni "
                f"eseguibili {sala.get('copertura_direzioni', '?')}/"
                f"{sala.get('direzioni_provate', '?')} contro "
                f"{sala.get('copertura_lettino_fermo', '?')} a lettino "
                f"fermo</span><br><br>")
        else:
            robot_html = (
                "<b style='color:#ff5c5c;'>ROBOT: VINCOLI NON APPLICATI</b><br>"
                "<span style='color:#ff5c5c; font-size:11px;'>I fasci sono "
                "stati scelti senza sapere se il braccio li puo' eseguire: "
                "una parte andra' persa nel Tab 4.</span><br><br>")
        rows = [robot_html,
            "<b style='color:#c8d0da;'>ZONE PROIBITE (nodi esclusi)</b><br>"
            f"Lettino (cono post.) : {m.get('scartati_lettino', 0)}<br>"
            f"Testa / gambe        : {m.get('scartati_testa_gambe', 0)}<br>"
            f"Sotto il piano letto : {m.get('scartati_piano_lettino', 0)}<br>"
            f"Braccia              : {m.get('scartati_braccia', 0)}<br>"
            f"<span style='color:#3ddc84;'>Nodi ammessi         : "
            f"{m['nodi_ammessi']} / {gen}</span><br>"
            f"<span style='color:#6f7a88; font-size:10px;'>quota lettino "
            f"{m.get('quota_lettino_mm', '?')} mm</span><br><br>",

            "<b style='color:#c8d0da;'>STADI DI OTTIMIZZAZIONE</b><br>"
            f"Mire (sfere)   : {m.get('punti_mira', '-')}"
            f" &rarr; usate {m.get('mire_usate', '-')}<br>"
            f"Candidati      : {m.get('candidati', '-')}<br>"
            f"Punti obiettivo: {m.get('punti_obiettivo', '-')}<br>"
            f"Sparsificati a : {m.get('selezionati', '-')}<br>"
            f"Nodi impiegati : {m.get('nodi_usati', '-')}<br>",
        ]
        sp = m.get("sparsificazione") or []
        if sp:
            rows.append(
                f"<span style='color:#6f7a88; font-size:10px;'>deficit "
                f"{sp[0]['deficit_pct']:.1f}% su {sp[0]['fasci']} fasci &rarr; "
                f"{sp[-1]['deficit_pct']:.1f}% su {sp[-1]['fasci']}</span><br>")
        rf = m.get("rifinitura") or []
        if rf:
            rows.append(
                f"<span style='color:#6f7a88; font-size:10px;'>rifinitura: dose "
                f"minima dal {100 * rf[0]['d_min_rel']:.0f}% al "
                f"{100 * rf[-1]['d_min_rel']:.0f}% della prescrizione</span><br>")
        rows.append("<br>")
        return "".join(rows)

    def _organ_dose_html(self, res):
        """Dose agli organi a rischio, calcolata sulle maschere del Tab 3D."""
        if not self.result or not self.result.get("organ_masks"):
            return ""
        try:
            stats = og.organ_dose_stats(
                self.result["organ_masks"], self.result["organ_affine"],
                res["dose_pct"], res["grid_affine"])
        except Exception:
            return ""

        rows = ["<b style='color:#c8d0da;'>DOSE AGLI ORGANI</b><br>"
                "<span style='color:#6f7a88; font-size:11px;'>"
                "massima / media, in % della prescrizione</span><br>"]
        for key, name, color, *_ in og.ORGAN_STYLE:
            s = stats.get(key)
            if not s or key == "body":
                continue
            col = ("#ff5c5c" if s["d_max_pct"] >= 100 else
                   "#ffb020" if s["d_max_pct"] >= 50 else "#8b95a3")
            rows.append(
                f"<span style='color:{color};'>&#9632;</span> "
                f"{name.split(' (')[0]:<16} "
                f"<span style='color:{col};'>{s['d_max_pct']:5.0f}%</span>"
                f" / {s['d_mean_pct']:.0f}%<br>")
        rows.append("<br>")
        return "".join(rows)

    # ------------------------------------------------------------------
    def on_export_clicked(self):
        """Esporta le mesh in STL (formato standard per CAD e simulatori)."""
        if not self.result:
            return
        base = os.path.splitext(os.path.basename(
            self.source["image_path"]))[0].replace(".nii", "")
        default = os.path.join(config.OUTPUT_DIR, f"{base}_tumor.stl")
        path, _ = QFileDialog.getSaveFileName(
            self, "Esporta mesh del tumore", default, "Mesh STL (*.stl)")
        if not path:
            return
        self.result["tumor_mesh"].export(path)

        saved = [path]
        # Esporta anche il busto come file separato, utile per la scena 3D.
        if self.result.get("body_mesh") is not None:
            body_path = path.replace(".stl", "_busto.stl")
            self.result["body_mesh"].export(body_path)
            saved.append(body_path)

        QMessageBox.information(self, "Esportazione completata",
                                "File salvati:\n" + "\n".join(saved))
