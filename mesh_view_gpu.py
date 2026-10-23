# ============================================================================
#  mesh_view_gpu.py - Vista 3D accelerata su GPU (PyVista / VTK).
#
#  PERCHE' SOSTITUISCE MATPLOTLIB 3D
#  ---------------------------------
#  Matplotlib non ha un motore 3D: proietta i poligoni e li ordina via software
#  (painter's algorithm) a ogni fotogramma. Il costo cresce col numero di facce
#  e non usa la scheda video. Sulla scena reale del progetto - 28.000 facce di
#  organi, 112 fasci, una superficie isodose - si misuravano 7-8 fps, cioe' una
#  rotazione a scatti.
#
#  VTK rasterizza in OpenGL sulla GPU. Oltre alla velocita' cambia la qualita':
#    - ILLUMINAZIONE reale con normali per vertice, invece di facce a tinta
#      piatta: la forma degli organi si legge dalle ombreggiature;
#    - DEPTH PEELING per la trasparenza. Qui e' decisivo, perche' corpo,
#      polmoni, mediastino e isodose sono tutti semitrasparenti e sovrapposti:
#      il painter's algorithm li compone in ordine sbagliato quando si
#      compenetrano, VTK li risolve pixel per pixel;
#    - ANTIALIASING, e una camera trackball con zoom a rotella nativo.
#
#  L'interfaccia pubblica e' identica a quella della vista Matplotlib, cosi'
#  il resto dell'applicazione non cambia e resta possibile ricadere sulla
#  vecchia implementazione se OpenGL non e' disponibile.
# ============================================================================
import numpy as np

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QSizePolicy

import pyvista as pv
from pyvistaqt import QtInteractor

from skimage import measure
import trimesh
from nibabel.affines import apply_affine

# Sfondo della scena, coerente col tema scuro dell'applicazione.
BG_TOP = "#141b22"
BG_BOTTOM = "#080b0e"

# Numero di passate di depth peeling: piu' passate = trasparenze piu' corrette
# fra superfici che si compenetrano, a un costo trascurabile su GPU dedicata.
DEPTH_PEELS = 12

TUMOR_COLOR = "#ff8a00"
BEAM_COLOR = "#00e5ff"
ISODOSE_COLOR = "#3ddc84"
MARKER_COLOR = "#00e5ff"

VIEW_VECTORS = {
    # nome            direzione di osservazione      alto
    "Isometrica": ((0.55, 0.62, 0.56), (0.0, 0.0, 1.0)),
    "Anteriore":  ((0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
    "Laterale":   ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
    "Superiore":  ((0.0, 0.0, 1.0), (0.0, 1.0, 0.0)),
}


def trimesh_to_pv(mesh):
    """Converte una mesh trimesh in PolyData di VTK.

    VTK vuole le facce in formato "conteggio + indici": ogni triangolo e'
    preceduto dal numero 3.
    """
    faces = np.empty((len(mesh.faces), 4), dtype=np.int64)
    faces[:, 0] = 3
    faces[:, 1:] = mesh.faces
    return pv.PolyData(np.asarray(mesh.vertices, dtype=np.float32),
                       faces.ravel())


class Mesh3DViewGPU(QWidget):
    """Scena 3D del paziente: organi, tumore, fasci e superficie isodose."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.body_mesh = None
        self.tumor_mesh = None
        self.com_world = None
        self.organ_colls = {}        # chiave organo -> attore VTK
        self._tumor_actor = None
        self._beam_actor = None
        self._isodose_actor = None
        self._body_key = "body"
        self._placeholder_actor = None

        self.plotter = QtInteractor(self)
        self.plotter.set_background(BG_BOTTOM, top=BG_TOP)
        self.plotter.setSizePolicy(QSizePolicy.Policy.Expanding,
                                   QSizePolicy.Policy.Expanding)
        try:
            self.plotter.enable_depth_peeling(DEPTH_PEELS)
        except Exception:
            pass                      # su driver poveri si continua senza
        try:
            self.plotter.enable_anti_aliasing("fxaa")
        except Exception:
            pass
        # Terna di orientamento in basso a destra: con una camera libera serve
        # un riferimento costante per sapere da che parte si sta guardando.
        try:
            self.plotter.add_axes(color="#8b95a3", labels_off=False)
        except Exception:
            pass

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.plotter.interactor)
        self._show_placeholder()

    # ------------------------------------------------------------------
    def _show_placeholder(self):
        try:
            self._placeholder_actor = self.plotter.add_text(
                "Nessun modello 3D generato.\n"
                "Eseguire l'inferenza nel Tab 1, poi \"Genera Modello 3D\".",
                position="upper_left", font_size=9, color="#5c6672")
        except Exception:
            self._placeholder_actor = None

    def _clear_scene(self):
        self.plotter.clear()
        self.organ_colls = {}
        self._tumor_actor = None
        self._beam_actor = None
        self._isodose_actor = None
        self._placeholder_actor = None

    # ==================================================================
    #  Costruzione della scena
    # ==================================================================
    def set_meshes(self, body_mesh, tumor_mesh, com_world=None,
                   organ_meshes=None):
        """Disegna organi e tumore, tutto in millimetri nello spazio paziente."""
        self.body_mesh = body_mesh
        self.tumor_mesh = tumor_mesh
        self.com_world = None if com_world is None else np.asarray(com_world,
                                                                   dtype=float)
        self._clear_scene()

        # --- Organi, dal piu' interno al piu' esterno ---
        for org in (organ_meshes or []):
            mesh = org.get("mesh")
            if mesh is None or not len(mesh.faces):
                continue
            actor = self.plotter.add_mesh(
                trimesh_to_pv(mesh), color=org["color"], opacity=org["alpha"],
                smooth_shading=True, specular=0.25, specular_power=18,
                ambient=0.22, diffuse=0.75, name=f"organ_{org['key']}")
            self.organ_colls[org["key"]] = actor

        # Compatibilita': se non arrivano organi si disegna il solo profilo.
        if "body" not in self.organ_colls and body_mesh is not None \
                and len(body_mesh.faces):
            self.organ_colls["body"] = self.plotter.add_mesh(
                trimesh_to_pv(body_mesh), color="#7fb6d6", opacity=0.10,
                smooth_shading=True, name="organ_body")

        # --- Tumore: unico oggetto opaco, e' il bersaglio ---
        if tumor_mesh is not None and len(tumor_mesh.faces):
            self._tumor_actor = self.plotter.add_mesh(
                trimesh_to_pv(tumor_mesh), color=TUMOR_COLOR, opacity=1.0,
                smooth_shading=True, specular=0.45, specular_power=30,
                name="tumor")

        # --- Centro di massa: il punto che va al robot ---
        if self.com_world is not None:
            self.plotter.add_mesh(
                pv.Sphere(radius=2.2, center=self.com_world),
                color=MARKER_COLOR, name="target_marker")

        self.set_view("Isometrica")
        self.plotter.reset_camera()
        self.render()

    # ------------------------------------------------------------------
    def set_beams(self, beams, visible=True):
        """Fasci come segmenti verso il rispettivo bersaglio.

        Tutti i fasci finiscono in UN solo attore: 112 attori separati
        costerebbero altrettante chiamate di disegno, mentre un'unica PolyData
        con il peso come scalare si colora da sola e si disegna in un colpo.
        """
        self.plotter.remove_actor("beams", render=False)
        self._beam_actor = None
        if not beams:
            self.render()
            return

        weights = np.array([b.weight for b in beams], dtype=float)
        wmax = max(float(weights.max()), 1e-12)

        pts, lines, scal = [], [], []
        for b, w in zip(beams, weights):
            d = np.asarray(b.target, float) - np.asarray(b.source, float)
            n = float(np.linalg.norm(d))
            if n < 1e-6:
                continue
            # Solo l'ultimo tratto verso il bersaglio: il segmento intero dal
            # nodo (800 mm) uscirebbe dall'inquadratura del busto.
            start = np.asarray(b.target, float) - (d / n) * min(n, 260.0)
            i = len(pts)
            pts.extend([start, np.asarray(b.target, float)])
            lines.extend([2, i, i + 1])
            scal.extend([w / wmax, w / wmax])

        if not pts:
            self.render()
            return
        poly = pv.PolyData(np.asarray(pts, dtype=np.float32))
        poly.lines = np.asarray(lines, dtype=np.int64)
        poly["peso"] = np.asarray(scal, dtype=np.float32)

        self._beam_actor = self.plotter.add_mesh(
            poly, scalars="peso", cmap=["#0d5f6b", BEAM_COLOR, "#d8fbff"],
            clim=(0.0, 1.0), line_width=2, opacity=0.55,
            show_scalar_bar=False, name="beams")
        self._beam_actor.SetVisibility(bool(visible))
        self.render()

    # ------------------------------------------------------------------
    def set_isodose(self, dose_pct, grid_affine, level=50.0, visible=True):
        """Superficie che racchiude la regione sopra la soglia di dose."""
        self.plotter.remove_actor("isodose", render=False)
        self._isodose_actor = None
        if dose_pct is None or not (dose_pct >= level).any():
            self.render()
            return
        try:
            verts, faces, _, _ = measure.marching_cubes(
                dose_pct.astype(np.float32), level=float(level))
        except (ValueError, RuntimeError):
            self.render()
            return

        verts_mm = apply_affine(np.asarray(grid_affine, dtype=np.float64), verts)
        mesh = trimesh.Trimesh(vertices=verts_mm, faces=faces, process=True)
        # Nessuna decimazione aggressiva: la GPU regge senza problemi decine di
        # migliaia di triangoli, e la superficie resta liscia.
        if len(mesh.faces) > 60000:
            try:
                mesh = mesh.simplify_quadric_decimation(face_count=60000)
            except Exception:
                pass

        self._isodose_actor = self.plotter.add_mesh(
            trimesh_to_pv(mesh), color=ISODOSE_COLOR, opacity=0.25,
            smooth_shading=True, specular=0.2, name="isodose")
        self._isodose_actor.SetVisibility(bool(visible))
        self.render()

    # ==================================================================
    #  Visibilita' e camera
    # ==================================================================
    def _set_actor_visible(self, actor, visible):
        if actor is not None:
            actor.SetVisibility(bool(visible))
            self.render()

    def set_organ_visible(self, key, visible):
        self._set_actor_visible(self.organ_colls.get(key), visible)

    def is_organ_visible(self, key):
        actor = self.organ_colls.get(key)
        return bool(actor.GetVisibility()) if actor is not None else False

    def set_body_visible(self, visible):
        self.set_organ_visible(self._body_key, visible)

    def set_tumor_visible(self, visible):
        self._set_actor_visible(self._tumor_actor, visible)

    def set_beams_visible(self, visible):
        self._set_actor_visible(self._beam_actor, visible)

    def set_isodose_visible(self, visible):
        self._set_actor_visible(self._isodose_actor, visible)

    def set_body_alpha(self, alpha):
        actor = self.organ_colls.get(self._body_key)
        if actor is not None:
            actor.GetProperty().SetOpacity(float(alpha))
            self.render()

    def set_view(self, name):
        vec, up = VIEW_VECTORS.get(name, VIEW_VECTORS["Isometrica"])
        try:
            self.plotter.view_vector(vec, viewup=up)
            self.plotter.reset_camera()
        except Exception:
            pass
        self.render()

    def reset_view(self):
        self.plotter.reset_camera()
        self.render()

    def render(self):
        """Ridisegna, ignorando l'assenza di un contesto OpenGL.

        In esecuzione headless (test) la finestra di rendering non esiste: la
        scena si costruisce comunque e resta verificabile, ma non la si puo'
        rasterizzare.
        """
        try:
            self.plotter.render()
        except Exception:
            pass
