# ============================================================================
# dvh_view.py - Vista dell'Istogramma Dose-Volume (DVH)
# ============================================================================
import numpy as np
import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from PyQt6.QtWidgets import QWidget, QVBoxLayout, QSizePolicy
from dosimetry import resample_to_grid

#: Soglie sulla probabilita' media che delimitano la banda di incertezza.
BAND_TAUS = (0.3, 0.5, 0.7)


def _primary_at(prob, affine, tau):
    """Lesione principale della mappa di probabilita' alla soglia tau."""
    import config
    import postproc
    cfg = config.POSTPROC.replace(prob_threshold=tau)
    mask, lesions = postproc.probs_to_mask(prob, cfg, affine=affine)
    if not lesions:
        return None, None
    return postproc.primary_mask(mask, affine)

class DvhView(QWidget):
    """Pannello per la visualizzazione dell'Istogramma Dose-Volume (DVH)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.fig = Figure(figsize=(6, 4), facecolor="#0b0e12")
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding,
                                  QSizePolicy.Policy.Expanding)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor("#151a22")
        self.fig.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.15)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addWidget(self.canvas)
        
        self._draw_placeholder()

    def _draw_placeholder(self):
        self.ax.clear()
        self.ax.set_facecolor("#151a22")
        self.ax.text(0.5, 0.5, "In attesa del calcolo dosimetrico", 
                     ha="center", va="center", color="#4a535f", 
                     fontsize=10, transform=self.ax.transAxes)
        self.ax.set_title("Istogramma Dose-Volume (DVH)", color="#c8d0da", fontsize=12, pad=10)
        self.ax.axis("off")
        self.canvas.draw_idle()

    def _compute_dvh(self, dose_grid, mask_on_dose_grid, max_dose=120.0, bins=200):
        if not mask_on_dose_grid.any():
            return np.array([]), np.array([])
            
        roi_doses = dose_grid[mask_on_dose_grid > 0.5]
        max_d = max(float(roi_doses.max()), max_dose)
        
        hist, edges = np.histogram(roi_doses, bins=bins, range=(0, max_d))
        vol_cum = np.cumsum(hist[::-1])[::-1]
        vol_pct = (vol_cum / roi_doses.size) * 100.0
        
        doses = edges[:-1]
        return doses, vol_pct

    def set_data(self, plan_res, source_res, organ_res):
        """Calcola e disegna il DVH sulla base del piano calcolato."""
        if not plan_res or not plan_res.get("dose_pct") is not None:
            self._draw_placeholder()
            return
            
        dose_pct = plan_res["dose_pct"]
        grid_affine = plan_res["grid_affine"]
        grid_shape = plan_res["grid_shape"]
        prescription_pct = plan_res.get("metrics", {}).get("prescription_pct", 80.0)
        
        self.ax.clear()
        self.ax.set_facecolor("#151a22")
        self.ax.grid(True, color="#2c3643", linestyle="--", alpha=0.7)
        self.ax.set_title("Istogramma Dose-Volume (DVH)", color="#c8d0da", fontsize=12, pad=10)
        self.ax.set_xlabel("Dose (% della prescrizione)", color="#c8d0da", fontsize=10)
        self.ax.set_ylabel("Volume (%)", color="#c8d0da", fontsize=10)
        self.ax.tick_params(colors="#8b95a3", labelsize=9)
        
        for spine in self.ax.spines.values():
            spine.set_color("#2c3643")
            
        max_plot_dose = 120.0
        
        # 1. Tumore
        if source_res and source_res.get("mask") is not None:
            aff = source_res["affine"]
            dose_top = float(max(dose_pct.max(), max_plot_dose))

            def dvh_of(mask):
                g = resample_to_grid(np.asarray(mask, np.float32), aff,
                                     grid_shape, grid_affine, order=0)
                # Stesso estremo per tutte le curve: la banda confronta
                # istogrammi e i bin devono coincidere.
                return self._compute_dvh(dose_pct, g,
                                         max_dose=dose_top)

            # Banda di incertezza del contorno: DVH dei tumori che si
            # otterrebbero con soglie diverse sulla probabilita' media delle
            # predizioni perturbate. Dice quanto la copertura dipende da
            # DOVE sta davvero il bordo, cioe' da cio' che la rete non sa.
            unc = source_res.get("uncertainty")
            if unc is not None:
                curves = []
                for tau in BAND_TAUS:
                    m, _ = _primary_at(unc["p_mean"], aff, tau)
                    if m is not None:
                        d, v = dvh_of(m)
                        if len(d):
                            curves.append(v)
                if len(curves) >= 2:
                    V = np.stack(curves)
                    self.ax.fill_between(
                        d, V.min(0), V.max(0), color="#ff3333", alpha=0.18,
                        linewidth=0,
                        label=f"Tumore, soglie {BAND_TAUS[0]:.1f}-"
                              f"{BAND_TAUS[-1]:.1f}")

            doses, vol_pct = dvh_of(source_res["mask"])
            if len(doses):
                self.ax.plot(doses, vol_pct, color="#ff3333", linewidth=2.5, label="Tumore")
                max_plot_dose = max(max_plot_dose, doses.max())

            ptv = plan_res.get("target_mask")
            if ptv is not None and ptv is not source_res["mask"]:
                d, v = dvh_of(ptv)
                if len(d):
                    self.ax.plot(d, v, color="#ff9f9f", linewidth=1.4,
                                 linestyle="--",
                                 label=plan_res.get("target_label", "PTV"))
                
        # 2. Organi a rischio
        if organ_res and organ_res.get("organ_masks") and organ_res.get("organ_affine") is not None:
            masks = organ_res["organ_masks"]
            aff = organ_res["organ_affine"]
            
            # I polmoni sono due maschere (organs.segment_organs): la chiave
            # "lungs" cercata prima non esisteva, e la curva non compariva.
            masks = dict(masks)
            if "lung_r" in masks and "lung_l" in masks:
                masks["lungs"] = masks["lung_r"] | masks["lung_l"]
            colors = {
                "lungs": ("#33ccff", "Polmoni"),
                "bones": ("#ffcc00", "Scheletro"),
                "mediastinum": ("#33ff33", "Mediastino"),
                "airways": ("#ff33cc", "Vie Aeree"),
                "body": ("#808080", "Corpo (Tessuto Sano)")
            }
            
            for key, (color, name) in colors.items():
                if key in masks and masks[key] is not None:
                    # Rimuoviamo il tumore dal calcolo dell'organo per avere l'OAR puro
                    if key == "body":
                         # Opzionale: calcolare il corpo meno il bersaglio
                         pass
                    
                    organ_grid = resample_to_grid(
                        masks[key].astype(np.float32), aff,
                        grid_shape, grid_affine, order=0
                    )
                    d, v = self._compute_dvh(dose_pct, organ_grid, max_dose=max_plot_dose)
                    if len(d):
                        self.ax.plot(d, v, color=color, linewidth=1.5, label=name)
        
        # Linea verticale per la prescrizione (es. 100% in scala relativa, che equivale alla prescription)
        # Attenzione: l'asse X e' gia' espresso in % della prescrizione grazie al calcolo in dosimetry.py 
        # (dose_pct = (dose / d_ref) * prescription_pct). 
        # Quindi la dose che copre il bersaglio all'isodose scelta si trova a `prescription_pct`.
        self.ax.axvline(x=prescription_pct, color="#ffffff", linestyle=":", alpha=0.5, 
                        label=f"Prescrizione ({prescription_pct}%)")

        # Imposta i limiti
        self.ax.set_xlim(0, min(max_plot_dose, 150.0))
        self.ax.set_ylim(0, 105)
        
        # Legenda
        leg = self.ax.legend(loc="upper right", facecolor="#151a22", edgecolor="#2c3643", 
                             fontsize=9, labelcolor="#c8d0da")
        
        self.canvas.draw_idle()
