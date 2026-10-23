# ============================================================================
#  report_view.py - Il resoconto di fine seduta, con il replay del tumore.
#
#  Si apre da solo quando la simulazione ROS termina (Tab 3), oppure a mano
#  con "Apri resoconto". Mostra:
#
#    - la SINTESI della seduta: fasci erogati, copertura finale contro quella
#      pianificata, tempi, rotazioni del lettino, margini e franchi;
#    - il REPLAY del bersaglio: gli stessi voxel colorati della finestra RViz,
#      dal rosso al verde, che si riempiono fascio dopo fascio. Si puo'
#      mettere in pausa, scorrere, accelerare, saltare a un fascio;
#    - l'ANDAMENTO: copertura, D95, dose minima e media dopo ogni fascio, con
#      il contributo di ciascuno; la linea temporale di trasferimenti,
#      erogazioni e rotazioni del lettino; margini, franchi e giunti del robot;
#    - la TABELLA dei fasci, nell'ordine di erogazione. Un clic su una riga
#      porta il replay a quel fascio.
#
#  Il replay non legge dosi salvate dal nodo: le ricostruisce dal piano e
#  dall'ordine di erogazione con dose_fill, la stessa ripartizione usata in
#  RViz (vedi treatment_log.py). Cursore del replay e cursori dei grafici
#  sono sincronizzati.
# ============================================================================
import os

import numpy as np

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QSlider,
    QComboBox, QTabWidget, QWidget, QTableWidget, QTableWidgetItem,
    QSplitter, QHeaderView, QFileDialog, QAbstractItemView, QSizePolicy,
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

import dose_fill as dfm
import treatment_log as tlog

#: Passi del cursore per ogni fascio: il replay interpola DENTRO l'erogazione,
#: come faceva il nodo, invece di saltare da un fascio all'altro.
SUB = 20
#: Fotogrammi al secondo del replay.
FPS = 30
#: Tratto del fascio disegnato attorno al bersaglio (mm). La sorgente sta a
#: 800 mm: disegnarlo intero costringerebbe la camera ad allontanarsi tanto
#: da non vedere piu' i voxel.
BEAM_BEFORE_MM, BEAM_AFTER_MM = 55.0, 30.0

BG = "#12141a"
FG = "#c9d1d9"
MUTED = "#8b949e"
GRID = "#30363d"
GOOD, WARN, BAD = "#3fb950", "#d29922", "#f85149"
COUCH_COLORS = ["#58a6ff", "#bc8cff", "#39c5cf", "#d29922", "#f778ba",
                "#3fb950"]

try:                                         # vista GPU, come nel Tab 2
    import pyvista as pv
    from pyvistaqt import QtInteractor
    GPU_OK = True
except Exception:                            # noqa: BLE001
    GPU_OK = False


def _style(ax):
    ax.set_facecolor(BG)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(True, color=GRID, lw=0.4, alpha=0.6)
    for sp in ax.spines.values():
        sp.set_color(GRID)
    ax.xaxis.label.set_color(FG)
    ax.yaxis.label.set_color(FG)
    ax.title.set_color(FG)


def _fmt(v, suf="", nd=1):
    return "-" if v is None else f"{v:.{nd}f}{suf}"


# ---------------------------------------------------------------------------
class ReplayView(QWidget):
    """Il bersaglio in 3D, voxel per voxel, con il fascio acceso."""

    def __init__(self, dose_fill, report, beams, shell_mesh=None, parent=None):
        super().__init__(parent)
        self.df = dose_fill
        self.report = report
        self.beams = beams
        self.steps = tlog.replay_doses(dose_fill, report, beams)
        self.order = [int(r["fascio"]) for r in report["fasci"]]
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.gpu = GPU_OK
        if self.gpu:
            try:
                self._build_gpu(shell_mesh)
                lay.addWidget(self.plotter.interactor)
            except Exception:                # noqa: BLE001
                self.gpu = False
        if not self.gpu:
            self._build_mpl()
            lay.addWidget(self.canvas)

    # -- stato a una posizione del cursore ---------------------------------
    def dose_at(self, pos):
        """Dose dei voxel alla posizione `pos` (in fasci, frazionaria)."""
        n = len(self.order)
        pos = float(np.clip(pos, 0.0, n))
        k = int(np.floor(pos))
        if k >= n:
            return self.steps[n], n - 1, 1.0
        u = pos - k
        return self.steps[k] + u * (self.steps[k + 1] - self.steps[k]), k, u

    # -- versione GPU ------------------------------------------------------
    def _build_gpu(self, shell_mesh):
        self.plotter = QtInteractor(self)
        self.plotter.set_background("#080b0e", top="#141b22")
        self.plotter.enable_depth_peeling(12)
        self.plotter.enable_anti_aliasing("ssaa")
        side = self.df.step_mm * 0.92
        cloud = pv.PolyData(np.asarray(self.df.points, dtype=np.float32))
        cube = pv.Cube(x_length=side, y_length=side, z_length=side)
        self._cells_per = cube.n_cells
        self.voxels = cloud.glyph(geom=cube, scale=False, orient=False)
        self._set_colors(self.steps[0])
        self.plotter.add_mesh(self.voxels, scalars="rgba", rgba=True,
                              show_edges=False, lighting=True,
                              name="voxel")
        if shell_mesh is not None:
            f = np.asarray(shell_mesh.faces)
            shell = pv.PolyData(np.asarray(shell_mesh.vertices, np.float32),
                                np.hstack([np.full((len(f), 1), 3), f])
                                .ravel())
            self.plotter.add_mesh(shell, color="#dfe6ee", opacity=0.10,
                                  smooth_shading=True, name="involucro")
        self._beam_actor = None
        self._text = self.plotter.add_text("", position="upper_left",
                                           font_size=10, color="#e6edf3")
        c = np.asarray(self.df.points).mean(axis=0)
        self.plotter.camera.focal_point = c
        self.plotter.camera.position = c + np.array([90.0, -70.0, 60.0])
        self.plotter.camera.up = (0.0, 0.0, 1.0)
        self.plotter.reset_camera_clipping_range()

    def _set_colors(self, dose):
        rgba = (self.df.colors(dose) * 255).astype(np.uint8)
        self.voxels.cell_data["rgba"] = np.repeat(rgba, self._cells_per,
                                                  axis=0)

    def _beam_segment(self, k):
        b = self.beams[self.order[k]]
        src = np.asarray(b.source, dtype=np.float64)
        tgt = np.asarray(b.target, dtype=np.float64)
        d = (tgt - src) / max(np.linalg.norm(tgt - src), 1e-9)
        return tgt - BEAM_BEFORE_MM * d, tgt + BEAM_AFTER_MM * d, b.cone_mm

    def show(self, pos, text):
        dose, k, u = self.dose_at(pos)
        n = len(self.order)
        acceso = pos < n and 0.0 < u < 1.0
        if self.gpu:
            self._set_colors(dose)
            if self._beam_actor is not None:
                self.plotter.remove_actor(self._beam_actor, render=False)
                self._beam_actor = None
            if acceso:
                a, b, cone = self._beam_segment(k)
                cyl = pv.Cylinder(center=(a + b) / 2.0, direction=b - a,
                                  radius=cone / 2.0,
                                  height=float(np.linalg.norm(b - a)))
                # Discreto: deve dire DA DOVE arriva la dose, non coprire i
                # voxel che la ricevono.
                self._beam_actor = self.plotter.add_mesh(
                    cyl, color="#ffd84d", opacity=0.10, name="fascio",
                    render=False)
            self._text.SetText(2, text)
            self.plotter.render()
        else:
            self._draw_mpl(dose, k if acceso else None, text)

    # -- ripiego senza GPU -------------------------------------------------
    def _build_mpl(self):
        self.fig = Figure(figsize=(5, 5), facecolor=BG)
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111, projection="3d")
        self._scatter = None

    def _draw_mpl(self, dose, k, text):
        ax = self.ax
        ax.clear()
        ax.set_facecolor(BG)
        p = self.df.points
        ax.scatter(p[:, 0], p[:, 1], p[:, 2], c=self.df.colors(dose),
                   s=14, depthshade=False)
        if k is not None:
            a, b, _ = self._beam_segment(k)
            ax.plot(*np.column_stack([a, b]), color="#ffd84d", lw=3)
        ax.set_title(text, color=FG, fontsize=9)
        ax.set_axis_off()
        self.canvas.draw_idle()

    def close_view(self):
        if self.gpu:
            try:
                self.plotter.close()
            except Exception:                # noqa: BLE001
                pass


# ---------------------------------------------------------------------------
class TreatmentReportDialog(QDialog):
    """Finestra di fine seduta: sintesi, replay, andamento, tabella."""

    def __init__(self, report, plan, beams, shell_mesh=None, parent=None):
        super().__init__(parent)
        self.report = report
        self.plan = plan
        self.rows = report["fasci"]
        self.n = len(self.rows)
        self.setWindowTitle("Resoconto della seduta")
        self.resize(1500, 920)
        # Stile ESPLICITO: la finestra deve reggersi anche senza il foglio di
        # stile dell'app. Senza, il riquadro delle schede usciva bianco e le
        # etichette delle schede inattive erano illeggibili.
        self.setStyleSheet(f"""
            QDialog {{ background: {BG}; color: {FG}; }}
            QLabel {{ color: {FG}; }}
            QTabWidget::pane {{ border: 1px solid {GRID}; background: {BG}; }}
            QTabBar::tab {{ background: #1b1f27; color: {MUTED};
                            padding: 6px 14px; border: 1px solid {GRID};
                            border-bottom: none; }}
            QTabBar::tab:selected {{ background: {BG}; color: #e6edf3; }}
            QTableWidget {{ background: {BG}; color: {FG};
                            gridline-color: {GRID};
                            selection-background-color: #1f6feb; }}
            QHeaderView::section {{ background: #1b1f27; color: {FG};
                                    border: 1px solid {GRID}; padding: 4px; }}
            QPushButton {{ background: #21262d; color: {FG};
                           border: 1px solid {GRID}; padding: 5px 14px;
                           border-radius: 4px; }}
            QPushButton:hover {{ background: #30363d; }}
            QComboBox {{ background: #21262d; color: {FG};
                         border: 1px solid {GRID}; padding: 3px 8px; }}
        """)

        self.df = dfm.DoseFill.from_plan(plan, beams)
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 10, 12, 10)

        self.lbl_head = QLabel()
        self.lbl_head.setTextFormat(Qt.TextFormat.RichText)
        self.lbl_head.setWordWrap(True)
        root.addWidget(self.lbl_head)
        self._render_head()

        split = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(split, 1)

        # --- replay --------------------------------------------------------
        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)
        if self.df is None:
            lv.addWidget(QLabel("Il piano non contiene la dose nel tumore: "
                                "replay non disponibile."))
            self.replay = None
        else:
            self.replay = ReplayView(self.df, report, beams, shell_mesh)
            lv.addWidget(self.replay, 1)
        ctr = QHBoxLayout()
        self.btn_play = QPushButton("Riproduci")
        self.btn_play.clicked.connect(self.toggle_play)
        self.btn_start = QPushButton("Inizio")
        self.btn_start.clicked.connect(lambda: self.slider.setValue(0))
        self.btn_end = QPushButton("Fine")
        self.btn_end.clicked.connect(
            lambda: self.slider.setValue(self.slider.maximum()))
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, self.n * SUB)
        self.slider.valueChanged.connect(self._on_pos)
        self.combo_speed = QComboBox()
        for lab, v in (("0.5x", 0.5), ("1x", 1.0), ("2x", 2.0), ("4x", 4.0),
                       ("8x", 8.0)):
            self.combo_speed.addItem(lab, v)
        self.combo_speed.setCurrentIndex(2)
        self.combo_speed.setToolTip("Fasci al secondo, a 1x")
        for w in (self.btn_start, self.btn_play, self.btn_end):
            ctr.addWidget(w)
        ctr.addWidget(self.slider, 1)
        ctr.addWidget(self.combo_speed)
        lv.addLayout(ctr)
        self.lbl_pos = QLabel()
        self.lbl_pos.setWordWrap(True)
        self.lbl_pos.setStyleSheet(f"color: {MUTED};")
        lv.addWidget(self.lbl_pos)
        split.addWidget(left)

        # --- andamento -----------------------------------------------------
        self.tabs = QTabWidget()
        self.tabs.addTab(self._tab_andamento(), "Andamento del bersaglio")
        self.tabs.addTab(self._tab_tempi(), "Tempi")
        self.tabs.addTab(self._tab_robot(), "Robot")
        self.tabs.addTab(self._tab_fasci(), "Fasci")
        split.addWidget(self.tabs)
        split.setSizes([640, 860])

        bottom = QHBoxLayout()
        note = QLabel(self._note())
        note.setStyleSheet(f"color: {MUTED}; font-size: 10px;")
        note.setWordWrap(True)
        bottom.addWidget(note, 1)
        btn_save = QPushButton("Salva immagine")
        btn_save.clicked.connect(self.save_image)
        btn_close = QPushButton("Chiudi")
        btn_close.clicked.connect(self.accept)
        bottom.addWidget(btn_save)
        bottom.addWidget(btn_close)
        root.addLayout(bottom)

        self.timer = QTimer(self)
        self.timer.setInterval(int(1000 / FPS))
        self.timer.timeout.connect(self._advance)
        self._on_pos(0)

    # ------------------------------------------------------------------
    def _render_head(self):
        s = self.report["sintesi"]
        fin = s.get("finale") or {}
        cov, plan_cov = fin.get("copertura_pct"), s.get(
            "copertura_pianificata_pct")
        col = GOOD if (cov or 0) >= 90 else (WARN if (cov or 0) >= 80 else BAD)
        tutti = s["fasci_erogati"] == s["fasci_pianificati"]
        angoli = ", ".join(f"yaw {a['yaw_deg']:.0f} ({a['fasci']})"
                           for a in s.get("angoli_lettino", []))

        def blocco(titolo, valore, sotto, colore=FG):
            return (f"<td style='padding:4px 22px 4px 0; vertical-align:top'>"
                    f"<div style='color:{MUTED}; font-size:10px'>{titolo}</div>"
                    f"<div style='color:{colore}; font-size:20px; "
                    f"font-weight:600'>{valore}</div>"
                    f"<div style='color:{MUTED}; font-size:10px'>{sotto}</div>"
                    f"</td>")

        celle = [
            blocco("FASCI EROGATI",
                   f"{s['fasci_erogati']} / {s['fasci_pianificati']}",
                   "tutti quelli pianificati" if tutti else
                   f"{s['fasci_non_raggiungibili']} non raggiungibili",
                   GOOD if tutti else BAD),
            blocco(("TUMORE (CTV)" if s.get("volume_giudicato") == "CTV"
                    else "BERSAGLIO") + " ALLA DOSE PRESCRITTA",
                   _fmt(cov, "%"),
                   f"pianificato {_fmt(plan_cov, '%')}", col),
            blocco("D95 · DOSE MINIMA",
                   f"{_fmt(fin.get('d95_pct'), '%', 0)} · "
                   f"{_fmt(fin.get('dose_min_pct'), '%', 0)}",
                   f"della prescrizione ({_fmt(s['prescrizione_pct'], '%', 0)})"),
            blocco("DURATA SIMULATA", _fmt(s["tempo_totale_s"], " s"),
                   f"trasferimenti {_fmt(s['tempo_trasferimenti_s'], ' s', 0)}"
                   f" · erogazione {_fmt(s['tempo_erogazione_s'], ' s', 0)}"
                   f" · lettino {_fmt(s['tempo_lettino_s'], ' s', 0)}"),
            blocco("LETTINO", f"{s['rotazioni_lettino']} rotazioni",
                   angoli or "fermo"),
            blocco("ROBOT", f"franco min {_fmt(s['franco_minimo_mm'], ' mm', 0)}",
                   f"margine min {_fmt(s['margine_minimo'], '', 3)} · "
                   f"movimento {_fmt(s['movimento_totale_deg'], '°', 0)} · "
                   f"{s['trasferimenti_con_ritiro']} ritiri"),
        ]
        rs = self.report.get("respiro")
        if rs:
            er = rs.get("errore_residuo") or {}
            ff = rs.get("fascio_fermo") or {}
            vmax = rs.get("velocita_max_giunti_deg_s") or [0]
            lim = rs.get("limiti_velocita_deg_s") or [1]
            uso = max(v / l for v, l in zip(vmax, lim)) if vmax else 0.0
            if rs.get("gating_mm"):
                # Col gating la sospensione e' il robot che ASPETTA il tumore:
                # e' voluta, e il tempo perso non costa dose.
                titolo = f"GATING (finestra {rs['gating_mm']:.1f} mm)"
                sotto = (f"a fascio sempre acceso {_fmt(ff.get('p95_3d'), ' mm')}"
                         f" · fascio acceso "
                         f"{_fmt(100 - (rs.get('sospensione_pct') or 0), '%')}"
                         f" · attese forzate {rs.get('erogazioni_forzate', 0)}")
            else:
                titolo = ("INSEGUIMENTO (" + rs.get("predittore", "") + ")"
                          if rs.get("inseguimento") else
                          "RESPIRO, FASCIO FERMO")
                sotto = (f"a fascio fermo {_fmt(ff.get('p95_3d'), ' mm')} · "
                         f"sospeso {_fmt(rs.get('sospensione_pct'), '%')} · "
                         f"giunti al {100 * uso:.0f}% della velocita' max")
            celle.append(blocco(
                titolo, f"p95 {_fmt(er.get('p95_3d'), ' mm')}", sotto,
                GOOD if (er.get("p95_3d") or 99) <= 3.0 else WARN))
        self.lbl_head.setText(
            f"<div style='color:{FG}; font-size:15px; font-weight:600'>"
            f"Resoconto della seduta</div><table><tr>{''.join(celle)}</tr>"
            f"</table>")

    def _note(self):
        s = self.report["sintesi"]
        m = self.report.get("meta", {})
        parti = []
        if s.get("tempo_reale_s") is not None:
            parti.append(f"tempo reale di simulazione "
                         f"{s['tempo_reale_s']:.0f} s")
        if m.get("scritto"):
            parti.append(f"scritto {m['scritto']}")
        if self.report.get("piano_generato") != self.plan.get("generato"):
            parti.append("ATTENZIONE: il piano e' cambiato dopo questa seduta, "
                         "il replay usa il piano attuale")
        parti.append("la dose voxel per voxel e' ricostruita dal piano e "
                     "dall'ordine di erogazione, come nella vista RViz")
        return " · ".join(parti)

    # ------------------------------------------------------------------
    def _canvas(self, rows=1, h=6.0):
        fig = Figure(figsize=(8, h), facecolor=BG)
        canvas = FigureCanvas(fig)
        canvas.setSizePolicy(QSizePolicy.Policy.Expanding,
                             QSizePolicy.Policy.Expanding)
        return fig, canvas

    def _tab_andamento(self):
        fig, canvas = self._canvas()
        x = np.arange(0, self.n + 1)
        def serie(key):
            return [0.0] + [r.get(key) or 0.0 for r in self.rows]
        cov = serie("copertura_pct")
        ax1 = fig.add_subplot(211)
        _style(ax1)
        ax1.plot(x, cov, color=GOOD, lw=2.0, label="bersaglio alla dose prescritta")
        # Stessa misura con la soglia al 95% della prescrizione: sale prima e
        # in modo piu' graduale, e mostra quanto il bersaglio e' VICINO alla
        # prescrizione quando la curva piena e' ancora a zero.
        ax1.plot(x, serie("copertura_95_pct"), color=GOOD, lw=1.2, ls="--",
                 alpha=0.8, label="bersaglio al 95% della prescrizione")
        ax1.plot(x, serie("d95_pct"), color="#58a6ff", lw=1.2, label="D95")
        ax1.plot(x, serie("dose_min_pct"), color=BAD, lw=1.0, label="dose minima")
        ax1.plot(x, serie("dose_media_pct"), color=WARN, lw=1.0, ls="--",
                 label="dose media")
        ax1.axhline(100.0, color=MUTED, lw=0.8, ls=":")
        ax1.set_ylabel("% (della prescrizione / del bersaglio)")
        ax1.set_xlim(0, self.n)
        ax1.set_title("Il bersaglio fascio dopo fascio", fontsize=10)
        ax1.legend(fontsize=7, facecolor=BG, edgecolor=GRID, labelcolor=FG,
                   loc="upper left")
        self._cur_and = ax1.axvline(0, color="#ffd84d", lw=1.2)

        ax2 = fig.add_subplot(212, sharex=ax1)
        _style(ax2)
        # Contributo di ogni fascio alla DOSE MEDIA: dice chi ha fatto il
        # lavoro. La copertura da sola e' a gradino - resta a zero finche'
        # non si supera la prescrizione - e nasconderebbe i primi fasci.
        media = np.array(serie("dose_media_pct"))
        contrib = np.diff(media)
        colors = [COUCH_COLORS[r["lettino"] % len(COUCH_COLORS)]
                  for r in self.rows]
        ax2.bar(np.arange(1, self.n + 1), contrib, color=colors, width=0.8)
        ax2.set_ylabel("dose media aggiunta (%)")
        ax2.set_xlabel("fascio, nell'ordine di erogazione")
        ax2.set_title("Contributo di ciascun fascio (colore = angolo di lettino)",
                      fontsize=10)
        top = int(np.argmax(contrib)) if len(contrib) else 0
        if len(contrib):
            r = self.rows[top]
            ax2.annotate(f"fascio {top + 1}: {r['collimatore_mm']:.0f} mm, "
                         f"{100 * contrib[top] / max(media[-1], 1e-9):.0f}% "
                         f"della dose", xy=(top + 1, contrib[top]),
                         xytext=(0.35, 0.85), textcoords="axes fraction",
                         color=FG, fontsize=8,
                         arrowprops=dict(arrowstyle="->", color=MUTED))
        self._cur_con = ax2.axvline(0, color="#ffd84d", lw=1.2)
        fig.tight_layout()
        self._canvas_and = canvas
        return canvas

    def _tab_tempi(self):
        fig, canvas = self._canvas()
        ev = self.report["eventi"]
        ax = fig.add_subplot(211)
        _style(ax)
        righe = {"trasferimento": (2, "#58a6ff"), "lettino": (1, "#bc8cff"),
                 "erogazione": (0, "#ffd84d")}
        for tipo, (y, col) in righe.items():
            bars = [(e["inizio_s"], e["durata_s"]) for e in ev
                    if e["tipo"] == tipo]
            if bars:
                ax.broken_barh(bars, (y - 0.35, 0.7), facecolors=col)
        ax.set_yticks([0, 1, 2])
        ax.set_yticklabels(["erogazione", "lettino", "trasferimento"],
                           color=FG)
        ax.set_xlabel("tempo simulato (s)")
        ax.set_title("Linea temporale della seduta", fontsize=10)
        self._cur_tempo = ax.axvline(0, color="#ff5c8a", lw=1.2)

        ax2 = fig.add_subplot(212)
        _style(ax2)
        x = np.arange(1, self.n + 1)
        tr = [r["trasferimento_s"] or 0.0 for r in self.rows]
        le = [r["rotazione_lettino_s"] or 0.0 for r in self.rows]
        tr_only = np.array(tr) - np.array(le)
        ax2.bar(x, tr_only, color="#58a6ff", label="spostamento del braccio")
        ax2.bar(x, le, bottom=tr_only, color="#bc8cff",
                label="rotazione del lettino")
        rit = [i + 1 for i, r in enumerate(self.rows) if r["con_ritiro"]]
        if rit:
            ax2.scatter(rit, [tr[i - 1] for i in rit], marker="v",
                        color=BAD, zorder=3, label="con ritiro")
        ax2.set_xlabel("fascio")
        ax2.set_ylabel("tempo prima del fascio (s)")
        ax2.legend(fontsize=7, facecolor=BG, edgecolor=GRID, labelcolor=FG)
        self._cur_tr = ax2.axvline(0, color="#ffd84d", lw=1.2)
        fig.tight_layout()
        self._canvas_tempi = canvas
        return canvas

    def _tab_robot(self):
        fig, canvas = self._canvas(h=7.0)
        x = np.arange(1, self.n + 1)
        ax1 = fig.add_subplot(311)
        _style(ax1)
        q = np.array([r["giunti_deg"] for r in self.rows], dtype=float)
        cols = ["#58a6ff", "#3fb950", "#d29922", "#f85149", "#bc8cff",
                "#39c5cf"]
        for j in range(q.shape[1]):
            ax1.plot(x, q[:, j], color=cols[j], lw=1.1, label=f"A{j + 1}")
        ax1.set_ylabel("giunto (gradi)")
        ax1.legend(ncol=6, fontsize=7, facecolor=BG, edgecolor=GRID,
                   labelcolor=FG, loc="upper center")
        ax1.set_title("Configurazione del braccio a ogni fascio", fontsize=10)

        ax2 = fig.add_subplot(312, sharex=ax1)
        _style(ax2)
        mg = [r["margine"] or 0.0 for r in self.rows]
        ax2.fill_between(x, 0, mg, color="#58a6ff", alpha=0.35)
        ax2.plot(x, mg, color="#58a6ff", lw=1.1)
        ax2.axhline(0.15, color=BAD, lw=0.8, ls="--")
        ax2.set_ylabel("margine dai\nfine corsa")

        ax3 = fig.add_subplot(313, sharex=ax1)
        _style(ax3)
        fr = [r["franco_mm"] if r["franco_mm"] is not None else np.nan
              for r in self.rows]
        ax3.plot(x, fr, color=GOOD, lw=1.2, marker="o", ms=2.5)
        ax3.set_ylabel("franco dagli\nostacoli (mm)")
        ax3.set_xlabel("fascio, nell'ordine di erogazione")
        self._cur_rob = [a.axvline(0, color="#ffd84d", lw=1.0)
                         for a in (ax1, ax2, ax3)]
        fig.tight_layout()
        self._canvas_rob = canvas
        return canvas

    def _tab_fasci(self):
        cols = ["ordine", "n. piano", "cono mm", "peso %", "lettino",
                "trasf. s", "trasf. °", "margine", "franco mm", "copertura",
                "D95"]
        # Il peso grezzo dell'ottimizzatore non ha unita' leggibile: si mostra
        # la quota del peso totale, che dice quanto conta il fascio.
        w_tot = sum((r["peso"] or 0.0) for r in self.rows) or 1.0
        t = QTableWidget(self.n, len(cols))
        t.setHorizontalHeaderLabels(cols)
        t.verticalHeader().setVisible(False)
        t.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        t.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        t.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch)
        for i, r in enumerate(self.rows):
            vals = [r["ordine"], r["fascio"], _fmt(r["collimatore_mm"]),
                    _fmt(100.0 * (r["peso"] or 0.0) / w_tot, "", 1),
                    f"yaw {_fmt(r['lettino_yaw_deg'], '', 0)}"
                    + (" ↻" if r["con_rotazione_lettino"] else ""),
                    _fmt(r["trasferimento_s"], "", 2),
                    _fmt(r["trasferimento_deg"], "", 0)
                    + (" ⤴" if r["con_ritiro"] else ""),
                    _fmt(r["margine"], "", 3), _fmt(r["franco_mm"], "", 0),
                    _fmt(r.get("copertura_pct"), "%"),
                    _fmt(r.get("d95_pct"), "%", 0)]
            for j, v in enumerate(vals):
                it = QTableWidgetItem(str(v))
                it.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                t.setItem(i, j, it)
        # Un clic su un fascio porta il replay a fine erogazione di quel fascio.
        t.cellClicked.connect(
            lambda row, _c: self.slider.setValue((row + 1) * SUB))
        self.table = t
        return t

    # ------------------------------------------------------------------
    def _on_pos(self, value):
        pos = value / SUB
        n = self.n
        k = min(int(pos), n - 1) if n else 0
        done = min(int(pos), n)
        cov = (self.rows[done - 1].get("copertura_pct") if done > 0 else 0.0)
        if self.replay is not None:
            dose, _, _ = self.replay.dose_at(pos)
            cov = 100.0 * self.df.coverage(dose)
        erog = f"fascio {k + 1}/{n} in erogazione" if pos < n and \
            pos != int(pos) else f"{done}/{n} fasci erogati"
        text = f"{erog}  -  bersaglio alla dose prescritta {cov:.0f}%"
        if self.replay is not None:
            self.replay.show(pos, text)
        r = self.rows[k] if n else {}
        chi = "ultimo fascio" if pos >= n else "fascio corrente"
        self.lbl_pos.setText(
            f"{text}   ·   {chi} (n. {r.get('fascio', '-')} nel piano): cono "
            f"{_fmt(r.get('collimatore_mm'), ' mm', 0)}, lettino yaw "
            f"{_fmt(r.get('lettino_yaw_deg'), '°', 0)}, "
            f"trasferimento {_fmt(r.get('trasferimento_s'), ' s', 2)}")
        xk = pos
        for ln in (self._cur_and, self._cur_con, self._cur_tr, *self._cur_rob):
            ln.set_xdata([xk, xk])
        t0 = r.get("inizio_s") or 0.0
        self._cur_tempo.set_xdata([t0, t0])
        for c in (self._canvas_and, self._canvas_tempi, self._canvas_rob):
            c.draw_idle()
        if n:
            self.table.selectRow(k)

    def toggle_play(self):
        if self.timer.isActive():
            self.timer.stop()
            self.btn_play.setText("Riproduci")
        else:
            if self.slider.value() >= self.slider.maximum():
                self.slider.setValue(0)
            self.timer.start()
            self.btn_play.setText("Pausa")

    def _advance(self):
        speed = float(self.combo_speed.currentData())
        step = max(int(round(speed * SUB / FPS)), 1)
        v = self.slider.value() + step
        if v >= self.slider.maximum():
            self.slider.setValue(self.slider.maximum())
            self.toggle_play()
            return
        self.slider.setValue(v)

    def save_image(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Salva il resoconto", "resoconto_seduta.png", "PNG (*.png)")
        if path:
            self.grab().save(path)

    def closeEvent(self, ev):
        self.timer.stop()
        if self.replay is not None:
            self.replay.close_view()
        super().closeEvent(ev)


# ---------------------------------------------------------------------------
def open_report(report_path, plan_path, mesh_dir=None, parent=None):
    """Carica resoconto e piano e costruisce la finestra (non la mostra)."""
    from plan_export import load_plan
    report = tlog.load_report(report_path)
    beams, plan = load_plan(plan_path)
    shell = None
    if mesh_dir:
        stl = os.path.join(mesh_dir, "tumor.stl")
        if os.path.exists(stl):
            import trimesh
            shell = trimesh.load(stl)
            shell.vertices = shell.vertices * 1000.0     # STL in metri
    return TreatmentReportDialog(report, plan, beams, shell, parent)
