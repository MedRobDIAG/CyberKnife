# ============================================================================
#  main.py - CyberKnife Control Center v1.0
#
#  Control Station desktop per il progetto di Medical Robotics.
#  Entry point dell'applicazione PyQt6: costruisce la Main Window con le tre
#  schede operative/placeholder e applica il tema grafico.
#
#  AVVIO:
#      python main.py
#      (oppure: .venv\Scripts\python.exe main.py)
# ============================================================================
import os
import sys

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QAction, QKeySequence
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QTabWidget, QLabel, QMessageBox, QToolBar,
    QFileDialog,
)

import config
from ui_tabs import PerceptionTab, DosimetryTab, KinematicsTab
from motion_tab import MotionTab


APP_NAME = "CyberKnife Control Center"
APP_VERSION = "1.0"


# ---------------------------------------------------------------------------
#  Foglio di stile (tema scuro, coerente con le workstation cliniche)
# ---------------------------------------------------------------------------
STYLESHEET = """
QMainWindow, QWidget      { background-color: #10141a; color: #c8d0da; }
QTabWidget::pane          { border: 1px solid #232a34; background: #10141a; top: -1px; }
QTabBar::tab              { background: #171d25; color: #8b95a3;
                            padding: 10px 22px; margin-right: 2px;
                            border: 1px solid #232a34; border-bottom: none;
                            font-size: 12px; }
QTabBar::tab:selected     { background: #10141a; color: #00e5ff;
                            border-top: 2px solid #00e5ff; }
QTabBar::tab:hover:!selected { color: #c8d0da; }

/* Sotto-schede interne (es. Modello 3D / Sezione): piu' discrete di quelle
   principali, per non competere con la navigazione di primo livello. */
QTabWidget#subTabs::pane  { border: 1px solid #1c232c; background: #10141a; }
QTabWidget#subTabs > QTabBar::tab { background: #141a22; color: #7d8794;
                            padding: 6px 16px; font-size: 11px;
                            border: 1px solid #1c232c; border-bottom: none; }
QTabWidget#subTabs > QTabBar::tab:selected { background: #10141a; color: #3ddc84;
                            border-top: 2px solid #3ddc84; }

QGroupBox                 { border: 1px solid #232a34; border-radius: 4px;
                            margin-top: 12px; padding-top: 10px;
                            font-size: 11px; color: #8b95a3; }
QGroupBox::title          { subcontrol-origin: margin; left: 10px;
                            padding: 0 6px; color: #8b95a3; }

QPushButton               { background: #1d242e; color: #c8d0da;
                            border: 1px solid #2e3846; border-radius: 4px;
                            padding: 8px 16px; font-size: 12px; }
QPushButton:hover         { background: #26303c; border-color: #3c4a5c; }
QPushButton:pressed       { background: #161c24; }
QPushButton:disabled      { background: #161a20; color: #4a535f;
                            border-color: #222933; }
QPushButton#primaryButton { background: #0d5f6b; color: #e6f7fa;
                            border: 1px solid #0f7f8f; font-weight: bold; }
QPushButton#primaryButton:hover   { background: #10788a; }
QPushButton#primaryButton:disabled{ background: #16323a; color: #4a6a72;
                                    border-color: #1d3d45; }

QProgressBar              { background: #171d25; border: 1px solid #232a34;
                            border-radius: 4px; text-align: center;
                            color: #c8d0da; font-size: 11px; }
QProgressBar::chunk       { background-color: #0f7f8f; border-radius: 3px; }

QLabel#infoPanel          { background: #0b0e12; border: 1px solid #232a34;
                            border-radius: 4px; padding: 12px; }
QLabel#sliceLabel         { color: #00e5ff; font-family: Consolas, monospace;
                            font-size: 11px; }
QLabel#hintLabel          { color: #6f7a88; font-size: 11px; font-style: italic; }

QSlider::groove:horizontal { background: #1d242e; height: 5px; border-radius: 2px; }
QSlider::handle:horizontal { background: #00e5ff; width: 13px; height: 13px;
                             margin: -5px 0; border-radius: 6px; }
QSlider::handle:horizontal:hover { background: #5cf0ff; }
QSlider::sub-page:horizontal { background: #0d5f6b; border-radius: 2px; }
QSlider::groove:horizontal:disabled { background: #161a20; }
QSlider::handle:horizontal:disabled { background: #333c48; }

QComboBox                 { background: #1d242e; color: #c8d0da;
                            border: 1px solid #2e3846; border-radius: 4px;
                            padding: 5px 10px; font-size: 11px; }
QComboBox:hover           { border-color: #3c4a5c; }
QComboBox QAbstractItemView { background: #171d25; color: #c8d0da;
                              selection-background-color: #0d5f6b;
                              border: 1px solid #2e3846; }
QScrollArea               { background: transparent; border: none; }
QScrollBar:vertical       { background: #0b0e12; width: 10px; margin: 0; }
QScrollBar::handle:vertical { background: #2e3846; min-height: 28px;
                              border-radius: 5px; }
QScrollBar::handle:vertical:hover { background: #3c4a5c; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: none; }

QCheckBox                 { color: #c8d0da; font-size: 11px; spacing: 6px; }
QCheckBox::indicator      { width: 14px; height: 14px; border-radius: 3px;
                            border: 1px solid #2e3846; background: #1d242e; }
QCheckBox::indicator:checked { background: #ff8a00; border-color: #ff8a00; }
QLabel#wipLabel           { color: #ffb020; font-size: 15px;
                            padding: 28px; border: 1px dashed #3c4a5c;
                            border-radius: 6px; background: #141922; }
QLabel#wipNote            { color: #ffb020; font-size: 10px; padding: 8px;
                            border: 1px dashed #3c4a5c; border-radius: 4px;
                            background: #141922; }
QFrame#separator          { color: #232a34; }
QStatusBar                { background: #0b0e12; color: #8b95a3;
                            border-top: 1px solid #232a34; font-size: 11px; }
QMenuBar                  { background: #0b0e12; color: #c8d0da; }
QMenuBar::item:selected   { background: #1d242e; }
QMenu                     { background: #171d25; color: #c8d0da;
                            border: 1px solid #232a34; }
QMenu::item:selected      { background: #1d242e; }
QToolBar                  { background: #0b0e12; border-bottom: 1px solid #232a34;
                            spacing: 6px; padding: 3px 8px; }
QToolBar QToolButton      { background: #1d242e; color: #c8d0da;
                            border: 1px solid #2e3846; border-radius: 4px;
                            padding: 4px 12px; font-size: 11px; }
QToolBar QToolButton:hover { background: #26303c; }
QToolBar QToolButton:disabled { color: #4a535f; }
"""

#: Dove si propongono i progetti salvati.
PROJECTS_DIR = os.path.join(config.OUTPUT_DIR, "progetti")


class MainWindow(QMainWindow):
    """Finestra principale: le quattro fasi della Control Station."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"{APP_NAME}  v{APP_VERSION}")
        self.resize(1440, 860)
        self.setMinimumSize(1100, 700)

        # ---------- Schede, nell'ordine della pipeline ----------
        # Il respiro sta PRIMA della dosimetria: la tecnica di gestione del
        # moto decide su quale bersaglio si ottimizza la dose (CTV, ITV,
        # CTV + margine d'inseguimento) e se in ROS il robot insegue.
        self.tabs = QTabWidget()
        self.tab_perception = PerceptionTab()
        self.tab_motion = MotionTab()
        self.tab_dosimetry = DosimetryTab()
        self.tab_kinematics = KinematicsTab()

        self.tabs.addTab(self.tab_perception, "  1 - Acquisizione e Percezione  ")
        self.tabs.addTab(self.tab_motion, "  2 - Respiro e Tecnica  ")
        self.tabs.addTab(self.tab_dosimetry, "  3 - Dosimetria e Simulazione Raggi  ")
        self.tabs.addTab(self.tab_kinematics, "  4 - Cinematica e Controllo ROS2  ")
        self.setCentralWidget(self.tabs)

        self.project_path = None
        self._dirty = False
        self._restoring = False

        # A inferenza completata, il Tab 1 passa i dati al visualizzatore 3D.
        self.tab_perception.resultAvailable.connect(
            self.tab_dosimetry.set_source_data)
        # A dosimetria completata, il Tab 2 restituisce la mappa di dose che il
        # Tab 1 sovrappone alle slice CT.
        self.tab_dosimetry.panel.doseAvailable.connect(
            self.tab_perception.set_dose_map)
        # Il piano completo va anche al Tab 3, che ne ricava la cinematica.
        self.tab_dosimetry.panel.planAvailable.connect(
            self.tab_kinematics.set_plan)
        # Il lettino dipende dal paziente: la sua quota e' il dorso, che esce
        # dalla segmentazione degli organi.
        self.tab_dosimetry.panel.organsAvailable.connect(
            self.tab_kinematics.set_organs)
        self.tab_perception.resultAvailable.connect(
            self.tab_kinematics.set_source_data)
        # Il respiro dipende da dove sta la lesione nel polmone; la tecnica
        # scelta fornisce il bersaglio alla dosimetria e arriva al Tab 4, che
        # la scrive nel piano per il nodo ROS.
        self.tab_perception.resultAvailable.connect(
            self.tab_motion.set_source_data)
        self.tab_dosimetry.panel.organsAvailable.connect(
            self.tab_motion.set_organs)
        self.tab_motion.configChanged.connect(self.tab_kinematics.set_motion)
        self.tab_kinematics.set_motion(self.tab_motion.ros_config())
        self.tab_dosimetry.panel.set_target_provider(
            self.tab_motion.planning_target)
        self.tab_motion.techniqueChanged.connect(self._on_technique_changed)

        # Ogni risultato nuovo rende il progetto "da salvare".
        for sig in (self.tab_perception.resultAvailable,
                    self.tab_dosimetry.panel.organsAvailable,
                    self.tab_dosimetry.panel.planAvailable,
                    self.tab_motion.resultChanged,
                    self.tab_kinematics.resultChanged):
            sig.connect(self._mark_dirty)
        self.tab_kinematics.sessionFinished.connect(self._offer_save)

        self._build_menu()
        self._build_toolbar()

        # ---------- Barra di stato ----------
        self.statusBar().showMessage("Pronto")
        # Backend 3D: va dichiarato, non lasciato indovinare. Se pyvista non e'
        # installato l'applicazione ricade su Matplotlib e continua a
        # funzionare, ma a ~7 fps invece di ~215: senza questa indicazione chi
        # aggiorna il codice senza aggiornare le dipendenze vede "la vecchia
        # visualizzazione" e non ha modo di capire perche'.
        from viewer3d import GPU_3D_AVAILABLE
        self.lbl_backend = QLabel(
            "3D: GPU (VTK)" if GPU_3D_AVAILABLE else "3D: Matplotlib (ripiego)")
        if not GPU_3D_AVAILABLE:
            self.lbl_backend.setStyleSheet("color:#ffb020;")
            self.lbl_backend.setToolTip(
                "PyVista non e' installato: la vista 3D usa il ripiego software.\n"
                "Per il rendering accelerato:  pip install -r requirements.txt")
        self.statusBar().addPermanentWidget(self.lbl_backend)

        # Stessa logica per la parte robotica: senza ROS 2 in WSL il Tab 3
        # continua a calcolare la cinematica ma non puo' aprire RViz. Il
        # controllo gira in un thread perche' avviare WSL costa un paio di
        # secondi e bloccherebbe la comparsa della finestra.
        self.lbl_ros = QLabel("ROS: verifica...")
        self.lbl_ros.setStyleSheet("color:#8b949e;")
        self.statusBar().addPermanentWidget(self.lbl_ros)
        self.ros_probe = RosProbe()
        self.ros_probe.done.connect(self._on_ros_probe)
        self.ros_probe.start()

        self.lbl_model = QLabel(self._model_status_text())
        self.statusBar().addPermanentWidget(self.lbl_model)

    def _on_ros_probe(self, available, detail):
        if available:
            self.lbl_ros.setText("ROS: pronto (WSL)")
            self.lbl_ros.setStyleSheet("color:#8b949e;")
            self.lbl_ros.setToolTip(detail)
        else:
            self.lbl_ros.setText("ROS: non disponibile")
            self.lbl_ros.setStyleSheet("color:#ffb020;")
            self.lbl_ros.setToolTip(
                "La simulazione del robot in RViz richiede ROS 2 Humble in "
                "WSL2.\nIl Tab 4 funziona lo stesso: risolve la cinematica e "
                "mostra\ntraiettorie e margini, ma non apre RViz.\n"
                "Istruzioni nel README, sezione «Fase 3».\n\n" + detail)

    # ------------------------------------------------------------------
    def _build_menu(self):
        file_menu = self.menuBar().addMenu("&File")
        act_open = QAction("Carica scansione CT...", self)
        act_open.setShortcut("Ctrl+O")
        # Delega al pulsante del Tab 1, riportando l'utente sulla scheda giusta.
        act_open.triggered.connect(self._menu_open)
        file_menu.addAction(act_open)
        file_menu.addSeparator()
        self.act_proj_open = QAction("Apri progetto...", self)
        self.act_proj_open.setShortcut("Ctrl+Shift+O")
        self.act_proj_open.setToolTip(
            "Riapre un progetto salvato: inferenza, incertezza, tecnica del "
            "moto,\nmodello 3D, piano e cinematica tornano come erano, "
            "senza ricalcolarli.")
        self.act_proj_open.triggered.connect(self.open_project)
        file_menu.addAction(self.act_proj_open)
        self.act_proj_save = QAction("Salva progetto", self)
        self.act_proj_save.setShortcut(QKeySequence.StandardKey.Save)
        self.act_proj_save.setToolTip(
            "Salva i risultati di tutte le fasi calcolate finora.")
        self.act_proj_save.triggered.connect(self.save_project)
        file_menu.addAction(self.act_proj_save)
        self.act_proj_save_as = QAction("Salva progetto con nome...", self)
        self.act_proj_save_as.setShortcut("Ctrl+Shift+S")
        self.act_proj_save_as.triggered.connect(
            lambda: self.save_project(ask=True))
        file_menu.addAction(self.act_proj_save_as)
        file_menu.addSeparator()
        act_quit = QAction("Esci", self)
        act_quit.setShortcut("Ctrl+Q")
        act_quit.triggered.connect(self.close)
        file_menu.addAction(act_quit)

        help_menu = self.menuBar().addMenu("&?")
        act_about = QAction("Informazioni", self)
        act_about.triggered.connect(self._show_about)
        help_menu.addAction(act_about)

    def _build_toolbar(self):
        """La barra in alto: le operazioni sul progetto, sempre a vista."""
        tb = QToolBar("Progetto")
        tb.setMovable(False)
        tb.addAction(self.act_proj_open)
        tb.addAction(self.act_proj_save)
        tb.addAction(self.act_proj_save_as)
        self.lbl_project = QLabel("Progetto non salvato")
        self.lbl_project.setStyleSheet("color:#6f7a88; padding-left:12px;")
        tb.addWidget(self.lbl_project)
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, tb)
        self._update_project_label()

    # ------------------------------------------------------------------
    #  Progetto: salvataggio e apertura
    # ------------------------------------------------------------------
    def _mark_dirty(self, *_):
        if self._restoring:
            return
        self._dirty = True
        self._update_project_label()

    def _update_project_label(self):
        name = (os.path.basename(self.project_path) if self.project_path
                else "Progetto non salvato")
        self.lbl_project.setText(name + (" *" if self._dirty else ""))
        self.setWindowTitle(f"{APP_NAME}  v{APP_VERSION}"
                            + (f"  -  {name}" if self.project_path else "")
                            + (" *" if self._dirty else ""))

    def _on_technique_changed(self, _key):
        """Tecnica cambiata: un piano gia' calcolato non la rispetta piu'."""
        panel = self.tab_dosimetry.panel
        if (not self._restoring and panel.dose_result is not None and
                panel.combo_target.currentData() == "technique"):
            self.statusBar().showMessage(
                "Tecnica di gestione del moto cambiata: il piano del Tab 3 e' "
                "stato calcolato per la precedente. Ricalcola la dosimetria.",
                10000)

    def _collect_state(self):
        return {
            "app": APP_VERSION,
            "percezione": self.tab_perception.export_state(),
            "respiro": self.tab_motion.export_state(),
            "dosimetria": self.tab_dosimetry.export_state(),
            "cinematica": self.tab_kinematics.export_state(),
        }

    def save_project(self, ask=False):
        """Salva lo stato di tutte le fasi. Restituisce True se salvato."""
        import project_io
        if self._running_workers():
            QMessageBox.information(
                self, "Calcolo in corso",
                "Attendi la fine del calcolo in corso: salvare a meta' "
                "registrerebbe uno stato incoerente fra le fasi.")
            return False
        if self.tab_perception.result is None:
            QMessageBox.information(self, "Niente da salvare",
                                    "Carica prima una TAC nel Tab 1.")
            return False
        path = self.project_path
        if ask or not path:
            os.makedirs(PROJECTS_DIR, exist_ok=True)
            base = os.path.basename(
                self.tab_perception.result["image_path"]).split(".")[0]
            path, _ = QFileDialog.getSaveFileName(
                self, "Salva progetto",
                path or os.path.join(PROJECTS_DIR, base + project_io.EXTENSION),
                f"Progetto CyberKnife (*{project_io.EXTENSION})")
            if not path:
                return False
        self.statusBar().showMessage("Salvataggio del progetto...")
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            path, omitted = project_io.save_project(path,
                                                    self._collect_state())
        except Exception as exc:                        # noqa: BLE001
            import traceback
            QMessageBox.critical(self, "Salvataggio non riuscito",
                                 f"{exc}\n\n{traceback.format_exc()}")
            return False
        finally:
            QApplication.restoreOverrideCursor()
        self.project_path, self._dirty = path, False
        self._update_project_label()
        mb = os.path.getsize(path) / 1e6
        self.statusBar().showMessage(f"Progetto salvato: {path} ({mb:.1f} MB)",
                                     8000)
        if omitted:
            # Ripiego VISIBILE: cio' che non si e' potuto salvare va detto.
            QMessageBox.warning(
                self, "Salvataggio parziale",
                "Questi elementi non sono stati salvati e andranno "
                "ricalcolati:\n\n" + "\n".join(omitted[:20]))
        return True

    def open_project(self):
        import project_io
        if not self._confirm_discard():
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "Apri progetto",
            PROJECTS_DIR if os.path.isdir(PROJECTS_DIR) else config.OUTPUT_DIR,
            f"Progetto CyberKnife (*{project_io.EXTENSION})")
        if not path:
            return
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        self._restoring = True
        warnings = []
        try:
            state, meta = project_io.load_project(path)
            st = state.get("percezione")
            if not st:
                raise ValueError("il progetto non contiene una TAC")
            self.statusBar().showMessage("Ricarico la TAC e i risultati...")
            warn = self.tab_perception.restore_state(st)
            if warn:
                warnings.append(warn)
            self.tab_motion.restore_state(state.get("respiro"))
            self.tab_dosimetry.restore_state(state.get("dosimetria"))
            work = os.path.join(PROJECTS_DIR, "_aperti", os.path.splitext(
                os.path.basename(path))[0])
            self.tab_kinematics.restore_state(state.get("cinematica"), work)
            if meta.get("omessi"):
                warnings.append("Elementi non salvati nel progetto, da "
                                "ricalcolare:\n" + "\n".join(meta["omessi"]))
        except Exception as exc:                        # noqa: BLE001
            import traceback
            QMessageBox.critical(self, "Progetto non aperto",
                                 f"{exc}\n\n{traceback.format_exc()}")
            return
        finally:
            self._restoring = False
            QApplication.restoreOverrideCursor()
        self.project_path, self._dirty = path, False
        self._update_project_label()
        self.statusBar().showMessage(
            f"Progetto aperto: {os.path.basename(path)} (salvato "
            f"{meta.get('salvato', '?')})", 8000)
        if warnings:
            QMessageBox.warning(self, "Progetto aperto con avvisi",
                                "\n\n".join(warnings))

    def _confirm_discard(self):
        """Chiede cosa fare delle modifiche non salvate. False = annulla."""
        if not self._dirty:
            return True
        r = QMessageBox.question(
            self, "Progetto non salvato",
            "Ci sono risultati non salvati. Salvarli prima di continuare?",
            QMessageBox.StandardButton.Save
            | QMessageBox.StandardButton.Discard
            | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Save)
        if r == QMessageBox.StandardButton.Cancel:
            return False
        if r == QMessageBox.StandardButton.Save:
            return self.save_project()
        return True

    def _offer_save(self):
        """Fine della pipeline (seduta ROS conclusa): si propone il salvataggio."""
        if not self._dirty:
            return
        r = QMessageBox.question(
            self, "Seduta completata",
            "La simulazione e' conclusa: tutte le fasi sono state eseguite.\n"
            "Salvare il progetto per riaprirlo senza ripetere i calcoli?",
            QMessageBox.StandardButton.Save | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Save)
        if r == QMessageBox.StandardButton.Save:
            self.save_project()

    def _menu_open(self):
        self.tabs.setCurrentWidget(self.tab_perception)
        self.tab_perception.on_load_clicked()

    def _model_status_text(self) -> str:
        """Indica nella status bar se i pesi del modello sono disponibili."""
        if os.path.exists(config.BEST_MODEL_PATH):
            return f"Modello: {os.path.basename(config.BEST_MODEL_PATH)} [OK]"
        return "Modello: NON TROVATO"

    def _show_about(self):
        QMessageBox.about(
            self, f"Informazioni su {APP_NAME}",
            f"<b>{APP_NAME}</b> v{APP_VERSION}<br><br>"
            "Control Station per simulazione di radiochirurgia stereotassica.<br>"
            "Fase 1: percezione e localizzazione del target tramite 3D U-Net "
            "(MONAI + PyTorch).<br><br>"
            "<span style='color:#8b95a3;'>Progetto universitario di "
            "Medical Robotics.</span>"
        )

    # ------------------------------------------------------------------
    def _running_workers(self):
        """Tutti i QThread di calcolo attualmente attivi."""
        panel = self.tab_dosimetry.panel
        candidates = [
            getattr(self.tab_perception, "worker", None),   # inferenza
            getattr(panel, "worker", None),                 # mesh 3D
            getattr(panel, "dose_worker", None),            # dosimetria
            getattr(self.tab_kinematics, "worker", None),   # cinematica
            getattr(self.tab_kinematics, "setup_worker", None),  # ottimizz. sala
            getattr(self, "ros_probe", None),               # verifica ROS
            getattr(self.tab_motion, "worker", None),       # respiro
            getattr(self.tab_motion, "scen_worker", None),  # scenari di dose
            getattr(self.tab_motion, "organ_worker", None), # organi (respiro)
            getattr(self.tab_motion, "margin_worker", None),# margine di moto
        ]
        return [w for w in candidates if w is not None and w.isRunning()]

    def closeEvent(self, event):
        """Attende la fine dei calcoli prima di chiudere.

        Distruggere un QThread ancora in esecuzione fa terminare il processo in
        modo anomalo: tutti i worker vanno attesi, non solo quello
        dell'inferenza.
        """
        if not self._confirm_discard():
            event.ignore()
            return
        running = self._running_workers()
        if running:
            reply = QMessageBox.question(
                self, "Elaborazione in corso",
                "Un'elaborazione e' ancora in corso. Uscire comunque?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.No:
                event.ignore()
                return
            for w in running:
                w.quit()
                w.wait(8000)
        event.accept()


class RosProbe(QThread):
    """Verifica in background se WSL con ROS 2 Humble e' utilizzabile."""

    done = pyqtSignal(bool, str)

    def run(self):
        import subprocess
        try:
            r = subprocess.run(
                ["wsl", "-d", "Ubuntu-22.04", "--",
                 "bash", "-lc", "source /opt/ros/humble/setup.bash "
                                "&& echo $ROS_DISTRO"],
                capture_output=True, text=True, timeout=25,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        except FileNotFoundError:
            self.done.emit(False, "WSL non e' installato su questo sistema.")
            return
        except (subprocess.SubprocessError, OSError) as exc:
            self.done.emit(False, f"Verifica non riuscita: {exc}")
            return
        if "humble" in (r.stdout or ""):
            self.done.emit(True, "ROS 2 Humble trovato in WSL (Ubuntu-22.04).")
        else:
            self.done.emit(
                False,
                (r.stderr or r.stdout or "").strip()[:300]
                or "ROS 2 Humble non trovato nella distribuzione Ubuntu-22.04.")


def _install_exception_hook():
    """Impedisce che un'eccezione in uno slot Qt faccia terminare il processo.

    PyQt6 chiama abort() quando un'eccezione Python sfugge da uno slot: la
    finestra svanisce senza messaggi e senza traceback, il che rende un errore
    banale (una chiave mancante in un pannello) indistinguibile da un crash
    nativo. Con questo hook l'errore viene mostrato e l'applicazione resta in
    piedi: si perde l'aggiornamento di quel pannello, non l'intera sessione.
    """
    import traceback

    def _hook(exc_type, exc, tb):
        msg = "".join(traceback.format_exception(exc_type, exc, tb))
        sys.stderr.write(msg)
        try:
            QMessageBox.critical(
                None, "Errore interno",
                f"{exc_type.__name__}: {exc}\n\n"
                "L'applicazione resta attiva; l'operazione in corso e' stata "
                "interrotta.")
        except Exception:
            pass

    sys.excepthook = _hook


def main():
    # Attributo da impostare prima di creare la QApplication (HiDPI).
    QApplication.setAttribute(
        Qt.ApplicationAttribute.AA_DontCreateNativeWidgetSiblings, True)
    _install_exception_hook()

    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setStyle("Fusion")
    app.setStyleSheet(STYLESHEET)

    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
