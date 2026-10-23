#!/usr/bin/env python3
# ============================================================================
#  treatment_node.py - Nodo ROS 2 che esegue il piano di trattamento.
#
#  COSA FA
#  -------
#  1. legge il piano prodotto dalla Fase 2 (JSON, millimetri, sistema RAS);
#  2. risolve la cinematica inversa per ogni fascio, riordinando la sequenza
#     per minimizzare il movimento del braccio;
#  3. anima il robot pubblicando /joint_states, che robot_state_publisher
#     trasforma in TF a partire dall'URDF;
#  4. pubblica l'anatomia e i fasci come MarkerArray per RViz.
#
#  DUE SISTEMI DI RIFERIMENTO, UNA SOLA CONVERSIONE
#  ------------------------------------------------
#  Il frame fisso e' `patient`: e' il RAS della TAC, dove vivono tumore,
#  organi e fasci. Il robot sta altrove nella stanza, e la sua collocazione e'
#  una TF statica patient -> base_link ricavata dalla stessa RobotBase che usa
#  la cinematica: se le due divergessero, RViz mostrerebbe il braccio che spara
#  accanto al tumore invece che dentro.
#
#  Il piano e' in MILLIMETRI (convenzione della CT), ROS lavora in METRI. La
#  conversione avviene solo al confine, in _mm(): dentro la cinematica tutto
#  resta in millimetri.
#
#  MACCHINA A STATI
#  ----------------
#      TRASFERIMENTO -> il braccio si muove verso il nodo successivo
#      EROGAZIONE    -> fermo sul nodo, il fascio e' acceso
#  Il tempo di trattamento reale e' dominato dai trasferimenti, per questo la
#  sequenza viene ottimizzata: la diagnostica riporta entrambi i contributi.
#
#  RESPIRO E INSEGUIMENTO
#  ----------------------
#  Se il piano porta il campo `respiro` (Tab 4 dell'app), il tumore si muove
#  (TF patient -> tumore_mobile) e durante l'EROGAZIONE il braccio lo insegue
#  con la catena di motion_model: torace -> predittore -> correlazione. La
#  logica sta in tracking_session.py, senza ROS, e i test la verificano su
#  Windows. Argomenti: respiro:=false spegne il moto, inseguimento:=false
#  lascia respirare il tumore a fascio fermo (per vedere cosa si perde).
# ============================================================================
import math
import os
import threading
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSHistoryPolicy
from rclpy.time import Time

from geometry_msgs.msg import TransformStamped, Point
from sensor_msgs.msg import JointState
from std_msgs.msg import ColorRGBA, Float64MultiArray, String
from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster
from visualization_msgs.msg import Marker, MarkerArray

from cyberknife_ros import kinematics as kin
from cyberknife_ros.plan_export import load_plan, setup_from_plan
from cyberknife_ros import dose_fill as dfill
from cyberknife_ros import treatment_log as tlog
from cyberknife_ros import motion_model as mm
from cyberknife_ros.tracking_session import TrackingSession

MM_TO_M = 0.001
#: Frame FISSO della scena: la stanza, con Z verticale e il pavimento a z=0.
#: RViz disegna sempre Z verso l'alto, quindi usare direttamente il sistema
#: della TAC farebbe apparire il paziente in piedi e il lettino come un muro:
#: in RAS +Z e' l'asse testa-piedi, che nella stanza e' orizzontale.
ROOM = "world"
#: Frame del paziente: RAS della TAC. Qui vivono tumore, organi e fasci.
FRAME = "patient"
#: Il tumore che respira: figlio di `patient`, traslato del moto vero.
TUMOR_FRAME = "tumore_mobile"


def _mm(p):
    """Millimetri (piano/cinematica) -> metri (ROS). Unico punto di conversione."""
    return np.asarray(p, dtype=np.float64) * MM_TO_M


def _pt(p_mm):
    v = _mm(p_mm)
    return Point(x=float(v[0]), y=float(v[1]), z=float(v[2]))


def _rgba(c, a=1.0):
    return ColorRGBA(r=float(c[0]), g=float(c[1]), b=float(c[2]), a=float(a))


def _quat_from_matrix(R):
    """Quaternione (x, y, z, w) da una matrice di rotazione 3x3.

    Si sceglie la componente maggiore come pivot: le formule diirette
    perdono precisione (o dividono per zero) quando la traccia e' vicina a -1,
    che qui capita davvero, perche' le rotazioni di montaggio sono di 90 e 120
    gradi esatti.
    """
    R = np.asarray(R, dtype=np.float64)
    t = float(np.trace(R))
    if t > 0.0:
        S = math.sqrt(t + 1.0) * 2.0
        w = 0.25 * S
        x = (R[2, 1] - R[1, 2]) / S
        y = (R[0, 2] - R[2, 0]) / S
        z = (R[1, 0] - R[0, 1]) / S
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        S = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / S
        x = 0.25 * S
        y = (R[0, 1] + R[1, 0]) / S
        z = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / S
        x = (R[0, 1] + R[1, 0]) / S
        y = 0.25 * S
        z = (R[1, 2] + R[2, 1]) / S
    else:
        S = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / S
        x = (R[0, 2] + R[2, 0]) / S
        y = (R[1, 2] + R[2, 1]) / S
        z = 0.25 * S
    return x, y, z, w


def _tf(parent, child, t_m, R, stamp):
    m = TransformStamped()
    m.header.stamp = stamp
    m.header.frame_id = parent
    m.child_frame_id = child
    m.transform.translation.x = float(t_m[0])
    m.transform.translation.y = float(t_m[1])
    m.transform.translation.z = float(t_m[2])
    qx, qy, qz, qw = _quat_from_matrix(R)
    m.transform.rotation.x, m.transform.rotation.y = qx, qy
    m.transform.rotation.z, m.transform.rotation.w = qz, qw
    return m


def _publish_markers(pub, arr):
    """Pubblica un MarkerArray marcando ogni marker come frame_locked.

    Senza `frame_locked` RViz trasforma un marker UNA VOLTA SOLA, quando lo
    riceve, e poi lo lascia fermo nel frame fisso. Nella finestra principale
    non si nota, perche' li' il frame fisso e' la stanza e non si muove.

    Nella Beam's Eye View invece il frame fisso e' `bev`, cioe' l'asse del
    fascio, che si sposta insieme alla testata: e' il mondo a dover ruotare
    attorno alla camera. Senza frame_locked l'anatomia - che arriva una volta
    sola, essendo latched - restava incollata dov'era al primo istante, e la
    vista sembrava immobile mentre il braccio si spostava.

    Si imposta qui, su tutto l'array, invece che marker per marker: cosi' non
    si puo' dimenticarne uno.
    """
    for m in arr.markers:
        m.frame_locked = True
    pub.publish(arr)


class TreatmentNode(Node):

    def __init__(self):
        super().__init__("cyberknife_treatment")

        share = self.declare_parameter("share_dir", "").value
        self.plan_path = self.declare_parameter(
            "plan", os.path.join(share, "plans", "piano_trattamento.json")).value
        self.mesh_dir = self.declare_parameter(
            "mesh_dir", os.path.join(share, "meshes")).value
        # n_roll = quante rotazioni attorno all'asse del fascio si provano.
        # Sui fasci reali di questo piano la ridondanza non cambia il numero di
        # pose risolvibili (misurato: 112/112 sia con 1 che con 16 tentativi),
        # quindi il valore basso e' quello giusto per la velocita'. Resta
        # parametrico perche' con vincoli di collisione tornera' a servire.
        self.n_roll = int(self.declare_parameter("n_roll", 4).value)
        self.rate_hz = float(self.declare_parameter("rate_hz", 50.0).value)
        #: gradi al secondo, riferiti al giunto piu' veloce del movimento
        self.joint_speed = float(self.declare_parameter("joint_speed_deg_s", 90.0).value)
        #: sosta sul nodo mentre il fascio e' acceso: e' l'erogazione
        self.dwell_s = float(self.declare_parameter("dwell_s", 1.2).value)
        #: tempo totale di fascio acceso (s), ripartito sui fasci in
        #: proporzione al peso (unita' monitor); 0 = sosta fissa dwell_s
        self.beam_on_s = float(self.declare_parameter("erogazione_s",
                                                      120.0).value)
        self.dwell_min_s = float(self.declare_parameter("dwell_min_s",
                                                        0.3).value)
        #: durata minima di un trasferimento, perche' lo spostamento si veda
        self.min_transfer_s = float(
            self.declare_parameter("min_transfer_s", 0.5).value)
        #: quanto dura una rotazione del lettino fra due gruppi di fasci
        self.couch_move_s = float(
            self.declare_parameter("couch_move_s", 3.0).value)
        #: pausa finale sul tumore coperto, prima di chiudere il lancio
        self.final_hold_s = float(
            self.declare_parameter("final_hold_s", 4.0).value)
        #: rotazione della camera attorno al tumore, gradi al secondo
        self.turntable = math.radians(float(
            self.declare_parameter("turntable_deg_s", 12.0).value))
        #: aggiornamenti al secondo della vista di dose (i colori di ~800
        #: voxel a 50 Hz sarebbero traffico sprecato: l'occhio non li segue)
        self.dose_hz = float(self.declare_parameter("dose_view_hz", 12.0).value)
        #: dove scrivere il resoconto della seduta. Vuoto = accanto al piano,
        #: seguendo il collegamento simbolico dell'installazione: cosi' il
        #: file finisce nella cartella del progetto su Windows, dove l'app lo
        #: legge a fine simulazione.
        self.report_path = str(self.declare_parameter("report_path", "").value)
        self.loop = bool(self.declare_parameter("loop", True).value)
        #: respiro del tumore, se il piano lo descrive
        self.breathing = bool(self.declare_parameter("respiro", True).value)
        #: il robot insegue il tumore (False: fascio fermo, tumore che respira)
        self.follow = bool(self.declare_parameter("inseguimento", True).value)
        #: predittore di latenza; vuoto = quello scritto nel piano
        self.predictor = str(self.declare_parameter("predittore", "").value)
        base_xyz = list(self.declare_parameter(
            "base_mm", [-800.0, -830.0, -150.0]).value)
        base_yaw = float(self.declare_parameter("base_yaw_deg", 0.0).value)

        # Base e angoli di lettino stanno nel PIANO, non nei parametri: i fasci
        # sono stati scelti sapendo da dove il robot li avrebbe eseguiti. I
        # parametri restano come ripiego per piani vecchi o per prove manuali.
        self.beams, self.plan = load_plan(self.plan_path)
        if not self.report_path:
            self.report_path = os.path.join(
                os.path.dirname(os.path.realpath(self.plan_path)),
                tlog.REPORT_NAME)
        base_plan, couch_plan = setup_from_plan(self.plan, kin)
        self.base = base_plan or kin.RobotBase(origin_mm=tuple(base_xyz),
                                               yaw_deg=base_yaw)
        self.couch_poses = couch_plan or [kin.CouchPose()]
        if base_plan is None:
            self.get_logger().warn(
                "Il piano non porta la configurazione di sala: uso i parametri "
                "del nodo. Rigeneralo per essere sicuro che la geometria "
                "coincida con quella con cui e' stato calcolato.")
        self._t0_ns = self.get_clock().now().nanoseconds
        self._t0_mono = time.monotonic()

        # --- Publisher -------------------------------------------------------
        # I marker dell'anatomia sono statici: QoS "transient local" li
        # consegna anche a RViz avviato DOPO il nodo, altrimenti la scena
        # resterebbe vuota fino al riavvio.
        latched = QoSProfile(depth=1, history=QoSHistoryPolicy.KEEP_LAST,
                             durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
        self.pub_js = self.create_publisher(JointState, "joint_states", 10)
        self.pub_static = self.create_publisher(MarkerArray,
                                                "cyberknife/anatomia", latched)
        # Topic SEPARATO per i fasci del piano: con QoS transient local e
        # depth=1 la coda conserva un solo messaggio, quindi pubblicare
        # anatomia e fasci sullo stesso topic farebbe arrivare a un RViz
        # avviato dopo il nodo soltanto l'ultimo dei due.
        self.pub_plan = self.create_publisher(MarkerArray,
                                              "cyberknife/piano", latched)
        self.pub_beams = self.create_publisher(MarkerArray,
                                               "cyberknife/fasci", 10)
        self.pub_state = self.create_publisher(String, "cyberknife/stato", 10)
        # Vista del tumore: i voxel colorati cambiano di continuo, l'involucro
        # trasparente no (latched, come l'anatomia).
        self.pub_dose = self.create_publisher(MarkerArray,
                                              "cyberknife/dose_tumore", 10)
        self.pub_shell = self.create_publisher(MarkerArray,
                                               "cyberknife/tumore_involucro",
                                               latched)
        self.tf_static = StaticTransformBroadcaster(self)
        # Il paziente NON e' piu' fermo: il lettino lo orienta, quindi la
        # sua trasformazione va ripubblicata a ogni passo.
        self.tf = TransformBroadcaster(self)
        self.pub_track = self.create_publisher(
            Float64MultiArray, "cyberknife/inseguimento", 10)

        self._load_and_solve()
        self._setup_breathing()
        self._publish_base_tf()
        self._publish_anatomy()
        self._publish_static_beams()

        self._t = 0.0
        self._leg = 0
        self._done = []          # indici dei fasci gia' erogati
        self._anim_t = 0.0       # tempo di animazione (NON _clock: e' di rclpy.Node)
        self._tick_n = 0
        self._wall0 = None       # istante reale del primo passo di animazione
        self._q_track = None     # posa corrente durante l'inseguimento
        self._track_leg = -1
        self._leg_res = {}       # posa -> residui registrati in erogazione
        self._leg_path = {}      # posa -> spostamenti del percorso del fascio
        self._factors = {}       # posa -> fattore di dose per voxel
        self._dose_every = max(1, int(round(self.rate_hz / max(self.dose_hz,
                                                                 0.1))))
        self._setup_dose_view()

        # L'animazione gira su un THREAD proprio, cadenzato da time.sleep(),
        # invece che su un timer di rclpy.
        #
        # Motivo, misurato in questo ambiente (ROS 2 Humble su WSL2): un
        # create_timer(0.02) eroga circa 4 Hz invece di 50, mentre time.sleep()
        # e' accurato allo 0.5% e gli orologi di sistema sono corretti. Il
        # collo di bottiglia sta nell'attesa interna di rclpy, non nel clock:
        # con il timer il piano da 112 fasci non arrivava in fondo nemmeno in
        # 100 secondi. Il thread rende la cadenza indipendente da quel difetto.
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def _run_loop(self):
        """Cadenza l'animazione a rate_hz, recuperando i ritardi accumulati."""
        dt = 1.0 / self.rate_hz
        next_t = time.monotonic()
        while self._running and rclpy.ok():
            try:
                self._tick()
            except Exception as exc:                  # non zittire l'errore
                self.get_logger().error(f"errore nel ciclo di animazione: {exc}")
                raise
            next_t += dt
            delay = next_t - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            else:
                # In ritardo: si riparte da adesso invece di rincorrere, per
                # non accumulare una raffica di passi tutti insieme.
                next_t = time.monotonic()

    def _finish(self):
        """Chiude il nodo a trattamento concluso, e con lui l'intero lancio.

        Non basta fermare l'animazione: il processo deve USCIRE, perche' il
        launch file lo osserva e al suo termine spegne anche RViz. Una
        simulazione che resta a girare va chiusa a mano e, peggio, non
        distingue "finita" da "bloccata".

        Si lascia un istante ai marker per arrivare a RViz: chiudere subito
        lascerebbe l'ultima immagine incompleta.
        """
        self._running = False
        # La pausa e' anche il tempo concesso ai marker per arrivare a RViz.
        time.sleep(max(0.4, self.final_hold_s))
        if rclpy.ok():
            rclpy.shutdown()

    def stop(self):
        self._running = False
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)

    # -----------------------------------------------------------------------
    def _stamp(self):
        """Marca temporale MONOTONA per gli header dei messaggi.

        Non si usa direttamente l'orologio di sistema: sotto WSL resta indietro
        rispetto a Windows e si riallinea a salti. Misurato su questo nodo, il
        timestamp tornava indietro 4 volte ogni 20 secondi; ogni volta
        robot_state_publisher registra "Moved backwards in time" e SVUOTA il
        buffer TF, quindi in RViz il braccio si congela e poi salta. Con una
        sosta di 0.4 s per fascio faceva uno scatto ogni ~7 fasci.

        Si legge l'ora di sistema una volta sola e si avanza con
        time.monotonic(), che per definizione non torna indietro.
        """
        ns = self._t0_ns + int((time.monotonic() - self._t0_mono) * 1e9)
        return Time(nanoseconds=ns).to_msg()

    def _publish_base_tf(self):
        """TF statiche: la stanza e il robot, che non si muovono.

        La rotazione world->patient NON sta qui: il lettino la cambia durante
        la seduta, quindi viene ripubblicata a ogni passo da _publish_couch_tf.

        R_wp0 e' la trasposta di RobotBase.MOUNT: se MOUNT porta la terna della
        stanza su quella del paziente, l'inversa fa il percorso opposto. Una
        sola costante governa cinematica e visualizzazione, e non possono
        discordare.
        """
        stamp = self._stamp()
        R_wp0 = np.asarray(self.base.mount, dtype=np.float64).T
        # L'origine della stanza si mette sulla base del robot: e' il punto
        # fermo della scena, e cosi' il pavimento cade a z = 0.
        self._R_wp0 = R_wp0
        self._t0 = -R_wp0 @ _mm(self.base.origin)
        self._iso_room = R_wp0 @ _mm(self.iso) + self._t0
        # `couch_base` NON viene pubblicato qui: nell'URDF il secondo
        # manipolatore e' appeso a base_link con un giunto fisso, perche' un
        # URDF ammette una sola radice. Pubblicarlo anche da qui creerebbe due
        # padri per lo stesso frame.
        self.tf_static.sendTransform([
            _tf(ROOM, "base_link", np.zeros(3), R_wp0 @ self.base.R, stamp),
        ])
        self._publish_couch_tf(self.couch_poses[0])

    def _publish_couch_tf(self, couch):
        """Dove sta il paziente, dato l'orientamento del lettino.

        Il lettino ruota attorno all'ISOCENTRO, che percio' resta fermo nella
        stanza: e' quello che fa un lettino di trattamento, compensando con le
        traslazioni. Il frame `isocentro` e' figlio di `patient` e serve da
        bersaglio alle camere di RViz.
        """
        stamp = self._stamp()
        R = self._R_wp0 @ couch.R
        t = self._iso_room - R @ _mm(self.iso)
        # `vista_tumore`: centrato sull'isocentro, verticale come la stanza, e
        # ruotato lentamente. La finestra del tumore lo usa come frame fisso,
        # quindi e' la scena a girare attorno alla camera: i tunnel di dose si
        # vedono in tre dimensioni invece che da un lato solo.
        th = self.turntable * getattr(self, "_anim_t", 0.0)
        c, s_ = math.cos(th), math.sin(th)
        R_turn = np.array([[c, -s_, 0.0], [s_, c, 0.0], [0.0, 0.0, 1.0]])
        self.tf.sendTransform([
            _tf(ROOM, FRAME, t, R, stamp),
            _tf(FRAME, "isocentro", _mm(self.iso), np.eye(3), stamp),
            _tf(ROOM, "vista_tumore", self._iso_room, R_turn, stamp),
        ])

    def _load_obstacles(self):
        """Lettino e paziente come scatole, dal manifesto della scena.

        Non vengono ricalcolati qui: li scrive export_scene.py a partire dalla
        stessa maschera del corpo da cui escono le mesh. Cosi' la scatola che
        il solutore evita e quella che RViz disegna sono la stessa cosa - se
        fossero due calcoli distinti, il braccio potrebbe schivare un lettino
        diverso da quello che si vede.
        """
        import json
        self.couch_arm = None
        path = os.path.join(self.mesh_dir, "scene.json")
        if not os.path.exists(path):
            self.get_logger().warn(
                "scene.json assente: nessun ostacolo, le collisioni con il "
                "lettino NON vengono evitate.")
            return []
        with open(path, encoding="utf-8") as fh:
            scene = json.load(fh)
        # Il manipolatore che porta il lettino. La sua geometria dipende dal
        # paziente (la colonna sta oltre il piede del lettino, la cui quota
        # viene dal dorso), quindi arriva dal manifesto: qui non ci sono ne'
        # nibabel ne' le maschere per ricalcolarla.
        lc = scene.get("lettino")
        if lc:
            self.couch_arm = kin.CouchArm(lc["piano_a_riposo_mm"],
                                          lc["pavimento_mm"],
                                          l1=lc["l1_mm"], l2=lc["l2_mm"])
        else:
            self.get_logger().warn(
                "Il manifesto non descrive il braccio del lettino: il secondo "
                "manipolatore non verra' animato. Rigenera con export_scene.py")
        boxes = [kin.Box.from_dict(o) for o in scene.get("ostacoli", [])]
        if not boxes:
            self.get_logger().warn(
                "Il manifesto non contiene ostacoli: rigenera con "
                "python export_scene.py")
        return boxes

    def _load_and_solve(self):
        self.iso = np.asarray(self.plan["isocentro_mm"], dtype=np.float64)
        self.get_logger().info(
            f"Piano: {len(self.beams)} fasci, isocentro {np.round(self.iso, 1)} mm")

        self.obstacles = self._load_obstacles()
        self.get_logger().info(
            f"Ostacoli attivi: {', '.join(o.name for o in self.obstacles) or 'nessuno'}")
        self.get_logger().info(
            "Rotazioni del lettino: "
            # Il paziente resta orizzontale: si mostra l'inclinazione solo se
            # per qualche motivo non fosse nulla (il piano non la prevede).
            + ", ".join(f"yaw {c.yaw_deg:.0f}" if abs(c.pitch_deg) < 0.5
                        else f"yaw {c.yaw_deg:.0f} / pitch {c.pitch_deg:.0f}"
                        for c in self.couch_poses))
        self.poses, diag = kin.solve_plan(self.beams, self.base,
                                          n_roll=self.n_roll,
                                          order_by_motion=True,
                                          obstacles=self.obstacles,
                                          couch_poses=self.couch_poses)
        self.diag = diag
        for k, v in diag.items():
            self.get_logger().info(f"  {k}: {v}")
        if not self.poses:
            raise RuntimeError(
                "Nessun fascio raggiungibile: controllare la collocazione della "
                "base (parametro base_mm).")
        if diag["non_raggiungibili"]:
            self.get_logger().warn(
                f"{diag['non_raggiungibili']} fasci non raggiungibili: NON "
                f"verranno erogati, il piano risulta parziale.")

        # Verifica di chiusura: la IK va sempre ricontrollata rifacendo la
        # diretta. Un errore qui e' silenzioso e produce un trattamento
        # plausibile ma sbagliato.
        eff = [c.base_in_patient(self.base) for c in self.couch_poses]
        errs = [kin.fk_error_mm(p["q"], p["beam"].source,
                                p["beam"].target - p["beam"].source,
                                eff[p["couch"]])
                for p in self.poses]
        pmax = max(e[0] for e in errs)
        amax = max(e[1] for e in errs)
        self.get_logger().info(
            f"  chiusura FK/IK: posizione {pmax:.2e} mm, angolo {amax:.2e} gradi")
        if pmax > 0.01:
            self.get_logger().error("Chiusura FK/IK fuori tolleranza!")

        # --- Linea temporale dell'esecuzione ---------------------------
        # Invece di una macchina a stati con indici e fasi, si costruisce una
        # volta sola l'elenco dei tratti da percorrere. Ogni tratto sa da dove
        # parte, dove arriva, quanto dura e se il fascio e' acceso. Il ciclo di
        # animazione diventa una semplice ricerca del tratto corrente, e i
        # passaggi di transito (ritira-ruota-riscendi) entrano naturalmente
        # come tratti in piu' invece di essere un caso speciale.
        # La costruzione vive in treatment_log: il resoconto di fine
        # seduta ne dipende, e cosi' si puo' riprodurre e verificare
        # anche senza ROS.
        # Sosta PROPORZIONALE AL PESO: e' il tempo di fascio acceso a rateo
        # di dose costante. Con una sosta uguale per tutti un fascio da un
        # sesto della dose durava una frazione di respiro.
        dwell = self.dwell_s
        if self.beam_on_s > 0:
            dwell = tlog.dwell_times([p["beam"].weight for p in self.poses],
                                     self.beam_on_s, self.dwell_min_s)
            self.get_logger().info(
                f"  erogazione proporzionale al peso: {self.beam_on_s:.0f} s "
                f"di fascio acceso, da {dwell.min():.1f} a {dwell.max():.1f} "
                f"s per fascio")
        self.timeline, n_legs = tlog.build_timeline(
            self.poses, self.joint_speed, self.min_transfer_s,
            self.couch_move_s, dwell)

        t_move = sum(e["dur"] for e in self.timeline
                     if e["tipo"] in ("trasferimento", "lettino"))
        t_dose = sum(e["dur"] for e in self.timeline
                     if e["tipo"] == "erogazione")
        n_via = sum(1 for p in self.poses if p.get("via"))
        n_sim = sum(1 for p in self.poses if p.get("couch_move"))
        t_couch = sum(e["dur"] for e in self.timeline
                      if e["tipo"] == "lettino")
        if n_sim:
            self.get_logger().info(
                f"  {n_sim} rotazioni del lettino eseguite in movimento "
                f"simultaneo col braccio ({t_couch:.0f}s di manovra)")
        self.get_logger().info(
            f"  tempi: {t_move:.0f}s di spostamenti su {n_legs} tratti "
            f"({n_via} con ritiro) + {t_dose:.0f}s di erogazione")

        self.w_max = max(float(b.weight) for b in self.beams) or 1.0

    # -----------------------------------------------------------------------
    def _setup_breathing(self):
        """Sessione di inseguimento, se il piano descrive un respiro."""
        self.session = None
        self.tumor_frame = FRAME
        d = self.plan.get("respiro")
        if not d or not self.breathing:
            if d and not self.breathing:
                self.get_logger().info("Respiro nel piano, spento da "
                                       "parametro: tumore immobile.")
            return
        cfg = mm.MotionConfig.from_dict(d)
        if self.predictor:
            cfg = cfg.replace(predictor=self.predictor)
        if cfg.predictor not in mm.predictor_names():
            # La GRU vive solo su Windows (torch): qui si ripiega, e lo si dice.
            self.get_logger().warn(
                f"Predittore '{cfg.predictor}' non disponibile nel nodo: "
                f"uso 'kalman'.")
            cfg = cfg.replace(predictor="kalman")
        # La tecnica ITV pianifica sull'intera traiettoria e NON insegue: il
        # piano lo dice, e l'argomento inseguimento:=false puo' solo spegnere
        # un inseguimento, non accenderne uno che il piano non prevede.
        self.follow = self.follow and bool(d.get("inseguimento", True))
        # Gating: fascio fermo che spara solo con il tumore stimato in
        # finestra. Lo decide la tecnica scritta nel piano.
        self.gating_mm = d.get("gating_mm")
        self._motion_cfg = cfg
        self.session = self._new_session()
        self.tumor_frame = TUMOR_FRAME
        amp = np.asarray(cfg.inhale_shift_mm)
        self.get_logger().info(
            f"Respiro: escursione {np.round(amp, 1)} mm, periodo "
            f"{cfg.period_s:.1f} s, predittore {cfg.predictor}, radiografie "
            f"ogni {cfg.kv_period_s:.0f} s, inseguimento "
            f"{'ATTIVO' if self.follow else 'SPENTO (fascio fermo)'}")

    def _new_session(self):
        dur = sum(e["dur"] for e in self.timeline) + 5.0
        return TrackingSession(self._motion_cfg, dur, kin, self.base,
                               self.couch_poses, self.obstacles,
                               tracking=self.follow,
                               gating_mm=getattr(self, "gating_mm", None))

    def _breath_step(self):
        """Offset vero e offset del fascio adesso; pubblica il tumore mobile."""
        if self.session is None:
            return np.zeros(3), np.zeros(3)
        true, beam = self.session.at(self._anim_t)
        self.tf.sendTransform([_tf(FRAME, TUMOR_FRAME, _mm(true), np.eye(3),
                                   self._stamp())])
        return true, beam

    def _tracked_pose(self, e, true, boff):
        """Posa del braccio durante un'erogazione, con l'inseguimento."""
        k = e["idx"]
        if self._track_leg != k:
            self._track_leg = k
            self._q_track = np.asarray(e["q1"], dtype=np.float64)
        q, hold, why = self.session.track(
            self.poses[k]["beam"], e.get("couch", 0), boff, self._q_track,
            1.0 / self.rate_hz)
        r = self.session.record(true, boff, hold, why)
        if r is not None:
            self._leg_res.setdefault(k, []).append(r)
            # Dove e' passato il fascio (inseguimento) o dove stava il tumore
            # (fascio fermo): serve alle tabelle di attenuazione delle ossa.
            path_off = boff if self.session.tracking else true
            self._leg_path.setdefault(k, []).append(np.asarray(path_off))
        self._q_track = q
        msg = Float64MultiArray()
        err = float(np.linalg.norm(r)) if r is not None else float("nan")
        msg.data = ([float(self._anim_t)] + [float(v) for v in true]
                    + [float(v) for v in boff] + [err, 1.0 if hold else 0.0])
        self.pub_track.publish(msg)
        return q, hold

    def _factor(self, pose_k):
        """Fattore di dose dei voxel per il fascio della posa, dal residuo."""
        if self.session is None or self.dose_fill is None:
            return 1.0
        res = self._leg_res.get(pose_k)
        if not res:
            return 1.0
        g = dfill.motion_factor(self.dose_fill, self._pose_beam[pose_k],
                                self._geom, res)
        # Ossa e tessuti attraversati dal fascio spostato (tabelle del piano).
        return g * dfill.path_factor(self.dose_fill, self._pose_beam[pose_k],
                                     self._leg_path.get(pose_k))

    # -----------------------------------------------------------------------
    def _publish_anatomy(self):
        """Mesh STL degli organi, gia' in metri, nel frame del paziente."""
        import json
        arr = MarkerArray()
        manifest = os.path.join(self.mesh_dir, "scene.json")
        if not os.path.exists(manifest):
            self.get_logger().warn(f"scene.json assente in {self.mesh_dir}: "
                                   f"nessuna anatomia mostrata")
            _publish_markers(self.pub_static, arr)
            return
        with open(manifest, encoding="utf-8") as fh:
            scene = json.load(fh)

        for i, org in enumerate(scene["organi"]):
            m = Marker()
            # Il tumore respira: il suo modello segue il frame mobile.
            m.header.frame_id = (self.tumor_frame if org["key"] == "tumor"
                                 else FRAME)
            # Un namespace per organo: RViz li elenca separatamente e si
            # possono accendere e spegnere uno a uno. Serve alla Beam's Eye
            # View, dove mediastino e profilo corporeo coprono il bersaglio
            # invece di aiutare, mentre le coste servono da riferimento.
            m.ns = org["key"]
            m.id = i
            m.type = Marker.MESH_RESOURCE
            m.action = Marker.ADD
            m.mesh_resource = f"package://cyberknife_ros/meshes/{org['file']}"
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = m.scale.z = 1.0     # STL gia' in metri
            m.color = _rgba(org["color"], org["alpha"])
            arr.markers.append(m)
        _publish_markers(self.pub_static, arr)
        self.get_logger().info(f"Anatomia: {len(arr.markers)} organi pubblicati")

    # -----------------------------------------------------------------------
    def _publish_static_beams(self):
        """Tutti i fasci del piano, colorati e ispessiti secondo il peso.

        E' la vista d'insieme: dove il piano concentra intensita' e da quali
        direzioni entra. Va su un topic latched separato da quello animato.
        """
        arr = MarkerArray()
        # Tutti i fasci del piano, in grigio-azzurro SPENTO.
        #
        # Prima il colore variava col peso e i fasci intensi diventavano
        # gialli: lo stesso giallo del fascio in erogazione, quindi sembravano
        # accesi tutti insieme. L'intensita' ora si legge solo dallo SPESSORE,
        # e il giallo resta riservato al fascio che sta sparando davvero.
        for tier, (lo, hi, width) in enumerate(
                ((0.0, 0.35, 0.0012), (0.35, 1.01, 0.0032))):
            m = Marker()
            m.header.frame_id = FRAME
            m.ns = "fasci_piano"
            m.id = tier
            m.type = Marker.LINE_LIST
            m.action = Marker.ADD
            m.pose.orientation.w = 1.0
            m.scale.x = width
            for b in self.beams:
                w = float(b.weight) / self.w_max
                if not (lo <= w < hi):
                    continue
                m.points.append(_pt(b.source))
                m.points.append(_pt(b.target))
            if m.points:
                m.color = _rgba((0.42, 0.52, 0.62), 0.22 + 0.10 * tier)
                arr.markers.append(m)

        # Ostacoli: le STESSE scatole che il solutore evita. Disegnarle e'
        # l'unico modo di verificare a occhio che il braccio le schivi davvero.
        #
        # Il lettino e' pieno, il paziente solo a SPIGOLI: e' un volume di
        # esclusione grande quanto tutto il lettino, e riempirlo nasconderebbe
        # proprio l'anatomia che ci sta dentro. A spigoli si vede il vincolo
        # senza perdere di vista coste e tumore.
        for k, box in enumerate(self.obstacles):
            c = _mm((box.lo + box.hi) / 2.0)
            d = _mm(box.hi - box.lo)
            m = Marker()
            m.header.frame_id = FRAME
            m.ns = "ostacoli"
            m.id = k
            m.action = Marker.ADD
            if box.name == "lettino":
                m.type = Marker.CUBE
                m.pose.position.x, m.pose.position.y, m.pose.position.z =                     map(float, c)
                m.pose.orientation.w = 1.0
                m.scale.x, m.scale.y, m.scale.z = map(float, d)
                # Semitrasparente: il lettino con il sostegno e' un volume
                # grosso, e opaco nasconderebbe la base del robot e il braccio
                # che gli gira attorno - cioe' proprio cio' che si vuole
                # controllare guardando la simulazione.
                m.color = _rgba((0.20, 0.23, 0.28), 0.55)
            else:
                m.type = Marker.LINE_LIST
                m.pose.orientation.w = 1.0
                m.scale.x = 0.008
                lo, hi = _mm(box.lo), _mm(box.hi)
                corners = [np.array([lo[0] if i & 1 else hi[0],
                                     lo[1] if i & 2 else hi[1],
                                     lo[2] if i & 4 else hi[2]]) for i in range(8)]
                # Due vertici sono uno spigolo se differiscono per un solo bit.
                for i in range(8):
                    for j in range(i + 1, 8):
                        if bin(i ^ j).count("1") != 1:
                            continue
                        for v in (corners[i], corners[j]):
                            m.points.append(Point(x=float(v[0]), y=float(v[1]),
                                                  z=float(v[2])))
                m.color = _rgba((0.25, 0.85, 0.50), 0.55)
            arr.markers.append(m)

        # Isocentro, per orientarsi nella scena.
        s = Marker()
        s.header.frame_id = FRAME
        s.ns = "isocentro"
        s.id = 0
        s.type = Marker.SPHERE
        s.action = Marker.ADD
        p = _mm(self.iso)
        s.pose.position.x, s.pose.position.y, s.pose.position.z = map(float, p)
        s.pose.orientation.w = 1.0
        s.scale.x = s.scale.y = s.scale.z = 0.006
        s.color = _rgba((1.0, 1.0, 0.2), 0.9)
        arr.markers.append(s)

        _publish_markers(self.pub_plan, arr)

    # -----------------------------------------------------------------------
    def _tick(self):
        """Un passo dell'animazione: si avanza sulla linea temporale.

        Il tempo si accumula contando i TICK (dt nominale) e non leggendo
        l'orologio: sotto WSL l'ora di sistema si riallinea a salti, e usarla
        come tempo trascorso farebbe saltare l'animazione a meta' piano.
        """
        if self._wall0 is None:
            self._wall0 = time.monotonic()
        self._t += 1.0 / self.rate_hz
        self._anim_t += 1.0 / self.rate_hz
        self._tick_n += 1
        true, boff = self._breath_step()
        e = self.timeline[self._leg]

        if self._t >= e["dur"]:
            self._t -= e["dur"]
            self._leg += 1
            if self._leg >= len(self.timeline):
                if not self.loop:
                    self._publish_joints(
                        self.timeline[-1]["q1"],
                        self.couch_poses[self.timeline[-1].get("couch", 0)])
                    # L'ultimo tratto e' un'erogazione, ma il ciclo termina
                    # prima del punto in cui i fasci vengono segnati come
                    # erogati: senza questo l'ultimo fascio mancherebbe dalla
                    # dose, e il tumore non arriverebbe mai al suo stato finale.
                    last = self.timeline[-1]
                    if last["tipo"] == "erogazione" and last["idx"] not in self._done:
                        self._done.append(last["idx"])
                        self._accumulate(last["idx"])
                        self._publish_delivered()
                    self._publish_active(-1)
                    self._running = False
                    self._publish_dose(-1, 0.0)
                    if self.dose_fill is not None:
                        self.get_logger().info(
                            f"  bersaglio alla dose prescritta: "
                            f"{100.0 * self.dose_fill.coverage(self._acc):.1f}%")
                    self._write_report()
                    msg = (f"trattamento completato: "
                           f"{len(self.poses)} fasci erogati")
                    self._say(msg)
                    self.get_logger().info(msg.upper())
                    self._finish()
                    return
                # Ciclo continuo: il resoconto si scrive a ogni giro
                # completato, prima di ricominciare.
                self._write_report()
                self._leg = 0
            prev = self.timeline[self._leg - 1] if self._leg else None
            if prev is not None and prev["tipo"] == "erogazione":
                self._done.append(prev["idx"])
                self._accumulate(prev["idx"])
                self._publish_delivered()
            e = self.timeline[self._leg]
            if self._leg == 0:                      # ricominciato il piano
                self._done = []
                self._leg_res, self._factors = {}, {}
                self._leg_path = {}
                self._reset_dose()
                self._publish_delivered()
                if self.session is not None:
                    self._anim_t = 0.0
                    self.session = self._new_session()
            if e["tipo"] == "erogazione" and e["idx"] % 20 == 0 and e["idx"]:
                self.get_logger().info(
                    f"  avanzamento: {e['idx']}/{len(self.poses)} fasci")

        u = min(self._t / e["dur"], 1.0) if e["dur"] > 0 else 1.0
        cur_couch = self.couch_poses[e.get("couch", 0)]
        hold = False
        if e["tipo"] == "erogazione":
            q = e["q1"]
            if self.session is not None:
                q, hold = self._tracked_pose(e, true, boff)
            if hold:
                # Il fascio e' spento: il tempo dell'erogazione NON avanza. Il
                # braccio aspetta e la dose di questo fascio arriva tutta,
                # dopo. Prima il tempo scorreva lo stesso e quella dose era
                # semplicemente persa.
                self._t = max(self._t - 1.0 / self.rate_hz, 0.0)
                u = min(self._t / e["dur"], 1.0) if e["dur"] > 0 else 1.0
        else:
            # Raccordo in accelerazione: un'interpolazione lineare pura fa
            # partire e fermare il braccio di scatto. La differenza e' presa
            # DIRETTA, non riavvolta: dopo la scelta delle determinazioni i
            # valori sono gia' angoli fisici, e riavvolgerli farebbe terminare
            # il tratto su un angolo diverso da quello previsto.
            sm = u * u * (3.0 - 2.0 * u)
            q = e["q0"] + (e["q1"] - e["q0"]) * sm
            if e["tipo"] == "lettino":
                # Tratto di manovra: il lettino ruota INSIEME al braccio, con
                # lo stesso parametro. E' il movimento che kinematics ha
                # verificato (transfer_collides interpola giunti e lettino
                # con la stessa frazione): il raccordo cambia i tempi, non il
                # cammino.
                u0, u1 = e["couch_u"]
                cur_couch = kin._interp_couch(
                    self.couch_poses[e["couch"]],
                    self.couch_poses[e["couch_to"]], u0 + (u1 - u0) * sm)
        self._publish_couch_tf(cur_couch)

        self._publish_joints(q, cur_couch)
        shift = boff if e["tipo"] == "erogazione" else np.zeros(3)
        self._publish_active(e["idx"], u, shift=shift, hold=hold)
        self._maybe_publish_dose(e["idx"], u)

    # -----------------------------------------------------------------------
    def _publish_joints(self, q, couch=None):
        """Configurazione di ENTRAMBI i manipolatori.

        I giunti del lettino si ricavano risolvendo la sua cinematica inversa
        sulla posa corrente, non interpolando i valori fra un angolo e l'altro:
        cosi' il piano disegnato da RViz coincide per costruzione con la
        trasformazione del paziente pubblicata su TF. Interpolando i giunti le
        due cose divergerebbero nel mezzo della rotazione.
        """
        js = JointState()
        js.header.stamp = self._stamp()
        names = list(kin.JOINT_NAMES)
        pos = [float(v) for v in q]
        if self.couch_arm is not None:
            qc = self.couch_arm.solve_pose(couch if couch is not None
                                           else self.couch_poses[0])
            if qc is not None:
                names += ["couch_lift", "couch_shoulder", "couch_elbow",
                          "couch_roll", "couch_pitch", "couch_yaw"]
                # La colonna e' un giunto PRISMATICO: in ROS va in metri,
                # mentre la cinematica lavora in millimetri.
                pos += [float(qc[0]) * MM_TO_M] + [float(v) for v in qc[1:]]
        js.name = names
        js.position = pos
        self.pub_js.publish(js)

    def _publish_delivered(self):
        """I fasci gia' erogati, in verde tenue.

        Serve a leggere l'avanzamento: senza, guardando la simulazione non si
        capisce se il trattamento sia al decimo o al centesimo fascio.
        """
        m = Marker()
        m.header.frame_id = FRAME
        m.header.stamp = self._stamp()
        m.ns = "fasci_erogati"
        m.id = 0
        m.type = Marker.LINE_LIST
        m.action = Marker.ADD if self._done else Marker.DELETEALL
        m.pose.orientation.w = 1.0
        m.scale.x = 0.0022
        for k in self._done:
            b = self.poses[k]["beam"]
            m.points.append(_pt(b.source))
            m.points.append(_pt(b.target))
        m.color = _rgba((0.30, 0.85, 0.45), 0.45)
        arr = MarkerArray()
        arr.markers.append(m)
        _publish_markers(self.pub_beams, arr)

    def _publish_active(self, active, u=0.0, shift=None, hold=False):
        """Il fascio in erogazione, come cilindro del diametro del collimatore.

        Una linea sottile non rende: il fascio ha un diametro reale (12-25 mm)
        ed e' quello che determina la conformita' del piano. Il cilindro parte
        dalla sorgente e si allunga fino al bersaglio nella prima meta' della
        sosta, cosi' l'erogazione si vede accadere invece di comparire e
        sparire fra due fotogrammi.
        """
        arr = MarkerArray()
        m = Marker()
        m.header.frame_id = FRAME
        m.header.stamp = self._stamp()
        m.ns = "fascio_attivo"
        m.id = 0
        m.type = Marker.CYLINDER
        m.action = Marker.ADD if active >= 0 else Marker.DELETE

        core = Marker()
        core.header.frame_id = FRAME
        core.header.stamp = self._stamp()
        core.ns = "fascio_attivo"
        core.id = 1
        core.type = Marker.LINE_LIST
        core.action = m.action
        core.pose.orientation.w = 1.0
        core.scale.x = 0.004

        if active >= 0:
            b = self.poses[active]["beam"]
            off = np.zeros(3) if shift is None else np.asarray(shift, float)
            src, tgt = _mm(b.source + off), _mm(b.target + off)
            grow = min(u / 0.4, 1.0)              # si allunga, poi resta acceso
            end = src + (tgt - src) * grow
            axis = end - src
            length = float(np.linalg.norm(axis))
            if length > 1e-6:
                mid = (src + end) / 2.0
                R = kin.frame_from_axis(axis)     # Z del cilindro sul fascio
                qx, qy, qz, qw = _quat_from_matrix(R)
                m.pose.position.x, m.pose.position.y, m.pose.position.z =                     map(float, mid)
                m.pose.orientation.x, m.pose.orientation.y = qx, qy
                m.pose.orientation.z, m.pose.orientation.w = qz, qw
                d = float(b.cone_mm) * MM_TO_M
                m.scale.x = m.scale.y = d
                m.scale.z = length
                # Fascio sospeso (inseguimento impossibile): rosso tenue.
                m.color = (_rgba((1.0, 0.25, 0.25), 0.25) if hold
                           else _rgba((1.0, 0.85, 0.25), 0.35))
                core.points = [_pt(b.source + off), Point(x=float(end[0]),
                                                    y=float(end[1]),
                                                    z=float(end[2]))]
                core.colors = [_rgba((1.0, 1.0, 0.7), 0.95)] * 2
            self._say(f"fascio {active + 1}/{len(self.poses)} in erogazione "
                      f"(collimatore {b.cone_mm:.0f} mm, peso "
                      f"{b.weight / self.w_max:.2f}) - "
                      f"erogati {len(self._done)}/{len(self.poses)}")
        arr.markers.extend([m, core])
        _publish_markers(self.pub_beams, arr)

    # -----------------------------------------------------------------------
    #  Vista del tumore: la dose che si accumula
    # -----------------------------------------------------------------------
    def _setup_dose_view(self):
        """Prepara la ripartizione della dose e l'involucro del tumore.

        La dose finale di ogni voxel arriva dal piano (calcolata dalla Fase
        2); qui la si ripartisce fra i fasci con il modello geometrico di
        dose_fill. Un piano che non la contiene non blocca la simulazione, ma
        lo si dice: la finestra resterebbe vuota senza spiegazione.
        """
        self.dose_fill = dfill.DoseFill.from_plan(self.plan, self.beams)
        if self.dose_fill is None:
            self.get_logger().warn(
                "Il piano non contiene la dose nel tumore (campo dose_tumore): "
                "la vista del tumore resta vuota. Ricalcola la dosimetria e "
                "riesporta il piano.")
            return
        self._pose_beam = [p["beam_index"] for p in self.poses]
        self._geom = ([b.source for b in self.beams],
                      [b.target for b in self.beams],
                      [b.cone_mm for b in self.beams],
                      [b.weight for b in self.beams])
        self._reset_dose()
        final = self.dose_fill.dose(self._pose_beam)
        self.get_logger().info(
            f"Vista del tumore: {self.dose_fill.n_points} voxel, a fine "
            f"trattamento {100.0 * self.dose_fill.coverage(final):.1f}% del "
            f"bersaglio alla dose prescritta")

        # Involucro: la superficie liscia del tumore, quasi trasparente, sopra
        # i voxel. Da' la forma "classica" della lesione mentre dentro si
        # vedono i tunnel.
        import json
        arr = MarkerArray()
        manifest = os.path.join(self.mesh_dir, "scene.json")
        if os.path.exists(manifest):
            with open(manifest, encoding="utf-8") as fh:
                organs = json.load(fh).get("organi", [])
            for org in organs:
                if org.get("key") != "tumor":
                    continue
                m = Marker()
                m.header.frame_id = self.tumor_frame
                m.ns = "involucro"
                m.id = 0
                m.type = Marker.MESH_RESOURCE
                m.action = Marker.ADD
                m.mesh_resource = (f"package://cyberknife_ros/meshes/"
                                   f"{org['file']}")
                m.pose.orientation.w = 1.0
                m.scale.x = m.scale.y = m.scale.z = 1.0
                m.color = _rgba((0.85, 0.88, 0.95), 0.13)
                arr.markers.append(m)
        _publish_markers(self.pub_shell, arr)
        self._publish_dose(-1, 0.0)

    def _reset_dose(self):
        if self.dose_fill is not None:
            self._acc = np.zeros(self.dose_fill.n_points)

    def _accumulate(self, pose_k):
        """Somma al tumore la dose del fascio appena concluso."""
        if self.dose_fill is not None and pose_k >= 0:
            # Con il respiro la dose del fascio si scala voxel per voxel: un
            # punto uscito dal cono per un errore d'inseguimento ne riceve
            # meno. Il fattore e' lo stesso che il resoconto ricalcola.
            g = self._factor(pose_k)
            self._factors[pose_k] = g
            self._acc = (self._acc + g *
                         self.dose_fill.contrib[:, self._pose_beam[pose_k]])

    def _maybe_publish_dose(self, active, u):
        if self._tick_n % self._dose_every == 0:
            self._publish_dose(active, u)

    def _publish_dose(self, active, u):
        """Voxel del tumore colorati dalla dose ricevuta fin qui.

        Il fascio acceso deposita la sua dose GRADUALMENTE, dopo che il
        cilindro giallo ha raggiunto il bersaglio (primo 40% della sosta): il
        tunnel si vede riempire mentre il fascio lo attraversa.
        """
        if self.dose_fill is None:
            return
        df = self.dose_fill
        d = self._acc
        if active is not None and active >= 0:
            ud = min(max((u - 0.4) / 0.6, 0.0), 1.0)
            if ud > 0.0:
                g = self._factor(active)
                d = d + ud * g * df.contrib[:, self._pose_beam[active]]
        rgba = df.colors(d)
        stamp = self._stamp()

        cubes = Marker()
        cubes.header.frame_id = self.tumor_frame
        cubes.header.stamp = stamp
        cubes.ns = "dose"
        cubes.id = 0
        cubes.type = Marker.CUBE_LIST
        cubes.action = Marker.ADD
        cubes.pose.orientation.w = 1.0
        side = df.step_mm * 0.92 * MM_TO_M
        cubes.scale.x = cubes.scale.y = cubes.scale.z = side
        pts = df.points * MM_TO_M
        cubes.points = [Point(x=float(q[0]), y=float(q[1]), z=float(q[2]))
                        for q in pts]
        cubes.colors = [ColorRGBA(r=float(c[0]), g=float(c[1]),
                                  b=float(c[2]), a=float(c[3])) for c in rgba]

        active_on = active is not None and active >= 0
        n_done = len(self._done) + (1 if active_on else 0)
        cov = 100.0 * df.coverage(d)
        finished = not getattr(self, "_running", True)
        head = Marker()
        head.header.frame_id = "vista_tumore"
        head.header.stamp = stamp
        head.ns = "didascalia"
        head.id = 0
        head.type = Marker.TEXT_VIEW_FACING
        head.action = Marker.ADD
        head.pose.position.z = 0.030
        head.pose.orientation.w = 1.0
        head.scale.z = 0.0055
        if finished:
            head.text = (f"TRATTAMENTO COMPLETATO - {cov:.0f}% del bersaglio "
                         f"alla dose prescritta")
            head.color = _rgba((0.35, 0.95, 0.50), 1.0)
        else:
            head.text = (f"fascio {n_done}/{len(self.poses)}  -  bersaglio "
                         f"alla dose prescritta {cov:.0f}%")
            head.color = _rgba((0.92, 0.94, 0.97), 1.0)

        legend = Marker()
        legend.header.frame_id = "vista_tumore"
        legend.header.stamp = stamp
        legend.ns = "didascalia"
        legend.id = 1
        legend.type = Marker.TEXT_VIEW_FACING
        legend.action = Marker.ADD
        legend.pose.position.z = -0.030
        legend.pose.orientation.w = 1.0
        legend.scale.z = 0.0035
        legend.text = ("rosso: nessuna dose   giallo: quasi   "
                       "verde: dose prescritta")
        legend.color = _rgba((0.65, 0.70, 0.78), 1.0)

        arr = MarkerArray()
        arr.markers.extend([cubes, head, legend])
        _publish_markers(self.pub_dose, arr)

    def _write_report(self):
        """Resoconto della seduta, per la finestra di fine trattamento.

        Non deve mai fermare la simulazione: un errore qui si registra e basta.
        """
        from datetime import datetime, timezone
        try:
            wall = (time.monotonic() - self._wall0
                    if self._wall0 is not None else None)
            report = tlog.build_report(
                self.plan, self.beams, self.poses, self.timeline, self.diag,
                self.couch_poses, dose_fill=self.dose_fill, wall_s=wall,
                respiro=(self.session.summary()
                         if self.session is not None else None),
                residui={k: dfill.pick_residuals(v).tolist()
                         for k, v in self._leg_res.items()},
                percorsi={k: dfill.pick_residuals(v).tolist()
                          for k, v in self._leg_path.items()},
                meta={
                    "scritto": datetime.now(timezone.utc).isoformat(
                        timespec="seconds"),
                    "piano": self.plan_path,
                    "rate_hz": self.rate_hz,
                    "dwell_s": self.dwell_s,
                    "joint_speed_deg_s": self.joint_speed,
                    "min_transfer_s": self.min_transfer_s,
                    "couch_move_s": self.couch_move_s,
                    "n_roll": self.n_roll,
                })
            tlog.write_report(report, self.report_path)
            self.get_logger().info(f"Resoconto scritto: {self.report_path}")
        except Exception as exc:                      # noqa: BLE001
            self.get_logger().error(f"Resoconto non scritto: {exc}")

    def _say(self, msg):
        self.pub_state.publish(String(data=msg))


def main(args=None):
    rclpy.init(args=args)
    node = TreatmentNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception:                      # shutdown chiesto dal nodo stesso
        if rclpy.ok():
            raise
    finally:
        node.stop()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
