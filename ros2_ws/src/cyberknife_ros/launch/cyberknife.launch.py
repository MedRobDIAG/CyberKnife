# ============================================================================
#  cyberknife.launch.py - Avvia la simulazione completa.
#
#  Tre nodi:
#    robot_state_publisher  legge l'URDF e trasforma /joint_states in TF
#    cyberknife_treatment   risolve la cinematica e anima il piano
#    rviz2                  visualizza (opzionale: rviz:=false per il collaudo)
#    rviz2_bev, rviz2_tumore  Beam's Eye View e tumore che si riempie di dose
#
#  ESEMPI
#    ros2 launch cyberknife_ros cyberknife.launch.py
#    ros2 launch cyberknife_ros cyberknife.launch.py dwell_s:=0.2 joint_speed_deg_s:=180.0
#    ros2 launch cyberknife_ros cyberknife.launch.py rviz:=false
#    ros2 launch cyberknife_ros cyberknife.launch.py loop:=true   (non si ferma)
#    ros2 launch cyberknife_ros cyberknife.launch.py inseguimento:=false
#        (il tumore respira ma il fascio resta fermo: si vede la dose persa)
#
#  FINE TRATTAMENTO
#  ----------------
#  Erogato l'ultimo fascio il nodo termina, e un gestore di evento spegne
#  l'INTERO lancio, RViz compreso. Una simulazione che resta a girare in
#  eterno costringe a chiuderla a mano e, peggio, non distingue "finita" da
#  "bloccata".
# ============================================================================
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, EmitEvent, RegisterEventHandler
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from typing import List


def _num(name, kind=float):
    """Argomento del launch tipizzato per il nodo.

    Gli argomenti arrivano come TESTO e rclpy deduce il tipo dal contenuto:
    `rate_hz:=50` diventa un intero e il nodo muore con
    InvalidParameterTypeException perche' il parametro e' dichiarato double.
    Chi lancia la simulazione non deve dover ricordare di scrivere `50.0`:
    dichiarando qui il tipo atteso, la conversione la fa il launch.
    """
    return ParameterValue(LaunchConfiguration(name), value_type=kind)


def generate_launch_description():
    share = get_package_share_directory("cyberknife_ros")
    urdf = os.path.join(share, "urdf", "cyberknife.urdf")
    with open(urdf, "r", encoding="utf-8") as fh:
        robot_description = fh.read()

    args = [
        DeclareLaunchArgument("rviz", default_value="true"),
        DeclareLaunchArgument("bev", default_value="true"),
        # Terza finestra: il tumore che si riempie di dose.
        DeclareLaunchArgument("tumore", default_value="true"),
        # Pausa sul tumore ormai verde prima di chiudere tutto: senza, il
        # momento in cui il bersaglio risulta coperto dura un fotogramma.
        DeclareLaunchArgument("final_hold_s", default_value="4.0"),
        # Rotazione della camera attorno al tumore (gradi al secondo).
        DeclareLaunchArgument("turntable_deg_s", default_value="12.0"),
        DeclareLaunchArgument("dose_view_hz", default_value="12.0"),
        # Resoconto di fine seduta. Vuoto = accanto al piano, cioe' nella
        # cartella del progetto su Windows, dove l'app lo va a leggere.
        DeclareLaunchArgument("report_path", default_value=""),
        DeclareLaunchArgument("loop", default_value="false"),
        DeclareLaunchArgument("dwell_s", default_value="0.6"),
        DeclareLaunchArgument("joint_speed_deg_s", default_value="90.0"),
        # Questi due erano parametri del nodo ma NON argomenti del launch:
        # passarli da riga di comando non aveva alcun effetto e la simulazione
        # restava lenta, con i tempi calcolati sui valori di default.
        DeclareLaunchArgument("min_transfer_s", default_value="0.5"),
        DeclareLaunchArgument("couch_move_s", default_value="3.0"),
        DeclareLaunchArgument("rate_hz", default_value="50.0"),
        # Ripiego per piani che non portano la configurazione di sala (formato
        # 1) o per prove manuali: normalmente base e angoli di lettino vengono
        # dal piano, perche' i fasci sono stati scelti sapendo da dove
        # eseguirli. Esposti comunque, cosi' si possono provare da riga di
        # comando senza toccare il codice.
        DeclareLaunchArgument("base_mm", default_value="[-1000.0, -830.0, -300.0]"),
        DeclareLaunchArgument("base_yaw_deg", default_value="0.0"),
        DeclareLaunchArgument("n_roll", default_value="4"),
        # Respiro (se il piano lo descrive): spento, oppure a fascio fermo per
        # vedere che cosa perde un trattamento senza inseguimento.
        DeclareLaunchArgument("respiro", default_value="true"),
        DeclareLaunchArgument("inseguimento", default_value="true"),
        # Vuoto = il predittore scritto nel piano (none/linear/kalman/nlms).
        DeclareLaunchArgument("predittore", default_value=""),
        # Tempo totale di fascio acceso, ripartito sui fasci in proporzione al
        # peso. 0 = sosta fissa dwell_s per ogni fascio (comportamento vecchio).
        DeclareLaunchArgument("erogazione_s", default_value="120.0"),
        DeclareLaunchArgument("dwell_min_s", default_value="0.3"),
        DeclareLaunchArgument(
            "plan", default_value=os.path.join(share, "plans",
                                               "piano_trattamento.json")),
    ]

    treatment = Node(
            package="cyberknife_ros",
            executable="treatment_node",
            name="cyberknife_treatment",
            output="screen",
            parameters=[{
                "share_dir": share,
                "plan": LaunchConfiguration("plan"),
                "mesh_dir": os.path.join(share, "meshes"),
                "loop": _num("loop", bool),
                "dwell_s": _num("dwell_s"),
                "joint_speed_deg_s": _num("joint_speed_deg_s"),
                "min_transfer_s": _num("min_transfer_s"),
                "couch_move_s": _num("couch_move_s"),
                "rate_hz": _num("rate_hz"),
                "base_mm": _num("base_mm", List[float]),
                "base_yaw_deg": _num("base_yaw_deg"),
                "n_roll": _num("n_roll", int),
                "final_hold_s": _num("final_hold_s"),
                "turntable_deg_s": _num("turntable_deg_s"),
                "dose_view_hz": _num("dose_view_hz"),
                "report_path": _num("report_path", str),
                "respiro": _num("respiro", bool),
                "inseguimento": _num("inseguimento", bool),
                "predittore": _num("predittore", str),
                "erogazione_s": _num("erogazione_s"),
                "dwell_min_s": _num("dwell_min_s"),
            }],
        )

    return LaunchDescription(args + [
        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            name="robot_state_publisher",
            output="screen",
            parameters=[{"robot_description": robot_description}],
        ),
        treatment,
        Node(
            package="rviz2",
            executable="rviz2",
            name="rviz2",
            output="screen",
            condition=IfCondition(LaunchConfiguration("rviz")),
            arguments=["-d", os.path.join(share, "rviz", "cyberknife.rviz")],
        ),
        # Seconda finestra con la Beam's Eye View. RViz disegna una sola vista
        # per finestra, quindi per averne due contemporaneamente servono due
        # processi. Guardare la simulazione da un solo punto di vista non basta:
        # dalla sala si capisce dove va il braccio, dal fascio si capisce se il
        # bersaglio e' centrato.
        Node(
            package="rviz2",
            executable="rviz2",
            name="rviz2_bev",
            output="log",
            condition=IfCondition(LaunchConfiguration("bev")),
            arguments=["-d", os.path.join(share, "rviz", "bev.rviz")],
        ),
        # Terza finestra: il tumore visto da vicino, colorato dalla dose
        # accumulata. E' la vista che dice COSA sta ottenendo il trattamento,
        # mentre le altre due dicono come il robot ci arriva.
        Node(
            package="rviz2",
            executable="rviz2",
            name="rviz2_tumore",
            output="log",
            condition=IfCondition(LaunchConfiguration("tumore")),
            arguments=["-d", os.path.join(share, "rviz", "tumore.rviz")],
        ),
        # Quando il nodo di trattamento esce - cioe' quando ha erogato
        # l'ultimo fascio - si spegne tutto il resto. Senza questo, RViz e
        # robot_state_publisher resterebbero aperti su una scena ferma.
        RegisterEventHandler(OnProcessExit(
            target_action=treatment,
            on_exit=[EmitEvent(event=Shutdown(
                reason="trattamento completato"))],
        )),
    ])
