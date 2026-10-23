# ============================================================================
#  urdf_gen.py - Genera l'URDF del manipolatore DALLE COSTANTI di kinematics.py.
#
#  PERCHE' GENERARLO E NON SCRIVERLO A MANO
#  ----------------------------------------
#  E' l'errore classico della parte robotica: l'URDF che RViz disegna e la
#  cinematica che calcola gli angoli sono due descrizioni dello stesso robot,
#  scritte in posti diversi. Appena una delle due cambia, RViz mostra un robot
#  in una posa e il piano ne calcola un'altra - e la discrepanza non da' alcun
#  errore, solo un risultato sbagliato.
#
#  Qui l'URDF e' un DERIVATO di kinematics.py: le lunghezze dei segmenti, gli
#  assi e i limiti di giunto vengono letti da li'. Non possono divergere.
#
#  CORRISPONDENZA CON LA CINEMATICA
#  --------------------------------
#  La catena riproduce esattamente
#      R03 = Rz(q1) Ry(q2+q3+phi)          R36 = Rz(q4) Ry(q5) Rz(q6)
#  dove phi = atan2(A3, D4) e' lo scostamento costante del gomito. Nell'URDF
#  quel phi non e' un giunto ma una rotazione FISSA fra il terzo braccio e il
#  polso, e lo scostamento diventa la traslazione (D4, 0, -A3).
#
#  URDF lavora in METRI: la conversione da millimetri avviene solo qui.
#
#  USO:  python urdf_gen.py            (riscrive urdf/ e la copia di kinematics)
# ============================================================================
import os
import shutil

import numpy as np

import kinematics as kin

MM = 1000.0     # millimetri -> metri

#: Quanto il disegno e' piu' sottile dell'ingombro usato per le collisioni.
#: I raggi visivi NON sono scelti a occhio: derivano da kin.LINK_RADII_MM, cosi'
#: quello che si vede in RViz corrisponde a quello che il planner considera
#: occupato. Se il disegno fosse piu' sottile del modello, il braccio
#: sembrerebbe schivare il lettino con un margine che in realta' non ha.
VISUAL_SHRINK = 0.92


def _r(k):
    """Raggio visivo della capsula k, in metri."""
    return float(kin.LINK_RADII_MM[k]) * VISUAL_SHRINK / MM


def _joint(name, parent, child, axis, xyz, rpy, lo, hi):
    return f"""
  <joint name="{name}" type="revolute">
    <parent link="{parent}"/>
    <child link="{child}"/>
    <origin xyz="{xyz}" rpy="{rpy}"/>
    <axis xyz="{axis}"/>
    <limit lower="{lo:.6f}" upper="{hi:.6f}" effort="1000.0" velocity="1.5"/>
  </joint>"""


def _link_cyl(name, length_m, radius_m, color, axis="z", offset=None):
    """Link visualizzato come cilindro lungo l'asse indicato."""
    if axis == "z":
        rpy, xyz = "0 0 0", f"0 0 {length_m/2:.6f}"
    elif axis == "x":
        rpy, xyz = "0 1.5707963 0", f"{length_m/2:.6f} 0 0"
    else:
        rpy, xyz = "1.5707963 0 0", f"0 {length_m/2:.6f} 0"
    if offset is not None:
        xyz = offset
    return f"""
  <link name="{name}">
    <visual>
      <origin xyz="{xyz}" rpy="{rpy}"/>
      <geometry><cylinder length="{max(length_m,1e-3):.6f}" radius="{radius_m:.6f}"/></geometry>
      <material name="{color}"/>
    </visual>
  </link>"""


def couch_offset_from_package(pkg_dir=None):
    """Posizione della colonna del lettino nel sistema del ROBOT, in metri.

    E' l'origine del giunto fisso che appende il secondo manipolatore
    all'albero dell'URDF. Si ricava dai file del pacchetto - la scena da' la
    colonna, il piano da' la base del robot - perche' entrambe dipendono dal
    paziente. Se mancano si restituisce None e l'URDF usa un ripiego.
    """
    import json
    here = os.path.dirname(os.path.abspath(__file__))
    pkg = pkg_dir or os.path.join(here, "ros2_ws", "src", "cyberknife_ros")
    try:
        scene = json.load(open(os.path.join(pkg, "meshes", "scene.json"),
                               encoding="utf-8"))
        plan = json.load(open(os.path.join(pkg, "plans",
                                           "piano_trattamento.json"),
                              encoding="utf-8"))
        col = np.asarray(scene["lettino"]["colonna_mm"], dtype=np.float64)
        r = plan["robot"]
        base = kin.RobotBase(origin_mm=tuple(r["base_mm"]),
                             yaw_deg=float(r.get("yaw_deg", 0.0)))
    except (OSError, KeyError, TypeError, ValueError):
        return None
    return base.to_robot(col) / MM


def build_urdf(couch_offset=None):
    """Compone l'URDF completo a partire dalle costanti della cinematica."""
    phi = float(np.arctan2(kin.A3, kin.D4))
    lim = kin.JOINT_LIMITS

    parts = ['<?xml version="1.0"?>',
             '<!-- GENERATO da urdf_gen.py: non modificare a mano.',
             '     Le quote derivano dalle costanti di kinematics.py. -->',
             '<robot name="cyberknife">']

    for nm, rgba in (("acciaio", "0.72 0.74 0.78 1"),
                     ("arancio", "1.0 0.54 0.0 1"),
                     ("scuro", "0.18 0.20 0.24 1"),
                     ("ciano", "0.0 0.90 1.0 1"),
                     ("lettino", "0.30 0.34 0.40 1")):
        parts.append(f'\n  <material name="{nm}"><color rgba="{rgba}"/></material>')

    # --- Base ---
    parts.append("""
  <link name="base_link">
    <visual>
      <origin xyz="0 0 0.125" rpy="0 0 0"/>
      <geometry><cylinder length="0.25" radius="0.30"/></geometry>
      <material name="scuro"/>
    </visual>
  </link>""")

    # --- A1: rotazione attorno a Z. Il link e' una L: colonna verticale alta
    #     D1 piu' sbraccio orizzontale A1 fino alla spalla. Lo sbraccio prima
    #     non era disegnato affatto, e il braccio sembrava interrompersi.
    parts.append(f"""
  <link name="link_1">
    <visual>
      <origin xyz="0 0 {kin.D1/2/MM:.6f}" rpy="0 0 0"/>
      <geometry><cylinder length="{kin.D1/MM:.6f}" radius="{_r(0):.4f}"/></geometry>
      <material name="acciaio"/>
    </visual>
    <visual>
      <origin xyz="{kin.A1/2/MM:.6f} 0 {kin.D1/MM:.6f}" rpy="0 1.5707963 0"/>
      <geometry><cylinder length="{kin.A1/MM:.6f}" radius="{_r(1):.4f}"/></geometry>
      <material name="acciaio"/>
    </visual>
  </link>""")
    parts.append(_joint("a1", "base_link", "link_1", "0 0 1",
                        "0 0 0", "0 0 0", lim[0, 0], lim[0, 1]))

    # --- A2: spalla. Origine spostata di (A1, 0, D1) ---
    parts.append(_link_cyl("link_2", kin.A2 / MM, _r(2), "acciaio", axis="x"))
    parts.append(_joint("a2", "link_1", "link_2", "0 1 0",
                        f"{kin.A1/MM:.6f} 0 {kin.D1/MM:.6f}", "0 0 0",
                        lim[1, 0], lim[1, 1]))

    # --- A3: gomito, all'estremita' del braccio ---
    parts.append(_link_cyl("link_3", kin.D4 / MM, _r(3), "acciaio", axis="x"))
    parts.append(_joint("a3", "link_2", "link_3", "0 1 0",
                        f"{kin.A2/MM:.6f} 0 0", "0 0 0",
                        lim[2, 0], lim[2, 1]))

    # --- Polso: traslazione (D4,0,-A3) e rotazione FISSA di phi ---
    # E' qui che lo scostamento del gomito viene assorbito: nella cinematica
    # compare come il termine costante +phi dentro R03.
    parts.append('\n  <link name="polso_base"/>')
    parts.append(f"""
  <joint name="offset_gomito" type="fixed">
    <parent link="link_3"/>
    <child link="polso_base"/>
    <origin xyz="{kin.D4/MM:.6f} 0 {-kin.A3/MM:.6f}" rpy="0 {phi:.6f} 0"/>
  </joint>""")

    # --- Polso sferico: A4 (Z), A5 (Y), A6 (Z) ---
    parts.append(_link_cyl("link_4", 0.12, _r(4) * 0.62, "acciaio", axis="z"))
    parts.append(_joint("a4", "polso_base", "link_4", "0 0 1",
                        "0 0 0", "0 0 0", lim[3, 0], lim[3, 1]))
    parts.append(_link_cyl("link_5", 0.12, _r(4) * 0.55, "acciaio", axis="z"))
    parts.append(_joint("a5", "link_4", "link_5", "0 1 0",
                        "0 0 0", "0 0 0", lim[4, 0], lim[4, 1]))
    parts.append(_link_cyl("link_6", kin.D6 / MM, _r(4) * 0.48, "acciaio", axis="z"))
    parts.append(_joint("a6", "link_5", "link_6", "0 0 1",
                        "0 0 0", "0 0 0", lim[5, 0], lim[5, 1]))

    # --- Testata dell'acceleratore, fino al punto focale ---
    parts.append(f"""
  <link name="linac">
    <visual>
      <origin xyz="0 0 {kin.LINAC_LENGTH/2/MM:.6f}" rpy="0 0 0"/>
      <geometry><cylinder length="{kin.LINAC_LENGTH/MM:.6f}" radius="{_r(4):.4f}"/></geometry>
      <material name="arancio"/>
    </visual>
  </link>
  <joint name="flangia" type="fixed">
    <parent link="link_6"/>
    <child link="linac"/>
    <origin xyz="0 0 {kin.D6/MM:.6f}" rpy="0 0 0"/>
  </joint>""")

    # --- Sorgente: il punto focale dei raggi X. E' il frame che deve
    #     coincidere con la posa richiesta dal piano. ---
    parts.append("""
  <link name="sorgente"/>""")
    parts.append(f"""
  <joint name="punto_focale" type="fixed">
    <parent link="linac"/>
    <child link="sorgente"/>
    <origin xyz="0 0 {kin.LINAC_LENGTH/MM:.6f}" rpy="0 0 0"/>
  </joint>""")

    # --- Frame per le camere di RViz -----------------------------------
    # RViz punta la camera lungo +X del frame bersaglio, mentre il fascio esce
    # lungo +Z della sorgente: agganciare la vista direttamente alla sorgente
    # inquadrerebbe di traverso. Questo frame ruotato risolve il disallineamento
    # una volta sola, e stando nell'URDF segue la catena senza codice in piu'.
    parts.append("""
  <link name="bev"/>
  <joint name="asse_fascio" type="fixed">
    <parent link="sorgente"/>
    <child link="bev"/>
    <origin xyz="0 0 0" rpy="0 -1.5707963 0"/>
  </joint>""")

    # --- Secondo manipolatore: il RoboCouch --------------------------------
    # Il lettino non e' un piano fisso: e' montato all'estremita' di un altro
    # braccio - colonna verticale prismatica, due rotoidali in pianta (spalla e
    # gomito) e polso roll-pitch-yaw. Le quote vengono dalle stesse costanti
    # della cinematica, quindi quello che RViz disegna e quello che il solutore
    # calcola non possono divergere.
    #
    # La radice `couch_base` viene collocata dal nodo con una TF statica: la
    # colonna sta oltre il piede del lettino, e quella posizione dipende dal
    # paziente, mentre l'URDF deve restarne indipendente.
    cl = kin.COUCH_JOINT_LIMITS
    # Un URDF ammette UNA SOLA radice: `couch_base` va appeso all'albero, non
    # lasciato come secondo albero indipendente (robot_state_publisher
    # abortisce senza dire perche'). Lo si aggancia a base_link con un giunto
    # fisso, la cui origine e' la posizione della colonna espressa nel sistema
    # del robot dell'acceleratore. Quella posizione dipende dal paziente, per
    # questo build_urdf accetta la geometria della scena.
    off = couch_offset if couch_offset is not None else (0.0, -2.0, 0.0)
    parts.append(f"""
  <joint name="ancoraggio_lettino" type="fixed">
    <parent link="base_link"/>
    <child link="couch_base"/>
    <origin xyz="{off[0]:.6f} {off[1]:.6f} {off[2]:.6f}" rpy="0 0 0"/>
  </joint>""")
    parts.append("""
  <link name="couch_base">
    <visual>
      <origin xyz="0 0 0.09" rpy="0 0 0"/>
      <geometry><cylinder length="0.18" radius="0.26"/></geometry>
      <material name="scuro"/>
    </visual>
  </link>""")
    parts.append(f"""
  <link name="couch_column">
    <visual>
      <origin xyz="0 0 -0.35" rpy="0 0 0"/>
      <geometry><cylinder length="0.70" radius="{_r(0) * 0.85:.4f}"/></geometry>
      <material name="scuro"/>
    </visual>
  </link>
  <joint name="couch_lift" type="prismatic">
    <parent link="couch_base"/>
    <child link="couch_column"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="{kin.COUCH_LIFT_RANGE_MM[0] / MM:.4f}" upper="{kin.COUCH_LIFT_RANGE_MM[1] / MM:.4f}" effort="8000.0" velocity="0.2"/>
  </joint>""")
    parts.append(_link_cyl("couch_arm1", kin.COUCH_L1_MM / MM,
                           float(kin.COUCH_LINK_RADII_MM[1]) * VISUAL_SHRINK / MM,
                           "acciaio", axis="x"))
    parts.append(_joint("couch_shoulder", "couch_column", "couch_arm1",
                        "0 0 1", "0 0 0", "0 0 0", cl[0, 0], cl[0, 1]))
    parts.append(_link_cyl("couch_arm2", kin.COUCH_L2_MM / MM,
                           float(kin.COUCH_LINK_RADII_MM[2]) * VISUAL_SHRINK / MM,
                           "acciaio", axis="x"))
    parts.append(_joint("couch_elbow", "couch_arm1", "couch_arm2", "0 0 1",
                        f"{kin.COUCH_L1_MM / MM:.6f} 0 0", "0 0 0",
                        cl[1, 0], cl[1, 1]))
    parts.append('\n  <link name="couch_wrist1"/>')
    parts.append(_joint("couch_roll", "couch_arm2", "couch_wrist1", "1 0 0",
                        f"{kin.COUCH_L2_MM / MM:.6f} 0 0", "0 0 0",
                        cl[2, 0], cl[2, 1]))
    parts.append('\n  <link name="couch_wrist2"/>')
    parts.append(_joint("couch_pitch", "couch_wrist1", "couch_wrist2", "0 1 0",
                        "0 0 0", "0 0 0", cl[3, 0], cl[3, 1]))
    # Il piano vero e proprio: e' la stessa scatola che il solutore evita.
    parts.append(f"""
  <link name="couch_top">
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry><box size="{kin.COUCH_LENGTH_MM / MM:.4f} {kin.COUCH_WIDTH_MM / MM:.4f} {kin.COUCH_THICKNESS_MM / MM:.4f}"/></geometry>
      <material name="lettino"/>
    </visual>
  </link>""")
    parts.append(_joint("couch_yaw", "couch_wrist2", "couch_piatto", "0 0 1",
                        "0 0 0", "0 0 0", cl[4, 0], cl[4, 1]))
    # SOSTEGNO: il piano sta sopra il polso, non attorno. Il braccio corre
    # sotto il lettino e lo regge da sotto - senza questo, il cilindro del
    # secondo braccio (raggio 150 mm) attraversava il corpo del paziente in
    # RViz. La quota e' la stessa costante che usa la cinematica, quindi
    # disegno e calcolo non possono divergere.
    parts.append(f"""
  <link name="couch_piatto">
    <visual>
      <origin xyz="0 0 {kin.COUCH_ARM_DROP_MM / MM / 2:.4f}" rpy="0 0 0"/>
      <geometry><cylinder length="{kin.COUCH_ARM_DROP_MM / MM:.4f}" radius="0.075"/></geometry>
      <material name="scuro"/>
    </visual>
  </link>
  <joint name="couch_sostegno" type="fixed">
    <parent link="couch_piatto"/>
    <child link="couch_top"/>
    <origin xyz="0 0 {kin.COUCH_ARM_DROP_MM / MM:.6f}" rpy="0 0 0"/>
  </joint>"""),

    parts.append("\n</robot>\n")
    return "".join(parts)


# ---------------------------------------------------------------------------
#  Cinematica diretta ricavata DALL'URDF, per il controllo incrociato
# ---------------------------------------------------------------------------
def urdf_forward_kinematics(q):
    """Percorre la catena descritta sopra, indipendentemente da kinematics.FK.

    Serve al test: se questa e la forward_kinematics analitica non coincidono,
    l'URDF e il calcolo descrivono robot diversi.
    """
    def T(xyz, rpy=(0, 0, 0), R=None):
        M = np.eye(4)
        if R is None:
            r, p, y = rpy
            R = kin.rot_z(y) @ kin.rot_y(p) @ kin.rot_x(r)
        M[:3, :3] = R
        M[:3, 3] = xyz
        return M

    phi = float(np.arctan2(kin.A3, kin.D4))
    q1, q2, q3, q4, q5, q6 = q
    M = np.eye(4)
    M = M @ T((0, 0, 0), R=kin.rot_z(q1))                       # a1
    M = M @ T((kin.A1, 0, kin.D1)) @ T((0, 0, 0), R=kin.rot_y(q2))   # a2
    M = M @ T((kin.A2, 0, 0)) @ T((0, 0, 0), R=kin.rot_y(q3))        # a3
    M = M @ T((kin.D4, 0, -kin.A3), rpy=(0, phi, 0))            # offset gomito
    M = M @ T((0, 0, 0), R=kin.rot_z(q4))                       # a4
    M = M @ T((0, 0, 0), R=kin.rot_y(q5))                       # a5
    M = M @ T((0, 0, 0), R=kin.rot_z(q6))                       # a6
    M = M @ T((0, 0, kin.D6))                                   # flangia
    M = M @ T((0, 0, kin.LINAC_LENGTH))                         # punto focale
    return M[:3, 3], M[:3, :3] @ np.array([0.0, 0.0, 1.0])


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    pkg = os.path.join(here, "ros2_ws", "src", "cyberknife_ros")
    urdf_dir = os.path.join(pkg, "urdf")
    os.makedirs(urdf_dir, exist_ok=True)

    # La cinematica va copiata nel pacchetto: dopo `colcon build` il nodo gira
    # da una cartella di installazione e non puo' risalire alla radice del
    # progetto. Un test verifica che la copia resti identica all'originale.
    # Il pacchetto deve essere AUTOSUFFICIENTE: dopo `colcon build` il nodo gira
    # da install/ e non puo' risalire alla radice del progetto. Tutto cio' che
    # gli serve viene copiato qui, e test_phase3.py verifica che le copie non
    # siano rimaste indietro. Non e' pedanteria: con il piano non sincronizzato
    # il nodo ha girato su una versione vecchia da 112 fasci mentre su Windows
    # ne erano stati calcolati 135, e i due risultati non tornavano.
    os.makedirs(os.path.join(pkg, "cyberknife_ros"), exist_ok=True)
    for fn in ("kinematics.py", "plan_export.py", "dose_fill.py",
               "treatment_log.py", "motion_model.py", "tracking_session.py"):
        dst = os.path.join(pkg, "cyberknife_ros", fn)
        shutil.copyfile(os.path.join(here, fn), dst)
        print(f"{fn} copiato nel pacchetto: {dst}")

    plan = os.path.join(here, "outputs", "piano_trattamento.json")
    dst = os.path.join(pkg, "plans", "piano_trattamento.json")

    def _generato(path):
        try:
            import json
            return json.load(open(path, encoding="utf-8")).get("generato", "")
        except (OSError, ValueError):
            return ""

    # Il pulsante di avvio del Tab 3 esporta DIRETTAMENTE nel pacchetto, non in
    # outputs/: il piano del pacchetto puo' quindi essere il piu' recente.
    # Copiarci sopra quello di outputs/ senza guardare riportava nel pacchetto
    # un paziente vecchio, con la scena di quello nuovo - il disallineamento
    # che faceva mancare i fasci e sparare lontano dal tumore.
    if os.path.exists(plan) and _generato(plan) < _generato(dst):
        print(f"il piano del pacchetto ({_generato(dst)}) e' piu' recente di "
              f"outputs/ ({_generato(plan)}): lo si conserva.")
    elif os.path.exists(plan):
        os.makedirs(os.path.join(pkg, "plans"), exist_ok=True)
        shutil.copyfile(plan, dst)
        print(f"piano di trattamento sincronizzato: {dst}")
    else:
        print("ATTENZIONE: outputs/piano_trattamento.json non trovato, "
              "il pacchetto conserva il piano precedente.")

    # L'URDF si genera PER ULTIMO: l'origine del giunto che appende il lettino
    # dipende dalla base del robot, che arriva dal piano appena copiato. Se lo
    # si generasse prima, la colonna resterebbe dove la voleva il piano
    # PRECEDENTE e servirebbero due passaggi per convergere.
    path = os.path.join(urdf_dir, "cyberknife.urdf")
    off = couch_offset_from_package(pkg)
    if off is None:
        print("ATTENZIONE: scena o piano mancanti, la colonna del lettino "
              "va in una posizione di ripiego.")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(build_urdf(off))
    print(f"URDF generato: {path}")
