# ============================================================================
#  test_phase3.py - Controlli sulla Fase 3 (cinematica + pacchetto ROS 2).
#
#  Gira su Windows, senza ROS: verifica tutto cio' che si puo' verificare
#  fuori da rclpy, cioe' la matematica e la coerenza dei file che finiscono
#  nel pacchetto. L'esecuzione del nodo si collauda in WSL con
#  ros2_ws/.verify3.sh.
#
#  USO:  .\.venv\Scripts\python.exe test_phase3.py
# ============================================================================
import json
import os
import sys

import numpy as np

import kinematics as kin
import urdf_gen as ug
from plan_export import load_plan, PLAN_FORMAT_VERSION

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.join(HERE, "ros2_ws", "src", "cyberknife_ros")
PLAN = os.path.join(HERE, "outputs", "piano_trattamento.json")
# Il piano del pacchetto puo' essere PIU' RECENTE di outputs/ (il Tab 3 esporta
# direttamente li'): si controlla quello che il nodo esegue davvero.
_PKG_PLAN = os.path.join(PKG, "plans", "piano_trattamento.json")
if os.path.exists(_PKG_PLAN) and (
        not os.path.exists(PLAN) or
        json.load(open(_PKG_PLAN, encoding="utf-8")).get("generato", "") >
        json.load(open(PLAN, encoding="utf-8")).get("generato", "")):
    PLAN = _PKG_PLAN


def drawn_volume_cm3(plan):
    """Volume che la mesh del tumore deve avere, secondo il piano.

    Con il bersaglio robusto il piano e' ottimizzato su CTV + banda (tv_cm3),
    ma il tumore disegnato e la vista della dose sono il CTV: il suo volume
    si ricava dai punti di dose. Per i piani senza CTV resta tv_cm3.
    """
    d = plan.get("dose_tumore") or {}
    if d.get("volume") == "CTV" and d.get("punti_mm"):
        return (len(d["punti_mm"]) * int(d.get("sottocampionamento", 1))
                * float(d["passo_mm"]) ** 3 / 1000.0)
    return float(plan["metriche"]["tv_cm3"])


_fails = []


def check(name, cond, extra=""):
    print(f"[{'OK ' if cond else 'FALLITO'}] {name} {extra}")
    if not cond:
        _fails.append(name)


# ---------------------------------------------------------------------------
print("--- 1) URDF e cinematica descrivono lo stesso robot ---")
# E' il controllo piu' importante del pacchetto: se divergono, RViz disegna
# una posa e il piano ne calcola un'altra, senza alcun errore visibile.
rng = np.random.default_rng(11)
pe, de = [], []
for _ in range(1000):
    q = np.array([rng.uniform(lo, hi) for lo, hi in kin.JOINT_LIMITS])
    s1, d1, _ = kin.forward_kinematics(q)
    s2, d2 = ug.urdf_forward_kinematics(q)
    pe.append(np.linalg.norm(s1 - s2))
    de.append(np.linalg.norm(d1 - d2))
# NB: si confrontano i vettori direzione, non l'angolo fra loro: arccos e'
# mal condizionato vicino a 1 e restituirebbe ~1e-8 rad di puro rumore.
check("posizione della sorgente coincide", max(pe) < 1e-6,
      f"max {max(pe):.2e} mm")
check("direzione del fascio coincide", max(de) < 1e-12,
      f"max {max(de):.2e}")

urdf_path = os.path.join(PKG, "urdf", "cyberknife.urdf")
# L'offset della colonna del lettino dipende dal paziente, quindi il confronto
# va fatto con lo stesso che usa urdf_gen: altrimenti il test fallirebbe
# sempre, pur essendo l'URDF corretto.
_off = ug.couch_offset_from_package()
check("l'URDF sul disco e' aggiornato",
      os.path.exists(urdf_path) and
      open(urdf_path, encoding="utf-8").read() == ug.build_urdf(_off),
      "(altrimenti: python urdf_gen.py)")
check("l'URDF ha una sola radice (serve a robot_state_publisher)",
      len(set(__import__("re").findall(r'<link name="([^"]+)"', ug.build_urdf(_off)))
          - set(__import__("re").findall(r'<child link="([^"]+)"',
                                         ug.build_urdf(_off)))) == 1)
check("l'URDF contiene il secondo manipolatore",
      all(n in ug.build_urdf(_off) for n in
          ("couch_lift", "couch_shoulder", "couch_elbow", "couch_top")))

# ---------------------------------------------------------------------------
print("\n--- 1b) Le quote sono quelle del robot reale ---")
# Il CyberKnife M6/S7 monta un KUKA KR 300 R2500 ultra. Le quote vengono
# dall'URDF pubblico del pacchetto di supporto ROS, e la portata dichiarata
# (R2500) le verifica: A1 + A2 + D4 deve dare 2500 mm.
check("portata coerente con il nome R2500",
      abs(kin.REACH_MM - 2500.0) < 15.0, f"{kin.REACH_MM:.0f} mm")
for _nome, _val, _atteso in (("D1", kin.D1, 675.0), ("A1", kin.A1, 350.0),
                             ("A2", kin.A2, 1150.0), ("A3", kin.A3, 41.0),
                             ("D4", kin.D4, 1000.0), ("D6", kin.D6, 240.0)):
    check(f"quota {_nome} del KR 300 R2500 ultra", abs(_val - _atteso) < 1e-6,
          f"{_val:.0f} mm")
check("corsa di A5 pari a +-122.5 gradi",
      abs(kin.JOINT_LIMITS_DEG[4, 1] - 122.5) < 1e-6,
      f"{kin.JOINT_LIMITS_DEG[4, 1]:.1f}")

# ---------------------------------------------------------------------------
print("\n--- 2) Il pacchetto ROS usa la stessa cinematica ---")
# kinematics.py viene COPIATO nel pacchetto perche' dopo colcon build il nodo
# gira da install/ e non puo' risalire alla radice del progetto. La copia deve
# restare identica, altrimenti WSL e Windows calcolano robot diversi.
for fn, src in (("kinematics.py", os.path.join(HERE, "kinematics.py")),
                ("plan_export.py", os.path.join(HERE, "plan_export.py")),
                ("motion_model.py", os.path.join(HERE, "motion_model.py")),
                ("tracking_session.py",
                 os.path.join(HERE, "tracking_session.py"))):
    dst = os.path.join(PKG, "cyberknife_ros", fn)
    same = (os.path.exists(dst) and
            open(dst, encoding="utf-8").read() == open(src, encoding="utf-8").read())
    check(f"copia di {fn} identica all'originale", same,
          "(altrimenti: python urdf_gen.py)")

# Anche il PIANO va sincronizzato: il nodo legge quello dentro il pacchetto.
# Con la copia vecchia il nodo ha girato su 112 fasci mentre su Windows ne
# erano stati calcolati 135, e i numeri non tornavano.
# Il pacchetto puo' pero' contenere un piano PIU' RECENTE di outputs/: il
# pulsante di avvio del Tab 3 lo esporta direttamente li'. Quello non e' un
# disallineamento; lo e' solo un piano nel pacchetto piu' VECCHIO.
_pkg_plan = os.path.join(PKG, "plans", "piano_trattamento.json")
_pp = json.load(open(_pkg_plan, encoding="utf-8"))     if os.path.exists(_pkg_plan) else None
_po = json.load(open(PLAN, encoding="utf-8"))
check("il piano nel pacchetto e' quello corrente",
      _pp is not None and (_pp == _po or
                           _pp.get("generato", "") > _po.get("generato", "")),
      "(altrimenti: python urdf_gen.py)")

# ---------------------------------------------------------------------------
print("\n--- 2b) Lettino orientabile (2R) ---")
_iso = np.array([94.563, 144.022, -65.961])
_b0 = kin.RobotBase()

# Ruotare il lettino e un movimento RIGIDO attorno all isocentro: il robot
# non si sposta davvero, e il paziente che gli gira intorno. La distanza
# base-isocentro deve percio restare identica.
_c = kin.CouchPose(45.0, 12.0, pivot_mm=_iso)
_b1 = _c.base_in_patient(_b0)
check("la rotazione del lettino e rigida",
      abs(np.linalg.norm(_b1.origin - _iso) -
          np.linalg.norm(_b0.origin - _iso)) < 1e-9,
      f"{np.linalg.norm(_b1.origin - _iso):.3f} mm")
check("l orientamento resta ortonormale",
      np.allclose(_b1.R @ _b1.R.T, np.eye(3)) and
      abs(np.linalg.det(_b1.R) - 1.0) < 1e-9)
check("il lettino a zero non cambia nulla",
      np.allclose(kin.CouchPose(pivot_mm=_iso).base_in_patient(_b0).origin,
                  _b0.origin))

# Il piano porta con se la configurazione di sala: senza, il nodo ROS
# potrebbe simulare una geometria diversa da quella con cui il piano e stato
# calcolato, e i fasci sono stati scelti proprio sapendo da dove eseguirli.
import plan_export as _pe
_base_plan, _couch_plan = _pe.setup_from_plan(
    json.load(open(PLAN, encoding="utf-8")), kin)
check("il piano contiene la base del robot", _base_plan is not None,
      str(np.round(_base_plan.origin, 0)) if _base_plan is not None else "")
check("il piano contiene gli angoli di lettino", len(_couch_plan) >= 1,
      ", ".join(f"yaw {c.yaw_deg:.0f}" for c in _couch_plan))
# Il lettino RUOTA, non si inclina: il paziente resta supino e orizzontale.
# Inclinarlo di 20 gradi guadagnava 5 direzioni su 353 e non e' una posa in
# cui si possa immobilizzare qualcuno: pitch e roll restano assi di
# correzione del polso, non variabili di ottimizzazione.
_tilt = [c for c in _couch_plan
         if abs(c.pitch_deg) > 0.5 or abs(c.roll_deg) > 0.5]
check("il piano non inclina il paziente", not _tilt,
      str([str(c) for c in _tilt]))
# E l'ottimizzatore non deve nemmeno proporla: il default e' solo rotazione.
import inspect as _ins
check("optimize_setup cerca solo rotazioni del lettino",
      "pitch_values = [0.0]" in _ins.getsource(kin.optimize_setup))
# ---------------------------------------------------------------------------
print("\n--- 2c) Il lettino e un secondo manipolatore ---")
# Il CyberKnife monta il lettino su un braccio a sei assi: colonna
# prismatica verticale, due rotoidali in pianta, polso roll-pitch-yaw.
_scene0 = json.load(open(os.path.join(PKG, 'meshes', 'scene.json'),
                        encoding='utf-8'))
_lc = _scene0.get('lettino')
check("la scena descrive il braccio del lettino", _lc is not None)
if _lc:
    _arm = kin.CouchArm(_lc['piano_a_riposo_mm'], _lc['pavimento_mm'],
                        l1=_lc['l1_mm'], l2=_lc['l2_mm'])
    # A riposo il piano deve stare esattamente dove lo mette la scena.
    _q0 = _arm.home_joints()
    check("il lettino raggiunge la posa a riposo", _q0 is not None)
    if _q0 is not None:
        _pp, _RR = _arm.forward(_q0)
        check("chiusura FK/IK del lettino sotto 1e-6 mm",
              np.linalg.norm(_pp - _arm.home) < 1e-6,
              f"{np.linalg.norm(_pp - _arm.home):.1e} mm")
        check("il piano a riposo non e ruotato",
              np.abs(_RR - np.eye(3)).max() < 1e-9)
    # La colonna deve stare a pavimento e oltre il piede del lettino.
    check("la colonna e a pavimento",
          abs(_arm.column[1] - _lc['pavimento_mm']) < 1e-6,
          f"Y {_arm.column[1]:.0f}")
    # Il meccanismo deve RIFIUTARE pose fuori dalla sua portata: se accettasse
    # tutto non sarebbe un vincolo e l ottimizzatore proporrebbe l impossibile.
    _far = kin.CouchPose(yaw_deg=170.0, pitch_deg=60.0,
                         shift_mm=(0.0, 900.0, 0.0), pivot_mm=_iso)
    check("il lettino rifiuta pose fuori portata",
          _arm.solve_pose(_far) is None)

    # IL BRACCIO DEL LETTINO NON DEVE BUCARE IL PAZIENTE. Prima i due bracci
    # in pianta stavano alla quota del CENTRO del piano e, con un raggio
    # d'ingombro di 150 mm, entravano 115 mm nel corpo: in RViz il link
    # attraversava il torace. Ora la catena corre COUCH_ARM_DROP_MM sotto il
    # piano, e si verifica che nulla sporga sopra la sua faccia inferiore -
    # con la funzione che genera l'ingombro del braccio, cosi' il controllo
    # usa la stessa geometria del solutore.
    _piano_sotto = float(_lc["piano_a_riposo_mm"][1]) - kin.COUCH_THICKNESS_MM / 2.0
    _sporgenze = []
    for _c in ([kin.CouchPose(pivot_mm=_iso)] + list(_couch_plan or [])):
        _sporgenze += kin.couch_arm_obstacles(_arm, _c,
                                              exclude_below=_piano_sotto)
    check("il braccio del lettino resta sotto il piano", not _sporgenze,
          str([o.name for o in _sporgenze]))
    _q0b = _arm.home_joints()
    if _q0b is not None:
        _pt = _arm.link_points(_q0b)
        _cima = max(float(_pt[k][1]) for k in (2, 3))             + float(kin.COUCH_LINK_RADII_MM[1])
        check("franco fra braccio del lettino e piano",
              _piano_sotto - _cima >= 5.0,
              f"{_piano_sotto - _cima:.1f} mm")

    # Gli angoli scelti nel piano devono essere realizzabili DAVVERO.
    _bad = [c for c in (_couch_plan or []) if _arm.solve_pose(c) is None]
    check("tutti gli angoli del piano sono realizzabili dal lettino",
          not _bad, str([str(c) for c in _bad]))
# ---------------------------------------------------------------------------
print("\n--- 3) Il piano di trattamento si rilegge ---")
beams, data = load_plan(PLAN)
check("versione del formato", data["format_version"] == PLAN_FORMAT_VERSION)
check("unita' dichiarate", data["unita"] == "mm", data["unita"])
check("fasci presenti", len(beams) > 0, f"{len(beams)} fasci")
d = np.array([b.direction for b in beams])
check("direzioni normalizzate",
      np.allclose(np.linalg.norm(d, axis=1), 1.0, atol=1e-5))
w = np.array([b.weight for b in beams])
check("pesi positivi e non tutti uguali", w.min() > 0 and w.std() > 0,
      f"min {w.min():.3g} max {w.max():.3g}")

# ---------------------------------------------------------------------------
print("\n--- 4) Cinematica inversa sul piano reale ---")
# Base e angoli di lettino vengono dal PIANO: e' la configurazione con cui il
# piano e' stato calcolato, e usare altro darebbe numeri che non c'entrano.
base = _base_plan or kin.RobotBase()
COUCH = _couch_plan or None
# Ostacoli reali della scena, dagli stessi dati che finiscono in RViz.
_scene = json.load(open(os.path.join(PKG, "meshes", "scene.json"),
                        encoding="utf-8"))
OBS = [kin.Box.from_dict(o) for o in _scene["ostacoli"]]
poses, diag = kin.solve_plan(beams, base, n_roll=4, obstacles=OBS,
                             couch_poses=COUCH)
EFF = [c.base_in_patient(base) for c in (COUCH or [kin.CouchPose()])]
check("tutti i fasci raggiungibili",
      diag["raggiungibili"] == len(beams),
      f"{diag['raggiungibili']}/{len(beams)}")
errs = [kin.fk_error_mm(p["q"], p["beam"].source,
                        p["beam"].target - p["beam"].source, EFF[p["couch"]])
        for p in poses]
check("chiusura FK/IK sotto 0.01 mm", max(e[0] for e in errs) < 0.01,
      f"max {max(e[0] for e in errs):.2e} mm")
check("tutte le pose entro i limiti di giunto",
      all(kin.within_limits(p["q"]) for p in poses))

mg = np.array([p["margine"] for p in poses])
check("tutte le pose strettamente dentro i limiti", mg.min() > 0.0,
      f"margine minimo {mg.min():.3f}")
# Soglia 0.10, non 0.15. Con il KR 240 R2700 che il modello usava prima il
# mediano stava a 0.162, ma quel braccio era 400 mm piu' lungo del reale: con
# le quote vere del KR 300 R2500 ultra, e il lettino modellato fino a terra, il
# manipolatore lavora piu' vicino ai propri limiti. Misurato 0.116, e non si
# sposta ne' con piu' rotazioni provate (0.122 con 16 invece di 4) ne' con una
# barriera piu' forte. E' una conseguenza della geometria, non una regressione;
# la soglia resta comunque utile perche' intercetterebbe un crollo.
check("meta' delle pose lontane dai fine corsa", np.median(mg) >= 0.10,
      f"margine mediano {np.median(mg):.3f}")

# Quali giunti impongono il minimo. Con lettino e paziente come ostacoli il
# margine minimo scende a ~0.01 e NON e' un difetto del solutore: e' la corsa
# del giunto A2 (135 gradi soli) che si esaurisce quando il braccio, partendo
# da terra, deve sollevarsi sopra il lettino. Misurato: alzare la base peggiora
# tutto (135 fasci erogabili scendono a 102), aumentare la barriera non sposta
# il minimo, e provare 12 rotazioni invece di 4 nemmeno. E' un limite della
# GEOMETRIA scelta per il manipolatore, e va detto invece che nascosto.
_lim = kin.JOINT_LIMITS
_half = (_lim[:, 1] - _lim[:, 0]) / 2
_ctr = (_lim[:, 1] + _lim[:, 0]) / 2
_per = 1 - np.abs(np.array([p["q"] for p in poses]) - _ctr) / _half
_binding = [f"{kin.JOINT_NAMES[j].upper()} ({_per[:, j].min():.3f})"
            for j in range(6) if _per[:, j].min() < 0.10]
print(f"   giunti che arrivano al fine corsa: {', '.join(_binding) or 'nessuno'}")

# Non basta che ogni POSA sia valida: conta anche il percorso fra una posa e la
# successiva. Un giunto puo' passare da 170 a -170 gradi con uno scarto ciclico
# di 20 gradi, ma quella strada attraversa 190 gradi, che il giunto non
# raggiunge. Qui si controlla la traiettoria continua, non i singoli punti.
qs = np.rad2deg(np.array([p["q"] for p in poses]))
lim = np.rad2deg(kin.JOINT_LIMITS)
fuori = [kin.JOINT_NAMES[j] for j in range(6)
         if qs[:, j].min() < lim[j, 0] - 1e-6 or qs[:, j].max() > lim[j, 1] + 1e-6]
check("la traiettoria continua resta entro i fine corsa", not fuori, str(fuori))

# ---------------------------------------------------------------------------
print("\n--- 4b) Collisioni con lettino e paziente ---")
check("gli ostacoli sono due (lettino, paziente)", len(OBS) == 2,
      str([o.name for o in OBS]))

# Il piano deve essere ESEGUIBILE: nessuna posa dentro un ostacolo e nessun
# TRATTO di percorso che ci passi attraverso. La verifica usa un
# campionamento molto piu' fitto di quello del solutore, cosi' il test non
# puo' superarsi solo perche' guarda esattamente negli stessi punti.
in_obs = [k for k, p in enumerate(poses)
          if kin.collides(p["q"], EFF[p["couch"]], OBS)]
check("nessuna posa dentro un ostacolo", not in_obs, f"{len(in_obs)} pose")

legs_ko = 0
for a, b in zip(poses[:-1], poses[1:]):
    if a["couch"] != b["couch"]:
        continue      # in mezzo il lettino ruota: tratto verificato a parte
    pts = [a["q"]] + list(b.get("via") or []) + [b["q"]]
    legs_ko += sum(1 for u, v in zip(pts[:-1], pts[1:])
                   if kin.path_collides(u, v, EFF[b["couch"]], OBS, n=200))
check("nessun trasferimento attraversa un ostacolo", legs_ko == 0,
      f"{legs_ko} tratti (verifica a 200 campioni)")

fr = [p["franco_mm"] for p in poses if p["franco_mm"] is not None]
check("franco positivo su tutte le pose", bool(fr) and min(fr) > 0.0,
      f"minimo {min(fr):.0f} mm" if fr else "")

# Il corridoio di transito deve essere libero per OGNI rotazione della base:
# e' cio' che rende sicuro per costruzione il tratto centrale del ritiro.
shell = kin.find_safe_shell(base, OBS)
check("esistono corridoi di ritiro liberi",
      bool(shell) and shell[0][2] > 0,
      f"{len(shell)} candidati, migliore {shell[0][2]:.0f} mm" if shell else "")

# Le due implementazioni dell'ingombro devono coincidere: la vettoriale e'
# usata ovunque per velocita', la scalare fa da oracolo.
_rng = np.random.default_rng(3)
_Q = np.array([[_rng.uniform(lo, hi) for lo, hi in kin.JOINT_LIMITS]
               for _ in range(300)])
_p1 = np.array([kin.link_points(q, base) for q in _Q])
_p2 = kin.link_points_batch(_Q, base)
check("geometria vettoriale identica alla scalare",
      np.abs(_p1 - _p2).max() < 1e-6,
      f"scarto {np.abs(_p1 - _p2).max():.1e} mm")
_c1 = np.array([kin.collides(q, base, OBS) for q in _Q])
check("collisioni vettoriali identiche alle scalari",
      bool((_c1 == kin.collides_batch(_Q, base, OBS)).all()),
      f"{_c1.sum()}/{len(_Q)} configurazioni in collisione")

# Quanto contava tenerne conto: senza ostacoli il piano SEMBRA eseguibile.
_p_free, _ = kin.solve_plan(beams, base, n_roll=4, couch_poses=COUCH)
_ko = sum(1 for p in _p_free if kin.collides(p["q"], base, OBS))
print(f"   ignorando gli ostacoli: {_ko}/{len(_p_free)} pose finirebbero "
      f"dentro il lettino o il paziente")
check("ignorare gli ostacoli produce davvero pose impossibili", _ko > 0)

# ---------------------------------------------------------------------------
print("\n--- 4c) Il braccio non passa sotto il lettino ---")
_couch = [o for o in OBS if o.name == "lettino"][0]
# Il lettino e' modellato dal pavimento al piano, non come una lastra sottile:
# con soli 70 mm di spessore il braccio ci passava sotto, che e' lecito per la
# geometria ma non in una sala di trattamento.
check("il lettino arriva fino a terra",
      (_couch.hi[1] - _couch.lo[1]) > 500.0,
      f"alto {(_couch.hi[1] - _couch.lo[1]):.0f} mm")

_sotto = 0
for a, b in zip(poses[:-1], poses[1:]):
    if a["couch"] != b["couch"]:
        continue          # fra i gruppi il lettino ruota: tratto a parte
    pts = [a["q"]] + list(b.get("via") or []) + [b["q"]]
    for u, v in zip(pts[:-1], pts[1:]):
        _Qp = u + (v - u) * np.linspace(0, 1, 60).reshape(-1, 1)
        # Base EFFETTIVA dell'angolo di lettino in corso: con quella a riposo
        # la geometria non torna e il controllo segnalerebbe attraversamenti
        # che non esistono.
        _P = kin.link_points_batch(_Qp, EFF[b["couch"]])
        _in = ((_P[..., 0] > _couch.lo[0]) & (_P[..., 0] < _couch.hi[0]) &
               (_P[..., 2] > _couch.lo[2]) & (_P[..., 2] < _couch.hi[2]) &
               (_P[..., 1] < _couch.hi[1]))
        _sotto += int(np.any(_in))
check("nessun tratto passa sotto il piano del lettino", _sotto == 0,
      f"{_sotto} tratti")

# Lo sbraccio della spalla spazza un cerchio di A1 + raggio attorno all'asse
# della base ogni volta che A1 ruota. Se la base e' troppo vicina al lettino,
# quel cerchio lo taglia comunque, e NESSUN trasferimento diventa percorribile.
_need = kin.A1 + kin.LINK_RADII_MM[1]
_dist = _couch.lo[0] - base.origin[0]
check("la base e' abbastanza lontana dal lettino", _dist >= _need,
      f"{_dist:.0f} mm disponibili, ne servono {_need:.0f}")

_shell = kin.find_safe_shell(base, OBS)[0]
_q1s = np.linspace(*kin.JOINT_LIMITS[0], 73)
_Qs = np.zeros((73, 6))
_Qs[:, 0], _Qs[:, 1], _Qs[:, 2] = _q1s, _shell[0], _shell[1]
check("la rotazione della base a braccio ritirato e' libera",
      bool((~kin.collides_batch(_Qs, base, OBS)).all()),
      f"franco mediano {_shell[2]:.0f} mm")

# ---------------------------------------------------------------------------
print("\n--- 4d) Le rotazioni del lettino sono percorribili ---")
# Non basta che i fasci siano raggiungibili con il lettino gia orientato: il
# lettino ci deve ARRIVARE. Un piano da 2.2 m che ruota di 90 gradi attorno
# all isocentro spazza mezza sala all altezza del braccio, e li c e il robot:
# misurato, per quell orientamento non esiste manovra valida con nessun
# assortimento di vie di fuga (888 candidati, 20 passi).
_pool = kin.escape_pool(base, COUCH or [kin.CouchPose(pivot_mm=_iso)], OBS)
check("esistono vie di fuga per il braccio", len(_pool) > 0,
      f"{len(_pool)} candidati")
_unreach = [c for c in (COUCH or []) if not c.is_identity()
            and not kin.couch_pose_attainable(base, OBS, c, _pool)]
check("il lettino raggiunge tutti gli angoli del piano", not _unreach,
      str([str(c) for c in _unreach]))

# Ogni cambio di angolo deve avere una manovra registrata, non un salto.
_jumps = 0
for _a, _b in zip(poses[:-1], poses[1:]):
    if _a['couch'] != _b['couch'] and not _b.get('via'):
        _jumps += 1
check("nessun cambio di angolo avviene senza manovra", _jumps == 0,
      f"{_jumps} salti")
# ---------------------------------------------------------------------------
print("\n--- 4e) Il pacchetto ROS e' coerente: piano, scena, sala ---")
# Il nodo legge il piano da plans/ e ostacoli, lettino e mesh da meshes/: se
# vengono da due TAC diverse il robot schiva un lettino che non c'e' e i fasci
# convergono lontano dal tumore disegnato. E' successo: piano di lung_087
# (esportato dal Tab 3) con la scena di lung_001, 47 fasci erogati su 108,
# copertura finale 0% contro il 95% pianificato, dose a 580 mm dal tumore.
_pk_beams, _pk = load_plan(_pkg_plan)
_pk_base, _pk_couch = _pe.setup_from_plan(_pk, kin)
_sc = json.load(open(os.path.join(PKG, "meshes", "scene.json"),
                     encoding="utf-8"))


def _same_file(a, b):
    return bool(a and b) and (os.path.normcase(os.path.abspath(os.path.join(HERE, a)))
                              == os.path.normcase(os.path.abspath(os.path.join(HERE, b))))


check("piano e scena vengono dalla stessa TAC",
      _same_file((_pk.get("sorgente") or {}).get("ct"), _sc.get("sorgente_ct")),
      f"piano {(_pk.get('sorgente') or {}).get('ct')} / "
      f"scena {_sc.get('sorgente_ct')}")
_tum_sc = [o for o in _sc["organi"] if o["key"] == "tumor"]
# Scena e piano misurano il volume su griglie diverse (TAC e griglia di
# dose): si tollera lo 0.15 cm3 della sezione 7 o il 2%, il maggiore.
_tv_pk = drawn_volume_cm3(_pk)
check("volume del tumore della scena uguale a quello del piano",
      bool(_tum_sc) and abs(_tum_sc[0]["volume_cm3"] - _tv_pk)
      < max(0.15, 0.02 * _tv_pk),
      f"scena {_tum_sc[0]['volume_cm3'] if _tum_sc else '-'} / "
      f"piano {_tv_pk:.2f} cm3")
# Il robot appoggia sul pavimento della scena: una base a un'altra quota e'
# il segno di una sala calcolata per un altro paziente.
check("la base del robot sta sul pavimento della scena",
      _pk_base is not None and _sc.get("lettino") is not None and
      abs(float(_pk_base.origin[1]) - float(_sc["lettino"]["pavimento_mm"])) < 2.0,
      f"base Y {float(_pk_base.origin[1]):.0f}, pavimento "
      f"{_sc['lettino']['pavimento_mm'] if _sc.get('lettino') else '-'}")


def _stl_vertices_mm(path):
    """Vertici di un STL binario, in mm (le mesh del pacchetto sono in m)."""
    raw = open(path, "rb").read()
    n = int(np.frombuffer(raw, dtype=np.uint32, count=1, offset=80)[0])
    rec = np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
    return np.frombuffer(raw, dtype=rec, count=n, offset=84)["v"] \
        .reshape(-1, 3).astype(np.float64) * 1000.0


# LA POSIZIONE DEL TUMORE: la mesh disegnata in RViz e i voxel su cui il piano
# ha calcolato la dose devono sovrapporsi, e ogni fascio deve mirare DENTRO la
# lesione. I punti di mira sono volutamente diversi fra loro (piano conformale
# multi-bersaglio: coprono la lesione a pezzi), ma nessuno puo' cadere fuori.
_tv = _stl_vertices_mm(os.path.join(PKG, "meshes", "tumor.stl"))
_tlo, _thi = _tv.min(axis=0), _tv.max(axis=0)
_dpts = np.asarray((_pk.get("dose_tumore") or {}).get("punti_mm") or
                   np.zeros((0, 3)))
_cen_d = float(np.linalg.norm(_dpts.mean(axis=0) - (_tlo + _thi) / 2.0)) \
    if len(_dpts) else np.inf
check("la mesh del tumore sta dove il piano ha calcolato la dose",
      len(_dpts) > 0 and _cen_d < 5.0 and
      bool(((_dpts >= _tlo - 3.0) & (_dpts <= _thi + 3.0)).all(axis=1).mean()
           > 0.99),
      f"centri a {_cen_d:.1f} mm")
_tg = np.array([b.target for b in _pk_beams])
_out = int((~((_tg >= _tlo - 3.0) & (_tg <= _thi + 3.0)).all(axis=1)).sum())
_near = (np.min(np.linalg.norm(_tg[:, None, :] - _dpts[None, :, :], axis=2),
                axis=1) if len(_dpts) else np.full(len(_tg), np.inf))
check("ogni fascio mira dentro il tumore della scena",
      _out == 0 and float(_near.max()) <= 5.0,
      f"{_out} mire fuori dalla mesh, distanza massima da un voxel del "
      f"tumore {float(_near.max()):.1f} mm, "
      f"{len({tuple(np.round(t, 1)) for t in _tg})} punti di mira distinti")

# Il nodo risolve QUESTO piano con QUESTI ostacoli: lo stesso calcolo qui,
# senza ROS. Sotto il 95% il test fallisce - un piano che il robot esegue a
# meta' non e' il piano che la dosimetria ha valutato.
_pk_obs = [kin.Box.from_dict(o) for o in _sc["ostacoli"]]
_pk_poses, _pk_diag = kin.solve_plan(_pk_beams, _pk_base or kin.RobotBase(),
                                     n_roll=4, obstacles=_pk_obs,
                                     couch_poses=_pk_couch or None)
_frac = _pk_diag["raggiungibili"] / max(len(_pk_beams), 1)
check("il nodo raggiunge almeno il 95% dei fasci del pacchetto",
      _frac >= 0.95,
      f"{_pk_diag['raggiungibili']}/{len(_pk_beams)} "
      f"({100 * _frac:.0f}%), bloccati nel percorso "
      f"{_pk_diag['bloccati_nel_percorso']}")
_ero = _pk["metriche"].get("fasci_erogabili")
check("il collaudo della dosimetria coincide con il nodo",
      _ero is None or _ero == _pk_diag["raggiungibili"],
      f"dosimetria {_ero}, nodo {_pk_diag['raggiungibili']}")

# Le manovre del lettino: ogni tratto va verificato con l'angolo di lettino
# che ha DAVVERO in quel momento. `couch_legs` dice quali tratti avvengono
# con il lettino in rotazione; gli altri sono a lettino fermo, all'angolo di
# partenza prima e a quello d'arrivo dopo.
_man_ko = 0
_cp = _pk_couch or [kin.CouchPose()]
for _a, _b in zip(_pk_poses[:-1], _pk_poses[1:]):
    if _a["couch"] == _b["couch"]:
        continue
    _pts = [_a["q"]] + list(_b.get("via") or []) + [_b["q"]]
    _s, _e = _b.get("couch_legs") or (0, len(_pts) - 1)
    _nm = _e - _s
    for _k in range(len(_pts) - 1):
        if _k < _s:
            _c0 = _c1 = _cp[_a["couch"]]
        elif _k >= _e:
            _c0 = _c1 = _cp[_b["couch"]]
        else:
            _c0 = kin._interp_couch(_cp[_a["couch"]], _cp[_b["couch"]],
                                    (_k - _s) / _nm)
            _c1 = kin._interp_couch(_cp[_a["couch"]], _cp[_b["couch"]],
                                    (_k - _s + 1) / _nm)
        _man_ko += int(kin.transfer_collides(_pts[_k], _pts[_k + 1],
                                             _pk_base, _pk_obs, _c0, _c1))
check("le manovre del lettino non attraversano ostacoli", _man_ko == 0,
      f"{_man_ko} tratti in collisione")

# ---------------------------------------------------------------------------
print("\n--- 5) La barriera sui fine corsa serve davvero ---")
# Senza barriera il solutore sceglie sempre il ramo di movimento minimo, anche
# quando appoggia il giunto A2 (135 gradi di corsa) sul suo limite.
poses_off, diag_off = kin.solve_plan(beams, base, n_roll=4,
                                     margin_barrier=0.0, obstacles=OBS,
                                     couch_poses=COUCH)
mg_off = np.array([p["margine"] for p in poses_off])
print(f"   senza barriera: mediano {np.median(mg_off):.3f}, "
      f"sotto 0.10 {(mg_off < 0.10).sum()}, minimo {mg_off.min():.3f}")
print(f"   con barriera:   mediano {np.median(mg):.3f}, "
      f"sotto 0.10 {(mg < 0.10).sum()}, minimo {mg.min():.3f}")
# Si guarda il MEDIANO e il numero di pose critiche, non il minimo. Con il
# lettino orientato il minimo e' saturo: c'e' una posa senza alternative, e
# nessun peso puo' migliorarla. La barriera agisce su tutte le altre - misurato
# 52 pose sotto la soglia senza, 47 con - ed e' quello il suo scopo.
check("la barriera allontana le pose dai fine corsa",
      np.median(mg) >= np.median(mg_off) and
      (mg < 0.10).sum() <= (mg_off < 0.10).sum())

# ---------------------------------------------------------------------------
print("\n--- 6) L'ordinamento della sequenza riduce il movimento ---")
_, diag_naive = kin.solve_plan(beams, base, n_roll=4, order_by_motion=False,
                               obstacles=OBS, couch_poses=COUCH)
gain = 100 * (1 - diag["movimento_totale_deg"] /
              max(diag_naive["movimento_totale_deg"], 1e-9))
check("l'ordinamento riduce il movimento totale", gain > 0,
      f"{diag_naive['movimento_totale_deg']:.0f} -> "
      f"{diag['movimento_totale_deg']:.0f} gradi ({gain:.0f}% in meno)")

# ---------------------------------------------------------------------------
print("\n--- 7) Scena anatomica per RViz ---")
mesh_dir = os.path.join(PKG, "meshes")
scene_path = os.path.join(mesh_dir, "scene.json")
if os.path.exists(scene_path):
    scene = json.load(open(scene_path, encoding="utf-8"))
    check("manifesto in metri", scene["unita"] == "m", scene["unita"])
    missing = [o["file"] for o in scene["organi"]
               if not os.path.exists(os.path.join(mesh_dir, o["file"]))]
    check("tutti gli STL dichiarati esistono", not missing, str(missing))
    tum = [o for o in scene["organi"] if o["key"] == "tumor"]
    check("il tumore e' nella scena", bool(tum))
    if tum:
        # Il volume deve coincidere con quello su cui e' stato calcolato il
        # piano: se divergono, robot e dosimetria stanno guardando due lesioni.
        tv_plan = drawn_volume_cm3(data)
        check("volume del tumore coerente col piano",
              abs(tum[0]["volume_cm3"] - tv_plan) < max(0.15, 0.02 * tv_plan),
              f"scena {tum[0]['volume_cm3']:.2f} vs piano {tv_plan:.2f} cm3")
else:
    check("scene.json presente", False, "(lancia: python export_scene.py)")

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
print("\n--- 8) La simulazione si spegne da sola ---")
_launch = open(os.path.join(PKG, "launch", "cyberknife.launch.py"),
               encoding="utf-8").read()
check("il ciclo infinito e disattivato di default",
      'DeclareLaunchArgument("loop", default_value="false")' in _launch)
check("il launch spegne tutto a nodo terminato",
      "OnProcessExit" in _launch and "Shutdown" in _launch and
      "target_action=treatment" in _launch)
_node = open(os.path.join(PKG, "cyberknife_ros", "treatment_node.py"),
             encoding="utf-8").read()
# Non basta fermare l animazione: il processo deve USCIRE, altrimenti il
# gestore di evento del launch non scatta e RViz resta aperto.
check("il nodo termina il processo a fine trattamento",
      "rclpy.shutdown()" in _node and "_finish" in _node)
check("il nodo legge la configurazione di sala dal piano",
      "setup_from_plan" in _node)

# Ogni parametro dichiarato dal nodo deve essere raggiungibile dal launch,
# altrimenti passarlo da riga di comando non ha effetto e si resta sui
# default senza capire perche': e' successo con min_transfer_s, e la
# simulazione restava lenta con i tempi calcolati sul valore sbagliato.
import re as _re
_declared = set(_re.findall(r'declare_parameter\(\s*"([a-z_]+)"', _node))
# Il launch li inoltra come LaunchConfiguration diretta oppure tramite
# l'involucro _num(), che ne dichiara anche il tipo.
_forwarded = set(_re.findall(r'LaunchConfiguration\("([a-z_]+)"\)', _launch))
_forwarded |= set(_re.findall(r'_num\("([a-z_]+)"', _launch))
_forwarded |= {"share_dir", "mesh_dir"}          # impostati dal launch a mano
_missing = sorted(_declared - _forwarded)
_untyped = sorted(set(_re.findall(
    r'"([a-z_]+)": LaunchConfiguration\("[a-z_]+"\)', _launch)) - {"plan"})
check("gli argomenti numerici del launch sono tipizzati", not _untyped,
      str(_untyped))
# La GUI deve scegliere i fasci SAPENDO da dove il robot li esegue. Senza,
# misurato su lung_001: 107 fasci pianificati, 70 erogati, 39% del peso di
# dose perso. Con la sala decisa prima della dosimetria: 52 su 52.
_dw = open(os.path.join(HERE, "dose_worker.py"), encoding="utf-8").read()
check("la dosimetria della GUI ottimizza la sala prima dei fasci",
      "optimize_setup" in _dw and "robot_base=" in _dw
      and "robot_couch_poses=" in _dw and "**robot" in _dw)
check("la dosimetria della GUI filtra con il braccio del lettino",
      "couch_arm=" in _dw)
# Il collaudo con la sequenza del robot deve avvenire PRIMA della dose finale,
# non dopo: con il solo avviso il piano prometteva una dose che il robot non
# dava. Misurato su lung_085: 2 fasci persi su 116, copertura dal 94.9%
# pianificato al 57.8% erogato; con il filtro 94.9% in entrambi.
_pe_src = open(os.path.join(HERE, "plan_export.py"), encoding="utf-8").read()
check("la dosimetria toglie i fasci che il robot non esegue",
      "deliverable_subset" in _dw and "beam_filter=beam_filter" in _dw
      and "deliverable_subset" in _pe_src and "beam_filter=" in _pe_src)
_kt = open(os.path.join(HERE, "kinematics_tab.py"), encoding="utf-8").read()
check("il Tab 3 adotta la sala per cui sono stati scelti i fasci",
      'res.get("robot_base")' in _kt and 'res.get("couch_poses")' in _kt)
check("l ottimizzazione di sala del Tab 3 usa il braccio del lettino",
      "couch_arm=self.couch_arm" in _kt)
check("tutti i parametri del nodo sono esposti dal launch", not _missing,
      str(_missing))

_tab = open(os.path.join(HERE, "kinematics_tab.py"), encoding="utf-8").read()
# L avvio deve esportare da solo: un pulsante separato e un invito a
# simulare il piano precedente convinti di guardare quello nuovo.
_launch_fn = _tab[_tab.index("def on_launch"):]
check("l avvio esporta il piano da solo",
      "_write_plan(ROS_PLAN)" in _launch_fn)
# ...e con il piano la SCENA, se e' di un altro paziente: esportare solo piano
# e URDF ha prodotto il pacchetto incoerente descritto nella sezione 4e.
check("l avvio rigenera la scena se e' di un altro paziente",
      "scene_matches(data)" in _launch_fn and "SceneWorker(" in _launch_fn)
check("il piano della GUI registra la TAC di origine",
      'setdefault("ct_path", res.get("image_path"))' in _tab)
# ---------------------------------------------------------------------------
print("\n--- 9) Il tumore si riempie di dose ---")
# La vista del tumore deve raccontare la dosimetria VERA: a fine trattamento i
# colori devono coincidere con la dose della Fase 2, non con una stima.
import dose_fill as _dfm
_pd = json.load(open(PLAN, encoding="utf-8"))
_dt = _pd.get("dose_tumore")
check("il piano porta la dose nei voxel del tumore",
      _dt is not None and len(_dt["punti_mm"]) > 100,
      f"{len(_dt['punti_mm'])} voxel" if _dt else "campo assente")
if _dt:
    _bd, _ = load_plan(PLAN)
    _df = _dfm.DoseFill.from_plan(_pd, _bd)
    _all = list(range(len(_bd)))
    _end = _df.dose(_all)
    _err = float(np.abs(_end - np.asarray(_dt["dose_pct"])).max())
    check("a fine trattamento la dose e' quella della dosimetria",
          _err < 1e-6, f"scarto {_err:.1e}")
    _cov = 100.0 * _df.coverage(_end)
    # Con il bersaglio robusto i voxel della vista sono il CTV: il confronto
    # va fatto con la copertura pianificata del CTV, non del volume di piano.
    _cov_plan = float(_pd["metriche"].get("copertura_ctv_pct",
                                          _pd["metriche"]["coverage_pct"])
                      if (_pd.get("dose_tumore") or {}).get("volume") == "CTV"
                      else _pd["metriche"]["coverage_pct"])
    check("la copertura finale coincide con quella del piano",
          abs(_cov - _cov_plan) < 1.0, f"{_cov:.1f}% contro {_cov_plan:.1f}%")
    check("nessun fascio toglie dose", bool((_df.contrib >= 0).all()))
    # I TUNNEL: la maggior parte dei fasci deve colpire una PARTE del tumore,
    # non colorarlo tutto in modo uniforme - altrimenti la vista non
    # mostrerebbe cio' che la ha motivata. Non si pretende da tutti: i coni da
    # 20-25 mm sono piu' larghi della lesione (18 mm) e la investono intera,
    # come e' fisicamente giusto. Misurato: 35 fasci su 52 lasciano un tunnel.
    _foot = []
    for _k in range(len(_bd)):
        _c = _df.contrib[:, _k]
        _foot.append(float(np.mean(_c > 0.5 * _c.max())) if _c.max() > 0 else 1.0)
    _foot = np.array(_foot)
    _n_tun = int((_foot < 0.8).sum())
    check("la maggior parte dei fasci lascia un tunnel",
          _n_tun * 2 > len(_bd),
          f"{_n_tun}/{len(_bd)} fasci, impronta mediana "
          f"{100 * np.median(_foot):.0f}% dei voxel")
    _half = 100.0 * _df.coverage(_df.dose(_all[:len(_all) // 2]))
    check("a meta' trattamento il tumore non e' ancora coperto",
          _half < _cov, f"{_half:.0f}% a meta', {_cov:.0f}% alla fine")
    _c0, _c1 = _df.colors(np.array([0.0, _df.rx]))
    check("rosso senza dose, verde alla prescrizione",
          _c0[0] > _c0[1] and _c1[1] > _c1[0])
_src = open(os.path.join(HERE, "dose_fill.py"), encoding="utf-8").read()
_dst = os.path.join(PKG, "cyberknife_ros", "dose_fill.py")
check("copia di dose_fill.py identica all'originale",
      os.path.exists(_dst) and open(_dst, encoding="utf-8").read() == _src,
      "(altrimenti: python urdf_gen.py)")
check("il launch apre la finestra del tumore",
      "rviz2_tumore" in _launch and
      os.path.exists(os.path.join(PKG, "rviz", "tumore.rviz")))

# ---------------------------------------------------------------------------
print("\n--- 10) Il resoconto di fine seduta ---")
# Il resoconto si costruisce con lo STESSO modulo che usa il nodo ROS
# (treatment_log), a partire dallo stesso piano e dalla stessa sequenza: se i
# numeri non tornano qui, non tornano nemmeno nella finestra di fine seduta.
import tempfile as _tf
import dose_fill as _dfr
import treatment_log as _tlog
_bR, _pR = load_plan(PLAN)
_baseR, _couchR = _pe.setup_from_plan(_pR, kin)
_posesR, _diagR = kin.solve_plan(_bR, _baseR, n_roll=4, obstacles=OBS,
                                 couch_poses=_couchR)
_tlR, _ = _tlog.build_timeline(_posesR, 90.0, 0.5, 3.0, 0.6)
_dfR = _dfr.DoseFill.from_plan(_pR, _bR)
_rep = _tlog.build_report(_pR, _bR, _posesR, _tlR, _diagR, _couchR,
                          dose_fill=_dfR, wall_s=1.0)
_sR = _rep["sintesi"]
check("il resoconto conta i fasci erogati",
      _sR["fasci_erogati"] == len(_posesR) and len(_rep["fasci"]) == len(_posesR),
      f"{_sR['fasci_erogati']}/{_sR['fasci_pianificati']}")
check("la copertura finale e' quella del piano",
      abs(_sR["finale"]["copertura_pct"] - _sR["copertura_pianificata_pct"]) < 0.5,
      f"{_sR['finale']['copertura_pct']}% contro "
      f"{_sR['copertura_pianificata_pct']}%")
_covR = [r["copertura_pct"] for r in _rep["fasci"]]
check("la copertura non diminuisce mai durante la seduta",
      all(b >= a - 1e-9 for a, b in zip(_covR, _covR[1:])))
_tot = sum(e["durata_s"] for e in _rep["eventi"])
check("i tempi della sintesi tornano con la linea temporale",
      abs(_tot - _sR["tempo_totale_s"]) < 0.05 and
      abs(_sR["tempo_trasferimenti_s"] + _sR["tempo_erogazione_s"]
          + _sR["tempo_lettino_s"] - _sR["tempo_totale_s"]) < 0.05,
      f"{_sR['tempo_totale_s']} s")
check("un evento di erogazione per fascio",
      sum(1 for e in _rep["eventi"] if e["tipo"] == "erogazione")
      == len(_posesR))
_stR = _tlog.replay_doses(_dfR, _rep)
check("il replay finisce sulla dose del piano",
      float(np.abs(_stR[-1] - _dfR.dose(range(len(_bR)))).max()) < 1e-9)
check("il replay parte da un bersaglio senza dose",
      float(np.abs(_stR[0]).max()) == 0.0)
with _tf.TemporaryDirectory() as _d:
    _pth = os.path.join(_d, _tlog.REPORT_NAME)
    _tlog.write_report(_rep, _pth)
    _back = _tlog.load_report(_pth)
check("il resoconto si rilegge identico", _back["sintesi"] == _sR)

_srcT = open(os.path.join(HERE, "treatment_log.py"), encoding="utf-8").read()
_dstT = os.path.join(PKG, "cyberknife_ros", "treatment_log.py")
check("copia di treatment_log.py identica all'originale",
      os.path.exists(_dstT) and open(_dstT, encoding="utf-8").read() == _srcT,
      "(altrimenti: python urdf_gen.py)")
check("il nodo usa la linea temporale condivisa e scrive il resoconto",
      "tlog.build_timeline" in _node and "_write_report()" in _node)

# Il file scritto DAVVERO dal nodo ROS, se c'e' ed e' di questo piano, deve
# dire le stesse cose della ricostruzione su Windows.
_rosR = os.path.join(PKG, "plans", _tlog.REPORT_NAME)
if os.path.exists(_rosR):
    _nodeR = _tlog.load_report(_rosR)
    # Con il respiro attivo la dose del nodo dipende dall'inseguimento, che
    # la ricostruzione qui non simula: si confrontano allora fasci e ordine,
    # non la copertura finale.
    _resp = bool((_nodeR.get("respiro") or _nodeR["sintesi"].get("respiro")
                  or {}).get("attivo"))
    if _nodeR.get("piano_generato") == _pR.get("generato"):
        check("il resoconto del nodo ROS coincide con quello ricostruito",
              _nodeR["sintesi"]["fasci_erogati"] == _sR["fasci_erogati"]
              and (_resp or _nodeR["sintesi"]["finale"] == _sR["finale"])
              and [r["fascio"] for r in _nodeR["fasci"]]
              == [r["fascio"] for r in _rep["fasci"]],
              f"{_nodeR['sintesi']['finale']['copertura_pct']}%"
              + (" (respiro attivo: copertura non confrontata)"
                 if _resp else ""))
    else:
        print("   resoconto del nodo presente ma di un altro piano: "
              "confronto saltato")
else:
    print("   nessun resoconto del nodo ROS: confronto saltato")

# La finestra si costruisce e il cursore arriva alla fine della seduta.
try:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtWidgets import QApplication as _QApp
    _app = _QApp.instance() or _QApp([])
    import report_view as _rv
    _dlg = _rv.TreatmentReportDialog(_rep, _pR, _bR)
    check("la finestra copre tutta la seduta",
          _dlg.slider.maximum() == len(_posesR) * _rv.SUB)
    _dlg.slider.setValue(_dlg.slider.maximum())
    _txt = _dlg.lbl_pos.text()
    check("a fine replay la finestra mostra la copertura finale",
          f"{_sR['finale']['copertura_pct']:.0f}%" in _txt, _txt[:60])
    _dlg.close()
except ImportError as _exc:
    print(f"   finestra non provata: {_exc}")

print("\n" + ("Tutti i controlli superati" if not _fails
              else f"FALLITI: {_fails}"))
sys.exit(1 if _fails else 0)
