# ============================================================================
#  plan_export.py - Esportazione del piano di trattamento per la Fase 3.
#
#  E' il contratto fra la parte di imaging/dosimetria (Windows, Python 3.14) e
#  quella robotica (ROS 2 Humble in WSL, Python 3.10). I due mondi non possono
#  condividere un processo - rclpy non esiste per Python 3.14 - quindi si
#  scambiano un file JSON con tutto cio' che serve al robot:
#
#    - i fasci: sorgente, bersaglio, diametro del collimatore, peso;
#    - il bersaglio e i suoi contorni, per la visualizzazione in RViz;
#    - le unita' e il sistema di riferimento, dichiarati esplicitamente.
#
#  Le coordinate sono in MILLIMETRI, sistema RAS del paziente: le stesse che
#  escono dalla Fase 1. La conversione in metri (unita' di ROS) avviene una
#  volta sola, nel nodo, per non avere due convenzioni in giro.
#
#  USO:
#      python plan_export.py --ct <ct.nii.gz> --mask <seg.nii.gz> --out piano.json
# ============================================================================
import json
from datetime import datetime, timezone

import numpy as np


PLAN_FORMAT_VERSION = 2

#: Punti del tumore salvati con la loro dose, per la vista di riempimento.
#: Oltre si sottocampiona: a 1.5 mm una lesione di 3 cm3 ne ha ~800.
MAX_DOSE_POINTS = 6000


def _tumor_dose(result):
    """Dose della Fase 2 in ogni voxel del tumore, sulla griglia di calcolo.

    Campo facoltativo del piano: lo usa la vista ROS in cui il tumore si
    colora man mano che i fasci lo attraversano. Lo stato finale di quella
    vista deve essere la dosimetria VERA, non una stima fatta nel nodo, che
    non ha ne' la TAC ne' il modello di convoluzione.
    """
    # Il TUMORE, non il bersaglio pianificato: con un margine (PTV, ITV)
    # il bordo del bersaglio perde dose sotto moto per costruzione - e' a
    # questo che serve il margine - e giudicarne la copertura faceva sembrare
    # perso cio' che invece era previsto. Misurato su un piano reale: nei
    # punti a oltre 2 mm dal bordo del PTV la copertura consegnata era 97%,
    # sull'intero PTV 89%.
    tum = result.get("ctv_grid")
    if tum is None:
        tum = result.get("tumor_grid")
    dose, aff = result.get("dose_pct"), result.get("grid_affine")
    if dose is None or tum is None or aff is None:
        return None
    idx = np.argwhere(np.asarray(tum) > 0.5)
    if not len(idx):
        return None
    step = max(1, int(np.ceil(len(idx) / MAX_DOSE_POINTS)))
    idx = idx[::step]
    aff = np.asarray(aff, dtype=np.float64)
    pts = idx @ aff[:3, :3].T + aff[:3, 3]
    vals = np.asarray(dose)[tuple(idx.T)]
    return {
        "volume": "CTV" if result.get("ctv_grid") is not None else "bersaglio",
        "passo_mm": round(float(np.linalg.norm(aff[:3, 0])), 3),
        "sottocampionamento": step,
        "punti_mm": np.round(pts, 2).tolist(),
        "dose_pct": np.round(vals.astype(np.float64), 3).tolist(),
    }


def plan_to_dict(result, ct_path=None, mask_path=None,
                 robot_base=None, couch_poses=None, motion=None):
    """Converte il risultato di compute_plan in una struttura serializzabile."""
    beams = []
    for i, b in enumerate(result["beams"]):
        beams.append({
            "id": i,
            "source_mm": [round(float(v), 3) for v in b.source],
            "target_mm": [round(float(v), 3) for v in b.target],
            "direction": [round(float(v), 6) for v in b.direction],
            "cone_mm": float(b.cone_mm),
            "weight": round(float(b.weight), 6),
            "path_cm": round(float(b.path_cm), 3),
            "bone_cm": round(float(b.bone_cm), 3),
            "target_id": int(b.target_id),
            "node_id": int(b.node_id),
        })

    m = result.get("metrics", {})
    return {
        "format_version": PLAN_FORMAT_VERSION,
        "generato": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "unita": "mm",
        "sistema_riferimento": "RAS paziente (+X destra, +Y anteriore, +Z craniale)",
        "sorgente": {"ct": ct_path, "maschera": mask_path},
        "isocentro_mm": [round(float(v), 3) for v in result["isocenter"]],
        "prescrizione_pct": float(m.get("prescription_pct", 80.0)),
        # Configurazione di SALA. Sta nel piano e non nei parametri del nodo
        # perche' e' parte del piano: i fasci sono stati scelti sapendo da dove
        # il robot li avrebbe eseguiti e con quali orientamenti del lettino.
        # Separarli significherebbe poter eseguire un piano con una geometria
        # diversa da quella per cui e' stato calcolato.
        "robot": ({"base_mm": [round(float(v), 3) for v in robot_base.origin],
                   "yaw_deg": round(float(np.rad2deg(robot_base.yaw)), 3)}
                  if robot_base is not None else None),
        "angoli_lettino": ([c.as_dict() for c in couch_poses]
                           if couch_poses else None),
        "metriche": {k: (v if isinstance(v, (int, float, str, list, dict))
                         else str(v))
                     for k, v in m.items()},
        "fasci": beams,
        "dose_tumore": _tumor_dose(result),
        # Respiro del paziente e tecnica di gestione del moto (Tab 2).
        # Campo facoltativo: senza, il nodo tratta un tumore immobile come
        # prima, quindi i piani gia' esportati restano validi e la versione
        # del formato non cambia.
        "respiro": _motion_dict(motion),
        "attenuazione_moto": _path_tables(result, motion),
    }


def _path_tables(result, motion):
    """Tabelle di attenuazione dei fasci sotto moto (solo su Windows: servono
    la griglia di densita' e scipy). None se il moto e' ignorato."""
    if motion is None or result.get("red_grid") is None:
        return None
    follows = (motion.get("inseguimento", True) if isinstance(motion, dict)
               else True)
    try:
        from motion_dose import path_attenuation_tables
    except ImportError:                       # nel nodo ROS non serve
        return None
    return path_attenuation_tables(
        result["red_grid"], result["grid_affine"], result["beams"],
        moving="entrambi" if follows else "bersaglio")


def _motion_dict(motion):
    """MotionConfig, oppure {"cfg", "inseguimento", "tecnica"} dal Tab 2.

    Il risultato e' la configurazione del respiro con due chiavi in piu':
    `inseguimento` (il robot segue il tumore? no per la tecnica ITV) e
    `tecnica`. MotionConfig.from_dict ignora le chiavi che non conosce.
    """
    if motion is None:
        return None
    if isinstance(motion, dict):
        d = motion["cfg"].as_dict()
        d["inseguimento"] = bool(motion.get("inseguimento", True))
        d["tecnica"] = motion.get("tecnica", "tracking")
        if motion.get("gating_mm") is not None:
            d["gating_mm"] = float(motion["gating_mm"])
        return d
    d = motion.as_dict()
    d["inseguimento"] = True
    return d


def save_plan(result, path, ct_path=None, mask_path=None,
              robot_base=None, couch_poses=None, motion=None):
    data = plan_to_dict(result, ct_path, mask_path, robot_base, couch_poses,
                        motion)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=1, ensure_ascii=False)
    return data


class LoadedBeam:
    """Fascio ricostruito dal JSON, con la stessa interfaccia di dosimetry.Beam.

    Il nodo ROS non importa dosimetry (che tira dentro scipy e il resto della
    pipeline di imaging): gli basta questa struttura minima.
    """

    __slots__ = ("source", "target", "direction", "cone_mm", "weight",
                 "path_cm", "bone_cm", "target_id", "node_id", "id")

    def __init__(self, d):
        self.id = int(d["id"])
        self.source = np.asarray(d["source_mm"], dtype=np.float64)
        self.target = np.asarray(d["target_mm"], dtype=np.float64)
        self.direction = np.asarray(d["direction"], dtype=np.float64)
        self.cone_mm = float(d["cone_mm"])
        self.weight = float(d["weight"])
        self.path_cm = float(d.get("path_cm", 0.0))
        self.bone_cm = float(d.get("bone_cm", 0.0))
        self.target_id = int(d.get("target_id", 0))
        self.node_id = int(d.get("node_id", -1))


def load_plan(path):
    """Rilegge un piano esportato. Restituisce (fasci, dati_completi)."""
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if data.get("format_version") != PLAN_FORMAT_VERSION:
        raise ValueError(
            f"Formato del piano non riconosciuto: atteso {PLAN_FORMAT_VERSION}, "
            f"trovato {data.get('format_version')}. Rigenera il piano con "
            f"plan_export.py oppure dal Tab 4.")
    return [LoadedBeam(b) for b in data["fasci"]], data


def setup_from_plan(data, kin):
    """Base del robot e angoli di lettino salvati nel piano.

    `kin` viene passato invece di importato: questo modulo gira anche dentro il
    nodo ROS, dove la cinematica e' una copia nel pacchetto, e un import fisso
    legherebbe il contratto a un percorso preciso.

    Restituisce (base, angoli) con None se il piano non li contiene - i piani
    di formato 1 non li avevano.
    """
    r = data.get("robot")
    base = None
    if r:
        base = kin.RobotBase(origin_mm=tuple(r["base_mm"]),
                             yaw_deg=float(r.get("yaw_deg", 0.0)))
    poses = [kin.CouchPose.from_dict(c) for c in (data.get("angoli_lettino") or [])]
    return base, poses


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    import time
    import nibabel as nib
    from scipy import ndimage

    import dosimetry as dz
    import organs as og

    ap = argparse.ArgumentParser(
        description="Calcola un piano ed esportalo in JSON per la Fase 3.")
    ap.add_argument("--ct", required=True)
    ap.add_argument("--mask", required=True)
    ap.add_argument("--out", default="outputs/piano_trattamento.json")
    ap.add_argument("--couch-poses", type=int, default=3, dest="couch_poses",
                    help="quanti angoli di lettino al massimo (1 = fermo)")
    ap.add_argument("--senza-collaudo", action="store_true",
                    dest="senza_collaudo",
                    help="non togliere i fasci che il robot non esegue (solo "
                         "per confronto: il piano promette piu' di cio' che "
                         "verra' erogato)")
    args = ap.parse_args()

    img = nib.load(args.ct)
    ct = np.asanyarray(img.dataobj, dtype=np.float32)
    msk = (nib.load(args.mask).get_fdata() > 0.5).astype(np.uint8)

    # Bersaglio = lesione principale, con il criterio condiviso.
    import postproc
    msk, n_les = postproc.primary_mask(msk, img.affine)
    if n_les > 1:
        print(f"{n_les} lesioni rilevate: trattata la principale")

    iso = dz.apply_affine_fast(
        img.affine,
        np.array(ndimage.center_of_mass(msk), dtype=np.float64).reshape(3, 1)
    ).ravel()

    print("Segmentazione degli organi...")
    masks, oaff, _ = og.segment_organs(ct, img.affine, 2.0)

    # Vincoli del ROBOT dentro la pianificazione: il lettino e il paziente
    # come ostacoli, e la base nella posizione reale. Senza questo passaggio
    # l'ottimizzatore distribuisce i pesi anche su fasci che il braccio non
    # puo' eseguire, e la dose calcolata non e' quella erogata.
    import kinematics as kin
    bounds = kin.bounds_mm(masks["body"], oaff)
    obstacles = kin.treatment_obstacles(bounds)
    # La quota della base non e' un parametro: il robot appoggia sul pavimento,
    # che sta sotto il lettino, che a sua volta appoggia sul dorso del paziente.
    template = kin.RobotBase(origin_mm=(kin.RobotBase().origin[0],
                                        kin.floor_y_mm(bounds),
                                        kin.RobotBase().origin[2]))
    print("Ottimizzazione della sala (base del robot + angoli di lettino)...")
    t_opt = time.time()
    # Il braccio del lettino va PASSATO: e' lui il filtro che scarta gli
    # orientamenti che il meccanismo non puo' assumere. Senza, l'ottimizzatore
    # ne sceglie di irrealizzabili - e' successo: un pitch di 20 gradi richiede
    # al polso un roll di 12, oltre i 10 di corsa, e il piano conteneva un
    # angolo che il controllo poi dichiarava impossibile.
    couch_arm = kin.treatment_couch_arm(bounds)
    setup = kin.optimize_setup(obstacles, iso, base_template=template,
                               max_couch_poses=args.couch_poses,
                               couch_arm=couch_arm,
                               status_cb=lambda m: print("  ", m))
    base, couch_poses = setup["base"], setup["couch_poses"]
    print(f"  in {time.time() - t_opt:.0f}s")
    print(f"  base {np.round(base.origin, 0)} mm")
    print(f"  angoli di lettino: {[str(c) for c in couch_poses]}")
    print(f"  orientamenti: {setup['angoli_realizzabili']} realizzabili dal "
          f"meccanismo, {setup['angoli_raggiungibili']} anche raggiungibili")
    print(f"  direzioni eseguibili: {setup['copertura_direzioni']}"
          f"/{setup['direzioni_provate']} con questi angoli, "
          f"{setup['copertura_lettino_fermo']} con il lettino fermo")
    print(f"Ostacoli: {', '.join(o.name for o in obstacles)}")

    # Collaudo con LO STESSO solutore e gli stessi parametri del nodo ROS
    # (n_roll=4, ostacoli della scena), DENTRO la dosimetria: i fasci che la
    # sequenza del robot non esegue escono prima della dose finale, e il
    # piano dichiara la dose che verra' erogata davvero.
    def _deliverable(beams):
        t1 = time.time()
        keep, dg = kin.deliverable_subset(
            beams, base, obstacles, couch_poses=couch_poses, n_roll=4,
            status_cb=lambda m: print("  ", m))
        print(f"  collaudo in {time.time() - t1:.0f}s")
        if not dg.get("verificato", True):
            print("ATTENZIONE: il collaudo non ha convergito, la simulazione "
                  "potrebbe non erogare tutti i fasci.")
        return keep

    t0 = time.time()
    res = dz.compute_plan(ct, msk, img.affine, iso, mode="conformal",
                          bone_mask=masks["bones"], bone_affine=oaff,
                          arms_mask=masks["arms"], body_mask=masks["body"],
                          robot_base=base, robot_obstacles=obstacles,
                          robot_couch_poses=couch_poses,
                          beam_filter=None if args.senza_collaudo
                          else _deliverable,
                          status_cb=lambda m: print("  ", m))
    print(f"Piano calcolato in {time.time() - t0:.1f}s")
    m = res["metrics"]
    print(f"Erogabili dal robot: {m.get('fasci_erogabili', '?')} fasci, "
          f"{m.get('fasci_scartati_dal_collaudo', 0)} scartati dal collaudo; "
          f"copertura {m['coverage_pct']:.1f}%, D95 {m['d95_pct']:.1f}%")

    data = save_plan(res, args.out, args.ct, args.mask,
                     robot_base=base, couch_poses=couch_poses)
    print(f"Esportati {len(data['fasci'])} fasci in {args.out}")
