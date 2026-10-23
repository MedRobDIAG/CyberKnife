# ============================================================================
#  treatment_log.py - Il resoconto di una seduta di trattamento.
#
#  Il nodo ROS lo scrive quando l'ultimo fascio e' stato erogato; l'app lo
#  rilegge e ne fa una finestra con il replay del tumore che si riempie di
#  dose e l'andamento del processo. Il modulo e' condiviso fra i due mondi -
#  Python 3.10 in WSL, 3.14 su Windows - quindi usa solo NumPy.
#
#  COSA C'E' DENTRO
#  ----------------
#    sintesi     fasci erogati e pianificati, tempi, rotazioni del lettino,
#                copertura finale, franco e margine minimi
#    fasci       una riga per fascio, NELL'ORDINE DI EROGAZIONE: collimatore,
#                peso, angolo di lettino, giunti, margine dai fine corsa,
#                franco dagli ostacoli, il trasferimento che lo precede, e lo
#                stato del bersaglio subito dopo (copertura, dose media,
#                minima, D95)
#    eventi      la linea temporale completa: trasferimenti, erogazioni,
#                tratti con il lettino in rotazione (braccio e lettino
#                insieme, come li verifica kinematics), con inizio e durata
#    respiro     (facoltativo) moto del tumore e qualita' dell'inseguimento:
#                errore residuo, sospensioni del fascio, velocita' dei giunti.
#                Per ogni fascio si salvano pochi campioni del residuo, da cui
#                il replay ricalcola la dose persa con la stessa funzione del
#                nodo (dose_fill.motion_factor).
#
#  COSA NON C'E', E PERCHE'
#  ------------------------
#  La dose voxel per voxel a ogni passo non viene salvata: sarebbero 52 x 808
#  numeri per un piano piccolo, e molti di piu' per uno grande. Il replay la
#  RICOSTRUISCE dal piano (che contiene la dose finale di ogni voxel) e
#  dall'ordine di erogazione, con la stessa ripartizione di dose_fill che usa
#  il nodo. Cosi' il replay coincide per costruzione con cio' che si e' visto
#  in RViz, e il file resta di pochi kilobyte.
# ============================================================================
import numpy as np

try:                                   # dentro il pacchetto ROS
    from . import dose_fill as _dfill
except ImportError:                    # su Windows
    import dose_fill as _dfill

REPORT_VERSION = 1
REPORT_NAME = "resoconto_trattamento.json"


def _f(x, nd=3):
    """Numero JSON-serializzabile, arrotondato."""
    if x is None:
        return None
    x = float(x)
    if not np.isfinite(x):
        return None
    return round(x, nd)


def couch_legs(pose, n_legs):
    """Intervallo [inizio, fine) dei tratti con il lettino in rotazione.

    E' la chiave "couch_legs" di kinematics.solve_plan, sugli indici dei tratti
    di [q_prev] + via + [q]; quando manca, il lettino ruota lungo TUTTI i
    tratti, simultaneo al braccio.
    """
    s, e = pose.get("couch_legs") or (0, n_legs)
    s, e = max(0, int(s)), min(n_legs, int(e))
    return (s, e) if e > s else (0, n_legs)


def dwell_times(weights, total_s, min_s=0.3):
    """Durata dell'erogazione di ogni fascio, proporzionale al suo peso.

    Il peso di un fascio e' la sua quota di unita' monitor: a rateo di dose
    costante, il tempo di fascio acceso e' proporzionale al peso. Con una
    sosta uguale per tutti, un fascio che porta il 17% della dose (misurato
    su un piano reale) durava 0.6 s come uno dello 0.1%, cioe' una frazione
    di respiro: il suo errore d'inseguimento non si mediava su piu' cicli e
    pesava sul risultato quanto un sesto dell'intero trattamento.
    """
    w = np.maximum(np.asarray(weights, dtype=np.float64), 0.0)
    if w.sum() <= 0:
        return np.full(len(w), float(min_s))
    return np.maximum(float(total_s) * w / w.sum(), float(min_s))


def build_timeline(poses, joint_speed_deg_s, min_transfer_s, couch_move_s,
                   dwell_s):
    """La linea temporale della seduta: (eventi, numero_di_tratti).

    Ogni evento sa da dove parte, dove arriva, quanto dura e se il fascio e'
    acceso. I passaggi di ritiro e le rotazioni del lettino entrano come
    tratti in piu', non come casi speciali.

    Una manovra fra due gruppi si anima ESATTAMENTE come la verifica
    kinematics (vedi couch_legs): nei tratti del suo intervallo il lettino
    ruota INSIEME al braccio (evento "lettino", con la frazione di rotazione
    a inizio e fine tratto), prima sta fermo all'angolo di partenza e dopo a
    quello d'arrivo. Prima il lettino ruotava a braccio fermo dopo il primo
    tratto, e i tratti seguenti tornavano all'angolo vecchio: una manovra mai
    verificata contro le collisioni.

    Stava dentro il nodo ROS: spostata qui perche' il resoconto dipende da
    lei, e senza ROS non si poteva riprodurre ne' verificare.
    """
    timeline = []
    n_legs = 0
    for k, pose in enumerate(poses):
        if k > 0:
            prev = poses[k - 1]
            pts = [prev["q"]] + list(pose.get("via") or []) + [pose["q"]]
            n = len(pts) - 1
            changed = pose["couch"] != prev["couch"]
            s, e = couch_legs(pose, n)
            for w in range(n):
                span = float(np.rad2deg(np.abs(
                    np.asarray(pts[w + 1]) - np.asarray(pts[w])).max()))
                dur = max(span / joint_speed_deg_s, min_transfer_s)
                ev = {"tipo": "trasferimento", "q0": pts[w],
                      "q1": pts[w + 1], "idx": -1,
                      "couch": prev["couch"] if w < e else pose["couch"]}
                if changed and s <= w < e:
                    # Il lettino ruota di una frazione uguale per tratto,
                    # come in couch_transition: il tratto dura quanto il piu'
                    # lento fra braccio e lettino.
                    dur = max(dur, couch_move_s / (e - s))
                    ev.update(tipo="lettino", couch=prev["couch"],
                              couch_to=pose["couch"],
                              couch_u=((w - s) / (e - s),
                                       (w - s + 1) / (e - s)))
                ev["dur"] = dur
                timeline.append(ev)
                n_legs += 1
        # Sosta sul nodo: e' l'erogazione, il fascio e' acceso. `dwell_s`
        # e' un valore unico oppure uno per posa (dwell_times).
        dw = (float(dwell_s[k]) if np.ndim(dwell_s) else float(dwell_s))
        timeline.append({"tipo": "erogazione", "q0": pose["q"],
                         "q1": pose["q"], "dur": dw, "idx": k,
                         "couch": pose["couch"]})
    return timeline, n_legs


def schedule(timeline):
    """La linea temporale con gli istanti d'inizio: (eventi, durata_totale)."""
    t = 0.0
    out = []
    for e in timeline:
        q0 = np.asarray(e["q0"], dtype=np.float64)
        q1 = np.asarray(e["q1"], dtype=np.float64)
        out.append({
            "tipo": e["tipo"],
            "inizio_s": _f(t),
            "durata_s": _f(e["dur"]),
            "posa": int(e.get("idx", -1)),
            "lettino": int(e.get("couch", 0)),
            "lettino_verso": (int(e["couch_to"]) if e.get("couch_to")
                              is not None else None),
            "escursione_deg": _f(np.rad2deg(np.abs(q1 - q0).max()), 2),
        })
        if e.get("couch_u") is not None:
            out[-1]["lettino_frazione"] = [_f(u, 4) for u in e["couch_u"]]
        t += float(e["dur"])
    return out, t


def dose_stats(dose, rx):
    """Stato del bersaglio: copertura, dose media e minima, D95 (% di rx)."""
    d = np.asarray(dose, dtype=np.float64)
    if not len(d) or rx <= 0:
        return {"copertura_pct": None, "copertura_95_pct": None,
                "dose_media_pct": None, "dose_min_pct": None, "d95_pct": None}
    return {
        "copertura_pct": _f(100.0 * np.mean(d >= rx - 1e-9), 2),
        # V95: bersaglio ad ALMENO il 95% della prescrizione. La copertura
        # alla prescrizione e' una soglia secca: con una dose omogenea tutti i
        # voxel la attraversano insieme e la curva resta a zero fino agli
        # ultimi fasci, poi salta. La V95 distingue "quasi coperto" da
        # "scoperto": su lung_085, senza 7 fasci, copertura 64% ma dose minima
        # al 96.5% della prescrizione.
        "copertura_95_pct": _f(100.0 * np.mean(d >= 0.95 * rx - 1e-9), 2),
        "dose_media_pct": _f(100.0 * d.mean() / rx, 2),
        "dose_min_pct": _f(100.0 * d.min() / rx, 2),
        # D95: la dose ricevuta ALMENO dal 95% del bersaglio.
        "d95_pct": _f(100.0 * np.percentile(d, 5) / rx, 2),
    }


def _geom(beams):
    return ([b.source for b in beams], [b.target for b in beams],
            [b.cone_mm for b in beams], [b.weight for b in beams])


def build_report(plan, beams, poses, timeline, diag, couch_poses,
                 dose_fill=None, wall_s=None, meta=None, respiro=None,
                 residui=None, percorsi=None):
    """Compone il resoconto di una seduta.

    `poses` e' l'uscita di kinematics.solve_plan, NELL'ORDINE DI EROGAZIONE;
    `timeline` e' la linea temporale del nodo; `dose_fill` (facoltativo) e'
    la ripartizione della dose fra i fasci, da cui si ricava l'andamento del
    bersaglio. `respiro` e `residui` (posa -> campioni del residuo) arrivano
    solo se il tumore respirava.
    """
    residui = residui or {}
    percorsi = percorsi or {}
    geom = _geom(beams) if residui else None
    eventi, t_tot = schedule(timeline)

    # Tempo e movimento che PRECEDONO ogni erogazione: tutto cio' che sta fra
    # l'erogazione precedente e questa. Un tratto "lettino" e' uno spostamento
    # del braccio con il lettino che ruota: conta come tratto e, a parte, come
    # tempo di rotazione del lettino.
    prima = {}
    acc = {"durata_s": 0.0, "tratti": 0, "escursione_deg": 0.0,
           "lettino_s": 0.0}
    for ev in eventi:
        if ev["tipo"] == "erogazione":
            prima[ev["posa"]] = dict(acc, inizio_s=ev["inizio_s"])
            acc = {"durata_s": 0.0, "tratti": 0, "escursione_deg": 0.0,
                   "lettino_s": 0.0}
        else:
            if ev["tipo"] == "lettino":
                acc["lettino_s"] += ev["durata_s"]
            acc["durata_s"] += ev["durata_s"]
            acc["tratti"] += 1
            acc["escursione_deg"] += ev["escursione_deg"] or 0.0

    rx = float(plan.get("prescrizione_pct", 80.0))
    dose = (np.zeros(dose_fill.n_points) if dose_fill is not None else None)
    fasci = []
    for k, p in enumerate(poses):
        bi = int(p["beam_index"])
        b = beams[bi]
        ci = int(p.get("couch", 0))
        c = couch_poses[ci] if ci < len(couch_poses) else None
        pr = prima.get(k, {})
        riga = {
            "ordine": k + 1,
            "fascio": bi,
            "collimatore_mm": _f(b.cone_mm, 1),
            "peso": _f(b.weight, 4),
            "lettino": ci,
            "lettino_yaw_deg": _f(c.yaw_deg, 1) if c is not None else None,
            "giunti_deg": [_f(v, 2) for v in np.rad2deg(np.asarray(p["q"]))],
            "margine": _f(p.get("margine"), 4),
            "franco_mm": _f(p.get("franco_mm"), 1),
            "con_ritiro": bool(p.get("via")),
            "con_rotazione_lettino": bool(p.get("couch_move")),
            "inizio_s": pr.get("inizio_s"),
            "trasferimento_s": _f(pr.get("durata_s"), 3),
            "trasferimento_tratti": int(pr.get("tratti", 0)),
            "trasferimento_deg": _f(pr.get("escursione_deg"), 2),
            "rotazione_lettino_s": _f(pr.get("lettino_s"), 3),
        }
        if k in residui:
            R = np.asarray(residui[k], dtype=np.float64)
            riga["residuo_campioni_mm"] = [[_f(v, 2) for v in r] for r in R]
            riga["residuo_medio_mm"] = _f(
                float(np.linalg.norm(R, axis=1).mean()), 2)
        if k in percorsi:
            riga["percorso_campioni_mm"] = [[_f(v, 2) for v in r]
                                            for r in np.asarray(percorsi[k])]
        if dose is not None:
            g = (_dfill.motion_factor(dose_fill, bi, geom, residui[k])
                 if k in residui else 1.0)
            g = g * _dfill.path_factor(dose_fill, bi, percorsi.get(k))
            dose = dose + g * dose_fill.contrib[:, bi]
            riga.update(dose_stats(dose, dose_fill.rx))
        fasci.append(riga)

    finale = (dose_stats(dose, dose_fill.rx) if dose is not None
              else dose_stats([], rx))
    marg = [r["margine"] for r in fasci if r["margine"] is not None]
    fr = [r["franco_mm"] for r in fasci if r["franco_mm"] is not None]
    t_tr = sum(e["durata_s"] for e in eventi if e["tipo"] == "trasferimento")
    t_er = sum(e["durata_s"] for e in eventi if e["tipo"] == "erogazione")
    t_le = sum(e["durata_s"] for e in eventi if e["tipo"] == "lettino")
    sintesi = {
        "fasci_pianificati": len(beams),
        "fasci_erogati": len(poses),
        "fasci_non_raggiungibili": int(diag.get("non_raggiungibili", 0)),
        "angoli_lettino": [
            {"indice": i, "yaw_deg": _f(c.yaw_deg, 1),
             "pitch_deg": _f(c.pitch_deg, 1),
             "fasci": sum(1 for r in fasci if r["lettino"] == i)}
            for i, c in enumerate(couch_poses)],
        "rotazioni_lettino": int(sum(1 for r in fasci
                                     if r["con_rotazione_lettino"])),
        "trasferimenti_con_ritiro": int(sum(1 for r in fasci
                                            if r["con_ritiro"])),
        "tempo_totale_s": _f(t_tot, 2),
        "tempo_trasferimenti_s": _f(t_tr, 2),
        "tempo_erogazione_s": _f(t_er, 2),
        "tempo_lettino_s": _f(t_le, 2),
        "tempo_reale_s": _f(wall_s, 2),
        "movimento_totale_deg": _f(diag.get("movimento_totale_deg"), 1),
        "margine_minimo": _f(min(marg), 4) if marg else None,
        "margine_mediano": _f(float(np.median(marg)), 4) if marg else None,
        "franco_minimo_mm": _f(min(fr), 1) if fr else None,
        "franco_mediano_mm": _f(float(np.median(fr)), 1) if fr else None,
        "prescrizione_pct": _f(rx, 1),
        "punti_bersaglio": int(dose_fill.n_points) if dose_fill else 0,
        "finale": finale,
        # Pianificata sugli STESSI punti della consegnata (dose della Fase 2
        # senza moto): cosi' la differenza e' solo cio' che ha fatto il moto.
        "copertura_pianificata_pct": (
            _f(dose_stats(dose_fill.final, dose_fill.rx)["copertura_pct"], 2)
            if dose_fill is not None else
            _f(plan.get("metriche", {}).get("coverage_pct"), 2)),
        "volume_giudicato": (plan.get("dose_tumore") or {}).get(
            "volume", "bersaglio"),
    }
    return {
        "formato": REPORT_VERSION,
        "piano_generato": plan.get("generato"),
        "meta": dict(meta or {}),
        "sintesi": sintesi,
        "fasci": fasci,
        "eventi": eventi,
        "respiro": respiro,
    }


def write_report(report, path):
    """Scrittura ATOMICA: l'app potrebbe leggerlo mentre il nodo lo scrive."""
    import json
    import os
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1, ensure_ascii=False)
    os.replace(tmp, path)


def load_report(path):
    import json
    with open(path, encoding="utf-8") as fh:
        rep = json.load(fh)
    if rep.get("formato") != REPORT_VERSION:
        raise ValueError(
            f"Resoconto in formato {rep.get('formato')}, atteso "
            f"{REPORT_VERSION}: rilancia la simulazione.")
    return rep


def replay_doses(dose_fill, report, beams=None):
    """Dose dei voxel dopo ciascun fascio, nell'ordine della seduta.

    Restituisce una matrice (fasci + 1, voxel): la riga 0 e' il bersaglio
    prima del primo fascio, la riga k dopo il k-esimo. E' la base del replay:
    fra due righe si interpola, come faceva il nodo durante l'erogazione.

    Con il respiro serve `beams` (la geometria del piano): ogni fascio si
    scala con il fattore ricavato dai suoi campioni di residuo, come nel nodo.
    """
    order = [int(r["fascio"]) for r in report["fasci"]]
    steps = np.zeros((len(order) + 1, dose_fill.n_points))
    geom = _geom(beams) if beams is not None else None
    for k, (bi, row) in enumerate(zip(order, report["fasci"])):
        g = 1.0
        if geom is not None and row.get("residuo_campioni_mm"):
            g = _dfill.motion_factor(dose_fill, bi, geom,
                                     row["residuo_campioni_mm"])
        g = g * _dfill.path_factor(dose_fill, bi,
                                   row.get("percorso_campioni_mm"))
        steps[k + 1] = steps[k] + g * dose_fill.contrib[:, bi]
    return steps
