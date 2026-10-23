# ============================================================================
#  test_motion.py - Verifica del moto respiratorio, dell'inseguimento e della
#  dose sotto moto, su casi con risposta nota a priori.
#
#  Come per le metriche: un errore qui produrrebbe numeri plausibili e falsi
#  (un segno invertito fra "tumore rispetto al fascio" e "fascio rispetto al
#  tumore" non si vede in nessun grafico). Si usano quindi situazioni in cui
#  il risultato e' calcolabile a mano.
#
#  USO:  .venv\Scripts\python.exe test_motion.py
# ============================================================================
import numpy as np

import motion_model as mm
import motion_dose as md

failures = []


def check(name, cond, detail=""):
    if cond:
        print(f"  OK   {name}")
    else:
        print(f"  FAIL {name}  {detail}")
        failures.append(name)


# Griglia di dose sintetica: 1 mm isotropo, con una traslazione qualsiasi.
AFF = np.array([[1.0, 0, 0, -40.0], [0, 1.0, 0, -40.0],
                [0, 0, 1.0, -40.0], [0, 0, 0, 1.0]])
N = 81
ax = np.arange(N) - 40.0
X, Y, Z = np.meshgrid(ax, ax, ax, indexing="ij")


print("\n[1] Traccia sintetica")
cfg = mm.MotionConfig(seed=3)
tr = mm.generate_trace(cfg, 120.0)
check("lunghezza = durata * fs", len(tr) == int(round(120 * cfg.fs_hz)),
      len(tr))
p2p = np.ptp(tr.tumor - tr.tumor.mean(0), axis=0)
check("asse dominante cranio-caudale", int(np.argmax(p2p)) == 2, p2p)
check("escursione SI vicina a quella impostata (+-40%)",
      0.6 * 12 < p2p[2] < 1.6 * 12 + 2, p2p[2])
# In inspirazione il tumore scende: correlazione negativa fra torace e Z.
r = np.corrcoef(tr.surrogate, tr.tumor[:, 2])[0, 1]
check("torace sale, tumore scende (corr < -0.8)", r < -0.8, r)
still = mm.generate_trace(cfg.replace(inhale_shift_mm=(0, 0, 0),
                                      drift_mm_per_min=(0, 0, 0)), 30.0)
check("ampiezza 0 e deriva 0 -> tumore fermo",
      np.abs(still.tumor).max() < 1e-9, np.abs(still.tumor).max())
f_base = mm.inhale_shift_for_location(0.0)
f_apex = mm.inhale_shift_for_location(1.0)
check("lobi inferiori si muovono piu' degli apici",
      abs(f_base[2]) > abs(f_apex[2]), (f_base, f_apex))

print("\n[2] Predittori")
fs = 26.0
t = np.arange(int(60 * fs)) / fs
T = 4.0
s = np.sin(2 * np.pi * t / T)
c = mm.MotionConfig(surrogate_noise_mm=0.01)
h = c.horizon_samples
check("orizzonte di 115 ms = 3 campioni a 26 Hz", h == 3, h)
none_p = mm.predict_series(mm.make_predictor("none", c), s)
e_none = mm.prediction_error(s, none_p, h, skip=200)
# senza predizione l'errore e' s(t+h) - s(t) = 2 cos(.) sin(w h / 2), il cui
# valore efficace e' sqrt(2) sin(pi h / (fs T))
expected = np.sqrt(2) * np.sin(np.pi * h / (fs * T))
check("nessun predittore: RMSE = teorico",
      abs(np.sqrt(np.mean(e_none ** 2)) - expected) < 0.01,
      (np.sqrt(np.mean(e_none ** 2)), expected))
for name in ("linear", "kalman", "nlms"):
    p = mm.predict_series(mm.make_predictor(name, c), s)
    e = mm.prediction_error(s, p, h, skip=600)
    rm = np.sqrt(np.mean(e ** 2))
    check(f"{name}: meglio di nessuna predizione su una sinusoide",
          rm < 0.5 * expected, (rm, expected))
p = mm.predict_series(mm.make_predictor("linear", c), 2.0 * t)
check("lineare: esatto su una rampa",
      np.abs(mm.prediction_error(2.0 * t, p, h, skip=20)).max() < 1e-9)
p = mm.predict_series(mm.make_predictor("nlms", c), s)
k = mm.make_predictor("nlms", c)
stream = np.array([k.step(v) for v in s])
check("predict_series coincide con lo streaming",
      np.allclose(p, stream))
try:
    import os as _os
    import motion_predictor as mp
    if _os.path.exists(mp.WEIGHTS_PATH):
        mp.register()
        g = mm.make_predictor("gru", c)
        blk = mm.predict_series(g, s[:300])
        g.reset()
        one = np.array([g.step(v) for v in s[:300]])
        check("GRU: calcolo in blocco = campione per campione",
              np.allclose(blk, one, atol=1e-4), np.abs(blk - one).max())
except ImportError:
    print("  --   GRU non verificata (torch assente)")

print("\n[3] Modello di correlazione")
cm = mm.CorrelationModel(mm.MotionConfig(corr_ridge=1e-6, corr_window=40))
rng = np.random.default_rng(0)
for v in rng.uniform(-1, 5, 30):
    inh = bool(rng.integers(2))
    pos = np.array([0.5 * v, 1.0 + 0.2 * v, -2.0 * v + (1.5 if inh else 0.0)])
    cm.add(v, inh, pos)
got_i = cm.predict(2.0, True)
got_e = cm.predict(2.0, False)
check("ramo inspirazione recupera la retta vera",
      np.allclose(got_i, [1.0, 1.4, -2.5], atol=0.05), got_i)
check("ramo espirazione recupera la retta vera",
      np.allclose(got_e, [1.0, 1.4, -4.0], atol=0.05), got_e)
check("serve almeno 3 punti", not mm.CorrelationModel().ready)

print("\n[4] Inseguimento")
ideal = mm.MotionConfig(seed=5, drift_mm_per_min=(0, 0, 0),
                        surrogate_drift_mm_per_min=0.0, surrogate_lead_s=0.0,
                        hysteresis_s=(0, 0, 0), kv_noise_mm=0.0,
                        surrogate_noise_mm=0.02, surrogate_curvature=0.0)
tr = mm.generate_trace(ideal, 180.0)
res = mm.simulate_tracking(tr, "kalman", ideal)
st_t = mm.error_stats(res.residual)
st_u = mm.error_stats(res.untracked)
check("inseguimento riduce il p95 almeno di 3 volte (caso ideale)",
      st_t["p95_3d"] < st_u["p95_3d"] / 3, (st_t["p95_3d"], st_u["p95_3d"]))
check("fascio spento finche' il modello non e' pronto",
      not res.active[0] and res.active[-1])
check("residuo = vero - stima",
      np.allclose(res.residual, (res.truth - res.estimate)[res.active]))
m = mm.van_herk_margin([1.0, 0.0, 2.0], [1.0, 1.0, 0.0])
check("van Herk: 2.5 Sigma + 0.7 sigma", np.allclose(m, [3.2, 0.7, 5.0]), m)
back = mm.MotionConfig.from_dict(ideal.as_dict())
check("MotionConfig: andata e ritorno JSON", back == ideal)

print("\n[5] Dose sotto moto")
dose = (100.0 * np.exp(-((X - 3) ** 2 + Y ** 2 + Z ** 2) / (2 * 8.0 ** 2))
        ).astype(np.float32)
pts = np.array([[0.0, 0, 0], [5.0, -2, 1], [-7.0, 3, 0]])
static = md.dose_at(dose, AFF, pts)
check("spostamento nullo -> dose statica",
      np.allclose(md.dose_under_motion(dose, AFF, pts, np.zeros((50, 3))),
                  static))
shift = np.tile([[2.0, 0.0, 0.0]], (100, 1))
check("spostamento costante -> dose traslata",
      np.allclose(md.dose_under_motion(dose, AFF, pts, shift),
                  md.dose_at(dose, AFF, pts + [2.0, 0, 0]), atol=1e-4))
# Oscillazione simmetrica attorno al massimo: la dose media si abbassa.
osc = np.stack([np.zeros(400), np.zeros(400),
                8.0 * np.sin(np.linspace(0, 20 * np.pi, 400))], 1)
blur = md.dose_under_motion(dose, AFF, np.array([[3.0, 0, 0]]), osc)
check("il moto sfuma il picco", blur[0] < 0.9 * dose.max(), blur)
off, w = md.displacement_pdf(osc, 1.0)
check("pdf: pesi a somma 1", abs(w.sum() - 1) < 1e-12)
check("pdf: media conservata",
      np.allclose(off.T @ w, osc.mean(0), atol=1e-9), (off.T @ w, osc.mean(0)))

print("\n[6] Maschere")
ball = ((X ** 2 + Y ** 2 + Z ** 2) <= 5 ** 2)
big = md.expand_mask(ball, AFF, [3.0, 3.0, 3.0])
ref = ((X ** 2 + Y ** 2 + Z ** 2) <= 8.0 ** 2)
check("espansione isotropa di 3 mm ~ sfera di 8 mm",
      abs(big.sum() - ref.sum()) / ref.sum() < 0.08, (big.sum(), ref.sum()))
ani = md.expand_mask(ball, AFF, [0.0, 0.0, 6.0])
ext = np.argwhere(ani)
check("espansione solo in Z",
      np.ptp(ext[:, 2]) == np.ptp(np.argwhere(ball)[:, 2]) + 12
      and np.ptp(ext[:, 0]) == np.ptp(np.argwhere(ball)[:, 0]),
      (np.ptp(ext, 0)))
line = np.stack([np.zeros(11), np.zeros(11), np.linspace(-10, 10, 11)], 1)
env = md.envelope_mask(ball, AFF, line)
ez = np.argwhere(env)[:, 2]
check("ITV: allungato di 20 mm lungo Z",
      np.ptp(ez) == np.ptp(np.argwhere(ball)[:, 2]) + 20, np.ptp(ez))
edge = np.zeros_like(ball)
edge[0:3, 40, 40] = True
env2 = md.envelope_mask(edge, AFF, np.array([[-5.0, 0, 0], [0, 0, 0]]))
check("ITV al bordo del volume: nessun rientro dal lato opposto",
      not env2[-10:].any(), np.argwhere(env2)[:, 0].max())

print("\n[7] Inseguimento sul piano del pacchetto ROS")
import os
import json
import kinematics as kin
import dose_fill as dfill
import treatment_log as tlog
import tracking_session as tsm
from plan_export import load_plan, setup_from_plan

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.join(HERE, "ros2_ws", "src", "cyberknife_ros")
beams, plan = load_plan(os.path.join(PKG, "plans", "piano_trattamento.json"))
base, couch = setup_from_plan(plan, kin)
couch = couch or [kin.CouchPose()]
b0 = beams[0]
d0 = b0.target - b0.source
eff = couch[0].base_in_patient(base)
q0, _ = kin.solve_ik(b0.source, d0, eff, n_roll=4)
check("IK di riferimento trovata", q0 is not None)
qs = kin.solve_ik_follow(b0.source, q0, eff)
check("follow con offset nullo = stessa posa",
      qs is not None and np.allclose(qs, q0, atol=1e-6))
off = np.array([2.0, -3.0, 10.0])
qf = kin.solve_ik_follow(b0.source + off, q0, eff)
pe, ae = kin.fk_error_mm(qf, b0.source + off, d0, eff)
check("follow: sorgente spostata esatta, direzione invariata",
      pe < 1e-6 and ae < 1e-4, (pe, ae))
check("follow: A6 non ruota di scatto (roll conservato)",
      abs(np.rad2deg(qf[5] - q0[5])) < 5.0, np.rad2deg(qf - q0))

poses, _ = kin.solve_plan(beams[:6], base, n_roll=4, couch_poses=couch[:1])
cfg = mm.MotionConfig(seed=2)
res_by_mode = {}
for follow in (True, False):
    S = tsm.TrackingSession(cfg, 60.0, kin, base, couch[:1], tracking=follow)
    t, dt, res = 0.0, 1.0 / 50.0, {}
    for k, p in enumerate(poses):
        t += 1.0
        q = p["q"]
        for _ in range(60):
            t += dt
            true, boff = S.at(t)
            q2, hold, why = S.track(p["beam"], 0, boff, q, dt)
            r = S.record(true, boff, hold, why)
            if not hold:
                q = q2
                res.setdefault(k, []).append(r)
    res_by_mode[follow] = (S.summary(), res)
sm_on, res_on = res_by_mode[True]
sm_off, _ = res_by_mode[False]
check("inseguimento: nessuna sospensione",
      sm_on["sospensione_pct"] == 0.0, sm_on["motivi_sospensione"])
check("inseguimento: velocita' dei giunti entro i limiti",
      all(v <= l for v, l in zip(sm_on["velocita_max_giunti_deg_s"],
                                 sm_on["limiti_velocita_deg_s"])))
check("inseguimento: residuo < fascio fermo",
      sm_on["errore_residuo"]["p95_3d"] < sm_off["errore_residuo"]["p95_3d"],
      (sm_on["errore_residuo"]["p95_3d"], sm_off["errore_residuo"]["p95_3d"]))
check("fascio fermo: il residuo e' il moto stesso",
      abs(sm_off["errore_residuo"]["p95_3d"]
          - sm_off["fascio_fermo"]["p95_3d"]) < 1e-9)

print("\n[8] Dose persa per il moto, nel nodo e nel replay")
fill = dfill.DoseFill.from_plan(plan, beams)
geom = ([b.source for b in beams], [b.target for b in beams],
        [b.cone_mm for b in beams], [b.weight for b in beams])
bi = poses[0]["beam_index"]
g0 = dfill.motion_factor(fill, bi, geom, np.zeros((5, 3)))
check("residuo nullo -> fattore 1", np.allclose(g0, 1.0), (g0.min(), g0.max()))
# Spostamento PERPENDICOLARE al fascio: lungo l'asse il tumore resta nel
# cono e perde solo l'inverso del quadrato.
ax_b = beams[bi].target - beams[bi].source
ax_b = ax_b / np.linalg.norm(ax_b)
perp = np.cross(ax_b, [0.0, 0.0, 1.0])
perp = 40.0 * perp / np.linalg.norm(perp)
gfar = dfill.motion_factor(fill, bi, geom, np.tile(perp, (5, 1)))
# Media PESATA sulla dose del fascio: i punti che il fascio gia' non colpiva
# ricevono solo la coda di diffusione, e il loro rapporto non conta.
cb = fill.contrib[:, bi]
lost = float((gfar * cb).sum() / cb.sum())
check("tumore fuori dal cono -> dose del fascio quasi persa", lost < 0.2,
      lost)
sub = {k: dfill.pick_residuals(v).tolist() for k, v in res_on.items()}
timeline, _ = tlog.build_timeline(poses, 90.0, 0.5, 3.0, 1.2)
rep = tlog.build_report(plan, beams, poses, timeline,
                        {"non_raggiungibili": 0}, couch[:1], dose_fill=fill,
                        respiro=sm_on, residui=sub)
rep = json.loads(json.dumps(rep))              # come dopo la scrittura
steps = tlog.replay_doses(fill, rep, beams)
direct = sum(dfill.motion_factor(fill, p["beam_index"], geom, res_on[k])
             * fill.contrib[:, p["beam_index"]] for k, p in enumerate(poses))
check("replay = dose accumulata dal nodo (stessi campioni)",
      # il resoconto arrotonda i residui a 0.01 mm: 0.05 punti di dose
      np.allclose(steps[-1], direct, atol=5e-2),
      np.abs(steps[-1] - direct).max())
check("resoconto: sezione respiro presente",
      rep["respiro"]["predittore"] == cfg.predictor)

print("\n[9] Erogazione proporzionale al peso")
dw = tlog.dwell_times([1.0, 3.0, 0.001], total_s=40.0, min_s=0.3)
check("sosta proporzionale al peso", abs(dw[1] / dw[0] - 3.0) < 1e-9, dw)
check("sosta minima rispettata", dw[2] == 0.3, dw)
check("tempo di fascio acceso = totale (a meno del minimo)",
      abs(dw[:2].sum() - 40.0 * 4.0 / 4.001) < 1e-6, dw.sum())
tl, _ = tlog.build_timeline(poses[:3], 90.0, 0.5, 3.0,
                            tlog.dwell_times([1.0, 2.0, 3.0], 12.0))
d_er = [e["dur"] for e in tl if e["tipo"] == "erogazione"]
check("la linea temporale usa le soste per posa",
      np.allclose(d_er, [2.0, 4.0, 6.0]), d_er)

print("\n[10] Gating: il robot aspetta il tumore")
gcfg = mm.MotionConfig(seed=7)
Sg = tsm.TrackingSession(gcfg, 120.0, kin, base, couch[:1], gating_mm=3.0)
check("gating: niente inseguimento", not Sg.tracking)
p0 = poses[0]
on, held, t = [], 0, 0.0
for _ in range(3000):
    t += 1 / 50.0
    true, boff = Sg.at(t)
    q, hold, why = Sg.track(p0["beam"], 0, boff, p0["q"], 1 / 50.0)
    check_q = np.allclose(q, p0["q"])
    if not check_q:
        break
    r = Sg.record(true, boff, hold, why)
    if hold:
        held += 1
    else:
        on.append(true)
check("gating: il braccio resta sulla posa pianificata", check_q)
check("gating: fascio sospeso per una parte del tempo", 0.2 < held / 3000 < 0.95,
      held / 3000)
p95_on = np.percentile(np.linalg.norm(np.array(on), axis=1), 95)
all_true = np.array([Sg.trace.tumor[i] for i in range(len(Sg.trace.t))])
p95_all = np.percentile(np.linalg.norm(all_true, axis=1), 95)
check("gating: a fascio acceso il tumore e' piu' vicino al piano",
      p95_on < 0.7 * p95_all, (p95_on, p95_all))
Sw = tsm.TrackingSession(gcfg, 60.0, kin, base, couch[:1], gating_mm=0.01,
                         max_wait_s=1.0)
forced = False
for i in range(200):
    true, boff = Sw.at(i / 50.0)
    _, hold, _ = Sw.track(p0["beam"], 0, boff, p0["q"], 1 / 50.0)
    forced |= not hold
check("attesa massima: una finestra impossibile non blocca la seduta",
      forced and Sw.n_forced > 0, Sw.n_forced)

print("\n[10b] Correzione del lettino contro la deriva (fascio fermo)")
dcfg = mm.MotionConfig(seed=4, drift_mm_per_min=(0.0, 0.0, 2.0))
res_drift = {}
for corr in (None, 1.5):
    Sd = tsm.TrackingSession(dcfg, 400.0, kin, base, couch[:1],
                             tracking=False, couch_correction_mm=corr)
    tt = np.arange(0, 380, 0.1)
    mean_z = np.mean([Sd.at(x)[0][2] for x in tt[-600:]])   # ultimi 60 s
    res_drift[corr] = (mean_z, len(Sd.couch_log))
check("senza correzione il tumore deriva di centimetri",
      abs(res_drift[None][0]) > 8.0, res_drift[None])
check("con la correzione resta vicino alla posizione pianificata",
      abs(res_drift[1.5][0]) < 3.0 and res_drift[1.5][1] >= 3,
      res_drift[1.5])
St = tsm.TrackingSession(dcfg, 200.0, kin, base, couch[:1])
check("inseguendo il lettino non si muove",
      St.couch_corr_mm is None)

print("\n[11] Ossa attraversate da un fascio che si sposta")
from dosimetry import Beam
red = np.full((60, 60, 60), 1.0, np.float32)
red[:, :, 40:44] = 1.8                         # lastra d'osso a z = 40..43
gaff = np.diag([2.0, 2.0, 2.0, 1.0])            # 2 mm, origine in 0
bz = Beam([60.0, 60.0, 118.0], [60.0, 60.0, 20.0], 10.0)   # lungo -z
bx = Beam([2.0, 60.0, 60.0], [100.0, 60.0, 60.0], 10.0)    # lungo +x
tabs = md.path_attenuation_tables(red, gaff, [bz, bx], moving="bersaglio")
n = len(tabs["griglia_mm"])
tz = np.asarray(tabs["fasci"][0]).reshape(n, n, n)
tx = np.asarray(tabs["fasci"][1]).reshape(n, n, n)
c = n // 2
check("spostamento nullo -> fattore 1", abs(tz[c, c, c] - 1.0) < 1e-6)
# fascio lungo -z: il bersaglio che si allontana dalla sorgente attraversa
# piu' tessuto, e l'osso resta nel percorso: fattore < 1
check("bersaglio piu' profondo -> meno fluenza",
      tz[c, c, 0] < tz[c, c, c] < tz[c, c, -1], tz[c, c, [0, c, -1]])
# fascio lungo +x: la lastra e' perpendicolare a z, e uno spostamento in z di
# 15 mm la porta... dentro il percorso? No: il fascio corre a z=60 mm e la
# lastra sta a 80-88 mm; spostato di +15 mm corre a 75 mm, ancora fuori.
check("fascio parallelo alla lastra e spostato di lato: nessun effetto",
      abs(tx[c, c, -1] - 1.0) < 0.01, tx[c, c, -1])
tabs_e = md.path_attenuation_tables(red, gaff, [bz], moving="entrambi")
te = np.asarray(tabs_e["fasci"][0]).reshape(n, n, n)
# Inseguire lungo l'asse del fascio NON annulla l'effetto: il tumore sale di
# 15 mm dentro il corpo (qui tutta la griglia e' tessuto a densita' 1) e
# attraversa 1.5 cm di tessuto in meno: exp(0.047 * 1.5) = 1.073.
check("inseguimento lungo l'asse: il tumore meno profondo riceve di piu'",
      abs(te[c, c, -1] - np.exp(0.047 * 1.5)) < 0.01, te[c, c, -1])

class _F:
    path_tables = tabs
check("path_factor: tabella interpolata sugli spostamenti",
      abs(dfill.path_factor(_F, 0, [[0, 0, 0]]) - 1.0) < 1e-9 and
      dfill.path_factor(_F, 0, [[0, 0, -15.0]]) < 1.0)
check("path_factor: senza tabelle vale 1",
      dfill.path_factor(object(), 0, [[0, 0, 5.0]]) == 1.0)

print("\n[12] Limite al peso del singolo fascio")
import planner
rng = np.random.default_rng(1)
# Un fascio uniforme (il bersaglio "ideale") e altri undici disomogenei che
# insieme possono sostituirlo: il solutore preferisce il primo.
A_t = (0.2 + 1.2 * rng.random((200, 12))).astype(np.float32)
A_t[:, 0] = 1.0
A_r = rng.random((50, 12)).astype(np.float32) * 0.01
vw = np.ones(200)
cols = list(range(12))
pen = np.full(12, 1e-3)
w0 = planner._solve_nnls(A_t, A_r, vw, cols, 100.0, 0.1, pen)
check("senza limite un fascio domina", w0.max() / w0.sum() > 0.5,
      w0.max() / w0.sum())
w1, ncap = planner.cap_weights(A_t, A_r, vw, 100.0, cols, pen, 0.1, w0,
                               np.ones(12), 0.2)
check("con limite al 20%: nessun fascio oltre", w1.max() / w1.sum() <= 0.205,
      w1.max() / w1.sum())
check("limite: la dose media resta vicina alla prescrizione",
      abs(float((A_t @ w1).mean()) - 100.0) < 10.0, float((A_t @ w1).mean()))

print("\n" + "=" * 60)
if failures:
    print(f"FALLITI: {len(failures)} -> {failures}")
    raise SystemExit(1)
print("Tutti i controlli superati.")
print("=" * 60)
