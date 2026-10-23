# ============================================================================
#  tracking_session.py - L'inseguimento del respiro dentro la seduta simulata.
#
#  E' il ponte fra motion_model (respiro, predittore, correlazione) e il
#  robot: a ogni passo dell'animazione dice dove sta il tumore, dove lo crede
#  il sistema, e quale configurazione di giunti porta il fascio li'.
#
#  Il nodo ROS lo usa cosi' com'e'; su Windows lo usano i test. Per questo il
#  modulo non importa ne' ROS ne' la cinematica: `kin` arriva da fuori, come
#  in plan_export.setup_from_plan, perche' nel pacchetto la cinematica e' una
#  copia con un altro nome di modulo.
#
#  REGOLE DEL CONTROLLO
#  --------------------
#  - La stima entra in vigore con la LATENZA del sistema: calcolata al
#    campione k, muove il fascio all'istante t_k + 115 ms.
#  - Durante l'erogazione la sorgente si sposta dell'offset stimato, con la
#    stessa orientazione della posa pianificata (kin.solve_ik_follow).
#  - Se la posa non esiste, o se raggiungerla chiederebbe a un giunto piu'
#    della sua velocita' massima, il fascio si SOSPENDE per quel passo (beam
#    hold) e il braccio resta dov'e'. Il sistema reale fa lo stesso.
#  - Il modello di correlazione si costruisce PRIMA di erogare: le immagini
#    iniziali vengono "acquisite" all'avvio, in un pre-rotolamento istantaneo.
#
#  DUE MODI DI GESTIRE IL MOTO
#  ---------------------------
#    inseguimento   il fascio segue la stima del tumore;
#    gating         il fascio resta FERMO sulla posizione pianificata e spara
#                   solo quando la stima del tumore e' entro `gating_mm` da
#                   li'. E' il robot che aspetta: misurato su 12 respiri da 5
#                   minuti (escursione 15 mm), p95 dell'errore da 8.6 mm a
#                   fascio fermo a 4.3 mm con finestra di 3 mm, al prezzo di
#                   un fascio acceso il 24% del tempo.
#  Il gating decide sulla STIMA (torace + correlazione), non sulla posizione
#  vera, che il sistema non conosce.
#
#  CORREZIONE DEL LETTINO (tecniche a fascio fermo)
#  ------------------------------------------------
#  Il tumore deriva lentamente (0.8 mm/min nel modello). Inseguendo, il
#  fascio lo segue comunque; a fascio fermo no, e una finestra di gating
#  fissa si svuota: misurato, fascio acceso dal 28% all'1% dopo 14 minuti, e
#  la seduta non finiva piu'. Come nei sistemi reali, a ogni radiografia si
#  confronta la posizione MEDIA stimata degli ultimi 20 s con quella
#  pianificata e, oltre `couch_correction_mm`, il RoboCouch riporta il
#  paziente in posizione. Tutto cio' che sta sul lettino (tumore e sua
#  stima) si sposta della correzione; il fascio, fermo nella stanza, no.
#  Dopo ogni correzione il modello di correlazione si ricostruisce con
#  immagini nuove (vedi _rebuild_model).
#
#  Il chiamante deve congelare il tempo dell'erogazione mentre il fascio e'
#  sospeso, altrimenti quella dose va persa invece di essere erogata dopo.
# ============================================================================
from collections import deque

import numpy as np

try:                                   # dentro il pacchetto ROS
    from . import motion_model as mm
except ImportError:                    # su Windows, come modulo di primo livello
    import motion_model as mm


class TrackingSession:

    def __init__(self, cfg, duration_s, kin, base, couch_poses,
                 obstacles=None, tracking=True, rng_seed=None,
                 gating_mm=None, max_wait_s=20.0, couch_correction_mm=1.5,
                 baseline_window_s=20.0):
        self.cfg = cfg
        self.kin = kin
        self.gating_mm = None if gating_mm is None else float(gating_mm)
        # Il gating tiene il fascio fermo: inseguire e aspettare insieme non
        # ha senso (il fascio seguirebbe un tumore che si aspetta).
        self.tracking = bool(tracking) and self.gating_mm is None
        self.max_wait_s = float(max_wait_s)
        # Correzione del lettino solo a fascio fermo: inseguendo non serve.
        self.couch_corr_mm = (None if self.tracking or couch_correction_mm
                              is None else float(couch_correction_mm))
        self.baseline_window_s = float(baseline_window_s)
        self.couch_shift = np.zeros(3)
        self.couch_log = []
        self._est_hist = deque()
        self._rebuild_kv = set()
        self._last_rebuild_t = -1e9
        self.estimate = np.zeros(3)
        self._wait = 0.0
        self.n_forced = 0
        self.obstacles = obstacles or None
        self.bases = [c.base_in_patient(base) for c in couch_poses]
        seed = cfg.seed if rng_seed is None else rng_seed
        self.rng = np.random.default_rng(seed + 11)

        init_t, self.t_start, _ = mm.kv_schedule(cfg, 1.0)
        # La seduta si allunga quando il fascio aspetta (gating: fascio acceso
        # ~1/4 del tempo; ricostruzioni del modello; pose irraggiungibili). La
        # traccia deve coprire anche quel tempo: quando finiva, il respiro si
        # fermava sull'ultimo campione e, se era fuori finestra, ogni fascio
        # aspettava il massimo - la seduta di gating non terminava piu'.
        stretch = 8.0 if self.gating_mm is not None else 2.0
        self.trace = mm.generate_trace(
            cfg, self.t_start + stretch * duration_s + 120.0)
        _, _, later = mm.kv_schedule(cfg, self.trace.t[-1])
        fs = cfg.fs_hz
        self.kv_idx = set(int(round(x * fs))
                          for x in np.concatenate([init_t, later]))
        self.tracker = mm.OnlineTracker(cfg, cfg.predictor)
        self.k = 0
        self.pending = deque()
        self.beam_offset = np.zeros(3)
        self.kv_log = []
        # statistiche della sola erogazione
        self.residuals = []
        self.untracked = []
        self.n_ticks = 0
        self.n_hold = 0
        self.hold_reasons = {}
        self.max_speed = np.zeros(6)
        self._advance(self.t_start)
        self.kv_log = []                  # le immagini iniziali non si contano

    # ------------------------------------------------------------------
    @property
    def predictor(self):
        return self.tracker.predictor.name

    def _advance(self, tt):
        tr, cfg = self.trace, self.cfg
        n = len(tr.t)
        while self.k < n and tr.t[self.k] <= tt:
            k = self.k
            est = self.tracker.step(tr.surrogate[k], tr.t[k])
            if k in self.kv_idx:
                meas = tr.tumor[k] + cfg.kv_noise_mm * self.rng.standard_normal(3)
                err = self.tracker.kv(meas)
                if err is not None:
                    self.kv_log.append((float(tr.t[k]), err))
                # Solo le radiografie PERIODICHE decidono una correzione:
                # quelle della ricostruzione servono al modello, e usarle
                # innescava correzioni a catena (77 in 15 minuti).
                if k not in self._rebuild_kv:
                    self._couch_check(float(tr.t[k]), err)
            ready = self.tracker.ready
            self.pending.append((tr.t[k] + cfg.latency_s,
                                 est if ready else np.zeros(3)))
            if ready:
                self._est_hist.append((float(tr.t[k]), np.asarray(est)))
                while self._est_hist and (self._est_hist[0][0] <
                                          tr.t[k] - self.baseline_window_s):
                    self._est_hist.popleft()
            self.k += 1
        while self.pending and self.pending[0][0] <= tt:
            self.beam_offset = np.asarray(self.pending.popleft()[1], float)
        # Stima nella stanza: il lettino ha gia' spostato il paziente.
        self.estimate = self.beam_offset - self.couch_shift

    def _couch_check(self, t, kv_err=None):
        """A una radiografia: la posizione media e' uscita di tolleranza?

        Media delle stime degli ultimi secondi, CORRETTA con l'errore che la
        radiografia ha appena misurato sul modello (kv_err = stima - misura):
        il modello di correlazione e' fatto soprattutto di immagini vecchie e
        sottostima la deriva. Senza questa correzione, con 2 mm/min di deriva
        il tumore restava in media a 6 mm dalla posizione pianificata.
        """
        if (self.couch_corr_mm is None or t < self.t_start
                or len(self._est_hist) < 10
                or t - self._last_rebuild_t < self.baseline_window_s + 5.0):
            return
        mean = np.mean([e for _, e in self._est_hist], axis=0)
        if kv_err is not None:
            mean = mean - np.asarray(kv_err)
        mean = mean - self.couch_shift
        if np.linalg.norm(mean) > self.couch_corr_mm:
            self.couch_shift = self.couch_shift + mean
            self.couch_log.append((t, mean.tolist()))
            self._rebuild_model(t)

    def _rebuild_model(self, t):
        """Dopo uno spostamento del lettino il modello di correlazione si
        RIFA', come nel sistema reale: nuove radiografie a fasi diverse
        (spaziate di 0.618 periodi), fascio sospeso finche' non e' pronto.

        Il modello vecchio e' fatto di immagini raccolte prima della deriva:
        misurato, tenendolo il gating vedeva fuori finestra un tumore appena
        ricentrato dal lettino (fascio acceso dal 26% al 6%), e riallinearlo
        all'ultima immagine peggiorava l'errore (p95 7.4 mm), perche' una
        sola immagine e' rumorosa e dipende dalla fase.
        """
        cfg = self.cfg
        self.tracker.model = mm.CorrelationModel(cfg)
        for i in range(max(3, cfg.n_init_kv // 2)):
            tk = t + (i + 1) * 0.618 * cfg.period_s
            k = int(round(tk * cfg.fs_hz))
            self.kv_idx.add(k)
            self._rebuild_kv.add(k)
        self._last_rebuild_t = t
        self._est_hist.clear()
        self.n_rebuild = getattr(self, "n_rebuild", 0) + 1

    def at(self, t_anim):
        """(offset vero del tumore, offset del fascio) all'istante t_anim."""
        tt = min(self.t_start + float(t_anim), float(self.trace.t[-1]))
        self._advance(tt)
        tr = self.trace
        true = np.array([np.interp(tt, tr.t, tr.tumor[:, a]) for a in range(3)])
        true = true - self.couch_shift
        beam = self.beam_offset.copy() if self.tracking else np.zeros(3)
        return true, beam

    def track(self, beam, couch_index, offset, q_prev, dt):
        """Posa che punta il fascio spostato di `offset`, o sospensione.

        Restituisce (q, sospeso, motivo). Con il gating il braccio resta sulla
        posa pianificata e il fascio si sospende finche' la stima del tumore
        e' fuori finestra; oltre `max_wait_s` di attesa consecutiva si eroga
        comunque (e lo si conta): una deriva fuori finestra non deve bloccare
        la seduta per sempre.
        """
        kin = self.kin
        if not self.tracker.ready:
            # Modello in ricostruzione (dopo un movimento del lettino).
            return q_prev, True, "modello in ricostruzione"
        if self.gating_mm is not None:
            if np.linalg.norm(self.estimate) > self.gating_mm:
                self._wait += dt
                if self._wait < self.max_wait_s:
                    return q_prev, True, "tumore fuori finestra"
                self.n_forced += 1
            self._wait = 0.0
            return q_prev, False, ""
        src = np.asarray(beam.source, float) + offset
        q = kin.solve_ik_follow(src, q_prev, self.bases[couch_index],
                                obstacles=self.obstacles)
        if q is None:
            return q_prev, True, "posa irraggiungibile"
        speed = np.rad2deg(np.abs(q - q_prev)) / max(dt, 1e-6)
        if np.any(speed > kin.JOINT_SPEED_LIMITS_DEG_S):
            return q_prev, True, "velocita' dei giunti"
        self.max_speed = np.maximum(self.max_speed, speed)
        return q, False, ""

    def record(self, true, beam, hold, reason=""):
        """Un passo di erogazione nelle statistiche della seduta."""
        self.n_ticks += 1
        if hold:
            self.n_hold += 1
            self.hold_reasons[reason] = self.hold_reasons.get(reason, 0) + 1
            return None
        r = np.asarray(true) - np.asarray(beam)
        self.residuals.append(r)
        self.untracked.append(np.asarray(true))
        return r

    def summary(self):
        kv = [np.linalg.norm(e) for _, e in self.kv_log]
        return {
            "attivo": True,
            "inseguimento": self.tracking,
            "predittore": self.predictor,
            "configurazione": self.cfg.as_dict(),
            "errore_residuo": mm.error_stats(self.residuals),
            "fascio_fermo": mm.error_stats(self.untracked),
            "passi_erogazione": int(self.n_ticks),
            "passi_sospesi": int(self.n_hold),
            "sospensione_pct": (100.0 * self.n_hold / self.n_ticks
                                if self.n_ticks else 0.0),
            "gating_mm": self.gating_mm,
            "correzioni_lettino": len(self.couch_log),
            "ricostruzioni_modello": int(getattr(self, "n_rebuild", 0)),
            "correzione_lettino_totale_mm": [float(v) for v in
                                             self.couch_shift],
            "erogazioni_forzate": int(self.n_forced),
            "motivi_sospensione": dict(self.hold_reasons),
            "velocita_max_giunti_deg_s": [float(v) for v in self.max_speed],
            "limiti_velocita_deg_s": [float(v) for v in
                                      self.kin.JOINT_SPEED_LIMITS_DEG_S],
            "radiografie": len(kv),
            "errore_modello_radiografie_mm": [float(v) for v in kv],
        }
