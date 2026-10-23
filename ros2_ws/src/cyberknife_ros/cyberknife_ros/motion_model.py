# ============================================================================
#  motion_model.py - Moto respiratorio del tumore e inseguimento ad anello
#  chiuso (il principio del sistema Synchrony del CyberKnife).
#
#  PERCHE' SERVE
#  -------------
#  Fino alla Fase 3 il tumore era immobile. Un tumore polmonare non lo e': nei
#  lobi inferiori si sposta di 1-2 cm a ogni respiro. Un robot che eroga su un
#  bersaglio fermo ha due sole scelte - allargare il campo a tutta la
#  traiettoria (ITV, molto polmone sano irradiato) o inseguire il tumore. Il
#  CyberKnife esiste per la seconda.
#
#  LA CATENA SIMULATA
#  ------------------
#     tumore (vero, invisibile)      surrogato esterno (LED sul torace)
#            |                                   |  continuo, 26 Hz
#            | immagini kV rade                  v
#            | (una al minuto)            PREDITTORE di latenza
#            v                                   |  s(t + 115 ms)
#     MODELLO DI CORRELAZIONE  <------------------+
#            |  posizione interna stimata
#            v
#     ROBOT: il fascio si sposta di quell'offset
#
#  Il tumore non si vede in continuo: l'unico segnale continuo e' il torace.
#  Il modello di correlazione traduce il torace in posizione interna e viene
#  ritarato ogni volta che arriva una radiografia. Il sistema ha un ritardo
#  (acquisizione + calcolo + movimento del braccio) che va compensato
#  PREDICENDO il respiro, non inseguendolo.
#
#  DATI SINTETICI, DA DICHIARARE
#  -----------------------------
#  Il dataset MSD e' una CT statica: non contiene respiro. Le traiettorie sono
#  generate con il modello di Lujan (Lujan et al., Med Phys 1999),
#      z(t) = z0 - b * cos^(2n)(pi t / tau - phi),
#  reso irregolare ciclo per ciclo, con isteresi fra gli assi e deriva lenta
#  della linea di base. Le ampiezze seguono gli ordini di grandezza di
#  Seppenwoolde et al. (IJROBP 2002): ~12 mm cranio-caudali nei lobi
#  inferiori, pochi mm in quelli superiori, 1-5 mm di isteresi. Sono valori
#  di letteratura, non misure su questi pazienti.
#
#  Il modulo usa SOLO NumPy: e' copiato nel pacchetto ROS 2 (Python 3.10 in
#  WSL) insieme a kinematics.py.
#
#  Convenzione: millimetri, sistema RAS del paziente (+X sinistra->destra del
#  paziente come nella CT, +Y anteriore, +Z craniale). Gli spostamenti sono
#  riferiti alla posizione del tumore nella CT di pianificazione.
# ============================================================================
from dataclasses import dataclass, asdict, field

import numpy as np


# ---------------------------------------------------------------------------
#  Configurazione
# ---------------------------------------------------------------------------
@dataclass
class MotionConfig:
    """Parametri del respiro e dell'inseguimento. Unica fonte di verita'."""

    # --- Campionamento -------------------------------------------------------
    # 26 Hz rende la latenza di 115 ms pari a 3 campioni (2.99): il predittore
    # lavora su un orizzonte intero senza interpolare.
    fs_hz: float = 26.0

    # --- Forma del respiro (Lujan) ------------------------------------------
    period_s: float = 4.0
    period_cv: float = 0.12        # variabilita' del periodo, ciclo per ciclo
    amplitude_cv: float = 0.12     # variabilita' dell'ampiezza, ciclo per ciclo
    shape_n: int = 2               # cos^(2n): il tumore sosta in espirazione
    # Spostamento a fine inspirazione rispetto all'espirazione, per asse RAS.
    # In inspirazione il diaframma scende: il tumore va in direzione caudale
    # (-Z) e, meno, anteriore.
    inhale_shift_mm: tuple = (1.0, 2.5, -12.0)
    # Ritardo degli assi X e Y rispetto a Z: e' l'isteresi, che trasforma il
    # percorso da segmento in orbita (inspirazione != espirazione).
    hysteresis_s: tuple = (0.20, 0.30, 0.0)
    # Deriva lenta della posizione di riposo (mm al minuto).
    drift_mm_per_min: tuple = (0.0, 0.3, 0.8)

    # --- Surrogato esterno ---------------------------------------------------
    surrogate_gain_mm: float = 8.0      # escursione del torace
    surrogate_lead_s: float = 0.15      # il torace anticipa il tumore
    surrogate_curvature: float = 0.25   # rapporto torace/tumore non lineare
    surrogate_noise_mm: float = 0.15
    # Il torace deriva in modo DIVERSO dal tumore: e' la ragione per cui il
    # modello di correlazione va ritarato durante la seduta.
    surrogate_drift_mm_per_min: float = -0.4

    # --- Immagini kV e modello di correlazione ------------------------------
    # E' la LEVA vera dell'inseguimento. Misurato su 12 tracce da 5 minuti
    # (nLMS): p95 dell'errore residuo 3.21 mm con una radiografia al minuto,
    # 2.63 mm ogni 30 s; senza deriva scenderebbe a ~2.1. L'errore e' dominato
    # dalla correlazione che invecchia, non dalla predizione della latenza.
    kv_period_s: float = 60.0
    kv_noise_mm: float = 0.5
    n_init_kv: int = 8             # immagini acquisite prima di erogare
    corr_window: int = 15          # punti conservati (FIFO)
    corr_ridge: float = 1e-2
    # Termine lineare nel TEMPO nel modello di correlazione, per stimare la
    # deriva differenziale torace/tumore. RISULTATO NEGATIVO misurato su 12
    # tracce da 5 minuti (nLMS, kV ogni 60 s): l'errore sistematico scende
    # (|media| 1.20 -> 0.87 mm) ma il p95 SALE (3.21 -> 4.49 mm), perche' una
    # pendenza stimata da una manciata di radiografie e' rumorosa. Spento.
    corr_time_term: bool = False
    # Emivita del peso dei punti nella finestra (s). 0 = pesi uniformi.
    corr_half_life_s: float = 0.0
    kv_error_limit_mm: float = 5.0  # oltre: il modello e' da ricostruire

    # --- Latenza e predizione ------------------------------------------------
    latency_s: float = 0.115
    # RMSE della predizione a 115 ms, mediana su 8 tracce da 5 minuti:
    #   none 0.611 | linear 0.331 | nlms 0.270 | kalman 0.266 mm
    # (la GRU di motion_predictor.py e' confrontata la').
    predictor: str = "kalman"

    seed: int = 0

    def replace(self, **kw):
        """Copia con alcuni campi sostituiti."""
        return MotionConfig(**{**asdict(self), **kw})

    def as_dict(self):
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, tuple):
                d[k] = list(v)
        return d

    @staticmethod
    def from_dict(d):
        if d is None:
            return MotionConfig()
        known = MotionConfig.__dataclass_fields__
        kw = {}
        for k, v in d.items():
            if k in known:
                kw[k] = tuple(v) if isinstance(v, list) else v
        return MotionConfig(**kw)

    @property
    def horizon_samples(self):
        return max(1, int(round(self.latency_s * self.fs_hz)))


# ---------------------------------------------------------------------------
#  Ampiezza in funzione della posizione della lesione
# ---------------------------------------------------------------------------
#: Escursione cranio-caudale ai due estremi del polmone (mm). Seppenwoolde
#: 2002: i tumori dei lobi inferiori, non fissati a strutture rigide, si
#: muovono di ~12 mm in media, con punte oltre i 20; quelli apicali di 2-5 mm.
SI_AMPLITUDE_APEX_MM = 4.0
SI_AMPLITUDE_BASE_MM = 18.0


def lesion_si_fraction(centroid_mm, lung_mask, affine, max_pts=200000):
    """Posizione cranio-caudale relativa della lesione nel polmone.

    0 = base (diaframma), 1 = apice. Si usa l'estensione Z, in mm, della
    maschera polmonare (aria, vedi organs.py): come BOUNDING BOX e' affidabile
    anche se un tumore solido e' un buco della maschera.
    """
    idx = np.argwhere(lung_mask)
    if len(idx) == 0:
        return 0.5
    if len(idx) > max_pts:
        idx = idx[:: len(idx) // max_pts + 1]
    z = idx @ affine[2, :3] + affine[2, 3]
    lo, hi = np.percentile(z, [0.5, 99.5])
    if hi - lo < 1.0:
        return 0.5
    return float(np.clip((centroid_mm[2] - lo) / (hi - lo), 0.0, 1.0))


def inhale_shift_for_location(si_fraction, cfg=None):
    """Vettore di spostamento a fine inspirazione per una lesione in quel punto.

    Le proporzioni fra gli assi (AP ~ 1/4, LR ~ 1/10 del SI) sono quelle di
    `cfg.inhale_shift_mm`; cambia solo la scala.
    """
    cfg = cfg or MotionConfig()
    si = SI_AMPLITUDE_BASE_MM + (SI_AMPLITUDE_APEX_MM - SI_AMPLITUDE_BASE_MM) \
        * float(np.clip(si_fraction, 0.0, 1.0))
    ref = np.asarray(cfg.inhale_shift_mm, float)
    scale = si / max(abs(ref[2]), 1e-6)
    return tuple(float(v) for v in ref * scale)


# ---------------------------------------------------------------------------
#  Generazione della traccia
# ---------------------------------------------------------------------------
@dataclass
class BreathingTrace:
    t: np.ndarray            # (N,) secondi
    tumor: np.ndarray        # (N,3) spostamento vero del tumore, mm
    surrogate: np.ndarray    # (N,) segnale del torace, mm (con rumore)
    inhaling: np.ndarray     # (N,) bool, vero stato respiratorio del tumore
    cfg: MotionConfig = field(default_factory=MotionConfig)

    @property
    def fs(self):
        return self.cfg.fs_hz

    def __len__(self):
        return len(self.t)


def _ar1(n, cv, rho, rng):
    """Processo AR(1) a media 0 e deviazione standard `cv`."""
    e = rng.standard_normal(n)
    x = np.empty(n)
    x[0] = e[0]
    k = np.sqrt(1.0 - rho * rho)
    for i in range(1, n):
        x[i] = rho * x[i - 1] + k * e[i]
    return cv * x


class _Breathing:
    """Fase e ampiezza del respiro come funzioni continue del tempo."""

    def __init__(self, cfg, duration_s, rng):
        n = int(duration_s / cfg.period_s * 2.0) + 16
        T = cfg.period_s * np.exp(_ar1(n, cfg.period_cv, 0.5, rng))
        self.T = np.maximum(T, 0.4 * cfg.period_s)
        A = 1.0 + _ar1(n, cfg.amplitude_cv, 0.6, rng)
        self.A = np.maximum(A, 0.3)
        # Si parte prima di t = 0 con fase casuale, e con margine per i
        # ritardi negativi (lead del surrogato, isteresi).
        pre = 3.0 * cfg.period_s + rng.uniform(0.0, self.T[0])
        self.starts = np.concatenate([[0.0], np.cumsum(self.T)]) - pre
        self.mid = self.starts[:-1] + 0.5 * self.T
        self.n = cfg.shape_n

    def cycle(self, tt):
        """Numero di cicli continuo: parte intera = ciclo, frazionaria = fase."""
        k = np.clip(np.searchsorted(self.starts, tt, side="right") - 1,
                    0, len(self.T) - 1)
        return k + (tt - self.starts[k]) / self.T[k]

    def amplitude(self, tt):
        return np.interp(tt, self.mid, self.A)

    def g(self, phase):
        """Forma di Lujan: 1 a fine inspirazione (fase 0), 0 in espirazione."""
        return np.cos(np.pi * phase) ** (2 * self.n)

    def value(self, tt):
        ph = np.mod(self.cycle(tt), 1.0)
        return self.amplitude(tt) * self.g(ph), ph


def generate_trace(cfg=None, duration_s=300.0, rng=None):
    """Traccia respiratoria sintetica: tumore vero e surrogato osservato."""
    cfg = cfg or MotionConfig()
    rng = rng if rng is not None else np.random.default_rng(cfg.seed)
    fs = cfg.fs_hz
    t = np.arange(int(round(duration_s * fs))) / fs
    br = _Breathing(cfg, duration_s, rng)

    shift = np.asarray(cfg.inhale_shift_mm, float)
    lag = np.asarray(cfg.hysteresis_s, float)
    drift = np.asarray(cfg.drift_mm_per_min, float) / 60.0

    tumor = np.empty((len(t), 3))
    for a in range(3):
        u, _ = br.value(t - lag[a])
        tumor[:, a] = shift[a] * u
    # Riferimento = posizione MEDIA del respiro regolare: una CT a respiro
    # libero ritrae il tumore mediato nel tempo. La deriva no: e' uno
    # spostamento sistematico reale, e va lasciato.
    tumor -= tumor.mean(axis=0)
    tumor += drift[None, :] * t[:, None]

    # Stato respiratorio vero, sull'asse dominante.
    _, ph = br.value(t - lag[int(np.argmax(np.abs(shift)))])
    inhaling = ph > 0.5          # g cresce nella seconda meta' del ciclo

    u, _ = br.value(t + cfg.surrogate_lead_s)
    k = cfg.surrogate_curvature
    s = cfg.surrogate_gain_mm * (u - k * u * u) / (1.0 - k)
    s = s + cfg.surrogate_drift_mm_per_min / 60.0 * t
    s = s + cfg.surrogate_noise_mm * rng.standard_normal(len(t))

    return BreathingTrace(t=t, tumor=tumor, surrogate=s,
                          inhaling=inhaling, cfg=cfg)


def load_trace_csv(path, cfg=None):
    """Traccia reale da CSV: colonne t, x, y, z[, surrogato].

    Senza surrogato si usa l'asse dominante del tumore piu' rumore: il
    predittore si puo' valutare, il modello di correlazione diventa banale e
    lo si deve dichiarare.
    """
    cfg = cfg or MotionConfig()
    d = np.genfromtxt(path, delimiter=",", names=True)
    cols = d.dtype.names
    t = np.asarray(d[cols[0]], float)
    tumor = np.stack([d[c] for c in cols[1:4]], axis=1).astype(float)
    fs = cfg.fs_hz
    tt = np.arange(t[0], t[-1], 1.0 / fs)
    tumor = np.stack([np.interp(tt, t, tumor[:, a]) for a in range(3)], 1)
    tumor -= tumor.mean(axis=0)
    if len(cols) > 4:
        s = np.interp(tt, t, d[cols[4]])
    else:
        a = int(np.argmax(tumor.std(axis=0)))
        s = -tumor[:, a] + cfg.surrogate_noise_mm * \
            np.random.default_rng(cfg.seed).standard_normal(len(tt))
    ds = np.gradient(s)
    return BreathingTrace(t=tt - tt[0], tumor=tumor, surrogate=s,
                          inhaling=ds > 0, cfg=cfg)


# ---------------------------------------------------------------------------
#  Predittori di latenza (sul surrogato)
#
#  Interfaccia comune, IN STREAMING: step(s_k) riceve un campione e restituisce
#  la stima di s(t_k + orizzonte). Nessun predittore vede il futuro: e' la
#  stessa interfaccia che usa il nodo ROS.
# ---------------------------------------------------------------------------
class NoPredictor:
    """Nessuna compensazione: il robot insegue la posizione di 115 ms fa."""
    name = "none"

    def __init__(self, cfg):
        self.h = cfg.horizon_samples

    def reset(self):
        pass

    def step(self, s):
        return float(s)


class LinearPredictor:
    """Estrapolazione lineare della pendenza recente."""
    name = "linear"

    def __init__(self, cfg, window_s=0.2):
        self.h = cfg.horizon_samples
        self.w = max(3, int(round(window_s * cfg.fs_hz)))
        x = np.arange(self.w) - (self.w - 1)
        self._x = x - x.mean()
        self._den = float(self._x @ self._x)
        self.reset()

    def reset(self):
        self.buf = []

    def step(self, s):
        self.buf.append(float(s))
        if len(self.buf) > self.w:
            self.buf.pop(0)
        if len(self.buf) < self.w:
            return float(s)
        y = np.asarray(self.buf)
        slope = float(self._x @ (y - y.mean())) / self._den
        fit_last = y.mean() + slope * self._x[-1]
        return fit_last + slope * self.h


class KalmanPredictor:
    """Filtro di Kalman con modello di oscillatore armonico piu' deriva.

    Stato [y, dy/dt, b]: il respiro e' un'oscillazione y attorno a una linea
    di base b che deriva lentamente. La pulsazione non e' fissa: si stima in
    linea dal periodo fra i passaggi ascendenti per la media (con isteresi),
    e la matrice di transizione si ricalcola quando cambia.
    """
    name = "kalman"

    def __init__(self, cfg, q_acc=40.0, q_bias=0.02, r=None):
        self.cfg = cfg
        self.dt = 1.0 / cfg.fs_hz
        self.h = cfg.horizon_samples
        self.q_acc = q_acc
        self.q_bias = q_bias
        self.r = (cfg.surrogate_noise_mm ** 2) if r is None else r
        self.r = max(self.r, 1e-4)
        self.reset()

    def reset(self):
        self.x = None
        self.P = np.diag([25.0, 100.0, 25.0])
        self.period = self.cfg.period_s
        self._set_omega(2 * np.pi / self.period)
        self.k = 0
        self.mean = None
        self.var = 1.0
        self.last_up = None
        self.above = False

    def _phi(self, w, dt):
        c, s = np.cos(w * dt), np.sin(w * dt)
        return np.array([[c, s / w, 0.0], [-w * s, c, 0.0], [0.0, 0.0, 1.0]])

    def _set_omega(self, w):
        self.w = w
        self.F = self._phi(w, self.dt)
        self.Fh = np.linalg.matrix_power(self.F, self.h)
        dt = self.dt
        q = self.q_acc
        self.Q = np.array([[q * dt ** 3 / 3, q * dt ** 2 / 2, 0.0],
                           [q * dt ** 2 / 2, q * dt, 0.0],
                           [0.0, 0.0, self.q_bias * dt]])

    def _track_period(self, s):
        # Media e varianza esponenziali su ~2 cicli.
        a = self.dt / (2.0 * self.period)
        if self.mean is None:
            self.mean = s
        self.mean += a * (s - self.mean)
        self.var += a * ((s - self.mean) ** 2 - self.var)
        thr = 0.3 * np.sqrt(max(self.var, 1e-6))
        t = self.k * self.dt
        if not self.above and s > self.mean + thr:
            self.above = True
            if self.last_up is not None:
                T = t - self.last_up
                if 0.4 * self.cfg.period_s < T < 2.5 * self.cfg.period_s:
                    self.period = 0.7 * self.period + 0.3 * T
                    self._set_omega(2 * np.pi / self.period)
            self.last_up = t
        elif self.above and s < self.mean - thr:
            self.above = False

    def step(self, s):
        s = float(s)
        self._track_period(s)
        self.k += 1
        if self.x is None:
            self.x = np.array([0.0, 0.0, s])
            return s
        x = self.F @ self.x
        P = self.F @ self.P @ self.F.T + self.Q
        H = np.array([1.0, 0.0, 1.0])
        S = H @ P @ H + self.r
        K = P @ H / S
        x = x + K * (s - H @ x)
        P = P - np.outer(K, H @ P)
        self.x, self.P = x, P
        xh = self.Fh @ x
        return float(xh[0] + xh[2])


class NlmsPredictor:
    """Filtro adattativo nLMS sulle differenze recenti.

    Regressore: gli ultimi M incrementi del segnale, normalizzati per la sua
    ampiezza corrente; bersaglio: l'incremento fra ora e ora + orizzonte. I
    pesi si aggiornano quando quel futuro diventa passato, cioe' con `h`
    campioni di ritardo. E' il predittore adattativo classico della
    letteratura sull'inseguimento respiratorio (Ernst et al.).
    """
    name = "nlms"

    def __init__(self, cfg, order=16, mu=0.08, eps=1e-3):
        self.cfg = cfg
        self.h = cfg.horizon_samples
        self.M = order
        self.mu = mu
        self.eps = eps
        self.reset()

    def reset(self):
        self.buf = []
        self.pending = []          # (regressore, s_k) in attesa della verita'
        # Parte da un'estrapolazione lineare: senza, i primi cicli
        # predirebbero "fermo".
        self.w = np.full(self.M, self.h / self.M)
        self.scale = None

    def _reg(self):
        y = np.asarray(self.buf[-(self.M + 1):])
        return np.diff(y)[::-1] / self.scale      # incremento piu' recente primo

    def step(self, s):
        s = float(s)
        self.buf.append(s)
        if len(self.buf) > self.M + self.h + 2:
            self.buf.pop(0)
        if len(self.buf) >= 2:
            d = abs(self.buf[-1] - self.buf[-2])
            a = 1.0 / (self.cfg.fs_hz * 2.0 * self.cfg.period_s)
            self.scale = d + 1e-3 if self.scale is None \
                else (1 - a) * self.scale + a * (d + 1e-3)
        if len(self.buf) < self.M + 1:
            return s
        u = self._reg()
        # aggiornamento con il regressore di h campioni fa
        self.pending.append((u, s))
        if len(self.pending) > self.h:
            u0, s0 = self.pending.pop(0)
            target = (s - s0) / self.scale
            e = target - self.w @ u0
            self.w += self.mu * e * u0 / (self.eps + u0 @ u0)
        return s + self.scale * float(self.w @ u)


_PREDICTORS = {
    "none": NoPredictor,
    "linear": LinearPredictor,
    "kalman": KalmanPredictor,
    "nlms": NlmsPredictor,
}


def register_predictor(name, factory):
    """Aggiunge un predittore (lo usa motion_predictor.py per la GRU)."""
    _PREDICTORS[name] = factory


def predictor_names():
    return list(_PREDICTORS)


def make_predictor(name, cfg=None):
    cfg = cfg or MotionConfig()
    if name not in _PREDICTORS:
        if name == "gru":
            raise KeyError("predittore 'gru' non registrato: importare "
                           "motion_predictor (richiede torch)")
        raise KeyError(f"predittore sconosciuto: {name!r}")
    return _PREDICTORS[name](cfg)


def predict_series(predictor, s):
    """Applica un predittore in streaming a tutta la serie.

    Un predittore puo' offrire series() per fare lo stesso calcolo in blocco
    (la GRU): il risultato deve coincidere con quello campione per campione.
    """
    if hasattr(predictor, "series"):
        return predictor.series(np.asarray(s, float))
    predictor.reset()
    return np.array([predictor.step(v) for v in s])


def prediction_error(s_true, s_pred, h, skip=0):
    """Errore della predizione a h campioni: s_pred[k] contro s_true[k+h]."""
    e = s_pred[skip:len(s_true) - h] - s_true[skip + h:]
    return e


# ---------------------------------------------------------------------------
#  Modello di correlazione surrogato -> tumore
# ---------------------------------------------------------------------------
class CorrelationModel:
    """Polinomio di secondo grado per asse, con due rami.

    Inspirazione ed espirazione hanno curve diverse (isteresi): con un unico
    polinomio l'errore resterebbe dell'ordine dell'isteresi stessa. Si tiene
    una finestra FIFO degli ultimi `corr_window` punti; un ramo con pochi punti
    ricade su un modello comune ai due, e con meno di 4 punti il modello e'
    lineare. Il termine quadratico e' regolarizzato: con pochi punti un
    polinomio libero estrapola in modo selvaggio.
    """

    def __init__(self, cfg=None):
        self.cfg = cfg or MotionConfig()
        self.S, self.I, self.Y, self.T = [], [], [], []
        self.coef = {}

    def __len__(self):
        return len(self.S)

    @property
    def ready(self):
        return len(self.S) >= 3

    def add(self, s, inhaling, pos, t=0.0):
        self.T.append(float(t))
        self.S.append(float(s))
        self.I.append(bool(inhaling))
        self.Y.append(np.asarray(pos, float))
        w = self.cfg.corr_window
        if len(self.S) > w:
            self.S, self.I, self.Y = self.S[-w:], self.I[-w:], self.Y[-w:]
            self.T = self.T[-w:]
        self._fit()

    def _design(self, z, tau, degree):
        cols = [np.ones_like(z), z] + ([z * z] if degree == 2 else [])
        if self.use_time:
            cols.append(tau)
        return np.stack(cols, 1)

    def _solve(self, s, tt, Y, degree):
        z = (s - self.mu) / self.sd
        X = self._design(z, (tt - self.t0) / 60.0, degree)
        hl = self.cfg.corr_half_life_s
        # Risultato NEGATIVO anche qui: con emivita 60-240 s il p95 non cambia
        # (3.21 -> 3.27-3.36 mm, kV ogni 60 s). Lasciato a 0.
        if hl <= 0:
            wt = np.ones(len(z))
        else:
            wt = 0.5 ** ((self.t_last - tt) / hl)
        wt = wt / wt.mean()
        R = np.zeros(X.shape[1])
        R[1:] = self.cfg.corr_ridge
        if degree == 2:
            R[2] = max(self.cfg.corr_ridge, 0.05)
        Xw = X * wt[:, None]
        A = Xw.T @ X + np.diag(R) * len(z)
        return np.linalg.solve(A, Xw.T @ Y), degree

    def _fit(self):
        S = np.asarray(self.S)
        I = np.asarray(self.I)
        T = np.asarray(self.T)
        Y = np.stack(self.Y)
        self.mu = S.mean()
        self.sd = max(S.std(), 1e-3)
        self.t0 = T.mean()
        self.t_last = T.max()
        # La deriva si stima solo se i punti coprono un intervallo di tempo
        # sufficiente: le 8 immagini iniziali, raccolte in 20 s, non bastano.
        self.use_time = (bool(self.cfg.corr_time_term)
                         and (T.max() - T.min()) >= 45.0)
        self.coef = {}
        if len(S) < 2:
            return
        deg = 2 if len(S) >= 4 else 1
        self.coef["all"] = self._solve(S, T, Y, deg)
        for key, sel in ((True, I), (False, ~I)):
            if sel.sum() >= 4:
                self.coef[key] = self._solve(S[sel], T[sel], Y[sel], 2)
            elif sel.sum() >= 3:
                self.coef[key] = self._solve(S[sel], T[sel], Y[sel], 1)

    def predict(self, s, inhaling, t=0.0):
        c = self.coef.get(bool(inhaling), self.coef.get("all"))
        if c is None:
            return np.zeros(3)
        W, deg = c
        z = np.array([(float(s) - self.mu) / self.sd])
        tau = np.array([(float(t) - self.t0) / 60.0])
        return (self._design(z, tau, deg) @ W)[0]


# ---------------------------------------------------------------------------
#  Inseguimento in linea
# ---------------------------------------------------------------------------
class OnlineTracker:
    """Il controllo ad anello chiuso, un campione alla volta.

    Lo usano sia la simulazione offline (simulate_tracking) sia il nodo ROS:
    step() riceve il torace e restituisce dove il fascio deve puntare fra
    `latency_s`; kv() riceve una radiografia e ritara la correlazione.
    """

    def __init__(self, cfg=None, predictor=None):
        self.cfg = cfg or MotionConfig()
        if predictor is None or isinstance(predictor, str):
            predictor = make_predictor(predictor or self.cfg.predictor,
                                       self.cfg)
        self.predictor = predictor
        self.model = CorrelationModel(self.cfg)
        n = max(3, int(round(0.15 * self.cfg.fs_hz)))
        self._slope_n = n
        x = np.arange(n) - (n - 1) / 2.0
        self._sx = x / (x @ x)
        self._hist = []
        self.kv_errors = []
        self.last_s = 0.0
        self.last_inhaling = False
        self.t = 0.0

    def reset(self):
        self.predictor.reset()
        self._hist = []
        self.t = 0.0

    def inhaling_now(self):
        if len(self._hist) < self._slope_n:
            return False
        return float(self._sx @ np.asarray(self._hist[-self._slope_n:])) > 0

    def step(self, s, t=None):
        """`t`: istante del campione (s). Senza, si conta a fs_hz."""
        self.t = self.t + 1.0 / self.cfg.fs_hz if t is None else float(t)
        s = float(s)
        self._hist.append(s)
        if len(self._hist) > 4 * self._slope_n:
            self._hist.pop(0)
        s_hat = self.predictor.step(s)
        self.last_s = s
        self.last_inhaling = self.inhaling_now()
        # Il ramo a fra 115 ms: se la predizione sale rispetto a ora, si sta
        # inspirando. Vicino ai picchi la differenza e' nel rumore e vale lo
        # stato attuale.
        d = s_hat - s
        if abs(d) > 2.0 * self.cfg.surrogate_noise_mm:
            inh = d > 0
        else:
            inh = self.last_inhaling
        return self.model.predict(s_hat, inh, self.t + self.cfg.latency_s)

    def kv(self, measured_pos, s=None, inhaling=None):
        """Nuova radiografia. Restituisce l'errore del modello PRIMA della
        ritaratura (e' quello che il sistema reale mostra all'operatore)."""
        s = self.last_s if s is None else s
        inhaling = self.last_inhaling if inhaling is None else inhaling
        err = None
        if self.model.ready:
            err = (self.model.predict(s, inhaling, self.t)
                   - np.asarray(measured_pos))
            self.kv_errors.append(err)
        self.model.add(s, inhaling, measured_pos, self.t)
        return err

    @property
    def ready(self):
        return self.model.ready


def kv_schedule(cfg, duration_s):
    """Istanti delle radiografie: prima la costruzione del modello, poi una
    ogni kv_period_s. Le immagini iniziali sono spaziate di 0.618 periodi,
    cosi' cadono in fasi respiratorie diverse invece che sempre nella stessa.
    """
    init = [1.0 + i * 0.618 * cfg.period_s for i in range(cfg.n_init_kv)]
    t_start = init[-1] + 1.0
    later = list(np.arange(t_start + cfg.kv_period_s, duration_s,
                           cfg.kv_period_s))
    return np.asarray(init), float(t_start), np.asarray(later)


@dataclass
class TrackingResult:
    t: np.ndarray            # (N,) istante di erogazione (t_k + latenza)
    truth: np.ndarray        # (N,3) tumore vero in quell'istante
    estimate: np.ndarray     # (N,3) dove punta il fascio
    active: np.ndarray       # (N,) bool, fascio erogabile (modello pronto)
    kv_t: np.ndarray
    kv_err: np.ndarray       # (m,3) errore del modello alle radiografie
    predictor: str = ""

    @property
    def residual(self):
        """Spostamento del tumore RISPETTO AL FASCIO: e' cio' che vede la dose."""
        return (self.truth - self.estimate)[self.active]

    @property
    def untracked(self):
        """Stesso respiro con fascio fermo sulla posizione di pianificazione."""
        return self.truth[self.active]

    def gated(self, window_mm):
        """Fascio fermo acceso solo con la STIMA del tumore entro la finestra:
        lo spostamento che vede la dose e' quello vero in quegli istanti."""
        on = self.active & (np.linalg.norm(self.estimate, axis=1)
                            <= window_mm)
        return self.truth[on]

    def duty_cycle(self, window_mm):
        """Frazione del tempo con il fascio acceso, a gating."""
        a = self.active
        return float(np.mean(np.linalg.norm(self.estimate[a], axis=1)
                             <= window_mm)) if a.any() else 0.0


def simulate_tracking(trace, predictor=None, cfg=None, rng=None):
    """Simula una seduta di inseguimento su una traccia."""
    cfg = cfg or trace.cfg
    rng = rng if rng is not None else np.random.default_rng(cfg.seed + 1)
    tracker = OnlineTracker(cfg, predictor)
    h = cfg.horizon_samples
    N = len(trace) - h
    init_t, t_start, later_t = kv_schedule(cfg, trace.t[-1])
    fs = cfg.fs_hz
    kv_idx = set(int(round(x * fs)) for x in np.concatenate([init_t, later_t]))

    est = np.zeros((N, 3))
    active = np.zeros(N, bool)
    kv_t, kv_err = [], []
    k_start = int(round(t_start * fs))
    for k in range(N):
        s = trace.surrogate[k]
        p = tracker.step(s, trace.t[k])
        if k in kv_idx:
            meas = trace.tumor[k] + cfg.kv_noise_mm * rng.standard_normal(3)
            e = tracker.kv(meas)
            if e is not None and k >= k_start:
                kv_t.append(trace.t[k])
                kv_err.append(e)
        if k >= k_start and tracker.ready:
            est[k] = p
            active[k] = True
    return TrackingResult(
        t=trace.t[h:h + N], truth=trace.tumor[h:h + N], estimate=est,
        active=active, kv_t=np.asarray(kv_t),
        kv_err=np.asarray(kv_err).reshape(-1, 3),
        predictor=tracker.predictor.name)


# ---------------------------------------------------------------------------
#  Statistiche
# ---------------------------------------------------------------------------
def error_stats(d):
    """Riassunto di una serie di spostamenti (N,3), in mm.

    La norma 3D si riporta con mediana e percentile 95; per asse si danno
    media (componente SISTEMATICA) e deviazione standard (componente
    CASUALE), che sono le due grandezze della ricetta dei margini di van Herk.
    """
    d = np.asarray(d, float).reshape(-1, 3)
    if len(d) == 0:
        return {"n": 0}
    n3 = np.linalg.norm(d, axis=1)
    return {
        "n": int(len(d)),
        "rms_3d": float(np.sqrt(np.mean(n3 ** 2))),
        "median_3d": float(np.median(n3)),
        "p95_3d": float(np.percentile(n3, 95)),
        "max_3d": float(n3.max()),
        "within_2mm": float(np.mean(n3 <= 2.0)),
        "within_5mm": float(np.mean(n3 <= 5.0)),
        "mean": [float(v) for v in d.mean(axis=0)],
        "sd": [float(v) for v in d.std(axis=0)],
        "p95_abs": [float(v) for v in np.percentile(np.abs(d), 95, axis=0)],
    }


def van_herk_margin(systematic_sd, random_sd):
    """Margine per asse, M = 2.5 Sigma + 0.7 sigma (van Herk, IJROBP 2000).

    Garantisce che il 90% dei pazienti riceva almeno il 95% della dose
    prescritta sul CTV. Vale per errori gaussiani e penombra di ~3.2 mm: e'
    una ricetta di popolazione, qui usata come ordine di grandezza.
    """
    return 2.5 * np.asarray(systematic_sd, float) + \
        0.7 * np.asarray(random_sd, float)
