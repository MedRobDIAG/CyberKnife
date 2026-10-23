# ============================================================================
#  dose_fill.py - Il tumore che si "riempie" di dose, fascio dopo fascio.
#
#  Serve alla vista del tumore nella simulazione ROS: ogni punto del bersaglio
#  parte rosso e diventa verde quando ha ricevuto la dose prescritta. I fasci
#  lasciano dei TUNNEL - le zone che attraversano - e dove i tunnel si
#  sovrappongono la dose si somma, finche' il bersaglio e' coperto.
#
#  PERCHE' NON BASTA "ATTRAVERSATO SI' / NO"
#  ----------------------------------------
#  Con un colore binario il tumore diventerebbe verde dopo pochi fasci: ogni
#  punto viene attraversato presto da qualcuno. Ma un singolo fascio da' solo
#  una piccola parte della dose; e' la SOVRAPPOSIZIONE di decine di fasci a
#  raggiungere la prescrizione, ed e' questo che la vista deve mostrare.
#
#  DA DOVE VIENE LA DOSE
#  ---------------------
#  Lo stato FINALE non e' stimato qui: e' la dose calcolata dalla Fase 2
#  (TERMA + convoluzione), salvata nel piano punto per punto. Questo modulo
#  decide solo come RIPARTIRLA fra i fasci, con un modello geometrico:
#    - profilo del fascio: pieno dentro il cono, penombra ai bordi;
#    - divergenza: il cono e' definito alla distanza del bersaglio;
#    - inverso del quadrato della distanza dalla sorgente;
#    - una piccola coda di diffusione fuori dal cono.
#  Cosi' a fine trattamento i colori coincidono ESATTAMENTE con la dosimetria:
#  se il piano lascia un punto freddo, resta giallo e non viene truccato di
#  verde. La geometria decide solo l'ordine in cui il tumore si colora.
#
#  Solo NumPy: gira identico nel nodo ROS (Python 3.10) e nei test su Windows.
# ============================================================================
import numpy as np

#: Larghezza della penombra (mm, dal 90% al 10% del profilo).
PENUMBRA_MM = 1.5
#: Coda di diffusione fuori dal cono: ampiezza relativa e lunghezza (mm).
#: Oltre al realismo garantisce che ogni punto abbia una quota da ripartire:
#: un punto che nessun cono geometrico tocca riceve comunque dose di scatter
#: nella dosimetria vera.
SCATTER_FRAC = 0.02
SCATTER_MM = 6.0

#: Scala dei colori: frazione della dose prescritta -> RGB.
COLOR_STOPS = (
    (0.00, (0.80, 0.10, 0.12)),     # nessuna dose: rosso
    (0.45, (0.95, 0.42, 0.10)),     # arancio
    (0.80, (0.98, 0.84, 0.20)),     # giallo
    (1.00, (0.20, 0.85, 0.35)),     # dose prescritta: verde
)


def beam_shares(points_mm, sources_mm, targets_mm, cones_mm, weights):
    """Contributo geometrico relativo di ogni fascio a ogni punto, (n, m)."""
    P = np.asarray(points_mm, dtype=np.float64)
    S = np.asarray(sources_mm, dtype=np.float64)
    T = np.asarray(targets_mm, dtype=np.float64)
    axis = T - S
    L = np.linalg.norm(axis, axis=1)
    d = axis / np.maximum(L, 1e-9)[:, None]

    V = P[:, None, :] - S[None, :, :]
    z = np.einsum("nmk,mk->nm", V, d)                 # profondita' sull'asse
    radial = V - z[..., None] * d[None, :, :]
    r = np.linalg.norm(radial, axis=2)                # distanza dall'asse
    z_safe = np.maximum(z, 1.0)

    # Il cono e' definito alla distanza del bersaglio e diverge con z.
    R = 0.5 * np.asarray(cones_mm, dtype=np.float64)[None, :] * z_safe / L[None, :]
    # Logistica con 10-90% su PENUMBRA_MM: 2*ln(9)*s = PENUMBRA_MM.
    s = PENUMBRA_MM / (2.0 * np.log(9.0))
    profile = 1.0 / (1.0 + np.exp(np.clip((r - R) / s, -50.0, 50.0)))
    tail = SCATTER_FRAC * np.exp(-np.maximum(r - R, 0.0) / SCATTER_MM)
    inv_sq = (L[None, :] / z_safe) ** 2
    w = np.asarray(weights, dtype=np.float64)[None, :]
    return w * (profile + tail) * inv_sq * (z > 0.0)


class DoseFill:
    """Dose finale di ogni punto del tumore, ripartita fra i fasci.

    `contrib[i, b]` e' la dose che il fascio b porta al punto i, in percentuale
    della prescrizione come la dosimetria. Per costruzione la somma su tutti i
    fasci restituisce esattamente la dose finale del piano.
    """

    def __init__(self, points_mm, dose_final_pct, prescription_pct,
                 sources_mm, targets_mm, cones_mm, weights, step_mm=1.5):
        self.points = np.asarray(points_mm, dtype=np.float64)
        self.final = np.asarray(dose_final_pct, dtype=np.float64)
        self.rx = float(prescription_pct)
        self.step_mm = float(step_mm)
        A = beam_shares(self.points, sources_mm, targets_mm, cones_mm, weights)
        tot = A.sum(axis=1, keepdims=True)
        m = A.shape[1]
        share = np.where(tot > 0.0, A / np.maximum(tot, 1e-300), 1.0 / max(m, 1))
        self.contrib = self.final[:, None] * share

    @property
    def n_points(self):
        return len(self.points)

    @classmethod
    def from_plan(cls, data, beams):
        """Dal piano esportato. None se il piano non porta la dose nel tumore."""
        dt = data.get("dose_tumore")
        if not dt or not beams:
            return None
        fill = cls(dt["punti_mm"], dt["dose_pct"],
                   data.get("prescrizione_pct", 80.0),
                   [b.source for b in beams], [b.target for b in beams],
                   [b.cone_mm for b in beams], [b.weight for b in beams],
                   step_mm=dt.get("passo_mm", 1.5))
        # Tabelle di attenuazione per fascio (motion_dose): facoltative.
        fill.path_tables = data.get("attenuazione_moto")
        return fill

    def dose(self, delivered, active=None, u=0.0):
        """Dose accumulata dopo i fasci `delivered` (indici di fascio del
        piano), piu' la frazione `u` del fascio `active` in corso."""
        idx = list(delivered)
        d = (self.contrib[:, idx].sum(axis=1) if idx
             else np.zeros(self.n_points))
        if active is not None and active >= 0 and u > 0.0:
            d = d + min(float(u), 1.0) * self.contrib[:, active]
        return d

    def coverage(self, dose):
        """Frazione del bersaglio che ha raggiunto la dose prescritta."""
        return float(np.mean(np.asarray(dose) >= self.rx - 1e-9))

    def colors(self, dose):
        """RGBA per punto. Il rosso resta semitrasparente, cosi' i tunnel
        colorati si leggono attraverso la parte non ancora irradiata."""
        f = np.clip(np.asarray(dose, dtype=np.float64) / self.rx, 0.0, 1.0)
        xs = [p for p, _ in COLOR_STOPS]
        rgb = np.stack([np.interp(f, xs, [c[k] for _, c in COLOR_STOPS])
                        for k in range(3)], axis=1)
        alpha = 0.30 + 0.62 * f
        return np.column_stack([rgb, alpha])


#: Campioni del residuo d'inseguimento conservati per ogni fascio: bastano a
#: riprodurre il fattore di dose nel replay, e il resoconto resta piccolo.
RESIDUAL_SAMPLES = 8


def pick_residuals(residuals, k=RESIDUAL_SAMPLES):
    """k campioni equispaziati dei residui registrati durante un'erogazione."""
    r = np.asarray(residuals, dtype=np.float64).reshape(-1, 3)
    if len(r) == 0:
        return np.zeros((1, 3))
    idx = np.linspace(0, len(r) - 1, min(k, len(r))).round().astype(int)
    return r[idx]


def motion_factor(fill, beam_index, beams_geom, residuals, clip=None):
    """Quanto della dose pianificata del fascio arriva a ciascun punto quando
    il tumore sta spostato di `residuals` (mm) rispetto al fascio.

    Rapporto fra il contributo geometrico nel punto spostato e quello nel
    punto nominale, mediato sui campioni. 1 = inseguimento perfetto; sotto 1
    il punto e' uscito dal cono, sopra 1 e' entrato nel cono di un fascio
    che prima lo sfiorava.

    NIENTE TAGLIO di default. Prima il rapporto era limitato a 1.25, per
    prudenza: con gli errori piccoli dell'inseguimento non cambiava nulla,
    ma a fascio fermo (ITV) un punto spostato di 10 mm riceve la dose
    proprio dai fasci che prima lo sfioravano, e il taglio la cancellava.
    Misurato su lung_090 (piano ITV, spostamento rigido di 10 mm) contro la
    dose vera della griglia: copertura del modello 52.5% col taglio, 97.6%
    senza, vera 100%; errore mediano -15.9 contro -3.0 punti. La seduta ITV
    risultava al 70.8% per questo.

    beams_geom: (sorgenti, bersagli, coni, pesi) del piano.
    """
    S, T, C, W = beams_geom
    b = int(beam_index)
    one = ([S[b]], [T[b]], [C[b]], [1.0])
    base = beam_shares(fill.points, *one)[:, 0]
    acc = np.zeros(fill.n_points)
    R = pick_residuals(residuals)
    for r in R:
        acc += beam_shares(fill.points + r, *one)[:, 0]
    g = acc / len(R) / np.maximum(base, 1e-12)
    return np.maximum(g, 0.0) if clip is None else np.clip(g, 0.0, clip)


def _trilinear(tab, grid, p):
    """Interpolazione trilineare su una griglia regolare, ai bordi bloccata."""
    g = np.asarray(grid, dtype=np.float64)
    step = g[1] - g[0]
    f = np.clip((np.asarray(p, dtype=np.float64) - g[0]) / step, 0.0,
                len(g) - 1 - 1e-9)
    i = f.astype(int)
    t = f - i
    out = 0.0
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                w = ((t[0] if dx else 1 - t[0]) * (t[1] if dy else 1 - t[1])
                     * (t[2] if dz else 1 - t[2]))
                out += w * tab[i[0] + dx, i[1] + dy, i[2] + dz]
    return out


def path_factor(fill, beam_index, offsets):
    """Attenuazione media del fascio (tabelle del piano) sugli spostamenti
    registrati durante la sua erogazione; 1 se il piano non le porta.

    Gli spostamenti si campionano come in pick_residuals, cosi' il nodo e il
    replay del resoconto usano gli stessi campioni.
    """
    t = getattr(fill, "path_tables", None)
    if not t or offsets is None or len(offsets) == 0:
        return 1.0
    grid = t["griglia_mm"]
    n = len(grid)
    tab = np.asarray(t["fasci"][int(beam_index)],
                     dtype=np.float64).reshape(n, n, n)
    O = pick_residuals(offsets)
    return float(np.mean([_trilinear(tab, grid, o) for o in O]))
