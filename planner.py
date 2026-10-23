# ============================================================================
#  planner.py - Ottimizzatore di piano a stadi (inverse planning strutturato).
#
#  Sostituisce l'ottimizzazione "in un colpo solo" con una pipeline in cinque
#  stadi, ciascuno con un compito preciso e verificabile:
#
#    STADIO 1  FATTIBILITA'     quali nodi del robot sono realmente utilizzabili
#    STADIO 2  DECOMPOSIZIONE   riempire il bersaglio di sfere -> punti di mira
#    STADIO 3  CANDIDATI        nodo x mira x cono, con i costi di ciascuno
#    STADIO 4  OTTIMIZZAZIONE   ottimo globale NNLS su tutti i candidati, poi
#                               sparsificazione a un numero erogabile di fasci
#    STADIO 5  RIFINITURA       ripesatura iterativa su punti freddi e caldi,
#                               con riparazione delle zone non coperte
#
#  Perche' a stadi e non un unico minimo quadrato: il problema ha vincoli di
#  natura diversa (geometrici, di copertura, di risparmio degli organi) che non
#  si possono esprimere tutti come termini quadratici. Separarli rende ogni
#  scelta ispezionabile - e il piano difendibile.
#
#  Convenzione degli assi (RAS, quella restituita da nibabel):
#      +X destra del paziente, +Y anteriore, +Z craniale.
# ============================================================================
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy import ndimage
from scipy.optimize import lsq_linear
from scipy.sparse.linalg import LinearOperator
from scipy.signal import fftconvolve

import config
import kinematics as kin
from dosimetry import (
    Beam, SAD, CONE_SIZES, snap_cone, beam_path_cm, beam_terma_roi,
    resample_to_grid, apply_affine_fast,
)


# ---------------------------------------------------------------------------
#  Vincoli geometrici: dove il robot NON puo' mettere la sorgente
# ---------------------------------------------------------------------------
class PlanLimits:
    """Zone proibite e limiti di macchina.

    couch_block_deg     semiapertura del cono posteriore occupato dal lettino;
    max_axial_tilt_deg  massima inclinazione del fascio fuori dal piano
                        assiale. E' il vincolo che esclude le direzioni quasi
                        parallele all'asse del corpo: la CT copre solo il
                        torace, ma il paziente continua, e un nodo posto a
                        800 mm lungo +Z o -Z cadrebbe dove ci sono la testa o
                        le gambe;
    couch_clearance_mm  quanto sotto il dorso passa il piano del lettino: la
                        sorgente deve stare al di sopra;
    max_arm_mm          massimo tratto di braccio attraversabile.

    I valori di riferimento stanno in config.PlannerConfig; questa classe resta
    come oggetto-valore per poterli variare in un singolo piano (ablazioni,
    prove) senza toccare la configurazione globale.
    """

    def __init__(self, couch_block_deg, max_axial_tilt_deg,
                 couch_clearance_mm, max_arm_mm):
        self.couch_block_deg = float(couch_block_deg)
        self.max_axial_tilt_deg = float(max_axial_tilt_deg)
        self.couch_clearance_mm = float(couch_clearance_mm)
        self.max_arm_mm = float(max_arm_mm)

    @classmethod
    def from_config(cls, cfg=None):
        cfg = cfg or config.PLANNER
        return cls(cfg.couch_block_deg, cfg.max_axial_tilt_deg,
                   cfg.couch_clearance_mm, cfg.max_arm_mm)


def _fibonacci_sphere(n):
    """Direzioni uniformi sulla sfera (spirale di Fibonacci)."""
    i = np.arange(n, dtype=np.float64) + 0.5
    phi = np.arccos(1.0 - 2.0 * i / n)
    theta = np.pi * (1.0 + 5.0 ** 0.5) * i
    return np.stack([np.sin(phi) * np.cos(theta),
                     np.sin(phi) * np.sin(theta),
                     np.cos(phi)], axis=1)


def feasible_nodes(target_mm, limits, body_mask=None, body_affine=None,
                   arms_grid=None, red_grid=None, grid_affine=None,
                   n_dense=420, n_keep=96, spacing_mm=2.5, sad=SAD):
    """STADIO 1 - Restituisce i versori dei nodi utilizzabili + diagnostica.

    Ogni regola scarta una zona fisicamente inaccessibile. Si parte da un
    insieme denso e si filtra, invece di costruire direttamente l'insieme
    ammesso: cosi' la copertura angolare residua resta uniforme e si puo'
    riferire quanti nodi ha eliminato ciascun vincolo.
    """
    target = np.asarray(target_mm, dtype=np.float64)
    u = _fibonacci_sphere(n_dense)
    n0 = len(u)
    diag = {"nodi_generati": n0}

    # --- Regola 1: lettino di trattamento (cono posteriore, -Y) ---
    cos_lim = np.cos(np.deg2rad(180.0 - limits.couch_block_deg))
    ok = u[:, 1] >= cos_lim
    diag["scartati_lettino"] = int((~ok).sum())

    # --- Regola 2: testa e gambe (direzioni quasi parallele all'asse Z) ---
    tilt_ok = np.abs(u[:, 2]) <= np.sin(np.deg2rad(limits.max_axial_tilt_deg))
    diag["scartati_testa_gambe"] = int((ok & ~tilt_ok).sum())
    ok &= tilt_ok

    # --- Regola 3: la sorgente deve stare sopra il piano del lettino ---
    if body_mask is not None and body_affine is not None:
        idx = np.array(np.nonzero(body_mask), dtype=np.float64)
        world = apply_affine_fast(np.asarray(body_affine, np.float64), idx)
        couch_y = float(world[1].min()) - limits.couch_clearance_mm
        src_y = target[1] + sad * u[:, 1]
        plane_ok = src_y > couch_y
        diag["scartati_piano_lettino"] = int((ok & ~plane_ok).sum())
        ok &= plane_ok
        diag["quota_lettino_mm"] = round(couch_y, 1)

    # --- Regola 4: attraversamento delle braccia ---
    if arms_grid is not None and red_grid is not None and arms_grid.any():
        arm_cm = np.full(n0, np.inf)
        idxs = np.flatnonzero(ok)
        for j in idxs:
            src = target + sad * u[j]
            arm_cm[j] = beam_path_cm(arms_grid, grid_affine, src, target,
                                     spacing_mm)
        arm_ok = arm_cm * 10.0 <= limits.max_arm_mm
        diag["scartati_braccia"] = int((ok & ~arm_ok).sum())
        ok &= arm_ok
    else:
        diag["scartati_braccia"] = 0

    u = u[ok]
    # Sottocampionamento uniforme: si conserva la distribuzione angolare.
    if len(u) > n_keep:
        u = u[np.linspace(0, len(u) - 1, n_keep).astype(int)]
    diag["nodi_ammessi"] = int(len(u))
    return u, diag


# ---------------------------------------------------------------------------
#  STADIO 2 - Decomposizione del bersaglio
# ---------------------------------------------------------------------------
def decompose_target(mask, affine, max_points=16, coverage=0.995,
                     min_radius_mm=1.5):
    """Impacchetta il bersaglio con sfere, garantendo la copertura del volume.

    Rispetto a un impacchettamento a soglia fissa, qui il ciclo prosegue
    finche' la frazione di volume coperta non raggiunge l'obiettivo: e' cio'
    che assicura che nessuna estremita' della lesione resti senza un punto di
    mira, e quindi senza fasci che la colpiscano.
    """
    m = np.asarray(mask) > 0.5
    if not m.any():
        return np.zeros((0, 3)), np.zeros(0), {}

    nz = np.array(np.nonzero(m))
    lo, hi = nz.min(axis=1), nz.max(axis=1) + 1
    sub = m[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]

    affine = np.asarray(affine, dtype=np.float64)
    spacing = np.linalg.norm(affine[:3, :3], axis=0)
    edt = ndimage.distance_transform_edt(np.pad(sub, 1), sampling=spacing)
    edt = edt[1:-1, 1:-1, 1:-1]

    grid = np.indices(sub.shape).astype(np.float32)
    total = float(sub.sum())
    covered = np.zeros_like(sub)
    points, radii = [], []

    while len(points) < max_points:
        remaining = sub & ~covered
        cand = np.where(remaining, edt, 0.0)
        r = float(cand.max())
        if r < min_radius_mm:
            break
        v = np.unravel_index(int(np.argmax(cand)), cand.shape)
        points.append(apply_affine_fast(
            affine, (np.array(v, dtype=np.float64) + lo).reshape(3, 1)).ravel())
        radii.append(r)

        # Si marca coperta la sfera inscritta, con un minimo che garantisce
        # progresso anche sui residui piu' sottili.
        rr = max(r, min_radius_mm)
        d2 = sum(((grid[k] - v[k]) * spacing[k]) ** 2 for k in range(3))
        covered |= (d2 <= rr ** 2)
        if float(covered.sum()) / total >= coverage:
            break

    if not points:
        c = np.array(ndimage.center_of_mass(m), dtype=np.float64)
        points = [apply_affine_fast(affine, c.reshape(3, 1)).ravel()]
        radii = [max(float(edt.max()), min_radius_mm)]

    diag = {"punti_mira": len(points),
            "copertura_sfere_pct": round(100.0 * float(covered.sum()) / total, 1)}
    return np.array(points), np.array(radii), diag


# ---------------------------------------------------------------------------
#  STADIO 3 - Candidati
# ---------------------------------------------------------------------------
def build_candidates(red_grid, grid_affine, nodes_u, targets, radii,
                     bone_grid=None, margin_mm=3.0, cone_choices=2,
                     keep_frac=0.6, spacing_mm=2.5, sad=SAD):
    """Genera i fasci candidati con il costo geometrico di ciascuno.

    Per ogni punto di mira si offrono PIU' diametri di cono (quello aderente
    allo spessore locale e i successivi): la scelta del collimatore diventa
    cosi' parte dell'ottimizzazione invece di essere decisa a priori.
    """
    beams = []
    for ti, (tp, r) in enumerate(zip(targets, radii)):
        base = snap_cone(2.0 * r + 2.0 * margin_mm)
        k0 = list(CONE_SIZES).index(base)
        cones = [CONE_SIZES[k] for k in range(k0, min(k0 + cone_choices,
                                                      len(CONE_SIZES)))]
        srcs = tp[None, :] + sad * nodes_u
        paths = np.array([beam_path_cm(red_grid, grid_affine, s, tp, spacing_mm)
                          for s in srcs])
        bones = (np.array([beam_path_cm(bone_grid, grid_affine, s, tp, spacing_mm)
                           for s in srcs])
                 if bone_grid is not None else np.zeros(len(srcs)))

        # Prefiltro sul percorso: entrare dal lato piu' vicino riduce il
        # tessuto sano attraversato. Non si tiene solo il minimo, perche'
        # servono direzioni diverse per far convergere la dose.
        order = np.argsort(paths)
        n_keep = max(8, int(round(len(order) * keep_frac)))
        for k in order[:n_keep]:
            for cone in cones:
                beams.append(Beam(srcs[k], tp, cone, weight=1.0,
                                  path_cm=float(paths[k]), node_id=int(k),
                                  target_id=ti, bone_cm=float(bones[k])))
    return beams


# ---------------------------------------------------------------------------
#  Matrice di influenza
# ---------------------------------------------------------------------------
def _roi_and_voxels(tumor_grid, red_grid, grid_affine, tumor_mask,
                    tumor_affine, spacing_mm, ring_mm, edge_boost,
                    max_tumor_pts, max_ring_vox, seed=0):
    """Definisce la regione di ottimizzazione e i punti su cui misurarla.

    Il bersaglio viene campionato alla risoluzione della CT, non a quella della
    griglia di dose: una lesione da 2.7 cm3 occupa solo ~170 voxel a 2.5 mm, e
    un obiettivo descritto da 170 numeri non permette di scolpire la dose. I
    punti CT (qualche migliaio) vengono poi valutati per interpolazione
    trilineare sulla griglia di dose. E' legittimo perche' il campo di dose e'
    gia' passato per il kernel di scatter, quindi e' liscio su scala di
    millimetri: interpolarlo non inventa informazione che non ci sia.
    """
    margin = int(np.ceil((ring_mm + 12.0) / spacing_mm))
    nz = np.array(np.nonzero(tumor_grid > 0.5))
    lo = np.maximum(nz.min(axis=1) - margin, 0)
    hi = np.minimum(nz.max(axis=1) + margin + 1, np.array(tumor_grid.shape))
    roi = (slice(lo[0], hi[0]), slice(lo[1], hi[1]), slice(lo[2], hi[2]))
    roi_shape = tuple(hi - lo)

    rng = np.random.default_rng(seed)

    # --- Punti bersaglio alla risoluzione CT, in coordinate della ROI ---
    tm = np.asarray(tumor_mask) > 0.5
    idx_ct = np.array(np.nonzero(tm), dtype=np.float64)
    if idx_ct.shape[1] > max_tumor_pts:
        sel = rng.choice(idx_ct.shape[1], max_tumor_pts, replace=False)
        idx_ct = idx_ct[:, sel]
    world = apply_affine_fast(np.asarray(tumor_affine, np.float64), idx_ct)
    t_coords = (apply_affine_fast(np.linalg.inv(grid_affine), world)
                - np.asarray(lo, dtype=np.float64).reshape(3, 1))

    # --- Corona di tessuto sano: qui la risoluzione della griglia basta,
    #     perche' la corona e' grande e serve solo a misurare lo spill ---
    tum_roi = tumor_grid[roi] > 0.5
    body_roi = red_grid[roi] > 0.05
    ring = (ndimage.binary_dilation(
        tum_roi, iterations=max(int(round(ring_mm / spacing_mm)), 1))
        & ~tum_roi & body_roi)
    ir = np.flatnonzero(ring.ravel())
    if len(ir) > max_ring_vox:
        ir = rng.choice(ir, max_ring_vox, replace=False)

    # --- Peso dei punti: il BORDO conta piu' dell'interno ---
    # I punti freddi si formano sempre alla periferia, e la normalizzazione
    # clinica riscala il piano finche' il bordo e' coperto: un bordo freddo si
    # traduce quindi in un centro incandescente.
    sp_ct = np.linalg.norm(np.asarray(tumor_affine)[:3, :3], axis=0)
    edt = ndimage.distance_transform_edt(tm, sampling=sp_ct)
    depth = ndimage.map_coordinates(edt.astype(np.float32), idx_ct, order=1)
    depth = depth / max(float(depth.max()), 1e-9)
    vox_w = 1.0 + edge_boost * (1.0 - depth)
    return roi, lo, roi_shape, t_coords, ir, vox_w, tum_roi


#: Contesto di sola lettura condiviso dai processi worker. Popolato una volta
#: per processo da _pool_init e mai modificato: e' questa immutabilita' a
#: rendere il calcolo privo di race condition.
_CTX = None


def _pool_init(red_grid, grid_affine, kernel, lo, roi_shape, t_coords, ir,
               spacing_mm, oversample):
    """Inizializzatore dei processi: riceve i dati pesanti UNA volta.

    Passarli invece a ogni task significherebbe serializzare la griglia di
    densita' per ogni fascio. Qui vengono trasferiti una sola volta per
    processo e poi riusati da tutti i task che quel processo esegue.
    """
    global _CTX
    _CTX = (red_grid, grid_affine, kernel, lo, roi_shape, t_coords, ir,
            spacing_mm, oversample)


def _beam_columns(args):
    """Calcola le colonne della matrice per un blocco di fasci.

    Riceve solo i PARAMETRI numerici dei fasci e ricostruisce gli oggetti in
    loco: evita di dipendere dalla serializzazione di una classe con __slots__
    e riduce il traffico fra processi a poche decine di byte per fascio.

    Ogni blocco produce un array PROPRIO e lo restituisce; nessun processo
    scrive in memoria condivisa. L'assemblaggio avviene solo nel processo
    padre, quindi non esiste stato mutabile condiviso da proteggere.
    """
    start, params = args
    (red_grid, grid_affine, kernel, lo, roi_shape, t_coords, ir,
     spacing_mm, oversample) = _CTX

    n = params.shape[0]
    a_t = np.empty((t_coords.shape[1], n), dtype=np.float32)
    a_r = np.empty((len(ir), n), dtype=np.float32)
    for k in range(n):
        p = params[k]
        beam = Beam(p[0:3], p[3:6], float(p[6]))
        terma = beam_terma_roi(red_grid, grid_affine, beam, lo, roi_shape,
                               spacing_mm, oversample=oversample)
        d = fftconvolve(terma, kernel, mode="same")
        a_t[:, k] = ndimage.map_coordinates(d, t_coords, order=1,
                                            mode="constant", cval=0.0)
        a_r[:, k] = d.ravel()[ir]
    return start, a_t, a_r


def _beam_params(beams):
    """Fasci -> array (N, 7): sorgente, bersaglio, diametro del cono.

    Sono gli unici campi che servono al calcolo della dose; peso, percorso e
    osso intervengono dopo, nel funzionale.
    """
    out = np.empty((len(beams), 7), dtype=np.float64)
    for i, b in enumerate(beams):
        out[i, 0:3] = b.source
        out[i, 3:6] = b.target
        out[i, 6] = b.cone_mm
    return out


def influence_matrix(beams, red_grid, grid_affine, kernel, lo, roi_shape,
                     t_coords, ir, spacing_mm, oversample=1.0, n_workers=0,
                     min_parallel=64, progress_cb=None, status_cb=None):
    """Dose di ciascun fascio nei punti campionati (una colonna per fascio).

    E' la fase piu' costosa dell'intero piano: un ray-tracing completo e una
    convoluzione per ciascun candidato. Con qualche migliaio di candidati il
    calcolo seriale supera il minuto, quindi viene distribuito su piu' processi.

    Servono PROCESSI e non thread: il lavoro sta in map_coordinates e
    fftconvolve, che sono CPU-bound e trattengono il GIL per gran parte del
    tempo.
    """
    n_b = len(beams)
    n_pts, n_ring = t_coords.shape[1], len(ir)

    workers = (n_workers if n_workers > 0
               else min(6, max((os.cpu_count() or 2) - 1, 1)))
    workers = min(workers, n_b)
    if workers <= 1 or n_b < min_parallel:
        return _influence_serial(beams, red_grid, grid_affine, kernel, lo,
                                 roi_shape, t_coords, ir, spacing_mm,
                                 oversample, progress_cb)

    A_t = np.empty((n_pts, n_b), dtype=np.float32)
    A_r = np.empty((n_ring, n_b), dtype=np.float32)
    params = _beam_params(beams)

    # Blocchi piu' piccoli del numero di processi: cosi' un blocco lento non
    # lascia gli altri core fermi in attesa alla fine del giro.
    chunk = max(8, int(np.ceil(n_b / (workers * 4))))
    tasks = [(s, params[s:s + chunk]) for s in range(0, n_b, chunk)]
    if status_cb:
        status_cb(f"Stadio 3/5 - matrice di influenza: {n_b} fasci su "
                  f"{workers} processi...")

    done = 0
    with ProcessPoolExecutor(
            max_workers=workers, initializer=_pool_init,
            initargs=(red_grid, grid_affine, kernel, lo, roi_shape, t_coords,
                      ir, spacing_mm, oversample)) as ex:
        for start, a_t, a_r in ex.map(_beam_columns, tasks):
            # Ogni blocco possiede un intervallo di colonne disgiunto, quindi
            # la scrittura non puo' collidere con quella di un altro blocco.
            A_t[:, start:start + a_t.shape[1]] = a_t
            A_r[:, start:start + a_r.shape[1]] = a_r
            done += a_t.shape[1]
            if progress_cb is not None:
                progress_cb(done, n_b)

    scale = np.maximum(A_t.max(axis=0), 1e-12)
    return A_t / scale, A_r / scale, scale


def _influence_serial(beams, red_grid, grid_affine, kernel, lo, roi_shape,
                      t_coords, ir, spacing_mm, oversample, progress_cb):
    """Versione di riferimento a processo singolo.

    Serve sia come percorso per piani piccoli, sia come oracolo nei test: la
    versione parallela deve produrre la stessa matrice bit per bit.
    """
    n_b = len(beams)
    A_t = np.zeros((t_coords.shape[1], n_b), dtype=np.float32)
    A_r = np.zeros((len(ir), n_b), dtype=np.float32)
    for j, beam in enumerate(beams):
        terma = beam_terma_roi(red_grid, grid_affine, beam, lo, roi_shape,
                               spacing_mm, oversample=oversample)
        d = fftconvolve(terma, kernel, mode="same")
        A_t[:, j] = ndimage.map_coordinates(d, t_coords, order=1,
                                            mode="constant", cval=0.0)
        A_r[:, j] = d.ravel()[ir]
        if progress_cb is not None:
            progress_cb(j + 1, n_b)
    scale = np.maximum(A_t.max(axis=0), 1e-12)
    return A_t / scale, A_r / scale, scale


# ---------------------------------------------------------------------------
#  STADIO 4 - Selezione greedy (matching pursuit vincolato)
# ---------------------------------------------------------------------------
def _solve_nnls(A_t, A_r, vox_w, cols, presc, w_ring, diag_pen, max_iter=40,
                ub=None):
    """Minimi quadrati NON NEGATIVI sul sottoinsieme di fasci scelto.

    Il sistema ha tre blocchi impilati: bersaglio (pesato), corona, e penalita'
    per fascio. Il terzo blocco e' DIAGONALE, e materializzarlo costava una
    matrice n x n quasi tutta di zeri - con qualche migliaio di candidati sono
    centinaia di MB e altrettanto lavoro sprecato a ogni prodotto matrice-vettore.

    Si usa quindi un LinearOperator: i tre blocchi vengono applicati come
    operazioni separate, senza mai costruire la matrice completa. La diagonale
    diventa una moltiplicazione elemento per elemento.
    """
    At = np.ascontiguousarray(A_t[:, cols])
    Ar = np.ascontiguousarray(A_r[:, cols])
    rw = np.sqrt(vox_w).astype(np.float32)
    sq_ring = float(np.sqrt(w_ring))
    dp = diag_pen[cols].astype(np.float64)

    n1, n2, nb = At.shape[0], Ar.shape[0], len(cols)
    Atw = At * rw[:, None]          # blocco bersaglio, gia' pesato

    def matvec(x):
        x = np.asarray(x, dtype=np.float32)
        return np.concatenate([Atw @ x, sq_ring * (Ar @ x), dp * x])

    def rmatvec(y):
        y = np.asarray(y, dtype=np.float64)
        return (Atw.T @ y[:n1].astype(np.float32)
                + sq_ring * (Ar.T @ y[n1:n1 + n2].astype(np.float32))
                + dp * y[n1 + n2:])

    op = LinearOperator((n1 + n2 + nb, nb), matvec=matvec, rmatvec=rmatvec,
                        dtype=np.float64)
    b = np.concatenate([np.full(n1, presc, dtype=np.float64) * rw,
                        np.zeros(n2), np.zeros(nb)])
    upper = np.inf if ub is None else np.asarray(ub, dtype=np.float64)
    sol = lsq_linear(op, b, bounds=(0.0, upper), method="trf",
                     lsmr_tol="auto", max_iter=max_iter, verbose=0)
    return np.maximum(sol.x, 0.0)


def cap_weights(A_t, A_r, vox_w, presc, cols, diag_pen, w_ring, w_sel, scale,
                max_frac, rounds=4):
    """Nessun fascio oltre `max_frac` delle unita' monitor totali.

    E' un vincolo di ROBUSTEZZA, non di qualita' statica. Misurato su un piano
    reale: un solo fascio (cono da 30 mm) portava il 17% della dose e i
    quattro piu' pesanti il 36%. Con il tumore che respira, l'errore
    d'inseguimento o un tratto di costa attraversato durante QUEL fascio pesa
    quanto un sesto del trattamento; distribuita su molti fasci la stessa
    dose media gli errori, come fanno i limiti di MU per fascio dei sistemi
    clinici.

    I pesi del solutore sono per colonne normalizzate (w_vero = w / scale):
    il limite va imposto sui pesi veri, e ridurne uno cambia il totale, quindi
    si itera.
    """
    if not max_frac or max_frac <= 0 or len(cols) == 0:
        return w_sel, 0
    sc = scale[cols]
    n_capped = 0
    for _ in range(rounds):
        wt = w_sel / sc
        tot = float(wt.sum())
        if tot <= 0 or wt.max() <= max_frac * tot * 1.001:
            break
        n_capped = int((wt > max_frac * tot).sum())
        ub = max_frac * tot * sc
        w_sel = _solve_nnls(A_t, A_r, vox_w, cols, presc, w_ring, diag_pen,
                            ub=ub)
    return w_sel, n_capped


def screen_candidates(A_t, A_r, vox_w, beams, keep_total, per_target=True):
    """Riduce i candidati prima del solve globale, senza perdere copertura.

    Il costo del solve NNLS cresce piu' che linearmente col numero di variabili,
    e con qualche migliaio di candidati diventa il collo di bottiglia dell'intero
    piano. Molti candidati sono pero' quasi-duplicati: stesso punto di mira,
    nodi vicini, stesso cono.

    Si assegna a ciascuno un punteggio - quanto copre il bersaglio, quanto
    sborda - e si tengono i migliori. Il quorum e' PER PUNTO DI MIRA: prendendo
    i migliori in assoluto si finirebbe con tutti i fasci puntati sulla mira
    piu' facile, e il resto della lesione resterebbe senza candidati.

    La selezione finale resta globale e convessa: qui si restringe solo
    l'insieme su cui viene eseguita.
    """
    n_b = A_t.shape[1]
    if keep_total <= 0 or n_b <= keep_total:
        return np.arange(n_b)

    cover = (vox_w[:, None] * A_t).sum(axis=0) / max(vox_w.sum(), 1e-9)
    spill = A_r.mean(axis=0) / (A_t.mean(axis=0) + 1e-12)

    def _norm(x):
        rng = float(x.max() - x.min())
        return (x - x.min()) / rng if rng > 1e-12 else np.zeros_like(x)

    score = _norm(cover) - 0.6 * _norm(spill)

    if not per_target:
        return np.sort(np.argsort(score)[::-1][:keep_total])

    tids = np.array([b.target_id for b in beams])
    uniq = np.unique(tids)
    quota = max(int(np.ceil(keep_total / len(uniq))), 4)
    keep = []
    for t in uniq:
        idx = np.flatnonzero(tids == t)
        order = idx[np.argsort(score[idx])[::-1]]
        keep.append(order[:quota])
    return np.sort(np.concatenate(keep))


def solve_and_sparsify(A_t, A_r, vox_w, presc, diag_pen, w_ring, budget=40,
                       rounds=3, status_cb=None):
    """STADIO 4 - Ottimo globale rilassato, poi riduzione a piano erogabile.

    Due passi, entrambi giustificabili:

      1. si risolve il problema NON NEGATIVO su TUTTI i candidati. Essendo
         convesso, la soluzione e' l'ottimo globale: nessuna scelta greedy puo'
         fare meglio in termini di distribuzione di dose;
      2. l'ottimo usa centinaia di fasci, non erogabili in seduta. Si tengono
         quindi i piu' pesanti fino al budget e si RISOLVE DI NUOVO su quel
         sottoinsieme, ripetendo qualche volta. E' un hard-thresholding
         iterativo: la seconda soluzione redistribuisce il peso dei fasci
         eliminati su quelli rimasti, invece di troncare e sperare.

    Il vantaggio rispetto a una selezione greedy e' che la qualita' del piano
    e' fissata dal passo 1, e il passo 2 la degrada in modo controllato e
    misurabile.
    """
    if status_cb:
        status_cb(f"Stadio 4/5 - ottimo su {A_t.shape[1]} candidati...")
    all_cols = np.arange(A_t.shape[1])
    w = _solve_nnls(A_t, A_r, vox_w, all_cols, presc, w_ring, diag_pen)

    hist = [{"fasci": int((w > 0).sum()),
             "deficit_pct": _deficit(A_t, all_cols, w, vox_w, presc)}]

    cols = all_cols
    for r in range(rounds):
        active = cols[w > 1e-9]
        if len(active) <= budget:
            cols, w = active, w[w > 1e-9]
            break
        order = np.argsort(w)[::-1][:budget]
        cols = np.sort(cols[order])
        w = _solve_nnls(A_t, A_r, vox_w, cols, presc, w_ring, diag_pen)
        hist.append({"fasci": int((w > 0).sum()),
                     "deficit_pct": _deficit(A_t, cols, w, vox_w, presc)})
        if status_cb:
            status_cb(f"Stadio 4/5 - riduzione {r + 1}: {len(cols)} fasci, "
                      f"deficit {hist[-1]['deficit_pct']:.1f}%")
    return list(np.atleast_1d(cols)), np.atleast_1d(w), hist


def _deficit(A_t, cols, w, vox_w, presc):
    """Radice del deficit quadratico medio (solo sottodosaggio), in %."""
    d = A_t[:, cols] @ w
    return round(float(np.sqrt(np.average(np.maximum(presc - d, 0) ** 2,
                                          weights=vox_w))), 2)


# ---------------------------------------------------------------------------
#  STADIO 5 - Rifinitura: ripesatura iterativa su freddi e caldi
# ---------------------------------------------------------------------------
def refine_weights(A_t, A_r, vox_w, presc, cols, diag_pen, w_ring,
                   passes=4, hot_limit=1.35, cold_gain=1.8, hot_gain=1.2):
    """Corregge punti freddi e caldi ripesando i voxel, non i vincoli.

    I minimi quadrati trattano tutti i voxel allo stesso modo, ma un voxel
    sottodosato e uno sovradosato hanno conseguenze cliniche diverse. Ad ogni
    passata si alza il peso dei voxel ancora freddi (che altrimenti costringono
    a riscalare tutto il piano) e si penalizzano quelli oltre il limite di
    sovradosaggio. E' la versione a minimi quadrati di un obiettivo
    dose-volume, che non sarebbe esprimibile in forma quadratica diretta.
    """
    w_vox = vox_w.copy()
    w_sel = _solve_nnls(A_t, A_r, w_vox, cols, presc, w_ring, diag_pen)
    hist = []
    for _ in range(passes):
        d = A_t[:, cols] @ w_sel
        cold = d < presc
        hot = d > hot_limit * presc
        if not cold.any() and not hot.any():
            break
        w_vox[cold] *= cold_gain
        w_vox[hot] *= hot_gain
        w_vox = np.clip(w_vox, 0.05, 1e4)
        w_sel = _solve_nnls(A_t, A_r, w_vox, cols, presc, w_ring, diag_pen)
        d = A_t[:, cols] @ w_sel
        hist.append({"freddi_pct": round(100.0 * float(cold.mean()), 1),
                     "caldi_pct": round(100.0 * float(hot.mean()), 1),
                     "d_min_rel": round(float(d.min() / presc), 3)})
    return w_sel, w_vox, hist


# ---------------------------------------------------------------------------
#  Orchestrazione
# ---------------------------------------------------------------------------
def optimize_plan(red_grid, grid_affine, tumor_mask, tumor_affine, tumor_grid,
                  kernel, cfg=None, limits=None, bone_grid=None,
                  arms_grid=None, body_mask=None, body_affine=None,
                  robot_base=None, obstacles=None, robot_n_roll=4,
                  couch_poses=None,
                  progress_cb=None, status_cb=None):
    """Esegue i cinque stadi e restituisce (fasci_attivi, diagnostica).

    Tutti i parametri numerici arrivano da `cfg` (config.PlannerConfig): questa
    funzione non ha piu' default propri, cosi' non puo' esistere un secondo
    valore per lo stesso parametro da qualche altra parte.
    """
    def _st(m):
        if status_cb:
            status_cb(m)

    cfg = cfg or config.PLANNER
    limits = limits or PlanLimits.from_config(cfg)
    spacing_mm = cfg.spacing_mm
    diag = {}
    presc = 100.0

    # ---------------- STADIO 2: decomposizione ----------------
    _st("Stadio 2/5 - decomposizione del bersaglio in sfere...")
    targets, radii, d2 = decompose_target(
        tumor_mask, tumor_affine, max_points=cfg.max_targets,
        coverage=cfg.target_coverage, min_radius_mm=cfg.min_sphere_radius_mm)
    diag.update(d2)
    if not len(targets):
        return [], diag

    # ---------------- STADIO 1: fattibilita' dei nodi ----------------
    _st("Stadio 1/5 - verifica dei nodi accessibili...")
    centroid = targets.mean(axis=0)
    nodes_u, d1 = feasible_nodes(centroid, limits, body_mask, body_affine,
                                 arms_grid, red_grid, grid_affine,
                                 n_dense=cfg.n_dense_nodes,
                                 n_keep=cfg.n_keep_nodes,
                                 spacing_mm=spacing_mm)
    diag.update(d1)
    if not len(nodes_u):
        return [], diag

    # ---------------- STADIO 3: candidati ----------------
    _st(f"Stadio 3/5 - generazione candidati ({len(nodes_u)} nodi x "
        f"{len(targets)} mire)...")
    beams = build_candidates(red_grid, grid_affine, nodes_u, targets, radii,
                             bone_grid=bone_grid, spacing_mm=spacing_mm,
                             margin_mm=cfg.cone_margin_mm,
                             cone_choices=cfg.cone_choices,
                             keep_frac=cfg.keep_frac)
    diag["candidati"] = len(beams)

    # ---------------- STADIO 3b: erogabilita' da parte del robot ----------
    # Un fascio ottimo sulla carta ma che il braccio non puo' eseguire e' un
    # fascio perso: se lo si lascia entrare nell'ottimizzazione, i pesi ci
    # finiscono sopra e la dose calcolata NON e' quella che il paziente
    # ricevera'. Filtrarlo qui - prima della matrice di influenza - e' anche
    # il punto piu' economico: si risparmia il calcolo su cio' che si scarta.
    if robot_base is not None:
        # Il lettino e' un asse mobile: un fascio va tenuto se esiste ALMENO UN
        # suo orientamento da cui il braccio lo esegue. Escludere i fasci che
        # richiedono di ruotare il paziente sarebbe una perdita gratuita:
        # misurato, con il lettino fermo restano eseguibili il 25% delle
        # direzioni, con tre angoli il 40%.
        eff = ([c.base_in_patient(robot_base) for c in couch_poses]
               if couch_poses else [robot_base])
        _st(f"Stadio 3/5 - filtro di erogabilita' del robot "
            f"({len(beams)} candidati, {len(eff)} angoli di lettino)...")
        # Non basta che la posa esista: deve essere COLLEGABILE, cioe' il
        # braccio deve poterci arrivare dal parcheggio. E' lo stesso criterio
        # con cui solve_plan assegna i fasci agli angoli di lettino; con la
        # sola IK il planner poteva scegliere pose che la sequenza del nodo
        # ROS non raggiunge.
        parks = ([kin.parking_poses(b_eff, obstacles) for b_eff in eff]
                 if obstacles else None)
        keep_rb, n_coll = [], 0
        for i, b in enumerate(beams):
            hit = False
            coll = False
            for j, b_eff in enumerate(eff):
                q, info = kin.solve_ik(b.source, b.direction, b_eff,
                                       n_roll=robot_n_roll, obstacles=obstacles)
                if q is not None and parks is not None:
                    q, _ = kin.connectable_pose(b.source, b.direction, b_eff,
                                                obstacles, parks[j],
                                                n_roll=robot_n_roll, q=q)
                if q is not None:
                    hit = True
                    break
                coll = coll or (info["scartate_per_collisione"] and
                                info["entro_i_limiti"])
            if hit:
                keep_rb.append(i)
            elif coll:
                n_coll += 1
        diag["scartati_robot"] = len(beams) - len(keep_rb)
        diag["scartati_robot_collisione"] = n_coll
        diag["angoli_lettino_usati"] = len(eff)
        if not keep_rb:
            _st("Nessun fascio candidato e' erogabile dal robot.")
            return [], diag
        beams = [beams[i] for i in keep_rb]
        diag["candidati_erogabili"] = len(beams)

    roi, lo, roi_shape, t_coords, ir, vox_w, tum_roi = _roi_and_voxels(
        tumor_grid, red_grid, grid_affine, tumor_mask, tumor_affine,
        spacing_mm, cfg.ring_mm, cfg.edge_boost,
        cfg.max_tumor_pts, cfg.max_ring_vox)

    _st(f"Stadio 3/5 - matrice di influenza ({len(beams)} fasci)...")
    A_t, A_r, scale = influence_matrix(beams, red_grid, grid_affine, kernel,
                                       lo, roi_shape, t_coords, ir, spacing_mm,
                                       n_workers=cfg.n_workers,
                                       min_parallel=cfg.parallel_min_beams,
                                       progress_cb=progress_cb,
                                       status_cb=status_cb)
    diag["punti_obiettivo"] = int(A_t.shape[0])

    # Penalita' per fascio: ridge (distribuisce il peso), percorso, osso.
    paths = np.array([b.path_cm for b in beams])
    bones = np.array([b.bone_cm for b in beams])

    def _norm(x):
        rng = float(x.max() - x.min())
        return (x - x.min()) / rng if rng > 1e-12 else np.zeros_like(x)

    diag_pen = ((cfg.lam_ridge + cfg.lam_path * _norm(paths)
                 + cfg.lam_bone * np.clip(bones / 3.0, 0, 2))
                * np.sqrt(A_t.shape[0]))

    # ---------------- STADIO 3c: pre-screening dei candidati --------------
    keep = screen_candidates(A_t, A_r, vox_w, beams, cfg.max_candidates)
    if len(keep) < len(beams):
        _st(f"Stadio 3/5 - pre-screening: {len(beams)} -> {len(keep)} candidati")
        beams = [beams[i] for i in keep]
        A_t, A_r = A_t[:, keep], A_r[:, keep]
        scale, diag_pen = scale[keep], diag_pen[keep]
    diag["candidati_dopo_screening"] = len(beams)

    # ---------------- STADIO 4: ottimo globale + sparsificazione ----------
    cols, w_sel, hist = solve_and_sparsify(A_t, A_r, vox_w, presc, diag_pen,
                                           cfg.w_ring, budget=cfg.beam_budget,
                                           rounds=cfg.sparsify_rounds,
                                           status_cb=status_cb)
    diag["selezionati"] = len(cols)
    diag["sparsificazione"] = hist

    # ---------------- STADIO 5: rifinitura + riparazione ----------------
    _st("Stadio 5/5 - rifinitura dei pesi...")
    w_sel, w_vox, rhist = refine_weights(A_t, A_r, vox_w, presc, cols,
                                         diag_pen, cfg.w_ring)
    diag["rifinitura"] = rhist

    # Riparazione: se restano voxel scoperti, si aggiungono fasci mirati
    # proprio su quelli invece di aumentare la dose globale.
    w_ring, min_weight_frac = cfg.w_ring, cfg.min_weight_frac
    for rnd in range(cfg.repair_rounds):
        d = A_t[:, cols] @ w_sel
        cold_frac = float((d < 0.90 * presc).mean())
        if cold_frac <= 0.05:
            break
        _st(f"Stadio 5/5 - riparazione zone fredde ({100 * cold_frac:.0f}%)...")
        cold = np.maximum(0.90 * presc - d, 0.0)
        num = A_t.T @ (vox_w * cold)
        den = np.sqrt((A_t ** 2).T @ vox_w) + 1e-12
        gain = num / den
        gain[cols] = -np.inf
        add = [int(j) for j in np.argsort(gain)[::-1][:8] if np.isfinite(gain[j])]
        if not add:
            break
        cols = cols + add
        w_sel, w_vox, rh = refine_weights(A_t, A_r, w_vox, presc, cols,
                                          diag_pen, w_ring, passes=2)
        diag[f"riparazione_{rnd + 1}"] = {"fasci_aggiunti": len(add),
                                          "storia": rh}

    # ---------------- Limite al peso del singolo fascio -----------------
    max_frac = getattr(cfg, "max_beam_frac", 0.0)
    if max_frac and max_frac > 0:
        _st(f"Stadio 5/5 - nessun fascio oltre il {100 * max_frac:.0f}% "
            f"della dose...")
        w_before = (w_sel / scale[cols])
        w_sel, n_capped = cap_weights(A_t, A_r, w_vox, presc, cols, diag_pen,
                                      cfg.w_ring, w_sel, scale, max_frac)
        w_after = w_sel / scale[cols]
        diag["limite_peso_fascio"] = {
            "frazione_max": max_frac,
            "prima": round(float(w_before.max() / max(w_before.sum(), 1e-12)),
                           3),
            "dopo": round(float(w_after.max() / max(w_after.sum(), 1e-12)), 3),
            "fasci_limitati": n_capped}

    # ---------------- Fasci attivi ----------------
    w_true = w_sel / scale[cols]
    wmax = float(w_true.max()) if len(w_true) else 0.0
    active = []
    for j, wj in zip(cols, w_true):
        if wmax > 0 and wj > min_weight_frac * wmax:
            beams[j].weight = float(wj)
            active.append(beams[j])
    if not active and len(cols):
        beams[cols[0]].weight = 1.0
        active = [beams[cols[0]]]

    diag["attivi"] = len(active)
    diag["coni_usati"] = sorted({b.cone_mm for b in active})
    diag["mire_usate"] = len({b.target_id for b in active})
    diag["nodi_usati"] = len({b.node_id for b in active})
    return active, diag
