# ============================================================================
#  dosimetry.py - Motore di calcolo della dose (TERMA + Convolution/Superposition)
#
#  FISICA IN BREVE
#  ---------------
#  L'unita' Hounsfield E' gia' una misura del coefficiente di attenuazione:
#        HU = 1000 * (mu - mu_acqua) / mu_acqua
#  Non serve quindi alcun modello di machine learning per ricavare mu: e'
#  un'inversione algebrica. C'e' pero' un passaggio obbligato: la TAC misura mu
#  a ~70 keV (dominato dall'effetto fotoelettrico), mentre il fascio terapeutico
#  e' a 6 MV (dominato dall'effetto Compton, quindi proporzionale alla densita'
#  elettronica). Si passa percio' per la densita' elettronica relativa (RED):
#
#        HU  ->  RED  ->  mu(6 MV) = RED * mu_acqua(6 MV)
#
#  MODELLO DI DOSE (a due stadi, come nei treatment planning system reali)
#  ----------------------------------------------------------------------
#   1. TERMA (Total Energy Released per unit MAss): energia ceduta dai fotoni
#      primari, calcolata per ray-tracing lungo la PROFONDITA' RADIOLOGICA
#      (profondita' equivalente in acqua), non lungo quella geometrica.
#   2. Convoluzione con un kernel di scatter: la dose non si deposita dove il
#      fotone interagisce, ma viene trasportata lateralmente dagli elettroni
#      secondari.
#
#  Ottimizzazione: convoluzione e somma sui fasci sono entrambe LINEARI, quindi
#        D = K (*) (somma_fasci T) = somma_fasci (K (*) T)
#  Si accumula la TERMA di tutti i fasci in un'unica griglia e si esegue UNA
#  SOLA convoluzione FFT finale, invece di una per fascio.
#
#  NB: questo modulo NON importa PyQt. Il calcolo gira anche nei processi
#  worker del planner, e su Windows ogni processo re-importa i moduli: farvi
#  entrare Qt significherebbe caricare l'intero toolkit grafico in sei processi
#  che non disegnano nulla. Il QThread che incapsula il calcolo per la GUI vive
#  in dose_worker.py.
#
#  USO (test da riga di comando, senza GUI):
#      python dosimetry.py --ct <ct.nii.gz> --mask <maschera_seg.nii.gz>
# ============================================================================
import numpy as np
from scipy import ndimage
from scipy.signal import fftconvolve
from scipy.special import erfc
from scipy.optimize import lsq_linear

import config


# ---------------------------------------------------------------------------
#  Costanti fisiche e macchina
# ---------------------------------------------------------------------------
# Coefficiente di attenuazione efficace dell'acqua a 6 MV (cm^-1).
# Ricavato dalla PDD del 6 MV: 67% a 10 cm di profondita', dmax a 1.5 cm
#   -> mu = -ln(0.67) / (10 - 1.5) cm = 0.047 cm^-1
MU_WATER_6MV = 0.047

# Distanza sorgente-asse (mm): valore tipico del CyberKnife.
SAD = 800.0

# Semiapertura del cono posteriore escluso: da li' il lettino di trattamento
# ostruisce il fascio.
COUCH_BLOCK_DEG = 35.0

# Penombra geometrica del collimatore (mm, sigma del profilo di bordo).
PENUMBRA_SIGMA = 1.6


# ===========================================================================
#  1) Fisica dei materiali: HU -> RED
# ===========================================================================
def hu_to_red(hu):
    """Converte unita' Hounsfield in densita' elettronica relativa (RED).

    Calibrazione bilineare (modello di Schneider semplificato), che e' lo
    standard clinico:
        HU < 0  (aria, polmone, grasso) : RED = 1 + HU/1000
        HU >= 0 (tessuti molli, osso)   : RED = 1 + HU/1950

    La pendenza cambia perche' sopra lo zero la densita' cresce piu' lentamente
    dell'attenuazione misurata: l'osso attenua molto per effetto fotoelettrico
    (dipendente da Z), ma la sua densita' elettronica e' solo ~1.5 volte quella
    dell'acqua. E' proprio questa la correzione che rende il dato utilizzabile
    a 6 MV.
    """
    hu = np.asarray(hu, dtype=np.float32)
    red = np.where(hu < 0.0, 1.0 + hu / 1000.0, 1.0 + hu / 1950.0)
    return np.clip(red, 0.0, 2.5).astype(np.float32)


# ===========================================================================
#  2) Griglia di calcolo isotropica
# ===========================================================================
def build_hu_grid(ct_data, affine, spacing_mm=2.5):
    """Ricampiona la CT (in HU) su una griglia ISOTROPICA.

    Perche' isotropica: la CT ha voxel anisotropi (es. 0.949 x 0.949 x 1.245 mm).
    Sia il ray-tracing sia il kernel di convoluzione presuppongono che uno
    spostamento di un voxel valga lo stesso in tutte le direzioni; senza
    ricampionamento il kernel risulterebbe deformato nello spazio fisico.
    Anche la segmentazione morfologica degli organi ha bisogno di voxel
    isotropi, perche' erosioni e dilatazioni agiscono per numero di voxel.

    Restituisce (hu_grid, grid_affine) con grid_affine che mappa gli indici
    della nuova griglia in millimetri.
    """
    affine = np.asarray(affine, dtype=np.float64)
    # Spacing corrente = norma delle colonne della parte lineare dell'affine.
    cur_spacing = np.linalg.norm(affine[:3, :3], axis=0)
    ratio = spacing_mm / cur_spacing              # indice nuovo -> indice vecchio

    new_shape = np.maximum(
        np.floor(np.array(ct_data.shape) / ratio).astype(int), 1)

    # Coordinate (in indici della CT originale) dei centri della nuova griglia.
    grids = np.meshgrid(
        *[np.arange(n, dtype=np.float32) * r for n, r in zip(new_shape, ratio)],
        indexing="ij",
    )
    coords = np.stack([g.ravel() for g in grids])

    hu = ndimage.map_coordinates(ct_data, coords, order=1, mode="constant",
                                 cval=-1000.0).reshape(new_shape)

    # L'affine della nuova griglia: stessa orientazione, scala dei passi.
    scale = np.eye(4)
    scale[:3, :3] = np.diag(ratio)
    grid_affine = affine @ scale
    return hu.astype(np.float32), grid_affine


def build_red_grid(ct_data, affine, spacing_mm=2.5):
    """Griglia isotropica di densita' elettronica relativa (RED)."""
    # Si converte DOPO il ricampionamento: lavorare sulla griglia ridotta evita
    # di allocare un secondo volume a piena risoluzione.
    hu, grid_affine = build_hu_grid(ct_data, affine, spacing_mm)
    return hu_to_red(hu), grid_affine


def resample_to_grid(volume, src_affine, grid_shape, grid_affine, order=0):
    """Porta un volume (es. la maschera del tumore) sulla griglia di calcolo."""
    idx = np.indices(grid_shape, dtype=np.float32).reshape(3, -1)
    world = apply_affine_fast(grid_affine, idx)
    src_idx = apply_affine_fast(np.linalg.inv(src_affine), world)
    out = ndimage.map_coordinates(volume, src_idx, order=order, mode="constant",
                                  cval=0)
    return out.reshape(grid_shape)


def apply_affine_fast(affine, coords):
    """Applica un'affine 4x4 a coordinate (3, N). Equivalente a nibabel
    apply_affine ma senza trasposizioni, per gestire milioni di punti."""
    A = np.asarray(affine, dtype=np.float64)
    return (A[:3, :3] @ coords) + A[:3, 3:4]


# Diametri dei collimatori fissi disponibili sul CyberKnife (mm).
CONE_SIZES = (5.0, 7.5, 10.0, 12.5, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 50.0, 60.0)


def suggest_collimator(mask, affine, isocenter_mm, margin_mm=5.0):
    """Sceglie il cono piu' piccolo che copre il bersaglio piu' un margine.

    Si usa il 95-esimo percentile del raggio (non il massimo) per non far
    dettare la scelta da qualche voxel isolato ai bordi della segmentazione.
    """
    idx = np.array(np.nonzero(mask > 0.5), dtype=np.float64)
    if idx.size == 0:
        return 20.0
    world = apply_affine_fast(np.asarray(affine, dtype=np.float64), idx)
    radii = np.linalg.norm(world - np.asarray(isocenter_mm).reshape(3, 1), axis=0)
    needed = 2.0 * float(np.percentile(radii, 95)) + 2.0 * margin_mm
    for c in CONE_SIZES:
        if c >= needed:
            return c
    return CONE_SIZES[-1]


# ===========================================================================
#  4) TERMA: ray-tracing con profondita' radiologica
# ===========================================================================
def _grid_world_bounds(grid_shape, grid_affine):
    """Bounding box in mm della griglia (gestisce assi ribaltati)."""
    n = np.array(grid_shape) - 1
    corners = np.array([[0, 0, 0], [n[0], 0, 0], [0, n[1], 0], [0, 0, n[2]],
                        [n[0], n[1], 0], [n[0], 0, n[2]], [0, n[1], n[2]], n],
                       dtype=np.float64).T
    world = apply_affine_fast(grid_affine, corners)
    return world.min(axis=1), world.max(axis=1)


def _ray_box_range(origin, direction, lo, hi):
    """Intervallo [t0, t1] in cui il raggio interseca il box (metodo slab)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = 1.0 / direction
        t_lo = (lo - origin) * inv
        t_hi = (hi - origin) * inv
    t_near = np.nanmax(np.minimum(t_lo, t_hi))
    t_far = np.nanmin(np.maximum(t_lo, t_hi))
    return max(t_near, 0.0), t_far


def _fibonacci_disc(n_rays, radius):
    """Campionamento uniforme di un disco (apertura del collimatore)."""
    k = np.arange(n_rays, dtype=np.float64) + 0.5
    r = radius * np.sqrt(k / n_rays)
    ang = np.pi * (1.0 + 5.0 ** 0.5) * k
    return r * np.cos(ang), r * np.sin(ang)


def _splat_trilinear(flat_grid, shape, idx, values):
    """Deposita i valori nella griglia con interpolazione trilineare.

    Si usa np.bincount sugli indici appiattiti: e' molto piu' veloce di
    np.add.at quando i punti sono centinaia di migliaia.
    """
    base = np.floor(idx).astype(np.int64)
    frac = idx - base
    nx, ny, nz = shape
    for dx in (0, 1):
        wx = frac[0] if dx else 1.0 - frac[0]
        ix = base[0] + dx
        for dy in (0, 1):
            wy = frac[1] if dy else 1.0 - frac[1]
            iy = base[1] + dy
            for dz in (0, 1):
                wz = frac[2] if dz else 1.0 - frac[2]
                iz = base[2] + dz
                ok = ((ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny) &
                      (iz >= 0) & (iz < nz))
                if not ok.any():
                    continue
                flat = (ix[ok] * ny + iy[ok]) * nz + iz[ok]
                w = (wx * wy * wz)[ok] * values[ok]
                flat_grid += np.bincount(flat, weights=w, minlength=flat_grid.size)


def rays_for_collimator(collimator_mm, spacing_mm, oversample=2.0):
    """Numero di raggi necessario a campionare l'apertura senza striature.

    La spaziatura fra raggi sul piano dell'isocentro deve restare sotto la
    dimensione del voxel, altrimenti le singole tracce restano visibili come
    strisce di sovra/sottodose. Si impone una spaziatura pari a
    spacing/oversample e si ricava il numero di raggi dall'area dell'apertura.
    """
    r_eff = collimator_mm * 1.45 / 2.0
    target = spacing_mm / oversample
    n = int(np.ceil(np.pi * r_eff ** 2 / (target ** 2)))
    return int(np.clip(n, 256, 6000))


class Beam:
    """Un singolo fascio: nodo sorgente, bersaglio, cono e peso.

    Ogni fascio ha il PROPRIO punto di mira e il PROPRIO diametro di
    collimatore: e' cio' che permette di conformare la dose a una forma
    irregolare invece di produrre una sfera.
    """

    __slots__ = ("source", "direction", "target", "cone_mm", "weight",
                 "path_cm", "bone_cm", "node_id", "target_id")

    def __init__(self, source, target, cone_mm, weight=1.0, path_cm=0.0,
                 node_id=-1, target_id=-1, bone_cm=0.0):
        self.source = np.asarray(source, dtype=np.float64)
        self.target = np.asarray(target, dtype=np.float64)
        d = self.target - self.source
        self.direction = d / max(np.linalg.norm(d), 1e-9)
        self.cone_mm = float(cone_mm)
        self.weight = float(weight)
        self.path_cm = float(path_cm)
        self.bone_cm = float(bone_cm)      # tratto di osso attraversato
        self.node_id = int(node_id)
        self.target_id = int(target_id)


def _trace_beam(red_grid, grid_affine, inv_affine, lo, hi, beam, spacing_mm,
                oversample=2.0):
    """Traccia un fascio e restituisce (indici_griglia, valori_TERMA).

    Per ogni raggio:
      1. si campiona la RED lungo il percorso (interpolazione trilineare);
      2. PROFONDITA' RADIOLOGICA = cumsum(RED * passo). E' la profondita'
         equivalente in acqua: attraversando il polmone (RED ~ 0.17) cresce
         circa 6 volte piu' lentamente di quella geometrica. E' esattamente la
         correzione di disomogeneita' che rende sensato il calcolo nel torace;
      3. fluenza = exp(-mu * d_rad): sola attenuazione esponenziale;
      4. TERMA = (mu/rho) * fluenza.
    """
    src, d = beam.source, beam.direction
    radius = beam.cone_mm / 2.0
    n_rays = rays_for_collimator(beam.cone_mm, spacing_mm, oversample)

    # Si campiona oltre il bordo geometrico per costruire la penombra.
    du, dv = _fibonacci_disc(n_rays, radius * 1.45)
    # Peso della penombra: transizione morbida al bordo dell'apertura invece
    # di un taglio netto (sorgente non puntiforme -> penombra geometrica).
    ray_weight = 0.5 * erfc((np.hypot(du, dv) - radius) / PENUMBRA_SIGMA)
    tot = ray_weight.sum()
    if tot <= 0:
        return None, None
    ray_weight /= tot

    # Base ortonormale del piano trasversale al fascio.
    tmp = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(tmp, d)) > 0.9:
        tmp = np.array([1.0, 0.0, 0.0])
    e1 = np.cross(d, tmp); e1 /= np.linalg.norm(e1)
    e2 = np.cross(d, e1)

    # Fascio DIVERGENTE: ogni raggio parte dalla sorgente e passa per un punto
    # dell'apertura proiettata sul piano del BERSAGLIO di questo fascio.
    plane = beam.target
    targets = (plane[None, :] + du[:, None] * e1[None, :]
               + dv[:, None] * e2[None, :])
    ray_dirs = targets - src[None, :]
    ray_dirs /= np.linalg.norm(ray_dirs, axis=1, keepdims=True)

    t0, t1 = _ray_box_range(src, d, lo, hi)
    if t1 <= t0:
        return None, None
    n_samp = max(int(np.ceil((t1 - t0) / spacing_mm)), 2)
    t = t0 + (np.arange(n_samp) + 0.5) * spacing_mm

    pts = src[None, None, :] + ray_dirs[:, None, :] * t[None, :, None]
    idx = apply_affine_fast(inv_affine, pts.reshape(-1, 3).T)

    red_s = ndimage.map_coordinates(
        red_grid, idx, order=1, mode="constant", cval=0.0
    ).reshape(len(ray_dirs), n_samp)

    step_cm = spacing_mm / 10.0
    d_rad_cm = (np.cumsum(red_s, axis=1) - 0.5 * red_s) * step_cm

    # NB: NON si moltiplica per (SAD/t)^2. I raggi partono divergenti da una
    # sorgente puntiforme, quindi la loro densita' geometrica per unita' di
    # volume cala gia' come 1/t^2: la legge dell'inverso del quadrato e'
    # implicita nel campionamento. Applicarla esplicitamente la conterebbe due
    # volte (1/t^4), gonfiando la dose d'ingresso.
    fluence = np.exp(-MU_WATER_6MV * d_rad_cm)

    # TERMA = (mu/rho) * fluenza, con mu/rho MASSICO (la dose e' energia per
    # unita' di MASSA). A 6 MV domina il Compton, quindi mu/rho e' quasi
    # identico fra polmone, tessuto molle e osso: e' una costante globale che
    # sparisce nella normalizzazione. Moltiplicare qui per la densita' (RED)
    # darebbe energia per unita' di VOLUME, facendo risalire artificialmente la
    # dose entrando nei tessuti densi. La densita' interviene gia', ed e' il
    # posto giusto, nella profondita' radiologica.
    terma = fluence * ray_weight[:, None]
    return idx, terma.ravel()


def accumulate_terma(red_grid, grid_affine, beams, spacing_mm=2.5,
                     oversample=2.0, progress_cb=None):
    """Somma pesata della TERMA di tutti i fasci su un'unica griglia."""
    shape = red_grid.shape
    lo, hi = _grid_world_bounds(shape, grid_affine)
    inv_affine = np.linalg.inv(grid_affine)
    terma_flat = np.zeros(int(np.prod(shape)), dtype=np.float64)

    for b, beam in enumerate(beams):
        if beam.weight <= 0.0:
            continue
        idx, terma = _trace_beam(red_grid, grid_affine, inv_affine, lo, hi,
                                 beam, spacing_mm, oversample)
        if idx is not None:
            _splat_trilinear(terma_flat, shape, idx, terma * beam.weight)
        if progress_cb is not None:
            progress_cb(b + 1, len(beams))

    return terma_flat.reshape(shape).astype(np.float32)


def beam_terma_roi(red_grid, grid_affine, beam, roi_origin, roi_shape,
                   spacing_mm=2.5, oversample=1.0):
    """TERMA di un singolo fascio, depositata solo nella regione d'interesse.

    Serve a costruire la matrice di influenza per l'ottimizzazione: calcolare
    ogni fascio sull'intera griglia richiederebbe centinaia di volumi completi.
    Il tracciamento resta comunque sull'intero percorso (la profondita'
    radiologica va accumulata dalla pelle), ma il deposito e' ristretto alla ROI.
    """
    lo, hi = _grid_world_bounds(red_grid.shape, grid_affine)
    inv_affine = np.linalg.inv(grid_affine)
    out_flat = np.zeros(int(np.prod(roi_shape)), dtype=np.float64)

    idx, terma = _trace_beam(red_grid, grid_affine, inv_affine, lo, hi, beam,
                             spacing_mm, oversample)
    if idx is None:
        return out_flat.reshape(roi_shape)

    local = idx - np.asarray(roi_origin, dtype=np.float64).reshape(3, 1)
    _splat_trilinear(out_flat, roi_shape, local, terma)
    return out_flat.reshape(roi_shape).astype(np.float32)


def beam_path_cm(red_grid, grid_affine, source, target, spacing_mm=2.5):
    """Profondita' radiologica (cm) dalla sorgente al bersaglio.

    E' la misura di "quanto tessuto attraversa il fascio prima di arrivare".
    Minimizzarla significa entrare dal lato del corpo piu' vicino alla lesione,
    riducendo il tessuto sano irradiato lungo il tragitto.
    """
    src = np.asarray(source, dtype=np.float64)
    tgt = np.asarray(target, dtype=np.float64)
    d = tgt - src
    dist = float(np.linalg.norm(d))
    if dist < 1e-6:
        return 0.0
    d /= dist
    n = max(int(np.ceil(dist / spacing_mm)), 2)
    t = (np.arange(n) + 0.5) * (dist / n)
    pts = src[None, :] + d[None, :] * t[:, None]
    idx = apply_affine_fast(np.linalg.inv(grid_affine), pts.T)
    red = ndimage.map_coordinates(red_grid, idx, order=1, mode="constant",
                                  cval=0.0)
    return float(red.sum() * (dist / n) / 10.0)


# ===========================================================================
#  4-bis) Pianificazione conformale
# ===========================================================================
def snap_cone(diameter_mm):
    """Arrotonda al collimatore fisico disponibile immediatamente superiore."""
    for c in CONE_SIZES:
        if c >= diameter_mm:
            return c
    return CONE_SIZES[-1]


def beam_node_directions(n_nodes=60, couch_block_deg=COUCH_BLOCK_DEG):
    """Versori dei nodi del robot (spirale di Fibonacci, cono lettino escluso)."""
    n_gen = int(n_nodes * 3 + 32)
    i = np.arange(n_gen, dtype=np.float64) + 0.5
    phi = np.arccos(1.0 - 2.0 * i / n_gen)
    theta = np.pi * (1.0 + 5.0 ** 0.5) * i
    u = np.stack([np.sin(phi) * np.cos(theta),
                  np.sin(phi) * np.sin(theta),
                  np.cos(phi)], axis=1)
    u = u[u[:, 1] >= np.cos(np.deg2rad(180.0 - couch_block_deg))]
    if len(u) > n_nodes:
        u = u[np.linspace(0, len(u) - 1, n_nodes).astype(int)]
    return u


def conformity_metrics(dose_pct, tumor_grid, prescription_pct, voxel_vol_mm3):
    """Indici di conformita' del piano.

    Paddick Conformity Index:  CI = TV_PIV^2 / (TV * PIV)
      TV     = volume del bersaglio
      PIV    = volume racchiuso dall'isodose di prescrizione
      TV_PIV = loro intersezione
    Vale 1 solo se il volume irradiato COINCIDE col tumore; scende sia se la
    dose deborda (irradia sano) sia se non copre (sottodosa). E' la misura
    diretta di "il volume irradiato coincide con la neoplasia".
    """
    tv = tumor_grid > 0.5
    piv = dose_pct >= prescription_pct
    n_tv = int(tv.sum())
    n_piv = int(piv.sum())
    n_int = int((tv & piv).sum())

    ci = (n_int ** 2) / (n_tv * n_piv) if n_tv and n_piv else 0.0
    return {
        "conformity_index": float(ci),
        "coverage_pct": 100.0 * n_int / n_tv if n_tv else 0.0,
        "spill_cm3": (n_piv - n_int) * voxel_vol_mm3 / 1000.0,
        "piv_cm3": n_piv * voxel_vol_mm3 / 1000.0,
        "tv_cm3": n_tv * voxel_vol_mm3 / 1000.0,
    }


# ===========================================================================
#  5) Kernel di scatter e dose finale
# ===========================================================================
def scatter_kernel(spacing_mm=2.5, extent_mm=30.0,
                   a_short=0.35, a_long=0.05, w_long=0.06):
    """Kernel di deposizione della dose (point spread function).

    Forma a doppio esponenziale, radialmente simmetrica:
        K(r) = (exp(-a_short * r) + w_long * exp(-a_long * r)) / r^2
    La componente a corto raggio rappresenta la deposizione degli elettroni
    secondari (pochi mm), quella a lungo raggio lo scatter dei fotoni diffusi
    (qualche cm). Il kernel e' normalizzato a somma unitaria: la convoluzione
    ridistribuisce l'energia senza crearla ne' distruggerla.
    """
    half = int(np.ceil(extent_mm / spacing_mm))
    ax = (np.arange(-half, half + 1)) * spacing_mm
    X, Y, Z = np.meshgrid(ax, ax, ax, indexing="ij")
    r = np.sqrt(X ** 2 + Y ** 2 + Z ** 2)
    r = np.maximum(r, spacing_mm * 0.5)         # evita la singolarita' in r=0
    k = (np.exp(-a_short * r) + w_long * np.exp(-a_long * r)) / r ** 2
    return (k / k.sum()).astype(np.float32)


def compute_dose(terma, kernel):
    """Convoluzione TERMA (*) kernel: da energia rilasciata a dose depositata."""
    return fftconvolve(terma, kernel, mode="same").astype(np.float32)


# ===========================================================================
#  6) Metriche di piano
# ===========================================================================
def plan_metrics(dose_pct, tumor_mask_grid):
    """Indicatori standard di qualita' del piano di trattamento."""
    tumor = tumor_mask_grid > 0.5
    metrics = {
        "dose_max_pct": float(dose_pct.max()),
        "n_tumor_voxels_grid": int(tumor.sum()),
    }
    if tumor.any():
        dt = dose_pct[tumor]
        # D95: dose ricevuta da almeno il 95% del volume tumorale. E' la
        # metrica di copertura per eccellenza in radioterapia.
        metrics["d95_pct"] = float(np.percentile(dt, 5))
        metrics["d_mean_pct"] = float(dt.mean())
        metrics["d_min_pct"] = float(dt.min())
        outside = dose_pct[~tumor]
        metrics["dose_max_outside_pct"] = float(outside.max()) if outside.size else 0.0
    return metrics


# ===========================================================================
#  7) Mappa di dose interpolabile sulle slice CT
# ===========================================================================
class DoseMap:
    """Incapsula la griglia di dose e la proietta su una slice CT su richiesta.

    Ricampionare la dose a piena risoluzione CT (512 x 512 x 304 float32)
    occuperebbe ~318 MB. Interpolare solo la slice visualizzata costa pochi
    millisecondi e nessuna memoria aggiuntiva.
    """

    def __init__(self, dose_pct, grid_affine, ct_shape, ct_affine):
        self.dose = dose_pct
        self.grid_affine = np.asarray(grid_affine, dtype=np.float64)
        self.ct_shape = tuple(ct_shape)
        self.ct_affine = np.asarray(ct_affine, dtype=np.float64)
        self._inv_grid = np.linalg.inv(self.grid_affine)

    def slice_for(self, axis, index, step=1):
        """Restituisce la mappa di dose sul piano della slice CT richiesta.

        axis: 0 = sagittale (i fisso), 1 = coronale (j fisso), 2 = assiale (k fisso).
        step > 1 campiona un pixel ogni 'step' lungo entrambi gli assi in piano:
        per estrarre CONTORNI la risoluzione piena non serve, e il costo scende
        con il quadrato del passo.
        """
        nx, ny, nz = self.ct_shape
        if axis == 0:
            a, b = np.meshgrid(np.arange(0, ny, step), np.arange(0, nz, step),
                               indexing="ij")
            ct_idx = np.stack([np.full(a.shape, index), a, b]).astype(np.float64)
        elif axis == 1:
            a, b = np.meshgrid(np.arange(0, nx, step), np.arange(0, nz, step),
                               indexing="ij")
            ct_idx = np.stack([a, np.full(a.shape, index), b]).astype(np.float64)
        else:
            a, b = np.meshgrid(np.arange(0, nx, step), np.arange(0, ny, step),
                               indexing="ij")
            ct_idx = np.stack([a, b, np.full(a.shape, index)]).astype(np.float64)

        out_shape = ct_idx.shape[1:]
        world = apply_affine_fast(self.ct_affine, ct_idx.reshape(3, -1))
        gidx = apply_affine_fast(self._inv_grid, world)
        vals = ndimage.map_coordinates(self.dose, gidx, order=1,
                                       mode="constant", cval=0.0)
        return vals.reshape(out_shape)


# ===========================================================================
#  8) Pipeline completa
# ===========================================================================
def compute_plan(ct_data, mask, affine, isocenter_mm, n_beams=40,
                 collimator_mm=20.0, mode="conformal", cfg=None,
                 prescription_pct=None, bone_mask=None, bone_affine=None,
                 arms_mask=None, body_mask=None, limits=None,
                 robot_base=None, robot_obstacles=None, robot_couch_poses=None,
                 beam_filter=None, progress_cb=None, status_cb=None):
    """Calcolo dosimetrico completo.

    mode="conformal" : multi-bersaglio, coni variabili, pesi ottimizzati.
    mode="single"    : isocentro unico e cono fisso (piano di riferimento, utile
                       per mostrare quanto la dose sferica sia meno conforme).

    `beam_filter(fasci) -> indici da tenere` si applica ai fasci scelti PRIMA
    della dose finale: e' il collaudo con la sequenza del robot
    (kinematics.deliverable_subset). Dose, normalizzazione e metriche escono
    cosi' dai soli fasci che il robot esegue, e il piano dichiarato e' quello
    erogato.

    I parametri numerici del planner vivono in config.PlannerConfig e arrivano
    qui tramite `cfg`; `prescription_pct` puo' essere sovrascritto perche' e'
    una scelta clinica per singolo piano, non un parametro di algoritmo.
    Restano espliciti `n_beams` e `collimator_mm`, che riguardano solo il piano
    di riferimento a isocentro singolo.
    """
    def _status(msg):
        if status_cb:
            status_cb(msg)

    cfg = cfg or config.PLANNER
    spacing_mm = cfg.spacing_mm
    if prescription_pct is None:
        prescription_pct = cfg.prescription_pct

    _status("Costruzione griglia di densita' elettronica...")
    red_grid, grid_affine = build_red_grid(ct_data, affine, spacing_mm)
    kernel = scatter_kernel(spacing_mm)
    voxel_vol_mm3 = abs(np.linalg.det(grid_affine[:3, :3]))

    tumor_grid = resample_to_grid(mask.astype(np.float32), affine,
                                  red_grid.shape, grid_affine, order=0)

    # Maschera ossea portata sulla griglia di calcolo: serve a misurare quanto
    # osso attraversa ogni fascio candidato.
    bone_grid = None
    if bone_mask is not None and bone_affine is not None:
        bone_grid = resample_to_grid(
            np.asarray(bone_mask, dtype=np.float32), bone_affine,
            red_grid.shape, grid_affine, order=0)

    opt_info = {}
    if mode == "single":
        # --- Piano di riferimento: un solo punto di mira, cono unico ---
        _status(f"Generazione di {n_beams} fasci non coplanari...")
        u = beam_node_directions(n_beams)
        iso = np.asarray(isocenter_mm, dtype=np.float64)
        beams = [Beam(iso + SAD * ui, iso, collimator_mm,
                      path_cm=beam_path_cm(red_grid, grid_affine,
                                           iso + SAD * ui, iso, spacing_mm),
                      node_id=i, target_id=0)
                 for i, ui in enumerate(u)]
        targets = iso.reshape(1, 3)
        radii = np.array([collimator_mm / 2.0])
    else:
        # --- Pianificazione inversa strutturata (planner.py, 5 stadi) ---
        from planner import optimize_plan, PlanLimits

        arms_grid = None
        if arms_mask is not None and bone_affine is not None:
            arms_grid = resample_to_grid(
                np.asarray(arms_mask, dtype=np.float32), bone_affine,
                red_grid.shape, grid_affine, order=0)

        # Senza maschera ossea la relativa penalita' non ha significato:
        # si azzera invece di penalizzare in base a un dato assente.
        plan_cfg = cfg if bone_grid is not None else cfg.replace(lam_bone=0.0)

        beams, opt_info = optimize_plan(
            red_grid, grid_affine, mask, affine, tumor_grid, kernel,
            cfg=plan_cfg,
            limits=limits or PlanLimits.from_config(cfg),
            bone_grid=bone_grid, arms_grid=arms_grid,
            body_mask=body_mask, body_affine=bone_affine,
            robot_base=robot_base, obstacles=robot_obstacles,
            couch_poses=robot_couch_poses,
            progress_cb=progress_cb, status_cb=status_cb)

        if not beams:
            raise RuntimeError(
                "Nessun fascio ammissibile: i vincoli geometrici (lettino, "
                "testa/gambe, braccia) hanno escluso ogni nodo. Allentare i "
                "limiti in PlanLimits.")
        if beam_filter is not None:
            # Il filtro del planner guarda i fasci uno per uno; la sequenza
            # del robot anche percorsi e manovre del lettino, e qualcuno puo'
            # restare fuori. Lo si toglie QUI, prima della dose finale: la
            # normalizzazione alla prescrizione riassorbe il peso perso, e il
            # piano non promette una dose che il robot non consegnera'.
            keep = list(beam_filter(beams))
            opt_info["fasci_scartati_dal_collaudo"] = len(beams) - len(keep)
            if len(keep) < len(beams):
                _status(f"{len(beams) - len(keep)} fasci non eseguibili dalla "
                        f"sequenza del robot: dose ricalcolata sui "
                        f"{len(keep)} erogabili.")
                beams = [beams[i] for i in keep]
            if not beams:
                raise RuntimeError(
                    "Nessun fascio eseguibile dalla sequenza del robot.")
            opt_info["fasci_erogabili"] = len(beams)
        targets = np.array([b.target for b in beams])
        radii = np.array([b.cone_mm / 2.0 for b in beams])

    # --- 4) Dose finale sull'intera griglia ---
    _status(f"Calcolo dose finale ({len(beams)} fasci attivi)...")
    terma = accumulate_terma(red_grid, grid_affine, beams,
                             spacing_mm=spacing_mm, oversample=1.6,
                             progress_cb=progress_cb)
    dose = compute_dose(terma, kernel)

    # --- 5) Normalizzazione alla copertura del bersaglio ---
    # Convenzione radiochirurgica: si normalizza in modo che l'isodose di
    # PRESCRIZIONE copra il bersaglio, e si giudica poi il piano dalla
    # conformita' e dai punti caldi. Normalizzare al massimo sarebbe fuorviante:
    # un singolo hotspot farebbe crollare in apparenza tutta la copertura.
    tum_mask = tumor_grid > 0.5
    if tum_mask.any():
        d_ref = float(np.percentile(dose[tum_mask], 5))   # dose che copre il 95%
    else:
        d_ref = float(dose.max())
    if d_ref <= 1e-12:
        d_ref = float(dose.max()) or 1.0
    dose_pct = (dose / d_ref * prescription_pct).astype(np.float32)

    _status("Metriche di piano...")
    metrics = plan_metrics(dose_pct, tumor_grid)
    metrics.update(conformity_metrics(dose_pct, tumor_grid, prescription_pct,
                                      voxel_vol_mm3))
    cones = sorted({b.cone_mm for b in beams})
    paths = np.array([b.path_cm for b in beams])
    weights = np.array([b.weight for b in beams])
    metrics.update({
        "mode": mode,
        "n_beams": len(beams),
        "n_targets": opt_info.get("mire_usate", len(np.unique(
            [b.target_id for b in beams]))),
        "cones_mm": cones,
        "collimator_mm": collimator_mm if mode == "single" else cones,
        "spacing_mm": spacing_mm,
        "prescription_pct": prescription_pct,
        "path_mean_cm": float(paths.mean()) if len(paths) else 0.0,
        "path_max_cm": float(paths.max()) if len(paths) else 0.0,
        # Osso attraversato, mediato sui fasci e PESATO sulla dose erogata:
        # un fascio quasi spento che tagli una costa conta poco.
        "bone_mean_cm": (float(np.average([b.bone_cm for b in beams],
                                          weights=np.maximum(weights, 1e-9)))
                         if len(beams) else 0.0),
        "weight_ratio": float(weights.max() / max(weights.min(), 1e-9))
                        if len(weights) else 1.0,
        # Indice di omogeneita': dose massima rapportata alla prescrizione.
        # In radiochirurgia valori di 1.2-1.6 sono normali e accettati.
        "homogeneity_index": float(dose_pct.max() / prescription_pct),
    })
    metrics.update(opt_info)

    return {
        "dose_pct": dose_pct,
        "grid_affine": grid_affine,
        "grid_shape": red_grid.shape,
        "red_grid": red_grid,
        "tumor_grid": tumor_grid,
        "beams": beams,
        "sources": np.array([b.source for b in beams]),
        "directions": np.array([b.direction for b in beams]),
        "weights": weights,
        "targets": np.asarray(targets),
        "radii": np.asarray(radii),
        "isocenter": np.asarray(isocenter_mm, dtype=np.float64),
        "metrics": metrics,
    }
# ---------------------------------------------------------------------------
#  Test da riga di comando (senza GUI)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    import time
    import nibabel as nib

    ap = argparse.ArgumentParser(description="Calcolo dosimetrico standalone.")
    ap.add_argument("--ct", required=True)
    ap.add_argument("--mask", required=True)
    ap.add_argument("--beams", type=int, default=60,
                    help="numero di nodi del robot (candidati per bersaglio)")
    ap.add_argument("--collimator", type=float, default=20.0,
                    help="cono fisso, usato solo in modalita' single")
    ap.add_argument("--spacing", type=float,
                    default=config.PLANNER.spacing_mm)
    ap.add_argument("--mode", choices=("conformal", "single"),
                    default="conformal")
    ap.add_argument("--prescription", type=float, default=80.0)
    args = ap.parse_args()

    ct_img = nib.load(args.ct)
    ct = np.asanyarray(ct_img.dataobj, dtype=np.float32)
    msk = (nib.load(args.mask).get_fdata() > 0.5).astype(np.uint8)

    # Isola la lesione principale: il centro di massa di tutte le componenti
    # cadrebbe in un punto privo di tumore. Il criterio sta in un posto solo
    # (postproc.primary_mask), perche' era scritto a mano in quattro file e in
    # un quinto - la dosimetria della GUI - mancava del tutto.
    import postproc
    msk, n_les = postproc.primary_mask(msk, ct_img.affine)
    if n_les > 1:
        print(f"{n_les} lesioni sopra la soglia di volume -> trattata la "
              f"principale ({int(msk.sum())} voxel)")

    iso = apply_affine_fast(
        ct_img.affine,
        np.array(ndimage.center_of_mass(msk), dtype=np.float64).reshape(3, 1)
    ).ravel()
    print("Isocentro (mm):", np.round(iso, 2))

    # La configurazione del planner e' quella di progetto; da riga di comando
    # si possono variare solo i parametri esplicitamente esposti.
    cli_cfg = config.PLANNER.replace(spacing_mm=args.spacing)

    t0 = time.time()
    res = compute_plan(ct, msk, ct_img.affine, iso, n_beams=args.beams,
                       collimator_mm=args.collimator, cfg=cli_cfg,
                       mode=args.mode, prescription_pct=args.prescription,
                       status_cb=lambda m: print("  ", m))
    print(f"Tempo totale: {time.time() - t0:.1f}s")
    for k, v in res["metrics"].items():
        print(f"  {k:<24}{v}")
