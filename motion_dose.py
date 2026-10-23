# ============================================================================
#  motion_dose.py - Che dose riceve DAVVERO un tumore che respira.
#
#  L'APPROSSIMAZIONE, DICHIARATA
#  -----------------------------
#  "Static dose cloud": la distribuzione di dose e' rigida e solidale con i
#  fasci; il tumore vi si muove dentro come corpo rigido. Un punto del tumore
#  che nella CT di pianificazione sta in x, spostato di d rispetto ai fasci,
#  riceve D(x + d). Mediando sul tempo:
#
#      D_vista(x) = sum_k  w_k * D(x + d_k)
#
#  dove {d_k, w_k} e' l'istogramma degli spostamenti RELATIVI fascio-tumore:
#      - fascio fermo (nessun inseguimento):  d = moto del tumore
#      - inseguimento:                         d = tumore - stima del robot
#  Il piano e' fatto di ~100 fasci erogati ciascuno per molti respiri, quindi
#  l'effetto di interplay (un fascio che vede una sola fase) e' trascurato.
#  Per il polmone sano la dose si valuta sulla distribuzione statica: il
#  polmone si muove con il tumore e un organo di litri e' poco sensibile a
#  spostamenti di pochi mm. Entrambe le scelte sono standard nella
#  letteratura sul moto (Bortfeld et al., 2004) e vanno dichiarate.
#
#  PERCHE' NON SI CONVOLVE LA GRIGLIA INTERA
#  -----------------------------------------
#  Serve la dose solo nei punti del bersaglio: si interpola in quei punti
#  spostati, K volte (K = celle occupate dell'istogramma, ~100-300). Qualche
#  milione di interpolazioni: pochi secondi, contro una convoluzione 3D per
#  ogni scenario.
# ============================================================================
import numpy as np
from scipy import ndimage

import motion_model as mm
from dosimetry import apply_affine_fast

#: Dose di prescrizione fisica per le metriche in Gy (SBRT polmonare tipica,
#: 3 x 18 Gy all'isodose di prescrizione). Le dosi del progetto sono relative;
#: questa scala serve SOLO a riportare V20 e dose media polmonare in unita'
#: cliniche, come dose fisica totale (non EQD2).
RX_GY = 54.0


# ---------------------------------------------------------------------------
#  Istogramma degli spostamenti
# ---------------------------------------------------------------------------
def displacement_pdf(disp, bin_mm=1.0):
    """Celle occupate dell'istogramma 3D: (offsets (K,3) mm, pesi (K,))."""
    d = np.asarray(disp, float).reshape(-1, 3)
    if len(d) == 0:
        return np.zeros((1, 3)), np.ones(1)
    q = np.round(d / bin_mm).astype(np.int64)
    keys, inv = np.unique(q, axis=0, return_inverse=True)
    inv = inv.ravel()
    w = np.bincount(inv, minlength=len(keys)).astype(float)
    # Il centro di ogni cella e' la media degli spostamenti che contiene, non
    # il nodo della griglia: cosi' la quantizzazione non sposta la media.
    off = np.stack([np.bincount(inv, weights=d[:, a], minlength=len(keys))
                    for a in range(3)], 1) / w[:, None]
    return off, w / w.sum()


# ---------------------------------------------------------------------------
#  Maschere
# ---------------------------------------------------------------------------
def _spacing(affine):
    return np.linalg.norm(np.asarray(affine)[:3, :3], axis=0)


def _world_to_index_dirs(affine, vec_mm):
    """Uno spostamento in mm (RAS) espresso in voxel della griglia."""
    A = np.asarray(affine, float)[:3, :3]
    return np.linalg.solve(A, np.asarray(vec_mm, float).T).T


def _crop(mask, pad_vox):
    idx = np.argwhere(mask)
    lo = np.maximum(idx.min(0) - pad_vox, 0)
    hi = np.minimum(idx.max(0) + pad_vox + 1, mask.shape)
    return tuple(slice(a, b) for a, b in zip(lo, hi))


def expand_mask(mask, affine, margin_mm):
    """Espansione anisotropa: ellissoide di semiassi margin_mm (X, Y, Z RAS).

    Distanza normalizzata = distanza euclidea con le coordinate divise per il
    margine di ciascun asse; si tiene cio' che sta entro 1. Gli assi della
    griglia sono assunti allineati a quelli RAS (vero per le CT del dataset,
    che differiscono al piu' per un segno).
    """
    mask = np.asarray(mask) > 0
    m = np.abs(np.asarray(margin_mm, float) * np.ones(3))
    if not mask.any() or np.all(m < 1e-6):
        return mask.copy()
    # asse della griglia -> asse RAS dominante
    A = np.abs(np.asarray(affine, float)[:3, :3])
    ras_of_axis = A.argmax(axis=0)
    sp = _spacing(affine)
    m_ax = np.maximum(m[ras_of_axis], 1e-3)
    pad = np.ceil(m_ax / sp).astype(int) + 1
    sl = _crop(mask, pad)
    sub = mask[sl]
    dist = ndimage.distance_transform_edt(~sub, sampling=sp / m_ax)
    out = np.zeros_like(mask)
    out[sl] = dist <= 1.0
    return out


def envelope_mask(mask, affine, disp, bin_mm=0.5):
    """ITV: unione del bersaglio traslato su tutte le posizioni del respiro."""
    mask = np.asarray(mask) > 0
    off, _ = displacement_pdf(disp, bin_mm)
    vox = np.round(_world_to_index_dirs(affine, off)).astype(int)
    vox = np.unique(vox, axis=0)
    pad = np.abs(vox).max(axis=0) + 1
    sl = _crop(mask, pad)
    # Padding esplicito prima dello scorrimento: np.roll fa rientrare dal lato
    # opposto cio' che esce, e il ritaglio puo' essere stato limitato dal
    # bordo del volume.
    pw = [(int(p), int(p)) for p in pad]
    sub = np.pad(mask[sl], pw)
    acc = np.zeros_like(sub)
    for v in vox:
        acc |= np.roll(sub, shift=tuple(int(x) for x in v), axis=(0, 1, 2))
    acc = acc[tuple(slice(p, p + n) for p, n in
                    zip(pad, mask[sl].shape))]
    out = np.zeros_like(mask)
    out[sl] = acc
    return out


# ---------------------------------------------------------------------------
#  Dose vista dal bersaglio in movimento
# ---------------------------------------------------------------------------
def mask_points(mask, affine, max_points=20000, seed=0):
    """Coordinate mondo (N,3) dei voxel della maschera, sottocampionate."""
    idx = np.argwhere(np.asarray(mask) > 0)
    if len(idx) > max_points:
        rng = np.random.default_rng(seed)
        idx = idx[rng.choice(len(idx), max_points, replace=False)]
    return apply_affine_fast(np.asarray(affine, float),
                             idx.T.astype(np.float64)).T


def dose_at(dose, grid_affine, pts):
    inv = np.linalg.inv(np.asarray(grid_affine, float))
    g = apply_affine_fast(inv, np.asarray(pts, float).T)
    return ndimage.map_coordinates(dose, g, order=1, mode="nearest")


def dose_under_motion(dose, grid_affine, pts, disp, bin_mm=1.0):
    """Dose media nel tempo ricevuta dai punti `pts` (mm) che si spostano di
    `disp` (N,3) rispetto alla distribuzione."""
    off, w = displacement_pdf(disp, bin_mm)
    acc = np.zeros(len(pts))
    for o, wk in zip(off, w):
        acc += wk * dose_at(dose, grid_affine, pts + o)
    return acc


def target_metrics(dvals, rx_pct):
    d = np.asarray(dvals, float)
    if d.size == 0:
        return {}
    return {
        "copertura_pct": float(100.0 * np.mean(d >= rx_pct)),
        "d95_pct": float(np.percentile(d, 5)),
        "d_min_pct": float(d.min()),
        "d_media_pct": float(d.mean()),
    }


def lung_metrics(dose, grid_affine, lung_mask, lung_affine, rx_pct,
                 exclude_mask=None, exclude_affine=None, rx_gy=RX_GY):
    """Dose media al polmone e V20 (Gy fisici totali, scala RX_GY)."""
    pts_idx = np.argwhere(np.asarray(lung_mask) > 0)
    if len(pts_idx) == 0:
        return {}
    pts = apply_affine_fast(np.asarray(lung_affine, float),
                            pts_idx.T.astype(np.float64)).T
    d = dose_at(dose, grid_affine, pts) / rx_pct * rx_gy
    return {
        "polmone_dose_media_gy": float(d.mean()),
        "polmone_v20_pct": float(100.0 * np.mean(d >= 20.0)),
        "polmone_v5_pct": float(100.0 * np.mean(d >= 5.0)),
    }


def piv_cm3(dose, grid_affine, rx_pct):
    vv = abs(np.linalg.det(np.asarray(grid_affine)[:3, :3]))
    return float((dose >= rx_pct).sum() * vv / 1000.0)


def evaluate(plan, eval_mask, eval_affine, disp, lungs=None, lung_affine=None,
             bin_mm=1.0):
    """Metriche di un piano con il bersaglio che si muove di `disp`.

    `eval_mask` e' il bersaglio da giudicare (il CTV vero, se disponibile):
    non necessariamente quello su cui il piano e' stato ottimizzato.
    """
    rx = plan["metrics"]["prescription_pct"]
    dose, gaff = plan["dose_pct"], plan["grid_affine"]
    pts = mask_points(eval_mask, eval_affine)
    dv = dose_under_motion(dose, gaff, pts, disp, bin_mm)
    out = target_metrics(dv, rx)
    tv = float(np.count_nonzero(eval_mask)
               * abs(np.linalg.det(np.asarray(eval_affine)[:3, :3])) / 1000.0)
    piv = piv_cm3(dose, gaff, rx)
    cov = out["copertura_pct"] / 100.0
    out.update({
        "tv_cm3": tv,
        "piv_cm3": piv,
        # Paddick con la copertura SOTTO MOTO: TV_PIV = copertura * TV.
        "paddick_ci": float(cov * cov * tv / piv) if piv > 0 else 0.0,
        "n_fasci": int(plan["metrics"].get("n_beams", 0)),
    })
    if lungs is not None:
        out.update(lung_metrics(dose, gaff, lungs, lung_affine, rx))
    return out


# ---------------------------------------------------------------------------
#  Margini di popolazione
# ---------------------------------------------------------------------------
def population_residual(cfg=None, predictor=None, n=20, duration_s=300.0,
                        seed0=1000, gating_mm=None):
    """Errore residuo d'inseguimento su una popolazione di sedute simulate.

    Sigma = deviazione standard, fra le sedute, dell'errore medio (componente
    sistematica); sigma = media quadratica delle deviazioni standard entro la
    seduta (componente casuale). Sono gli ingredienti della ricetta di van
    Herk. I semi partono da `seed0`, lontano da quelli usati per valutare.
    """
    cfg = cfg or mm.MotionConfig()
    means, sds, duty = [], [], []
    for i in range(n):
        c = cfg.replace(seed=seed0 + i)
        tr = mm.generate_trace(c, duration_s)
        r = mm.simulate_tracking(tr, predictor or c.predictor, c)
        # Con il gating il fascio e' fermo: conta lo spostamento VERO negli
        # istanti in cui la stima e' in finestra.
        if gating_mm is None:
            st = mm.error_stats(r.residual)
        else:
            st = mm.error_stats(r.gated(gating_mm))
            duty.append(r.duty_cycle(gating_mm))
        means.append(st["mean"])
        sds.append(st["sd"])
    means, sds = np.asarray(means), np.asarray(sds)
    Sigma = means.std(axis=0)
    sigma = np.sqrt(np.mean(sds ** 2, axis=0))
    return {
        "Sigma_mm": [float(v) for v in Sigma],
        "sigma_mm": [float(v) for v in sigma],
        "margine_mm": [float(v) for v in mm.van_herk_margin(Sigma, sigma)],
        "n_sedute": int(n),
        "gating_mm": gating_mm,
        "fascio_acceso": float(np.median(duty)) if duty else 1.0,
    }


# ---------------------------------------------------------------------------
#  Scenari
# ---------------------------------------------------------------------------
SCENARIOS = ("statico", "ctv_senza_inseguimento", "itv", "inseguimento")

SCENARIO_LABELS = {
    "statico": "Statico (ideale)",
    "ctv_senza_inseguimento": "CTV, nessuna compensazione",
    "itv": "ITV, fascio fermo",
    "inseguimento": "Inseguimento + margine",
    "adattivo": "Inseguimento + margine adattivo",
    "gating": "Gating (fascio fermo in finestra)",
}


def scenario_targets(ctv, affine, planning_disp, margin_mm):
    """Bersagli da pianificare per ciascuno scenario."""
    return {
        "statico": ctv,
        "ctv_senza_inseguimento": ctv,
        "itv": envelope_mask(ctv, affine, planning_disp),
        "inseguimento": expand_mask(ctv, affine, margin_mm),
    }


def compare_scenarios(ct, affine, ctv, trace, tracking, plan_fn,
                      eval_mask=None, lungs=None, lung_affine=None,
                      planning_trace=None, margin_mm=None,
                      scenarios=SCENARIOS, extra_targets=None,
                      status_cb=None):
    """Pianifica e valuta i quattro scenari.

    plan_fn(target_mask) -> risultato di compute_plan. Si passa una funzione
    per lasciare al chiamante organi, sala e configurazione del planner.

    - planning_trace: la "4DCT" da cui si ricava l'ITV. Deve essere una
      traccia DIVERSA da quella di trattamento: l'ITV si disegna prima, e la
      seduta vera respira in modo un po' diverso (deriva, cicli piu' ampi).
    - margin_mm: margine dell'inseguimento (van Herk). Default: da
      population_residual.
    - extra_targets: {nome: maschera} di bersagli in piu', tutti erogati con
      inseguimento (per esempio il PTV adattivo all'incertezza).
    """
    def _st(m):
        if status_cb:
            status_cb(m)

    eval_mask = ctv if eval_mask is None else eval_mask
    planning_trace = planning_trace or trace
    if margin_mm is None:
        margin_mm = population_residual(trace.cfg)["margine_mm"]
    targets = scenario_targets(ctv, affine, planning_trace.tumor, margin_mm)
    disp = {
        "statico": np.zeros((1, 3)),
        "ctv_senza_inseguimento": tracking.untracked,
        "itv": tracking.untracked,
        "inseguimento": tracking.residual,
    }
    scenarios = list(scenarios)
    for name, m in (extra_targets or {}).items():
        # (maschera) oppure (maschera, spostamenti): il gating ha i suoi.
        if isinstance(m, tuple):
            m, d = m
        else:
            d = tracking.residual
        targets[name] = m
        disp[name] = d
        scenarios.append(name)
    plans, results = {}, {}
    cache = {}
    for name in scenarios:
        key = "ctv" if targets[name] is ctv else name
        if key not in cache:
            _st(f"Scenario '{SCENARIO_LABELS.get(name, name)}': "
                f"pianificazione...")
            cache[key] = plan_fn(targets[name])
        plans[name] = cache[key]
        _st(f"Scenario '{SCENARIO_LABELS.get(name, name)}': dose sotto moto...")
        r = evaluate(plans[name], eval_mask, affine, disp[name],
                     lungs, lung_affine)
        vv = abs(np.linalg.det(np.asarray(affine)[:3, :3])) / 1000.0
        r["ptv_cm3"] = float(np.count_nonzero(targets[name]) * vv)
        results[name] = r
    return {"results": results, "plans": plans, "targets": targets,
            "margin_mm": [float(v) for v in np.ravel(margin_mm)],
            "displacements": disp}


def dvh_under_motion(plan, mask, affine, disp, bins=240, max_pct=150.0):
    """DVH cumulativo del bersaglio in movimento: (dose, volume %)."""
    pts = mask_points(mask, affine)
    d = dose_under_motion(plan["dose_pct"], plan["grid_affine"], pts, disp)
    edges = np.linspace(0, max(max_pct, float(d.max())), bins + 1)
    hist, _ = np.histogram(d, edges)
    vol = np.cumsum(hist[::-1])[::-1] / max(len(d), 1) * 100.0
    return edges[:-1], vol


# ---------------------------------------------------------------------------
#  Ossa e tessuti attraversati da un fascio che si sposta
# ---------------------------------------------------------------------------
#: Spostamenti (mm, per asse RAS) su cui si tabula l'attenuazione.
PATH_GRID_MM = tuple(float(v) for v in range(-15, 16, 5))


def path_attenuation_tables(red_grid, grid_affine, beams, moving="entrambi",
                            grid_mm=PATH_GRID_MM, step_mm=2.0):
    """Per ogni fascio, quanto cambia la fluenza primaria sul tumore quando
    il tumore (e, con l'inseguimento, il fascio) si sposta di d.

    La nuvola di dose statica non lo vede: il fascio attraversa sempre gli
    stessi tessuti. Ma se il fascio si sposta di 1 cm per seguire il tumore,
    puo' passare dentro una costa che nel piano non attraversava, o uscirne;
    se resta fermo e il tumore scende, il tumore si trova dietro altro
    tessuto. Si traccia la profondita' radiologica (densita' elettronica
    integrata, come beam_path_cm) su una griglia di spostamenti e si tabula

        f(d) = exp(-mu_acqua * (profondita'(d) - profondita'(0)))

    cioe' l'attenuazione primaria rispetto al piano. E' un'approssimazione
    dichiarata: lo scatter segue la fluenza primaria, e il cono resta quello.

    moving: "entrambi" (inseguimento: sorgente e bersaglio traslano insieme)
            "bersaglio" (fascio fermo: si sposta solo il punto sul tumore).
    Restituisce il dizionario da mettere nel piano.
    """
    from dosimetry import MU_WATER_6MV
    g = np.asarray(grid_mm, dtype=np.float64)
    O = np.stack(np.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3)
    inv = np.linalg.inv(np.asarray(grid_affine, dtype=np.float64))
    tables = []
    for b in beams:
        src = np.asarray(b.source, dtype=np.float64)
        tgt = np.asarray(b.target, dtype=np.float64)
        dist = float(np.linalg.norm(tgt - src))
        n = max(int(np.ceil(dist / step_mm)), 2)
        u = (np.arange(n) + 0.5) / n                       # (n,)
        S = src[None, :] + (O if moving == "entrambi" else 0.0)
        T = tgt[None, :] + O
        pts = S[:, None, :] + (T - S)[:, None, :] * u[None, :, None]
        idx = inv[:3, :3] @ pts.reshape(-1, 3).T + inv[:3, 3:4]
        red = ndimage.map_coordinates(red_grid, idx, order=1,
                                      mode="constant", cval=0.0)
        seg = np.linalg.norm(T - S, axis=1) / n / 10.0     # cm per passo
        depth = red.reshape(len(O), n).sum(axis=1) * seg
        d0 = depth[len(O) // 2]                            # spostamento nullo
        tables.append(np.round(np.exp(-MU_WATER_6MV * (depth - d0)), 4)
                      .tolist())
    return {"griglia_mm": list(g), "spostamento": moving,
            "mu_cm": float(MU_WATER_6MV), "fasci": tables}
