# ============================================================================
#  organ_mesh.py - Ricostruzione 3D degli organi: qualita' misurata, non a occhio.
#
#  PERCHE' ESISTE QUESTO MODULO
#  ----------------------------
#  La prima versione trattava tutte le strutture nello stesso modo: maschera
#  binaria sfumata con una gaussiana, marching cubes a 0.5, decimazione a un
#  numero fisso di facce, Taubin per otto iterazioni. Misurato su lung_001,
#  quella catena dava:
#
#      struttura     facce   fedelta'   asprezza   componenti   chiusa
#      scheletro     10000    1.97 mm     42.1 d       126        no
#      mediastino     3000    1.88 mm     37.6 d        13        no
#      polmone d.     3000    1.10 mm     19.7 d        20        no
#      corpo          7000    0.84 mm      6.9 d         1        no
#
#  e nel rendering lo scheletro mostrava grandi SCHEGGE PIATTE al posto di
#  scapole e clavicole. Le cause, tutte verificate:
#
#  1. NESSUNA IMBOTTITURA. Il marching cubes tagliava di netto ai bordi del
#     volume: nessuna superficie era chiusa, e il volume non era nemmeno
#     misurabile.
#  2. GUSCI A DOPPIA PARETE. La soglia dell'osso prende il corticale ma non il
#     midollo, quindi ogni costa era un tubo cavo: una superficie sottile e
#     aperta, che la decimazione collassa in triangoli che scavalcano il vuoto.
#     Sono le schegge.
#  3. DECIMAZIONE GLOBALE. Un budget unico su 126 componenti: le grandi se lo
#     prendevano tutto, le piccole diventavano spazzatura.
#  4. RITIRO. Gaussiana su maschera binaria + Taubin tirano la superficie verso
#     l'interno (scarto -0.45 mm sull'osso, -0.67 sul mediastino).
#  5. VALORI HOUNSFIELD IGNORATI. La maschera dice "osso / non osso", la CT
#     dice QUANTO osso c'e' in ogni voxel. L'isosuperficie sui valori veri ha
#     precisione sub-voxel per costruzione.
#
#  COME FUNZIONA ORA
#  -----------------
#  Ogni struttura ha la propria RICETTA, e ogni ricetta e' misurata, non
#  scelta a intuito (`python organ_mesh.py --tune`). La catena e':
#
#    1. CAMPO "DENTRO POSITIVO". Dai valori HU dove hanno senso: f = hu - soglia
#       per le strutture dense (osso, pelle), f = soglia - hu per quelle
#       piene d'aria (polmoni, vie aeree). Dove la densita' non distingue
#       nulla - il mediastino e' tessuto molle come la parete toracica - si usa
#       la trasformata di distanza firmata della maschera, in millimetri:
#       sfumarla non sposta il contorno come farebbe su una maschera binaria.
#    2. RITAGLIO E IMBOTTITURA. Si lavora sul solo parallelepipedo dell'organo
#       (piu' veloce e meno memoria) con un bordo di sfondo: cosi' la
#       superficie si CHIUDE anche dove l'organo tocca il bordo della CT.
#    3. RIEMPIMENTO DELL'INTERNO. Per le strutture solide si forza l'interno
#       oltre la soglia: l'osso diventa pieno invece che cavo, il polmone non
#       si riempie di superfici attorno ai vasi. Resta un guscio unico, che
#       la decimazione non puo' sbriciolare.
#    4. MARCHING CUBES a livello 0 sul campo continuo.
#    5. PULIZIA: facce degeneri e duplicate via, vertici uniti, componenti
#       sotto un volume fisico scartate, superficie orientata.
#    6. DECIMAZIONE PER COMPONENTE (vtkQuadricDecimation), con budget
#       proporzionale all'area: una costa non perde le sue facce a favore del
#       bacino, e la superficie non si apre.
#    7. LISCIATURA WINDOWED-SINC, dopo la decimazione e non prima: sulla mesh
#       densa, con triangoli da mezzo millimetro, qualunque lisciatura agisce
#       su scala microscopica e non cambia nulla (misurato: asprezza da 43.9 a
#       44.5 gradi con 20 iterazioni di Taubin). Sui triangoli millimetrici
#       della mesh decimata agisce dove serve, e taglia le alte frequenze
#       spaziali invece di mediare, quindi non ritira il volume.
#    8. RIPARAZIONE: i pochi lati non-manifold lasciati dalla decimazione
#       vengono rimossi e i buchi turati, finche' la superficie e' chiusa.
#
#  LE METRICHE
#  -----------
#  fedelta' (mm)   distanza fra superficie ricostruita e riferimento, su punti
#                  campionati uniformemente sull'AREA delle facce (non sui
#                  vertici: dopo una decimazione i vertici sono pochi e mal
#                  distribuiti, e la media sui vertici premierebbe le mesh
#                  peggiori) contro la trasformata di distanza del riferimento.
#  asprezza (gradi) angolo diedro medio fra facce adiacenti: quanto la
#                  superficie e' "a gradini". Sotto ~15 gradi e' liscia, sopra
#                  40 e' a scalini di voxel.
#  tenuta          chiusura, numero di componenti, facce degeneri, errore di
#                  volume rispetto alla maschera.
#
#  Fedelta' e asprezza sono in CONFLITTO - lisciare allontana dal dato - per
#  questo la scelta dei parametri e' un'ottimizzazione con un compromesso
#  dichiarato, non una preferenza estetica.
# ============================================================================
import numpy as np
from scipy import ndimage

import trimesh


# ---------------------------------------------------------------------------
#  Ricette: una per struttura
# ---------------------------------------------------------------------------
#  source   'dense' -> superficie dove hu = level, con l'organo piu' DENSO
#           'air'   -> superficie dove hu = level, con l'organo piu' RAREFATTO
#           'mask'  -> nessun appoggio nei valori HU: distanza firmata della
#                      maschera (il mediastino e' tessuto molle come la parete
#                      toracica: la densita' non li distingue)
#  fill     riempie l'interno dell'organo: il guscio diventa solido
#  sigma_mm sfumatura del campo, in millimetri (non in voxel: cosi' la ricetta
#           non cambia significato se si cambia la risoluzione della griglia)
#  faces    budget di facce
#  taubin   iterazioni di lisciatura
#  min_ml   volume minimo di una componente perche' sia tenuta
#
#  I valori vengono da `--tune`: vedi la tabella in
#  archivio/documenti/README_precedente.md.
RECIPES = {
    "bones": dict(source="dense", level=200.0, fill=True, median_mm=0.0,
                  sigma_mm=0.7, faces=26000, smooth_iters=10, pass_band=0.2,
                  min_ml=0.8, band_mm=2.5, close_mm=1.2),
    "body": dict(source="dense", level=-320.0, fill=True, median_mm=0.0,
                 sigma_mm=1.0, faces=12000, smooth_iters=10, pass_band=0.02,
                 min_ml=50.0, band_mm=4.0, close_mm=1.2),
    "lung_r": dict(source="air", level=-400.0, fill=True, median_mm=0.0,
                   sigma_mm=1.4, faces=9000, smooth_iters=5, pass_band=0.02,
                   min_ml=5.0, band_mm=3.0, close_mm=2.5),
    "lung_l": dict(source="air", level=-400.0, fill=True, median_mm=0.0,
                   sigma_mm=1.4, faces=9000, smooth_iters=5, pass_band=0.02,
                   min_ml=5.0, band_mm=3.0, close_mm=2.5),
    "airways": dict(source="air", level=-400.0, fill=False, median_mm=0.0,
                    sigma_mm=1.4, faces=6000, smooth_iters=10, pass_band=0.1,
                    min_ml=0.3, band_mm=2.0),
    "mediastinum": dict(source="mask", level=0.0, fill=True, sigma_mm=1.6,
                        faces=7000, smooth_iters=5, pass_band=0.02,
                        min_ml=3.0, band_mm=0.0),
    "tumor": dict(source="mask", level=0.0, fill=True, sigma_mm=0.8,
                  faces=6000, smooth_iters=20, pass_band=0.05, min_ml=0.05,
                  band_mm=0.0),
}

#: Risoluzione della griglia su cui si RICOSTRUISCE, in mm. Piu' fine di
#: quella su cui si SEGMENTA (2.0 mm): la segmentazione a 2 mm e' robusta e
#: veloce, ma il dettaglio della superficie si perde. Separare le due
#: risoluzioni da' il meglio dei due mondi.
MESH_SPACING_MM = 1.2


# ---------------------------------------------------------------------------
#  Metriche
# ---------------------------------------------------------------------------
def voxel_size(grid_affine):
    """Dimensione fisica del voxel lungo i tre assi della griglia (mm)."""
    return np.linalg.norm(np.asarray(grid_affine, dtype=np.float64)[:3, :3],
                          axis=0)


def signed_distance_field(reference, grid_affine):
    """Distanza in MILLIMETRI dalla superficie del riferimento binario.

    Negativa dentro, positiva fuori. Due trasformate di distanza euclidee con
    `sampling` pari alla dimensione fisica del voxel: cosi' la distanza e' in
    mm anche su griglie anisotrope, e confrontabile fra casi diversi.
    """
    ref = np.asarray(reference, dtype=bool)
    vox = voxel_size(grid_affine)
    if not ref.any():
        return np.full(ref.shape, np.inf, dtype=np.float32)
    out = ndimage.distance_transform_edt(~ref, sampling=vox)
    inn = ndimage.distance_transform_edt(ref, sampling=vox)
    return (out - inn).astype(np.float32)


def sample_field(field, grid_affine, points_mm, order=1):
    """Valore di un campo di griglia nei punti fisici dati (mm)."""
    aff = np.asarray(grid_affine, dtype=np.float64)
    inv = np.linalg.inv(aff)
    p = np.asarray(points_mm, dtype=np.float64)
    idx = (inv[:3, :3] @ p.T) + inv[:3, 3:4]
    return ndimage.map_coordinates(field, idx, order=order, mode="nearest")


def surface_fidelity_mm(mesh, reference, grid_affine, n_samples=40000,
                        seed=0, sdf=None):
    """Quanto la mesh si scosta dalla superficie di riferimento, in mm."""
    if mesh is None or len(mesh.faces) == 0:
        return {"media_mm": float("nan"), "p95_mm": float("nan"),
                "bias_mm": float("nan")}
    if sdf is None:
        sdf = signed_distance_field(reference, grid_affine)
    pts, _ = trimesh.sample.sample_surface(mesh, n_samples, seed=seed)
    d = sample_field(sdf, grid_affine, pts)
    return {"media_mm": float(np.abs(d).mean()),
            "p95_mm": float(np.percentile(np.abs(d), 95)),
            "bias_mm": float(d.mean())}


def roughness_deg(mesh):
    """Angolo diedro medio fra facce adiacenti: l'asprezza della superficie."""
    if mesh is None or len(mesh.faces) < 4:
        return float("nan")
    pairs = mesh.face_adjacency
    if len(pairs) == 0:
        return float("nan")
    n = mesh.face_normals
    cos = np.clip(np.einsum("ij,ij->i", n[pairs[:, 0]], n[pairs[:, 1]]),
                  -1.0, 1.0)
    return float(np.rad2deg(np.arccos(cos)).mean())


def spike_fraction(mesh, ratio=8.0, area_mult=4.0):
    """Frazione di area occupata da triangoli a scheggia.

    Una scheggia e' la firma della decimazione che scavalca un vuoto: una
    faccia con il lato lungo molto maggiore dell'altezza E con un'area molto
    maggiore di quella tipica. Il solo allungamento non basta come criterio:
    in una zona piatta il decimatore produce triangoli allungati del tutto
    innocui, e contandoli il mediastino risultava all'8% di schegge mentre nel
    rendering non se ne vedeva nessuna. Si pesa sull'AREA perche' sono le
    schegge grandi a rovinare l'immagine.
    """
    if mesh is None or len(mesh.faces) == 0:
        return float("nan")
    v = mesh.vertices[mesh.faces]
    e = np.stack([np.linalg.norm(v[:, 1] - v[:, 0], axis=1),
                  np.linalg.norm(v[:, 2] - v[:, 1], axis=1),
                  np.linalg.norm(v[:, 0] - v[:, 2], axis=1)], axis=1)
    long_ = e.max(axis=1)
    area = mesh.area_faces
    # altezza relativa al lato lungo: 2*area / lato
    h = 2.0 * area / np.maximum(long_, 1e-12)
    bad = ((long_ / np.maximum(h, 1e-12)) > ratio) & \
        (area > area_mult * np.median(area))
    tot = float(area.sum())
    return float(area[bad].sum() / tot) if tot > 0 else float("nan")


def integrity(mesh, reference=None, grid_affine=None):
    """Tenuta della mesh: chiusura, componenti, degeneri, errore di volume."""
    if mesh is None or len(mesh.faces) == 0:
        return {"facce": 0, "componenti": 0, "chiusa": False,
                "degeneri": 0, "err_volume_pct": float("nan")}
    areas = mesh.area_faces
    rep = {
        "facce": int(len(mesh.faces)),
        "componenti": int(mesh.body_count),
        "chiusa": bool(mesh.is_watertight),
        "degeneri": int((areas <= 1e-9).sum()),
        "err_volume_pct": float("nan"),
    }
    if reference is not None and grid_affine is not None:
        vox = abs(np.linalg.det(np.asarray(grid_affine,
                                           dtype=np.float64)[:3, :3]))
        v_ref = float(np.asarray(reference, dtype=bool).sum()) * vox
        if mesh.is_watertight and v_ref > 0:
            rep["err_volume_pct"] = 100.0 * (abs(mesh.volume) - v_ref) / v_ref
    return rep


def mesh_report(mesh, reference, grid_affine, n_samples=40000, sdf=None):
    """Tutte le metriche in un solo dizionario, pronto da stampare."""
    rep = {}
    rep.update(surface_fidelity_mm(mesh, reference, grid_affine, n_samples,
                                   sdf=sdf))
    rep["asprezza_deg"] = roughness_deg(mesh)
    rep["schegge_pct"] = 100.0 * spike_fraction(mesh)
    rep.update(integrity(mesh, reference, grid_affine))
    return rep


def format_report(key, rep):
    """Una riga di tabella leggibile per una struttura."""
    return (f"{key:<13}{rep['facce']:>7}f  fedelta' "
            f"{rep['media_mm']:>5.2f} mm (p95 {rep['p95_mm']:>5.2f})  "
            f"scarto {rep['bias_mm']:>+5.2f}  "
            f"asprezza {rep['asprezza_deg']:>5.1f}d  "
            f"schegge {rep['schegge_pct']:>5.1f}%  "
            f"comp {rep['componenti']:>3}  "
            f"{'chiusa' if rep['chiusa'] else 'APERTA':<7} "
            f"vol {rep['err_volume_pct']:>+6.1f}%")


# ---------------------------------------------------------------------------
#  Costruzione del campo
# ---------------------------------------------------------------------------
def _crop_box(mask, margin_vox):
    """Parallelepipedo della maschera con un bordo, limitato al volume."""
    idx = np.argwhere(mask)
    lo = np.maximum(idx.min(axis=0) - margin_vox, 0)
    hi = np.minimum(idx.max(axis=0) + margin_vox + 1, np.array(mask.shape))
    return lo, hi


def _shift_affine(grid_affine, lo):
    """Affine della sottogriglia che parte dall'indice `lo`."""
    aff = np.asarray(grid_affine, dtype=np.float64).copy()
    aff[:3, 3] = aff[:3, :3] @ np.asarray(lo, dtype=np.float64) + aff[:3, 3]
    return aff


def build_field(mask, hu, grid_affine, recipe):
    """Campo continuo con DENTRO POSITIVO, ritagliato e imbottito.

    Restituisce (campo, affine_del_ritaglio). La superficie da estrarre e'
    sempre il livello 0: cosi' l'orientamento delle facce e' noto per
    costruzione e non dipende dal tipo di struttura.
    """
    vox = voxel_size(grid_affine)
    sigma_vox = np.asarray(recipe["sigma_mm"], dtype=np.float64) / vox
    # Il bordo deve contenere la coda della gaussiana, altrimenti la
    # sfumatura "vede" il taglio del ritaglio e sposta il contorno.
    margin = int(np.ceil(max(3.0 * sigma_vox.max(),
                             recipe.get("band_mm", 0.0) / vox.min()))) + 3
    lo, hi = _crop_box(mask, margin)
    sl = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
    m = np.asarray(mask, dtype=bool)[sl]
    aff = _shift_affine(grid_affine, lo)

    src = recipe["source"]
    if src == "mask":
        # Nessun appoggio nella densita': distanza firmata in mm, dentro
        # positiva. Sfumarla sposta il contorno molto meno che sfumare una
        # maschera binaria, dove la gaussiana tira le superfici curve verso
        # l'interno.
        f = -signed_distance_field(m, aff)
    else:
        h = np.asarray(hu, dtype=np.float32)[sl]
        f = (h - recipe["level"]) if src == "dense" else (recipe["level"] - h)
        # MEDIANO opzionale, prima di tutto il resto. Serve all'osso: il
        # granulato dell'osso spongioso oscilla attorno alla soglia e produce
        # una superficie ruvida che la gaussiana puo' solo smussare, perdendo
        # i bordi. Il mediano toglie il granulato e i bordi restano dove sono,
        # perche' non fa una media ma sceglie un valore realmente presente.
        med = float(recipe.get("median_mm", 0.0))
        if med > 0:
            size = np.maximum((med / vox).round().astype(int) * 2 + 1, 1)
            f = ndimage.median_filter(f, size=tuple(int(v) for v in size))

        # NORMALIZZAZIONE DEI DUE LATI. Un gradino asimmetrico sposta lo zero
        # verso il lato di modulo minore: sulla pelle il tessuto molle vale
        # +320 e l'aria -680, e la superficie si ritirava di 0.56 mm.
        # Saturare a +-clip non basta - misurato sul fantoccio sferico: con 60
        # HU di contrasto interno e aria a -1000, il raggio usciva 24.67 invece
        # di 24.99 - perche' il lato debole resta sotto la saturazione. Si
        # riscalano percio' i due lati SEPARATAMENTE, in modo che entrambi i
        # pianori valgano 1 in modulo: lo zero cade a meta' fra i due campioni
        # qualunque sia il contrasto, mentre nella fascia di transizione il
        # campo resta proporzionale ai valori HU e mantiene la precisione
        # sub-voxel. I percentili, invece dei massimi, evitano che un voxel di
        # rumore decida la scala.
        clip = float(recipe.get("clip_hu", 300.0))
        f = np.clip(f, -clip, clip)
        pos = f > 0
        s_in = np.percentile(f[pos], 75) if pos.any() else 1.0
        s_out = np.percentile(-f[~pos], 75) if (~pos).any() else 1.0
        f = np.where(pos, f / max(float(s_in), 1e-6),
                     f / max(float(s_out), 1e-6)).astype(np.float32)
        f = np.clip(f, -1.5, 1.5)
        clip = 1.5                      # d'ora in poi il campo e' adimensionale
        # Fuori dall'organo il campo va spento: senza questo, un vaso con
        # contrasto accanto a una costa entrerebbe nell'osso, e il lettino
        # entrerebbe nella pelle.
        band = recipe.get("band_mm", 0.0)
        if band > 0:
            it = max(int(round(band / vox.min())), 1)
            near = ndimage.binary_dilation(m, iterations=it)
            f = np.where(near, f, -clip)
        if recipe.get("fill"):
            # L'INTERNO VA RIEMPITO DAVVERO. Prima si erodeva la maschera di un
            # voxel e si forzava quello: dentro un corpo vertebrale l'osso
            # spongioso oscilla attorno alla soglia, quindi restava una
            # superficie grumosa - nel rendering le vertebre erano un
            # "cavolfiore". Qui si chiude il solido (le fessure corticali si
            # saldano), si turano le cavita' - il midollo - e solo l'interno
            # cosi' ottenuto viene forzato oltre la soglia. Resta un guscio
            # unico, che la decimazione non puo' sbriciolare.
            it = max(int(round(recipe.get("close_mm", 1.5) / vox.min())), 1)
            solid = ndimage.binary_closing(f > 0, iterations=it)
            solid = ndimage.binary_fill_holes(solid)
            if band > 0:
                solid &= near
            # Interno = interno secondo la DENSITA' unito a interno secondo la
            # MASCHERA. Servono entrambi: la densita' riempie il midollo di una
            # vertebra, dove la maschera grossolana non arriva; la maschera
            # riempie i polmoni dentro il corpo, che la densita' dichiara
            # "fuori" e che non vengono turati come cavita' perche' LA TRACHEA
            # LI COLLEGA all'aria esterna. Senza il secondo pezzo la mesh del
            # corpo si portava dietro la superficie interna dei polmoni:
            # fedelta' 20.95 mm invece di 0.7.
            inside = (ndimage.binary_erosion(solid, iterations=1)
                      | ndimage.binary_erosion(m, iterations=1))
            f = np.where(inside, clip, f)

    # Imbottitura di sfondo: chiude la superficie anche dove l'organo tocca il
    # bordo della CT (il corpo lo fa sempre, in testa e ai piedi).
    pad = 2
    f = np.pad(f.astype(np.float32), pad, mode="constant",
               constant_values=float(min(f.min(), -1.0)))
    aff = _shift_affine(aff, (-pad, -pad, -pad))
    if np.any(sigma_vox > 0):
        f = ndimage.gaussian_filter(f, sigma=tuple(sigma_vox))
    return f, aff


# ---------------------------------------------------------------------------
#  Pulizia e decimazione
# ---------------------------------------------------------------------------
def _clean(mesh, close=True):
    """Facce degeneri e duplicate via, vertici uniti, buchi turati, normali.

    Il rattoppo dei buchi serve dopo la decimazione: vtkQuadricDecimation
    lascia qualche lato scoperto - misurato, 8 lati su 7000 facce sul
    mediastino - e basta quello perche' la superficie non sia piu' chiusa e il
    volume non sia calcolabile. Sono buchi di pochi triangoli, non squarci.
    """
    mesh.merge_vertices()
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.update_faces(mesh.unique_faces())
    mesh.remove_unreferenced_vertices()
    # Lati NON-MANIFOLD (un lato condiviso da piu' di due facce) e buchi si
    # inseguono: togliendo le facce non-manifold si apre un anello, turando
    # l'anello puo' ricomparire un lato non-manifold. Si ripete qualche volta,
    # e si smette appena la superficie e' chiusa. Senza questo, sul mediastino
    # restavano 3 lati di bordo per 7.6 mm totali che ne' VTK ne' trimesh
    # riuscivano a chiudere, perche' confinanti con 5 lati non-manifold.
    for _ in range(3):
        if not close or mesh.is_watertight:
            break
        e = mesh.edges_sorted
        _, inv, cnt = np.unique(e, axis=0, return_inverse=True,
                                return_counts=True)
        bad = np.nonzero((cnt > 2)[inv.ravel()])[0] // 3
        if len(bad):
            keep = np.ones(len(mesh.faces), dtype=bool)
            keep[np.unique(bad)] = False
            mesh.update_faces(keep)
            mesh.remove_unreferenced_vertices()
        try:
            filled = _from_pv(_to_pv(mesh).fill_holes(8.0).triangulate())
            if len(filled.faces) >= len(mesh.faces):
                mesh = filled
        except Exception:
            pass
        if not mesh.is_watertight:
            trimesh.repair.fill_holes(mesh)
    trimesh.repair.fix_normals(mesh)
    return mesh


def _drop_small(mesh, min_ml, grid_affine=None):
    """Scarta le componenti sotto un volume fisico.

    Il criterio e' in millilitri e non "tieni le prime N": le coste sono 24
    componenti legittime, il rumore ne produce centinaia, e una soglia di
    volume distingue le due cose senza sapere quante siano.
    """
    if min_ml <= 0:
        return mesh
    parts = mesh.split(only_watertight=False)
    if len(parts) <= 1:
        return mesh
    keep = []
    for p in parts:
        # Le componenti aperte non hanno volume: si stima dal parallelepipedo,
        # che per una scheggia e' comunque piccolo.
        v = abs(p.volume) if p.is_watertight else float(np.prod(
            np.maximum(p.extents, 1e-6))) * 0.3
        if v / 1000.0 >= min_ml:
            keep.append(p)
    if not keep:
        return mesh
    return trimesh.util.concatenate(keep)


def _to_pv(mesh):
    import pyvista as pv
    f = np.asarray(mesh.faces)
    return pv.PolyData(np.asarray(mesh.vertices, dtype=np.float64),
                       np.hstack([np.full((len(f), 1), 3), f]).ravel())


def _from_pv(poly):
    f = np.asarray(poly.faces).reshape(-1, 4)[:, 1:]
    # process=True unisce i vertici coincidenti. Senza, la mesh risulta
    # APERTA per costruzione - ogni faccia ha i propri vertici e nessun lato
    # e' condiviso - ed era il vero motivo per cui dopo la decimazione niente
    # era piu' watertight.
    return trimesh.Trimesh(vertices=np.asarray(poly.points), faces=f,
                           process=True)


def _decimate(mesh, faces):
    """Decimazione PER COMPONENTE con budget proporzionale all'area.

    Decimando la mesh intera, il decimatore spende le facce dove ce n'e' piu'
    bisogno in senso globale: il bacino si prende tutto e una costa diventa
    una scheggia. Componente per componente, ognuna tiene la propria quota.
    """
    if faces <= 0 or len(mesh.faces) <= faces:
        return mesh
    parts = mesh.split(only_watertight=False)
    if len(parts) <= 1:
        return _decimate_one(mesh, faces)
    areas = np.array([p.area for p in parts], dtype=np.float64)
    quota = np.maximum((areas / areas.sum() * faces).astype(int), 60)
    out = []
    for p, q in zip(parts, quota):
        out.append(_decimate_one(p, int(q)) if len(p.faces) > q else p)
    return trimesh.util.concatenate(out)


#: Ripieghi effettivamente usati, da mostrare a chi guarda il programma.
#: Un ripiego silenzioso e' peggio di un errore: il risultato peggiora e non
#: si capisce perche'. Li raccoglie build_all e li manda allo status_cb.
FALLBACK_NOTES = set()


def _decimate_one(mesh, faces):
    """Decimazione quadrica di VTK, che NON apre la superficie.

    Il decimatore di trimesh (fast_simplification) rompeva la chiusura: la
    mesh arrivava watertight con 1.24 milioni di facce e ne usciva aperta.
    vtkQuadricDecimation conserva la topologia e, con volume_preservation,
    non fa collassare il guscio verso l'interno.
    """
    n = len(mesh.faces)
    if n <= faces:
        return mesh
    try:
        poly = _to_pv(mesh).decimate(1.0 - float(faces) / n,
                                     volume_preservation=True)
        out = _from_pv(poly.triangulate())
        if len(out.faces) > 0:
            return out
    except Exception as exc:
        FALLBACK_NOTES.add(
            f"decimazione senza VTK ({type(exc).__name__}): le superfici "
            f"restano aperte e piu' aspre. Installa pyvista.")
    try:
        return mesh.simplify_quadric_decimation(face_count=int(faces))
    except Exception:
        return mesh


def _smooth(mesh, iters, pass_band):
    """Lisciatura windowed-sinc (VTK), lo strumento standard per le isosuperfici.

    Perche' non il Taubin di trimesh: applicato alla mesh densa non lisciava
    nulla (asprezza da 43.9 a 44.5 gradi con 20 iterazioni), perche' agisce sul
    vicinato di ogni vertice e su triangoli da 0.5 mm quel vicinato e'
    microscopico. Il filtro windowed-sinc lavora sulla mesh DECIMATA, cioe' su
    triangoli millimetrici, e taglia le frequenze spaziali sopra `pass_band`
    invece di mediare: e' costruito per non ritirare il volume.
    """
    if iters <= 0 or len(mesh.faces) < 4:
        return mesh
    try:
        poly = _to_pv(mesh).smooth_taubin(
            n_iter=int(iters), pass_band=float(pass_band),
            feature_smoothing=False, boundary_smoothing=True,
            non_manifold_smoothing=False, normalize_coordinates=True)
        out = _from_pv(poly.triangulate())
        if len(out.faces) > 0:
            return out
    except Exception as exc:
        FALLBACK_NOTES.add(
            f"lisciatura senza VTK ({type(exc).__name__}): il Taubin di "
            f"trimesh liscia molto meno. Installa pyvista.")
    trimesh.smoothing.filter_taubin(mesh, iterations=int(iters))
    return mesh


#: Griglie su cui l'ottimizzatore cerca, parametro per parametro.
TUNE_GRIDS = {
    "median_mm": [0.0, 1.2, 2.4],
    "sigma_mm": [0.3, 0.45, 0.7, 1.0, 1.4, 2.0],
    "pass_band": [0.02, 0.05, 0.1, 0.2, 0.4],
    "smooth_iters": [0, 5, 10, 20, 40],
    "close_mm": [1.2, 1.8, 3.0],
}

#: Quanto pesa la levigatezza rispetto alla fedelta', struttura per struttura.
#  Non e' una preferenza estetica arbitraria: dipende da cosa RAPPRESENTA la
#  superficie. La pelle e il mediastino sono contorni lisci, e un'asprezza
#  residua li' e' rumore di ricostruzione; una costa e una vertebra hanno
#  dettaglio vero a scala millimetrica, e lisciarle lo cancella. Il tumore sta
#  in mezzo: e' il bersaglio, la sua forma conta, ma viene da una maschera
#  binaria e i gradini sono artefatti.
TUNE_WEIGHTS = {
    "bones": dict(rough=0.45, spike=0.5),
    "body": dict(rough=1.60, spike=0.8),
    "lung_r": dict(rough=1.00, spike=0.6),
    "lung_l": dict(rough=1.00, spike=0.6),
    "airways": dict(rough=0.60, spike=0.6),
    "mediastinum": dict(rough=1.60, spike=0.8),
    "tumor": dict(rough=1.20, spike=0.8),
}


def cost(rep, weights):
    """Funzione costo di una ricostruzione: piu' bassa e' meglio.

    La fedelta' e' in millimetri e fa da unita' di misura; l'asprezza si
    normalizza a 40 gradi (il valore di una superficie a scalini di voxel),
    le schegge al 10% dell'area. Mesh aperte e sbriciolamento sono penalita'
    dure: una superficie non chiusa non ha volume, e cento frammenti non sono
    un organo.
    """
    if rep is None or not rep["facce"]:
        return float("inf")
    c = float(rep["media_mm"])
    c += weights["rough"] * float(rep["asprezza_deg"]) / 40.0
    c += weights["spike"] * float(rep["schegge_pct"]) / 10.0
    c += 0.25 * abs(float(rep["bias_mm"]))          # ne' gonfia ne' ritirata
    if not rep["chiusa"]:
        c += 0.40
    c += 0.004 * max(int(rep["componenti"]) - 1, 0)
    return c


def tune_organ(key, mask, hu, grid_affine, recipe=None, grids=None,
               weights=None, passes=2, log=print):
    """Cerca i parametri migliori per UNA struttura, a discesa di coordinate.

    Si parte dalla ricetta corrente e si ottimizza un parametro alla volta,
    tenendo fermi gli altri, per due giri. Non e' una ricerca esaustiva - la
    griglia completa sarebbe 450 ricostruzioni per organo, circa un'ora - ma
    ogni passo e' una MISURA, non un giudizio a occhio, e il risultato e'
    riproducibile: la funzione costo e le griglie stanno in questo file.
    """
    rec = dict(recipe or RECIPES[key])
    grids = grids or TUNE_GRIDS
    w = weights or TUNE_WEIGHTS.get(key, dict(rough=1.0, spike=0.6))
    ref = reference_mask(mask, hu, rec)
    sdf = signed_distance_field(ref, grid_affine)

    def evaluate(r):
        mesh = build_mesh(mask, hu, grid_affine, r)
        rep = mesh_report(mesh, ref, grid_affine, sdf=sdf)
        return cost(rep, w), rep, mesh

    best_c, best_rep, best_mesh = evaluate(rec)
    log(f"  {key}: partenza costo {best_c:.3f}  {format_report('', best_rep)}")
    # Solo i parametri che questa struttura usa davvero: il mediano e la
    # chiusura hanno senso solo sul campo di densita', non sulla distanza
    # firmata di una maschera.
    hu_based = rec.get("source") != "mask"
    names = [n for n in ("median_mm", "sigma_mm", "smooth_iters", "pass_band",
                         "close_mm")
             if n in grids
             and (n not in ("close_mm", "median_mm") or hu_based)
             and (n != "close_mm" or rec.get("fill"))]
    for it in range(passes):
        improved = False
        for name in names:
            for val in grids[name]:
                if val == rec.get(name):
                    continue
                trial = dict(rec, **{name: val})
                c, rep, mesh = evaluate(trial)
                if c < best_c - 1e-4:
                    best_c, best_rep, best_mesh = c, rep, mesh
                    rec, improved = trial, True
                    log(f"    {name} -> {val}: costo {c:.3f}  "
                        f"{format_report('', rep)}")
        if not improved:
            break
    log(f"  {key}: scelta {{" + ", ".join(
        f"{n}: {rec[n]}" for n in names if n in rec)
        + f"}}  costo {best_c:.3f}")
    return rec, best_rep, best_mesh


def reference_mask(mask, hu, recipe):
    """La superficie di VERITA' per una struttura, sulla griglia data.

    Non e' la maschera di segmentazione: quella vive su una griglia grossolana
    (2 mm) ed e' essa stessa un'approssimazione. La verita' e' il criterio
    fisico applicato ai valori HU - "osso dove la densita' supera la soglia,
    entro la regione segmentata" - ed e' contro questa che va misurata la
    fedelta', sia della catena nuova sia di quella vecchia. Misurare contro la
    maschera grossolana premierebbe chi la copia, non chi descrive il paziente.
    """
    m = np.asarray(mask, dtype=bool)
    if recipe["source"] == "mask" or hu is None:
        return m
    h = np.asarray(hu, dtype=np.float32)
    dense = recipe["source"] == "dense"
    ref = (h > recipe["level"]) if dense else (h < recipe["level"])
    ref &= ndimage.binary_dilation(m, iterations=1)
    if recipe.get("fill"):
        ref = ndimage.binary_fill_holes(ref) | ndimage.binary_erosion(m, iterations=1)
    return ref


def organ_grid(ct_data, ct_affine, mask, seg_affine, spacing_mm,
               margin_mm=6.0):
    """Griglia fine LOCALE attorno a una struttura: (hu, maschera, affine).

    Non si ricampiona l'intera TAC: a 1.2 mm sarebbero ~52 milioni di voxel
    per la sola densita', e altrettanti per ogni maschera. Ritagliando attorno
    all'organo si resta su qualche decina di MB, e il marching cubes lavora
    solo dove serve.

    Il ricampionamento usa `affine_transform`, che compone le due affini senza
    materializzare le coordinate di ogni voxel: la versione con
    map_coordinates chiedeva 360 MB di soli indici.
    """
    m = np.asarray(mask, dtype=bool)
    if not m.any():
        return None, None, None
    seg_aff = np.asarray(seg_affine, dtype=np.float64)
    idx = np.argwhere(m)
    # Gli otto vertici del parallelepipedo in coordinate fisiche: con un'affine
    # ruotata il minimo per indice non e' il minimo nel mondo.
    lo_i, hi_i = idx.min(axis=0), idx.max(axis=0)
    corners = np.array([[lo_i[0] if i & 1 else hi_i[0],
                         lo_i[1] if i & 2 else hi_i[1],
                         lo_i[2] if i & 4 else hi_i[2]] for i in range(8)],
                       dtype=np.float64)
    world = (seg_aff[:3, :3] @ corners.T + seg_aff[:3, 3:4]).T
    lo_w = world.min(axis=0) - margin_mm
    hi_w = world.max(axis=0) + margin_mm

    # Griglia locale allineata agli assi del mondo (RAS), isotropica.
    aff = np.eye(4, dtype=np.float64)
    aff[:3, :3] = np.diag([spacing_mm] * 3)
    aff[:3, 3] = lo_w
    shape = tuple(int(np.ceil((hi_w[k] - lo_w[k]) / spacing_mm)) + 1
                  for k in range(3))

    def _resample(vol, src_affine, order):
        src = np.asarray(src_affine, dtype=np.float64)
        inv = np.linalg.inv(src)
        mat = inv[:3, :3] @ aff[:3, :3]
        off = inv[:3, :3] @ aff[:3, 3] + inv[:3, 3]
        return ndimage.affine_transform(vol, mat, offset=off,
                                        output_shape=shape, order=order,
                                        mode="constant",
                                        cval=(-1000.0 if order else 0))

    hu = _resample(np.asarray(ct_data, dtype=np.float32),
                   np.asarray(ct_affine, dtype=np.float64), 1)
    m_fine = _resample(m.astype(np.uint8), seg_aff, 0).astype(bool)
    return hu, m_fine, aff


def build_all(masks, seg_affine, ct_data=None, ct_affine=None,
              spacing_mm=MESH_SPACING_MM, keys=None, faces_scale=1.0,
              recipes=None, progress_cb=None, status_cb=None):
    """Ricostruisce tutte le strutture, ognuna con la sua ricetta.

    Le strutture con appoggio nei valori HU vengono ricostruite su una griglia
    fine locale ricavata dalla TAC; quelle senza (il mediastino, il tumore)
    restano sulla griglia della loro maschera, dove la distanza firmata da'
    gia' una superficie continua.

    Restituisce una lista di dict nello stesso formato di
    organs.build_organ_meshes, cosi' il resto del programma non cambia.
    """
    import organs as og

    recipes = recipes or RECIPES
    order = [k for k, *_ in og.ORGAN_STYLE]
    todo = [k for k in order
            if k in masks and k in recipes and np.any(masks[k])]
    if keys:
        todo = [k for k in todo if k in keys]

    out = []
    for i, key in enumerate(todo):
        info = og.ORGAN_INFO[key]
        if status_cb:
            status_cb(f"Ricostruzione: {info['name']}...")
        rec = dict(recipes[key])
        rec["faces"] = max(int(rec.get("faces", 0) * faces_scale), 400)
        if rec["source"] == "mask" or ct_data is None:
            mesh = build_mesh(masks[key], None, seg_affine,
                              dict(rec, source="mask"))
            ref, ref_aff = masks[key], seg_affine
        else:
            hu, m_fine, aff = organ_grid(ct_data, ct_affine, masks[key],
                                         seg_affine, spacing_mm)
            mesh = build_mesh(m_fine, hu, aff, rec)
            ref, ref_aff = m_fine, aff
        if mesh is None or len(mesh.faces) == 0:
            continue
        out.append({"key": key, "name": info["name"], "color": info["color"],
                    "alpha": info["alpha"], "zorder": info["zorder"],
                    "mesh": mesh, "riferimento": ref, "rif_affine": ref_aff})
        if progress_cb:
            progress_cb(i + 1, len(todo))
    if FALLBACK_NOTES and status_cb:
        for note in sorted(FALLBACK_NOTES):
            status_cb("ATTENZIONE: " + note)
    return out


def build_mesh(mask, hu, grid_affine, recipe, smooth=True):
    """Dalla maschera (e dai valori HU) alla mesh in millimetri."""
    from skimage import measure

    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return None
    field, aff = build_field(mask, hu, grid_affine, recipe)
    try:
        verts, faces, _, _ = measure.marching_cubes(field, level=0.0,
                                                    step_size=1)
    except (ValueError, RuntimeError):
        return None
    if not len(faces):
        return None

    verts_mm = (np.asarray(aff, dtype=np.float64)[:3, :3] @ verts.T
                + np.asarray(aff, dtype=np.float64)[:3, 3:4]).T
    # Un'affine con determinante negativo ribalta l'orientamento: le facce
    # andrebbero "al rovescio" e il rendering mostrerebbe l'interno.
    if np.linalg.det(np.asarray(aff, dtype=np.float64)[:3, :3]) < 0:
        faces = faces[:, ::-1]
    mesh = trimesh.Trimesh(vertices=verts_mm, faces=faces, process=True)
    mesh = _clean(mesh)
    mesh = _drop_small(mesh, recipe.get("min_ml", 0.0))
    # Prima si decima, POI si liscia: sulla mesh densa (1.2 milioni di facce,
    # triangoli da mezzo millimetro) qualunque lisciatura agisce su una scala
    # microscopica e non cambia nulla. Sui triangoli millimetrici della mesh
    # decimata agisce dove serve.
    mesh = _decimate(mesh, int(recipe.get("faces", 0)))
    if smooth:
        mesh = _smooth(mesh, int(recipe.get("smooth_iters", 0)),
                       recipe.get("pass_band", 0.05))
    mesh = _clean(mesh)
    return mesh


# ---------------------------------------------------------------------------
#  Riga di comando: misura le ricette correnti oppure cercane di migliori
# ---------------------------------------------------------------------------
def _load_case(ct_path, spacing_mm, cache=None, log=print):
    """CT + maschere, con cache su file: segmentare costa ~40 s."""
    import pickle

    import nibabel as nib

    img = nib.load(ct_path)
    ct = np.asanyarray(img.dataobj, dtype=np.float32)
    if cache and __import__("os").path.exists(cache):
        masks, gaff = pickle.load(open(cache, "rb"))
        log(f"maschere dalla cache: {cache}")
    else:
        import organs as og
        masks, gaff, _ = og.segment_organs(ct, img.affine, spacing_mm,
                                           status_cb=lambda m: log("   " + m))
        if cache:
            pickle.dump((masks, gaff), open(cache, "wb"))
    return ct, img.affine, masks, gaff


def _grids_for(key, masks, seg_affine, ct, ct_affine, spacing_mm):
    """Maschera, densita' e affine su cui lavorare per una struttura."""
    if RECIPES[key]["source"] == "mask":
        return masks[key], None, seg_affine
    hu, m_fine, aff = organ_grid(ct, ct_affine, masks[key], seg_affine,
                                 spacing_mm)
    return m_fine, hu, aff


if __name__ == "__main__":
    import argparse
    import json
    import time

    ap = argparse.ArgumentParser(
        description="Qualita' della ricostruzione 3D degli organi: misura le "
                    "ricette correnti (--report) o cercane di migliori (--tune).")
    ap.add_argument("--ct", required=True)
    ap.add_argument("--mask", default=None,
                    help="maschera del tumore (Fase 1): serve per la ricetta "
                         "'tumor', che non esce dalla segmentazione per soglie")
    ap.add_argument("--spacing", type=float, default=2.0,
                    help="griglia di SEGMENTAZIONE (mm)")
    ap.add_argument("--mesh-spacing", type=float, default=MESH_SPACING_MM,
                    dest="mesh_spacing", help="griglia di RICOSTRUZIONE (mm)")
    ap.add_argument("--cache", default=None,
                    help="file .pkl in cui tenere le maschere segmentate")
    ap.add_argument("--tune", action="store_true")
    ap.add_argument("--keys", default=None,
                    help="elenco separato da virgole (default: tutte)")
    ap.add_argument("--out", default=None,
                    help="dove scrivere le ricette trovate (JSON)")
    args = ap.parse_args()

    ct, ct_aff, masks, gaff = _load_case(args.ct, args.spacing, args.cache)
    if args.mask:
        # Il tumore vive sulla griglia della TAC, non su quella di
        # segmentazione: la sua ricetta si misura li'.
        import nibabel as _nib
        from tumor_3d import largest_component as _lc
        _tm = (_nib.load(args.mask).get_fdata() > 0.5)
        masks = dict(masks, tumor=_lc(_tm, ct_aff).astype(bool))
    keys = ([k for k in args.keys.split(",")] if args.keys
            else [k for k in RECIPES if k in masks and np.any(masks[k])])

    chosen, t0 = {}, time.time()
    for key in keys:
        if key == "tumor":
            mask, hu, aff = masks["tumor"], None, ct_aff
        else:
            mask, hu, aff = _grids_for(key, masks, gaff, ct, ct_aff,
                                       args.mesh_spacing)
        if mask is None or not np.any(mask):
            continue
        if args.tune:
            rec, rep, _ = tune_organ(key, mask, hu, aff)
            chosen[key] = rec
        else:
            rec = RECIPES[key]
            mesh = build_mesh(mask, hu, aff, rec)
            rep = mesh_report(mesh, reference_mask(mask, hu, rec), aff)
            print(format_report(key, rep))

    print(f"\nin {time.time() - t0:.0f}s")
    if chosen:
        print("\nRicette trovate (da incollare in RECIPES):")
        for key, rec in chosen.items():
            print(f'    "{key}": dict(' + ", ".join(
                f"{k}={v!r}" for k, v in rec.items()) + "),")
        if args.out:
            json.dump(chosen, open(args.out, "w", encoding="utf-8"), indent=1)
            print(f"\nscritte in {args.out}")
