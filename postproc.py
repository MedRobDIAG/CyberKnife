# ============================================================================
#  postproc.py - UNICA definizione del post-processing della predizione.
#
#  PERCHE' ESISTE QUESTO MODULO
#  Il post-processing era scritto tre volte (inference.py, inference_worker.py,
#  tumor_3d.py) e in nessuna di queste forme veniva applicato in validazione.
#  Conseguenza: il Dice riportato durante il training descriveva una pipeline
#  che non era quella realmente eseguita. Concentrando tutto qui, la coerenza
#  fra validazione e deployment smette di essere una questione di disciplina e
#  diventa una proprieta' del codice: chiamano tutti la stessa funzione.
#
#  COSA CAMBIA RISPETTO A PRIMA: "componente piu' grande" -> "componenti sopra
#  un volume fisico".
#  La regola precedente (tenere solo la componente connessa maggiore) ha due
#  difetti gravi per un sistema che alimenta un planner:
#    1. Non puo' MAI dire "nessun tumore". Acceso un solo voxel, quel voxel
#       diventa il bersaglio e il planner ci costruisce sopra un piano.
#    2. Se un falso positivo cresce piu' della lesione vera, l'isocentro salta
#       sul falso positivo senza alcun segnale d'allarme. E' il fallimento
#       registrato in archivio/documenti/STATO_PROGETTO.md (D95 ~ 0%,
#       isocentro a 74 mm dal
#       bersaglio): non un caso limite teorico, e' gia' successo.
#  La soglia in mm3 e' invece una grandezza FISICA e interpretabile: 100 mm3 =
#  0.1 cm3 = sfera di 5.8 mm di diametro, cioe' esattamente la piu' piccola
#  lesione realmente annotata in questo dataset. Sotto quel volume non c'e'
#  nulla da trovare, e quel che si trova e' rumore.
# ============================================================================
from dataclasses import dataclass, asdict

import numpy as np
from scipy import ndimage


@dataclass
class PostProcConfig:
    """Parametri del post-processing. Unica fonte di verita' (come PlannerConfig).

    prob_threshold : soglia sulla probabilita' softmax della classe tumore.
        0.50 equivale all'argmax usato finora. NON e' un valore sacro: e' una
        manopola che scambia sensibilita' contro falsi positivi, e va tarata
        sulle predizioni out-of-fold (evaluate.py --sweep), non lasciata al
        default per inerzia.
    min_lesion_mm3 : volume minimo di una componente perche' sia considerata
        una lesione. 100 mm3 = sfera di 5.8 mm.
    lung_roi_filter : se True, scarta le componenti il cui centroide cade fuori
        dalla maschera polmonare dilatata. E' un filtro SOFT, applicato per
        componente: non si moltiplica mai la maschera voxel per voxel (la
        maschera polmonare di organs.py e' una maschera d'ARIA, e il tumore ne
        e' un buco: moltiplicare cancellerebbe il 65% dei voxel tumorali).
    lung_dilate_mm : dilatazione applicata alla maschera polmonare prima del
        test di appartenenza. 20 mm copre le lesioni pleuriche e mediastiniche.
    """

    prob_threshold: float = 0.50
    min_lesion_mm3: float = 100.0
    lung_roi_filter: bool = False
    lung_dilate_mm: float = 20.0

    def replace(self, **kw):
        return PostProcConfig(**{**asdict(self), **kw})



@dataclass
class GateConfig:
    """Criterio di accettazione del bersaglio (cancello di confidenza).

    Sta qui e non in gate.py per una ragione precisa: config.py deve poterlo
    importare, e gate.py tira dentro splits/dataset/predict, creando un ciclo
    config -> gate -> splits -> dataset -> config. postproc.py invece non
    importa nulla del progetto, quindi resta importabile da chiunque.

    Il razionale della regola e le misure che la giustificano stanno in
    gate.py e in config.GATE.
    """

    min_volume_cm3: float = 1.0
    min_prob_mean: float = 0.0

    def accepts(self, lesion: dict | None) -> bool:
        """Il bersaglio e' abbastanza affidabile da essere consegnato?"""
        if lesion is None:
            return False
        return (lesion["volume_cm3"] >= self.min_volume_cm3
                and lesion["prob_mean"] >= self.min_prob_mean)

    def reason(self, lesion: dict | None) -> str:
        """Perche' e' stato rifiutato (stringa per l'interfaccia)."""
        if lesion is None:
            return "nessuna lesione sopra la soglia di volume"
        if lesion["volume_cm3"] < self.min_volume_cm3:
            return (f"lesione primaria di {lesion['volume_cm3']:.2f} cm3, "
                    f"sotto il minimo di {self.min_volume_cm3:.2f} cm3")
        if lesion["prob_mean"] < self.min_prob_mean:
            return (f"confidenza media {lesion['prob_mean']:.2f}, "
                    f"sotto il minimo di {self.min_prob_mean:.2f}")
        return ""

    def replace(self, **kw):
        return GateConfig(**{**asdict(self), **kw})


def _voxel_volume_mm3(affine) -> float:
    """Volume di un voxel in mm3 = |det| della parte lineare dell'affine.

    Vale anche per affine ruotate/oblique, dove il prodotto degli spacing non
    sarebbe corretto.
    """
    return float(abs(np.linalg.det(np.asarray(affine, dtype=np.float64)[:3, :3])))


def _world_to_index(affine, world_pts):
    """Coordinate mondo (3, N) -> indici di griglia (3, N)."""
    inv = np.linalg.inv(np.asarray(affine, dtype=np.float64))
    return (inv[:3, :3] @ np.asarray(world_pts, dtype=np.float64)) + inv[:3, 3:4]


def probs_to_mask(prob_tumor, cfg: PostProcConfig | None = None,
                  voxel_volume_mm3: float | None = None,
                  affine=None, lung_mask=None, lung_affine=None):
    """Probabilita' della classe tumore -> (maschera uint8, elenco lesioni).

    prob_tumor : array 3D di probabilita' in [0,1] (canale 1 della softmax),
        gia' riportato nello spazio della CT di riferimento.
    affine     : affine 4x4 di quello spazio. Serve per il volume del voxel e
        per esprimere i centroidi in millimetri.

    Restituisce:
      mask    : uint8 (0/1), stessa griglia dell'input, con le sole componenti
                sopravvissute ai filtri.
      lesions : lista di dict ORDINATA per volume decrescente. Ogni voce ha
                label, n_voxels, volume_mm3, com_voxel, com_world, prob_mean,
                prob_max. Lista vuota = nessun tumore rilevato, ed e' un esito
                legittimo che il chiamante deve saper gestire.
    """
    cfg = cfg or PostProcConfig()
    prob = np.asarray(prob_tumor)

    if voxel_volume_mm3 is None:
        if affine is None:
            raise ValueError("Serve 'affine' oppure 'voxel_volume_mm3'.")
        voxel_volume_mm3 = _voxel_volume_mm3(affine)

    mask = np.zeros(prob.shape, dtype=np.uint8)
    binary = prob >= cfg.prob_threshold
    if not binary.any():
        return mask, []

    lab, n = ndimage.label(binary)
    if n == 0:
        return mask, []

    ids = np.arange(1, n + 1)

    # Statistiche di TUTTE le componenti in poche passate sul volume. Estrarre
    # una maschera booleana per componente costerebbe un attraversamento
    # ciascuna: su 100+ componenti di rumore e' la differenza fra millisecondi
    # e decine di secondi.
    sizes = np.asarray(ndimage.sum(np.ones_like(lab, dtype=np.uint8), lab, ids))
    volumes = sizes * voxel_volume_mm3

    keep = volumes >= cfg.min_lesion_mm3
    if not keep.any():
        return mask, []

    kept_ids = ids[keep]
    coms = np.asarray(ndimage.center_of_mass(
        np.ones_like(lab, dtype=np.uint8), lab, kept_ids), dtype=np.float64)
    coms = coms.reshape(-1, 3)                      # center_of_mass appiattisce con 1 id
    p_mean = np.asarray(ndimage.mean(prob, lab, kept_ids), dtype=np.float64).ravel()
    p_max = np.asarray(ndimage.maximum(prob, lab, kept_ids), dtype=np.float64).ravel()

    if affine is not None:
        A = np.asarray(affine, dtype=np.float64)
        worlds = (A[:3, :3] @ coms.T + A[:3, 3:4]).T
    else:
        worlds = np.full_like(coms, np.nan)

    # --- Filtro polmonare opzionale, per COMPONENTE ---------------------------
    if cfg.lung_roi_filter and lung_mask is not None and lung_affine is not None:
        inside = _centroids_in_lung(worlds, lung_mask, lung_affine,
                                    cfg.lung_dilate_mm)
        if inside.any():
            kept_ids, coms, worlds = kept_ids[inside], coms[inside], worlds[inside]
            volumes = volumes[keep][inside]
            sizes_kept = sizes[keep][inside]
            p_mean, p_max = p_mean[inside], p_max[inside]
        else:
            # Nessuna componente nel polmone: si preferisce restituire vuoto
            # piuttosto che un bersaglio che si sa essere fuori sede.
            return mask, []
    else:
        volumes = volumes[keep]
        sizes_kept = sizes[keep]

    lesions = [
        {
            "label": int(l),
            "n_voxels": int(s),
            "volume_mm3": float(v),
            "volume_cm3": float(v) / 1000.0,
            "com_voxel": c.tolist(),
            "com_world": w.tolist(),
            "prob_mean": float(pm),
            "prob_max": float(px),
        }
        for l, s, v, c, w, pm, px in zip(
            kept_ids, sizes_kept, volumes, coms, worlds, p_mean, p_max)
    ]
    lesions.sort(key=lambda d: d["volume_mm3"], reverse=True)

    mask[np.isin(lab, kept_ids)] = 1
    return mask, lesions


def _centroids_in_lung(worlds, lung_mask, lung_affine, dilate_mm: float):
    """Quali centroidi (coordinate mondo) cadono nel polmone dilatato.

    La dilatazione si esprime in millimetri e si converte in iterazioni di
    voxel usando lo spacing EFFETTIVO della griglia polmonare: la stessa
    tolleranza in mm deve valere qualunque sia la risoluzione con cui la
    maschera e' stata calcolata.
    """
    lung = np.asarray(lung_mask, dtype=bool)
    spacing = np.linalg.norm(np.asarray(lung_affine, dtype=np.float64)[:3, :3],
                             axis=0).mean()
    iters = max(int(round(dilate_mm / max(spacing, 1e-6))), 1)
    # binary_dilation con iterazioni alte e' costoso; su una griglia a 2-3 mm
    # sono ~7-10 iterazioni, accettabile e fatto una volta per caso.
    grown = ndimage.binary_dilation(lung, iterations=iters)

    idx = _world_to_index(lung_affine, np.asarray(worlds).T)
    idx_r = np.rint(idx).astype(int)
    ok = np.ones(idx_r.shape[1], dtype=bool)
    for a in range(3):
        ok &= (idx_r[a] >= 0) & (idx_r[a] < grown.shape[a])
    out = np.zeros(idx_r.shape[1], dtype=bool)
    if ok.any():
        out[ok] = grown[tuple(idx_r[:, ok])]
    return out


def primary_lesion(lesions: list[dict]) -> dict | None:
    """La lesione che definisce l'isocentro per il planner mono-isocentro.

    E' la piu' voluminosa, NON il centroide dell'unione delle lesioni: la media
    fra due lesioni distinte cade nel tessuto sano che sta in mezzo, che e'
    esattamente il modo di produrre un piano con copertura nulla.
    """
    return lesions[0] if lesions else None


def primary_mask(mask, affine=None, min_lesion_mm3=None):
    """La maschera della sola lesione PRINCIPALE: il bersaglio del piano.

    Restituisce `(maschera_uint8, n_lesioni_significative)`.

    Perche' esiste: il piano di questo progetto e' MONO-ISOCENTRO. L'isocentro
    viene dalla lesione principale (vedi `primary_lesion`), e il bersaglio deve
    essere la stessa lesione. Passando al planner tutte le componenti si
    ottiene un piano che cerca di coprire noduli distinti da un unico centro:
    i pesi si disperdono, la copertura misurata e' una media su bersagli che
    nessun fascio raggiunge insieme, e i numeri non sono confrontabili con
    quelli di un piano vero.

    La regola era scritta a mano in QUATTRO punti diversi - la riga di comando
    della dosimetria, quella dell'esportazione del piano, l'esportazione della
    scena e il modello 3D - e in un quinto punto, la dosimetria lanciata dalla
    GUI, non era scritta affatto: il Tab 2 calcolava il piano su tutte e sei
    le lesioni di lung_001 (5.98 cm3) mentre la riga di comando ne trattava
    una (2.73 cm3). Da qui la funzione unica.

    `n_lesioni_significative` serve a DIRLO all'utente: una lesione scartata in
    silenzio e' una lesione non trattata di cui nessuno sa nulla.
    """
    m = np.asarray(mask) > 0
    if not m.any():
        return np.zeros(m.shape, dtype=np.uint8), 0
    lab, n = ndimage.label(m)
    if n <= 1:
        return m.astype(np.uint8), 1
    ids = np.arange(1, n + 1)
    sizes = np.asarray(ndimage.sum(np.ones_like(lab, dtype=np.uint8), lab, ids))
    primary = int(ids[int(np.argmax(sizes))])
    n_signif = n
    if affine is not None:
        lim = (PostProcConfig().min_lesion_mm3 if min_lesion_mm3 is None
               else float(min_lesion_mm3))
        n_signif = int((sizes * _voxel_volume_mm3(affine) >= lim).sum())
    return (lab == primary).astype(np.uint8), n_signif


def mask_from_lesions(shape, lab, lesions) -> np.ndarray:
    """Ricostruisce la maschera da un sottoinsieme di lesioni gia' calcolate.

    Utile allo sweep di soglia in evaluate.py, dove si rivaluta piu' volte lo
    stesso volume etichettato senza ri-eseguire ndimage.label.
    """
    mask = np.zeros(shape, dtype=np.uint8)
    if lesions:
        mask[np.isin(lab, [d["label"] for d in lesions])] = 1
    return mask
