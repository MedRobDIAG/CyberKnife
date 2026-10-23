# ============================================================================
#  organs.py - Segmentazione degli organi toracici dalla CT e mesh 3D.
#
#  METODO: soglie in unita' Hounsfield + morfologia. NESSUNA rete neurale.
#  Le densita' dei tessuti sono cosi' separate in HU che una soglia basta:
#
#      aria        ~ -1000 HU        polmone     ~ -900 / -500 HU
#      grasso      ~  -100 HU        tessuti molli ~ 0 / +100 HU
#      osso spongioso ~ +300 HU      osso corticale > +700 HU
#
#  LIMITE DICHIARATO: la sola densita' NON distingue organi di tessuto molle
#  fra loro (cuore, fegato, esofago, grossi vasi stanno tutti intorno a
#  0-60 HU). Qui il "mediastino" e' quindi ricostruito per criterio GEOMETRICO
#  (il tessuto molle racchiuso fra i due polmoni), non per densita': e' una
#  approssimazione utile a localizzare cuore e grossi vasi, non un contorno
#  clinico. Per organi di tessuto molle separati servirebbe un modello
#  addestrato (es. TotalSegmentator) o un atlante deformabile.
# ============================================================================
import numpy as np
from scipy import ndimage
from skimage import measure
import trimesh

from dosimetry import build_hu_grid, apply_affine_fast


# ---------------------------------------------------------------------------
#  Definizione degli organi: soglie, colore, resa grafica.
#  L'ordine e' anche l'ordine di disegno: dal piu' interno al piu' esterno,
#  cosi' le strutture profonde restano visibili per trasparenza.
# ---------------------------------------------------------------------------
#  Il budget di facce del CORPO e' il piu' alto: e' la superficie piu' estesa e
#  articolata (spalle e braccia). Con una decimazione troppo spinta le braccia
#  si appiattiscono e le ossa che contengono finiscono per sporgere dalla pelle.
ORGAN_STYLE = [
    # key            nome                       colore     alpha  facce  z  sigma
    ("airways",      "Vie aeree (trachea)",     "#ffd166", 0.85,  2000,  3, 0.6),
    ("mediastinum",  "Mediastino (cuore)",      "#e05263", 0.40,  3000,  4, 1.0),
    ("bones",        "Scheletro",               "#e8e2d0", 0.55, 10000,  5, 0.9),
    ("lung_r",       "Polmone destro",          "#6fd3e8", 0.18,  3000,  6, 0.9),
    ("lung_l",       "Polmone sinistro",        "#63e0b8", 0.18,  3000,  6, 0.9),
    ("body",         "Corpo (pelle)",           "#7fb6d6", 0.10,  7000,  8, 0.9),
]
ORGAN_INFO = {k: dict(name=n, color=c, alpha=a, faces=f, zorder=z, sigma=s)
              for k, n, c, a, f, z, s in ORGAN_STYLE}

BODY_HU = -320       # soglia corpo / aria esterna
AIR_HU = -400        # soglia aria interna (polmoni, vie aeree)

# Soglia OSSO a ISTERESI (due livelli).
#   BONE_CORE_HU : solo osso indiscutibile -> definisce i "semi"
#   BONE_GROW_HU : soglia permissiva, ma accettata SOLO se connessa a un seme
# Una soglia unica non funziona: abbassandola si raccolgono migliaia di
# frammenti di rumore (vasi, calcificazioni, bordi in volume parziale),
# alzandola si amputano le coste, che sono sottili. Con l'isteresi il rumore
# isolato non ha semi e viene scartato, mentre le coste restano intere perche'
# collegate ai corpi vertebrali.
BONE_CORE_HU = 350
BONE_GROW_HU = 200
BONE_MIN_ML = 1.0    # volume minimo di una componente ossea accettabile

#: Quanto vicino deve stare una sacca d'aria al polmone perche' sia parenchima
#: e non aria addominale (mm). Misurato su tre pazienti: il parenchima
#: sfuggito alla separazione sta a 3.5 mm, l'aria dello stomaco fra 20 e 103.
LUNG_ADJ_MM = 8.0

#: Volume oltre il quale il "mediastino" non e' piu' un mediastino (cm3).
#: Cuore, grossi vasi e grasso mediastinico stanno sotto il litro.
MEDIASTINUM_MAX_ML = 1200.0


# ---------------------------------------------------------------------------
def _world_axis(affine, world_index=0):
    """Trova l'asse della griglia meglio allineato a un asse del mondo.

    Serve per il criterio "compreso fra i due polmoni", che ha senso lungo
    l'asse sinistra-destra del PAZIENTE (X in convenzione RAS). Gli assi della
    griglia non coincidono necessariamente con quelli del mondo: su una CT
    orientata diversamente, dare per scontato l'asse 0 produrrebbe un
    mediastino sbagliato.
    """
    lin = np.asarray(affine, dtype=np.float64)[:3, :3]
    return int(np.argmax(np.abs(lin[world_index, :])))


def _largest_components(mask, n_keep=1, min_frac=0.0):
    """Mantiene le n componenti connesse piu' grandi."""
    lab, n = ndimage.label(mask)
    if n == 0:
        return [np.zeros_like(mask, dtype=bool)]
    sizes = ndimage.sum(np.ones_like(mask, dtype=np.uint8), lab,
                        index=range(1, n + 1))
    order = np.argsort(sizes)[::-1]
    out = []
    biggest = sizes[order[0]]
    for k in order[:n_keep]:
        if sizes[k] >= min_frac * biggest:
            out.append(lab == (k + 1))
    return out or [np.zeros_like(mask, dtype=bool)]


def _split_lungs(air, ax):
    """Separa i due polmoni, che spesso formano un'unica componente connessa.

    Sull'immagine i polmoni comunicano attraverso il recesso anteriore o
    tramite trachea e carena, quindi una semplice etichettatura ne trova uno
    solo. Si procede in due tempi:
      1. si erode progressivamente finche' la maschera non si spezza in due
         parti confrontabili (le connessioni sono sottili, i polmoni no);
      2. se l'erosione non basta, si taglia sul piano sagittale mediano,
         individuato dal minimo del profilo di area lungo l'asse destra-sinistra
         (li' stanno mediastino e colonna, dove il polmone e' assente).
    In entrambi i casi si ridilata dentro la maschera d'aria originale, per
    recuperare il bordo perso con l'erosione.
    """
    for iters in (2, 3, 4, 5, 6):
        opened = ndimage.binary_opening(air, iterations=iters)
        comps = _largest_components(opened, n_keep=2, min_frac=0.25)
        if len(comps) == 2:
            return [ndimage.binary_dilation(c, iterations=iters) & air
                    for c in comps]

    # Fallback geometrico: taglio sul piano sagittale mediano.
    lung = _largest_components(ndimage.binary_opening(air, iterations=2), 1)[0]
    if not lung.any():
        return [lung]
    axes = tuple(i for i in range(3) if i != ax)
    profile = lung.sum(axis=axes)                 # area per sezione sagittale
    c = int(round(ndimage.center_of_mass(lung)[ax]))
    span = max(len(profile) // 8, 3)
    lo, hi = max(c - span, 1), min(c + span, len(profile) - 1)
    cut = lo + int(np.argmin(profile[lo:hi])) if hi > lo else c

    idx = np.arange(lung.shape[ax]).reshape(
        [-1 if i == ax else 1 for i in range(3)])
    left = lung & (idx < cut)
    right = lung & (idx >= cut)
    return [m for m in (left, right) if m.any()]


#: Apertura morfologica usata per staccare le braccia dal tronco (mm). Il
#: collegamento fra braccio e parete toracica, dove il braccio appoggia, e'
#: largo pochi millimetri; il braccio e il tronco sono molto piu' spessi.
ARM_BREAK_MM = 14.0


def _segment_arms(body, spacing_mm, core=None, min_area_ml=0.4):
    """Separa le braccia dal tronco, sezione assiale per sezione assiale.

    LA VERSIONE PRECEDENTE NON FUNZIONAVA: etichettava ogni sezione e chiamava
    "braccio" tutto cio' che non era la componente maggiore. Ma nelle fette
    alte il braccio APPOGGIA sulla parete toracica, i due si toccano, e
    l'etichettatura ne trova uno solo. Misurato su lung_001: 224 voxel di
    braccia su 2.37 milioni di corpo, cioe' il vincolo era spento e nel piano
    `scartati_braccia` risultava 0.
    Non e' solo estetica: un fascio che entra attraversando un braccio
    deposita dose su un arto sano e perde intensita' inutilmente.

    Ora si procede in tre passi per ogni sezione:
      1. APERTURA morfologica di ~14 mm: il ponte sottile fra braccio e torace
         si rompe, mentre i corpi veri - spessi molti centimetri - resistono;
      2. si riconosce il TRONCO come la parte che contiene polmoni e coste,
         non come la piu' grande: alle spalle il braccio puo' essere piu'
         voluminoso della sezione toracica;
      3. ogni voxel del corpo viene assegnato alla parte APERTA piu' vicina,
         cosi' la pelle e il grasso persi con l'apertura tornano al proprio
         corpo invece di restare orfani.
    """
    it = max(int(round(ARM_BREAK_MM / spacing_mm)), 1)
    area_vox = min_area_ml * 1000.0 / (spacing_mm ** 3)
    arms = np.zeros_like(body)
    for k in range(body.shape[2]):
        sl = body[:, :, k]
        if not sl.any():
            continue
        opened = ndimage.binary_opening(sl, iterations=it)
        lab, n = ndimage.label(opened)
        if n <= 1:
            continue
        ids = np.arange(1, n + 1)
        sizes = np.asarray(ndimage.sum(np.ones_like(sl, np.uint8), lab, ids))
        if core is not None and core[:, :, k].any():
            ov = np.asarray(ndimage.sum(core[:, :, k].astype(np.uint8), lab,
                                        ids))
            trunk = ids[int(np.argmax(ov))] if ov.max() > 0 else \
                ids[int(np.argmax(sizes))]
        else:
            trunk = ids[int(np.argmax(sizes))]
        extra = ids[(ids != trunk) & (sizes >= area_vox)]
        if not len(extra):
            continue
        # Assegnazione al piu' vicino: la trasformata di distanza restituisce
        # gli INDICI del punto piu' vicino appartenente a una parte aperta,
        # e da quelli si legge l'etichetta.
        near = ndimage.distance_transform_edt(~opened, return_distances=False,
                                              return_indices=True)
        arms[:, :, k] = sl & np.isin(lab[tuple(near)], extra)
    return arms


def above_lung_floor(lungs, az, affine, margin_mm=10.0, spacing_mm=2.0,
                     min_depth_mm=40.0, window_mm=60.0):
    """Cio' che NON sta sotto il diaframma: il limite caudale del mediastino.

    Il diaframma e' una CUPOLA: in una sezione assiale vicina alle basi
    polmonari ai lati c'e' polmone e al centro c'e' il fegato, quindi nessun
    criterio "per fetta" li separa - misurato, limitare il mediastino alle
    fette con abbondante polmone toglieva 2 cm3 su 2609.

    Si stima quindi una SUPERFICIE: in ogni colonna (x, y) il voxel di polmone
    piu' caudale segna il diaframma. Con due accortezze, entrambe nate da una
    misura:

    1. Solo le colonne che contengono polmone per almeno `min_depth_mm`
       valgono come riferimento. Prendendo tutte le colonne, quelle in cui il
       polmone c'e' solo in alto - sopra il cuore - davano un "pavimento"
       altissimo e tagliavano il cuore: il mediastino di lung_001 crollava da
       662 a 10 cm3.
    2. Il pavimento viene poi esteso alle colonne senza riferimento prendendo
       il MINIMO su un intorno: fra due stime si sceglie la piu' caudale,
       cioe' quella che taglia meno. Il cuore appoggia sul diaframma nella sua
       parte piu' bassa, e un'estensione ottimista lo amputerebbe.
    """
    lungs = np.asarray(lungs, dtype=bool)
    if not lungs.any():
        return np.ones(lungs.shape, dtype=bool)
    n = lungs.shape[az]
    idx = np.broadcast_to(np.arange(n).reshape(
        [-1 if i == az else 1 for i in range(3)]), lungs.shape)
    # Verso caudale: se +Z del mondo cresce con l'indice, i piedi stanno agli
    # indici bassi.
    cranial_up = (np.sign(np.asarray(affine, dtype=np.float64)[2, az])
                  or 1.0) > 0
    depth = lungs.sum(axis=az)
    valid = depth >= max(min_depth_mm / max(spacing_mm, 1e-6), 1.0)
    win = max(int(round(window_mm / max(spacing_mm, 1e-6))), 1)
    marg = margin_mm / max(spacing_mm, 1e-6)

    if cranial_up:
        raw = np.where(lungs, idx, n + 1).min(axis=az)
        raw = np.where(valid, raw, n + 1)
        floor = ndimage.minimum_filter(raw, size=win, mode="nearest")
        return idx >= np.expand_dims(floor - marg, axis=az)
    raw = np.where(lungs, idx, -1).max(axis=az)
    raw = np.where(valid, raw, -1)
    ceil = ndimage.maximum_filter(raw, size=win, mode="nearest")
    return idx <= np.expand_dims(ceil + marg, axis=az)


def anterior_to_spine(body, bones, ax, ay, spacing_mm, affine,
                      margin_mm=4.0, band_mm=35.0):
    """Meta' del volume che sta DAVANTI alla faccia anteriore della colonna.

    Serve a delimitare il mediastino: senza questo vincolo inghiottiva i
    solchi paravertebrali e il canale vertebrale, e in 3D ne usciva una massa
    bitorzoluta invece del profilo di cuore e grossi vasi.

    La colonna e' l'osso centrale POSTERIORE. La sola centralita' non basta:
    nella fascia mediana c'e' anche lo STERNO, che e' anteriore, e prendendolo
    come riferimento il limite taglia via il cuore - misurato, il mediastino
    crollava da 1076 a 26 cm3.

    E' una funzione pubblica perche' la verifica in test_organ_mesh.py deve
    controllare LO STESSO criterio: riscrivendolo a mano si finisce per
    controllarne un altro (ed e' successo).
    """
    ant_sign = np.sign(np.asarray(affine, dtype=np.float64)[1, ay]) or 1.0
    com = ndimage.center_of_mass(body)
    idx_ax = np.broadcast_to(np.arange(body.shape[ax]).reshape(
        [-1 if i == ax else 1 for i in range(3)]), body.shape)
    idx_ay = np.broadcast_to(np.arange(body.shape[ay]).reshape(
        [-1 if i == ay else 1 for i in range(3)]), body.shape)

    behind = (idx_ay < com[ay]) if ant_sign > 0 else (idx_ay > com[ay])
    spine = bones & (np.abs(idx_ax - com[ax]) <= band_mm / spacing_mm) & behind

    front = np.ones_like(body)
    az_m = 3 - ax - ay
    margin = margin_mm / spacing_mm
    # Nella sezione 2D gli assi rimasti conservano il loro ordine: qui si
    # ricava quale dei due e' l'asse antero-posteriore.
    ay_in_plane = [i for i in range(3) if i != az_m].index(ay)
    for k in range(body.shape[az_m]):
        sl = tuple(slice(None) if i != az_m else k for i in range(3))
        sp = spine[sl]
        if not sp.any():
            continue
        pos = np.nonzero(sp)[ay_in_plane]
        # "Anteriore" e' il verso in cui cresce Y del mondo: l'indice di
        # griglia puo' essere concorde o discorde, e il segno dell'affine lo
        # dice.
        if ant_sign > 0:
            front[sl] = idx_ay[sl] > (pos.max() - margin)
        else:
            front[sl] = idx_ay[sl] < (pos.min() + margin)
    return front


#: Sezione plausibile della trachea (cm2): sotto e' rumore, sopra e' polmone.
AIRWAY_SEED_CM2 = (0.3, 8.0)
#: Meta' larghezza della fascia centrale in cui cercare la trachea (mm).
AIRWAY_BAND_MM = 40.0
#: Tetto di sicurezza al volume dell'albero aereo ricostruito (cm3).
AIRWAY_MAX_ML = 120.0


def _segment_airways(air, body, ax, az, spacing_mm, affine,
                     max_ml=AIRWAY_MAX_ML, status_cb=None):
    """Trachea e bronchi principali: accrescimento con CONTROLLO DI ESPLOSIONE.

    Due versioni precedenti non reggevano su pazienti diversi:

    1. "l'aria che resta togliendo i polmoni" si spezzava vicino alla carena,
       dove il dilatato dei polmoni la mangia (7 componenti invece di una);
    2. l'accrescimento dentro quell'aria residua TRACIMAVA quando i polmoni
       erano sotto-segmentati: su lung_003 restavano 1729 cm3 di parenchima fra
       i candidati e le vie aeree uscivano da 998 cm3 invece di ~40. Il calibro
       non salva: a 2 mm il parenchima ha canali d'aria sottili in ogni
       direzione, e anche limitando a 4 mm restavano 860 cm3.

    Qui si cresce dentro TUTTA l'aria interna - senza bisogno della maschera
    dei polmoni, che quindi non deve piu' essere calcolata prima - e si guarda
    QUANTO il volume cresce a ogni passo. Misurato su lung_001 e lung_003:
    dentro l'albero bronchiale la crescita e' di 0.2-0.6 cm3 per passo e
    stabile; appena si entra nel parenchima accelera (1.7, 3, 7, 15 cm3). Si
    ferma quando la crescita supera tre volte la propria linea di base, che e'
    il momento dello sfondamento.

    Il seme e' la prima sezione centrale con area PLAUSIBILE per una trachea:
    pretenderla solo "piccola" non bastava - su lung_004 il seme finiva su un
    singolo voxel di rumore (0.04 cm2) e non cresceva nulla.
    """
    def _st(m):
        if status_cb:
            status_cb(m)

    # La griglia di segmentazione e' isotropa per costruzione (build_hu_grid).
    vox_ml = (spacing_mm ** 3) / 1000.0
    if not air.any():
        return np.zeros_like(air)

    # Direzione craniale: il segno con cui l'indice di griglia si traduce in Z
    # del mondo (RAS: +Z verso la testa).
    sign = np.sign(np.asarray(affine, dtype=np.float64)[2, az]) or 1.0
    n_sl = air.shape[az]
    order = range(n_sl - 1, -1, -1) if sign > 0 else range(n_sl)

    # Fascia centrale: la trachea corre sulla linea mediana.
    idx_ax = np.arange(air.shape[ax]).reshape(
        [-1 if i == ax else 1 for i in range(3)])
    central = np.broadcast_to(
        np.abs(idx_ax - ndimage.center_of_mass(body)[ax])
        <= AIRWAY_BAND_MM / spacing_mm, air.shape)

    lo_vox = AIRWAY_SEED_CM2[0] * 100.0 / (spacing_mm ** 2)
    hi_vox = AIRWAY_SEED_CM2[1] * 100.0 / (spacing_mm ** 2)
    seed = np.zeros_like(air)
    for k in order:
        sl = tuple(slice(None) if i != az else k for i in range(3))
        s = air[sl] & central[sl]
        if not s.any():
            continue
        lab, n = ndimage.label(s)
        sizes = np.asarray(ndimage.sum(np.ones_like(s, np.uint8), lab,
                                       np.arange(1, n + 1)))
        ok = np.nonzero((sizes >= lo_vox) & (sizes <= hi_vox))[0]
        if not len(ok):
            continue
        seed[sl] = (lab == int(ok[int(np.argmax(sizes[ok]))]) + 1)
        break
    if not seed.any():
        _st("ATTENZIONE: trachea non trovata, vie aeree non ricostruite.")
        return np.zeros_like(air)

    cur = seed
    deltas, base = [], None
    for _ in range(200):
        new = ndimage.binary_dilation(cur, iterations=1) & air
        d = float(new.sum() - cur.sum()) * vox_ml
        if d <= 0.0:
            cur = new
            break
        # LINEA DI BASE CONGELATA, misurata nel primo tratto - quello dentro la
        # trachea, dove la crescita e' stabile - e poi mai piu' aggiornata.
        # Con una mediana MOBILE il controllo non scattava: la tracimazione
        # accelera gradualmente (0.6, 1.7, 3.2, 7, 15 cm3 per passo), la
        # mediana mobile la inseguiva e ogni singolo passo sembrava normale.
        # Misurato: tutti i casi finivano contro il tetto di 120 cm3.
        if base is None and len(deltas) >= 12:
            base = float(np.median(deltas[4:12]))
        if base is not None and d > max(3.0 * base, 1.5):
            break                           # sfondamento nel parenchima
        cur = new
        deltas.append(d)
        if float(cur.sum()) * vox_ml > max_ml:
            _st(f"ATTENZIONE: vie aeree al tetto di {max_ml:.0f} cm3: "
                f"probabile sfondamento non rilevato.")
            break
    return cur


# ---------------------------------------------------------------------------
def segment_organs(ct_data, affine, spacing_mm=2.0, status_cb=None):
    """Segmenta gli organi toracici. Restituisce (masks, grid_affine, volumi).

    masks: dict chiave -> maschera booleana sulla griglia isotropica.
    """
    def _st(m):
        if status_cb:
            status_cb(m)

    _st("Ricampionamento CT per la segmentazione...")
    hu, grid_affine = build_hu_grid(ct_data, affine, spacing_mm)

    # --- 1) CORPO: tutto cio' che e' piu' denso dell'aria esterna ------------
    _st("Estrazione del profilo corporeo...")
    body = hu > BODY_HU
    # Riempimento SLICE PER SLICE lungo l'asse assiale: in 2D i polmoni sono
    # racchiusi dalla parete toracica e vengono chiusi. Un fill 3D non li
    # chiuderebbe, perche' l'aria polmonare comunica con l'esterno tramite la
    # trachea.
    for k in range(body.shape[2]):
        body[:, :, k] = ndimage.binary_fill_holes(body[:, :, k])
    body = _largest_components(body, 1)[0]        # scarta lettino e artefatti

    # --- 2) ARIA INTERNA: polmoni + vie aeree --------------------------------
    _st("Separazione dei polmoni...")
    air = (hu < AIR_HU) & body
    ax = _world_axis(affine, 0)          # asse griglia ~ sinistra-destra
    ay = _world_axis(affine, 1)          # asse griglia ~ posteriore-anteriore
    lungs = _split_lungs(air, ax)

    # Destro / sinistro dal centroide lungo l'asse X del paziente (RAS: +X =
    # destra del paziente). Il segno dell'affine dice come l'indice di griglia
    # si traduce in X del mondo: puo' essere invertito.
    sign = np.sign(np.asarray(affine)[0, ax]) or 1.0
    lung_r = lung_l = np.zeros_like(body)
    if len(lungs) == 2:
        c0 = ndimage.center_of_mass(lungs[0])[ax] * sign
        c1 = ndimage.center_of_mass(lungs[1])[ax] * sign
        lung_r, lung_l = (lungs[0], lungs[1]) if c0 > c1 else (lungs[1], lungs[0])
    elif len(lungs) == 1:
        lung_r = lungs[0]

    lungs_any = lung_r | lung_l

    # --- 3) VIE AEREE: aria interna non polmonare ----------------------------
    _st("Ricostruzione delle vie aeree...")
    vox_ml = abs(np.linalg.det(grid_affine[:3, :3])) / 1000.0
    az = 3 - ax - ay                     # asse assiale (testa-piedi)
    # Accrescimento da un seme sulla trachea, invece di "l'aria che resta":
    # vicino alla carena il residuo si spezzava in frammenti.
    # Non serve piu' la maschera dei polmoni: l'accrescimento si ferma da
    # solo quando sfonda nel parenchima. Su lung_003 i polmoni erano
    # sotto-segmentati e la vecchia versione, che dipendeva da loro, dava
    # 998 cm3 di "vie aeree".
    airways = _segment_airways(air, body, ax, az, spacing_mm, affine,
                               status_cb=status_cb)

    # Le sacche d'aria che non sono ne' polmone ne' via aerea sono parenchima
    # sfuggito alla separazione: tornano al polmone piu' vicino, invece di
    # restare fuori da tutto.
    leftover = air & ~ndimage.binary_dilation(lungs_any, iterations=2) \
        & ~ndimage.binary_dilation(airways, iterations=1)

    lab, n = ndimage.label(ndimage.binary_opening(leftover, iterations=1))
    if n:
        ids = np.arange(1, n + 1)
        # Dimensioni e distanze di TUTTE le componenti in poche passate:
        # estrarre una maschera booleana per componente costerebbe un
        # attraversamento del volume ciascuna.
        sizes = np.asarray(ndimage.sum(np.ones_like(lab, np.uint8), lab, ids))
        vols = sizes * vox_ml

        # ADIACENZA, non vicinanza del centroide. L'ARIA ADDOMINALE - bolla
        # gastrica e gas intestinale - finiva assegnata al polmone: la vecchia
        # regola guardava solo quale polmone avesse il centroide piu' vicino
        # lungo l'asse destra-sinistra, e una bolla di 556 cm3 sotto il
        # diaframma la superava senza problemi. Da li' nasceva anche un
        # "mediastino" da 2609 cm3: tutto il fegato risultava racchiuso fra i
        # due "polmoni".
        #
        # Misurato su lung_001, lung_003 e lung_006, la separazione e' netta:
        # il parenchima sfuggito alla separazione sta a 3.5 mm dal polmone -
        # e' il guscio tolto dalla dilatazione - mentre l'aria addominale sta
        # fra 20 e 103 mm. Una soglia di adiacenza le distingue senza dover
        # riconoscere il diaframma.
        vox_mm = np.abs(np.asarray(grid_affine)[:3, :3]).sum(axis=0)
        d_l = (ndimage.distance_transform_edt(~lung_l, sampling=vox_mm)
               if lung_l.any() else np.full(lab.shape, np.inf, np.float32))
        d_r = (ndimage.distance_transform_edt(~lung_r, sampling=vox_mm)
               if lung_r.any() else np.full(lab.shape, np.inf, np.float32))
        dmin_l = np.asarray(ndimage.minimum(d_l, lab, ids))
        dmin_r = np.asarray(ndimage.minimum(d_r, lab, ids))
        adiacenti = (vols >= 0.5) & (np.minimum(dmin_l, dmin_r) <= LUNG_ADJ_MM)

        give_l = ids[adiacenti & (dmin_l <= dmin_r)]
        give_r = ids[adiacenti & (dmin_l > dmin_r)]
        if len(give_l):
            lung_l = lung_l | np.isin(lab, give_l)
        if len(give_r):
            lung_r = lung_r | np.isin(lab, give_r)
        scartati = float(vols[(vols >= 0.5) & ~adiacenti].sum())
        if scartati > 20.0:
            _st(f"{scartati:.0f} cm3 di aria non polmonare (stomaco, "
                f"intestino) esclusi dai polmoni")
    lungs_any = lung_r | lung_l

    # --- 4) SCHELETRO --------------------------------------------------------
    _st("Estrazione dello scheletro...")
    # Soglia a isteresi: i semi sono osso certo (> BONE_CORE_HU), poi si cresce
    # nella regione permissiva (> BONE_GROW_HU) ma solo per contiguita'. Il
    # rumore isolato non tocca alcun seme e viene escluso in partenza.
    core = (hu > BONE_CORE_HU) & body
    grow = (hu > BONE_GROW_HU) & body
    bones = ndimage.binary_propagation(core, mask=grow)
    # NIENTE apertura morfologica qui: eroderebbe 2 mm, cioe' l'intero spessore
    # corticale di una costa, spezzando gli archi costali in frammenti.
    #
    # La soglia cattura solo l'osso CORTICALE: il midollo, meno denso, resta
    # escluso e ogni costa diventa un tubo cavo a doppia parete, cioe' una
    # superficie sottile che la decimazione riduce a schegge. Riempire il
    # midollo dimezza la superficie e da' una mesh molto piu' pulita.
    #
    # Il riempimento va fatto in 3D e NON sezione per sezione: in una sezione
    # assiale coste, sterno e colonna formano un anello quasi chiuso, e un
    # riempimento 2D annegherebbe l'intera cavita' toracica nello "scheletro".
    # In 3D la cavita' resta collegata all'esterno dagli spazi intercostali,
    # quindi non viene toccata, mentre il midollo - realmente chiuso dentro
    # l'osso - viene riempito.
    bones = ndimage.binary_closing(bones, iterations=1)
    filled = ndimage.binary_fill_holes(bones)
    holes = filled & ~bones
    lab_h, n_h = ndimage.label(holes)
    if n_h:
        h_ids = np.arange(1, n_h + 1)
        h_vol = np.asarray(ndimage.sum(np.ones_like(lab_h, np.uint8), lab_h,
                                       h_ids)) * vox_ml
        # Rete di sicurezza: si riempiono solo le cavita' piccole (midollo),
        # mai volumi grandi che sarebbero errori di segmentazione.
        keep_h = h_ids[h_vol < 5.0]
        if len(keep_h):
            bones |= np.isin(lab_h, keep_h)
    # Le coste sono strutture sottili e separate: si tengono tutte quelle sopra
    # una soglia di volume, non solo la componente maggiore.
    lab, n = ndimage.label(bones)
    if n:
        ids = np.arange(1, n + 1)
        sizes = np.asarray(ndimage.sum(np.ones_like(bones, np.uint8), lab, ids))
        bones = np.isin(lab, ids[sizes * vox_ml >= BONE_MIN_ML])

    # --- 5) MEDIASTINO (approssimazione geometrica) --------------------------
    _st("Delimitazione del mediastino...")
    soft = body & ~lungs_any & ~airways & ~bones & (hu > -200) & (hu < 300)
    # "Racchiuso dai polmoni": deve esistere polmone da entrambi i lati, sia
    # lungo l'asse destra-sinistra sia lungo quello antero-posteriore. E' un
    # criterio GEOMETRICO, non di densita': la densita' non distinguerebbe il
    # cuore dalla parete toracica. Il vincolo antero-posteriore serve a
    # escludere sterno, muscoli dorsali e tessuto sottocutaneo, che altrimenti
    # rientrerebbero fra i due polmoni.
    def _enclosed(mask_, axis):
        before = np.cumsum(mask_, axis=axis) > 0
        flip = [slice(None)] * 3
        flip[axis] = slice(None, None, -1)
        after = np.cumsum(mask_[tuple(flip)], axis=axis)[tuple(flip)] > 0
        return before & after

    # Il secondo vincolo esclude sterno, muscoli dorsali e sottocute: sono
    # tessuto molle che sta comunque "fra i due polmoni" lungo X, ma e'
    # superficiale, mentre il mediastino e' profondo. Si richiede quindi che il
    # voxel disti almeno ~12 mm dalla superficie corporea.
    deep = ndimage.binary_erosion(
        body, iterations=max(int(round(12.0 / spacing_mm)), 1))

    # TERZO vincolo: DAVANTI ALLA COLONNA. Senza di esso il mediastino
    # inghiottiva i solchi paravertebrali e il canale vertebrale - nelle
    # sezioni il contorno avvolgeva il corpo vertebrale - e in 3D ne usciva
    # una massa bitorzoluta invece del profilo di cuore e grossi vasi. Il
    # limite si ricava dalla colonna stessa, sezione per sezione: la faccia
    # anteriore del corpo vertebrale.
    front = anterior_to_spine(body, bones, ax, ay, spacing_mm, affine)
    # QUARTO vincolo: sopra il diaframma. Senza, il fegato e la milza - che
    # nelle sezioni delle basi polmonari stanno davvero fra i due polmoni -
    # finivano nel mediastino, che arrivava a 2598 cm3 su lung_003.
    sopra = above_lung_floor(lungs_any, 3 - ax - ay, affine,
                             spacing_mm=spacing_mm)

    mediastinum = soft & _enclosed(lungs_any, ax) & deep & front & sopra
    mediastinum = ndimage.binary_opening(mediastinum, iterations=2)
    mediastinum = _largest_components(mediastinum, 1)[0]
    # PLAUSIBILITA', dichiarata invece che taciuta. Cuore, grossi vasi e
    # grasso mediastinico stanno sotto il litro; oltre, la maschera ha
    # inglobato tessuto sotto-diaframmatico - fegato o milza - e il motivo e'
    # un limite di metodo, non un parametro da correggere: alla quota del
    # diaframma cuore e fegato sono un UNICO blocco di tessuto molle
    # contiguo, con la stessa densita'. Tre criteri geometrici provati (limite
    # per fetta sull'area polmonare, pavimento del polmone per colonna,
    # pavimento esteso con finestra) non li separano, e non possono: servirebbe
    # un modello addestrato o un atlante. Qui si avvisa, perche' da questa
    # maschera esce anche la DOSE AL MEDIASTINO mostrata nel pannello.
    v_med = float(mediastinum.sum()) * vox_ml
    if v_med > MEDIASTINUM_MAX_ML:
        _st(f"ATTENZIONE: mediastino {v_med:.0f} cm3, oltre i "
            f"{MEDIASTINUM_MAX_ML:.0f} plausibili: include tessuto "
            f"sotto-diaframmatico. La dose al mediastino e' sovrastimata "
            f"come volume e va letta con cautela.")
    # Le cavita' interne (bronchi, vasi esclusi dalla soglia) diventerebbero
    # superfici dentro il cuore: si turano, l'organo e' un volume pieno.
    mediastinum = ndimage.binary_fill_holes(mediastinum)

    # --- 6) BRACCIA (per i vincoli di fattibilita' dei fasci) ---------------
    _st("Individuazione delle braccia...")
    # Polmoni e coste dicono dove passa il TRONCO: senza questo riferimento,
    # alle spalle la sezione del braccio puo' essere piu' grande di quella
    # del torace e verrebbe scambiata per tronco.
    arms = _segment_arms(body, spacing_mm, core=(lungs_any | bones))

    masks = {
        "body": body,
        "trunk": body & ~arms,
        "arms": arms,
        "lung_r": lung_r,
        "lung_l": lung_l,
        "airways": airways,
        "bones": bones,
        "mediastinum": mediastinum,
    }
    volumes = {k: float(v.sum()) * abs(np.linalg.det(grid_affine[:3, :3])) / 1000.0
               for k, v in masks.items()}
    return masks, grid_affine, volumes


# ---------------------------------------------------------------------------
def build_organ_meshes(masks, grid_affine, smoothing=8, ct_data=None,
                       ct_affine=None, faces_scale=1.0, progress_cb=None,
                       status_cb=None):
    """Costruisce una mesh per organo, in coordinate fisiche (mm).

    Con la TAC (`ct_data`) si usa la catena di `organ_mesh`: isosuperficie sui
    valori Hounsfield, griglia di ricostruzione piu' fine della griglia di
    segmentazione, e una ricetta misurata per ogni struttura. Misurato su
    lung_001, contro la stessa superficie di riferimento: fedelta' dello
    scheletro da 1.62 a 0.60 mm, delle vie aeree da 1.17 a 0.42 mm, schegge
    da decimazione dal 4.4% all'1.0% dell'area.

    Senza la TAC resta la catena storica (maschera sfumata e marching cubes a
    0.5), che e' peggiore ma non richiede la densita': serve a chi chiama la
    funzione con le sole maschere.

    Restituisce una lista di dict pronti per il rendering, gia' ordinati dal
    piu' interno al piu' esterno.
    """
    if ct_data is not None and ct_affine is not None:
        import organ_mesh as omsh
        return omsh.build_all(masks, grid_affine, ct_data, ct_affine,
                              faces_scale=faces_scale,
                              progress_cb=progress_cb, status_cb=status_cb)
    if status_cb:
        status_cb("Nessuna TAC disponibile: ricostruzione dalla sola "
                  "maschera, meno fedele (vedi organ_mesh.py).")
    out = []
    keys = [k for k, *_ in ORGAN_STYLE if k in masks and masks[k].any()]
    for i, key in enumerate(keys):
        info = ORGAN_INFO[key]
        if status_cb:
            status_cb(f"Mesh: {info['name']}...")
        m = masks[key]
        # Pre-smoothing gaussiano PRIMA del Marching Cubes.
        #
        # Su una maschera binaria l'isosuperficie segue gli spigoli dei voxel e
        # risulta a gradini: e' la causa principale dell'aspetto frastagliato.
        # Sfumando la maschera in un campo continuo, l'isolivello 0.5 cade fra i
        # voxel e la superficie diventa liscia con precisione sub-voxel.
        # Lisciare la MESH a posteriori non e' equivalente: partirebbe da una
        # geometria gia' degradata e i gradini si trasformerebbero in ondulazioni.
        field = ndimage.gaussian_filter(m.astype(np.float32),
                                        sigma=info.get("sigma", 0.9))
        try:
            verts, faces, _, _ = measure.marching_cubes(field, level=0.5,
                                                        step_size=1)
        except (ValueError, RuntimeError):
            continue

        verts_mm = apply_affine_fast(np.asarray(grid_affine, dtype=np.float64),
                                     verts.T).T
        mesh = trimesh.Trimesh(vertices=verts_mm, faces=faces, process=True)

        # Decimazione: matplotlib 3D rallenta molto oltre ~20k poligoni totali,
        # quindi ogni organo ha il proprio budget di facce.
        if len(mesh.faces) > info["faces"]:
            try:
                mesh = mesh.simplify_quadric_decimation(face_count=info["faces"])
            except Exception:
                pass
        if smoothing:
            trimesh.smoothing.filter_taubin(mesh, iterations=smoothing)

        out.append({"key": key, "name": info["name"], "color": info["color"],
                    "alpha": info["alpha"], "zorder": info["zorder"],
                    "mesh": mesh})
        if progress_cb:
            progress_cb(i + 1, len(keys))
    return out


# ---------------------------------------------------------------------------
def organ_dose_stats(masks, organ_affine, dose_pct, dose_affine):
    """Dose ricevuta da ciascun organo (Dmax e Dmedia, in % di prescrizione).

    Gli organi a rischio si giudicano cosi': non basta che il tumore sia
    coperto, occorre sapere quanto ricevono polmone sano, midollo e cuore.
    """
    stats = {}
    inv = np.linalg.inv(np.asarray(dose_affine, dtype=np.float64))
    for key, m in masks.items():
        # Non tutte le maschere sono organi da mostrare: 'trunk' e 'arms'
        # servono ai vincoli geometrici del planner e non hanno una voce di
        # stile. Si accetta comunque la chiave grezza come etichetta, invece di
        # sollevare un'eccezione su una maschera legittima.
        if key not in ORGAN_INFO or not m.any():
            continue
        idx = np.array(np.nonzero(m), dtype=np.float64)
        # Sottocampionamento: per una statistica di dose non serve ogni voxel.
        if idx.shape[1] > 60000:
            sel = np.linspace(0, idx.shape[1] - 1, 60000).astype(int)
            idx = idx[:, sel]
        world = apply_affine_fast(np.asarray(organ_affine, dtype=np.float64), idx)
        d = ndimage.map_coordinates(dose_pct, apply_affine_fast(inv, world),
                                    order=1, mode="constant", cval=0.0)
        stats[key] = {"name": ORGAN_INFO[key]["name"],
                      "d_max_pct": float(d.max()),
                      "d_mean_pct": float(d.mean())}
    return stats


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    import time
    import nibabel as nib

    ap = argparse.ArgumentParser(description="Segmentazione organi toracici.")
    ap.add_argument("--ct", required=True)
    ap.add_argument("--spacing", type=float, default=2.0)
    args = ap.parse_args()

    img = nib.load(args.ct)
    ct = np.asanyarray(img.dataobj, dtype=np.float32)
    t0 = time.time()
    masks, gaff, vols = segment_organs(ct, img.affine, args.spacing,
                                       status_cb=lambda m: print("  ", m))
    print(f"Segmentazione in {time.time() - t0:.1f}s")
    for k, v in vols.items():
        print(f"  {ORGAN_INFO[k]['name']:<26}{v:8.1f} cm3")
