# ============================================================================
#  config.py - Configurazione centralizzata della pipeline.
#  Tenere qui tutti gli iperparametri/percorsi rende gli altri moduli "puliti".
# ============================================================================
import os

# ---------------------------------------------------------------------------
#  PERCORSI DATASET (MSD Task06 Lung)
# ---------------------------------------------------------------------------
# Adatta DATA_ROOT alla cartella dove hai scompattato il dataset.
# Struttura attesa:
#   <DATA_ROOT>/imagesTr/lung_xxx.nii.gz   (volumi CT)
#   <DATA_ROOT>/labelsTr/lung_xxx.nii.gz   (maschere ground-truth)
DATA_ROOT = os.environ.get(
    "MSD_LUNG_ROOT",
    r"C:\Users\ilgra\Documents\Progetto Medical\Task06_Lung\Task06_Lung",
)
IMAGES_DIR = os.path.join(DATA_ROOT, "imagesTr")
LABELS_DIR = os.path.join(DATA_ROOT, "labelsTr")

# Cartella dove salvare checkpoint del modello e output.
OUTPUT_DIR = os.path.join(
    r"C:\Users\ilgra\Documents\Progetto Medical", "outputs"
)
# Modello di produzione. Pre-addestrato su LIDC-IDRI (221 serie, noduli con
# malignita' >= 3.5) + NSCLC-Radiomics (415 serie, GTV delineati da oncologi
# radioterapisti), poi addestrato su TUTTI i 63 casi MSD con augmentation
# anatomica, 250 epoche. Checkpoint = ultima epoca: nessuna selezione su
# validazione, quindi nessun rischio di eleggere un picco di rumore come fece
# lo storico "best_dice 0.586".
#
# I due dataset esterni sono COMPLEMENTARI per dimensione della lesione, e si
# vede nei risultati per terzile: LIDC (mediana 1.8 cm3) ha alzato il terzile
# piccolo, NSCLC (mediana 39 cm3) quello grande.
#
# Prestazioni ONESTE: cross-validation a 5 fold con la stessa ricetta, vedi
# outputs/eval/finetune_all_oof/summary.json. Questo modello ha visto tutti i
# casi MSD in addestramento, quindi qualunque numero misurato su di essi
# sarebbe privo di significato.
#     Dice 0.623 (mediana 0.724) | sensibilita' lesionale 0.877 | 0.75 FP/scan
#     isocentro dentro il tumore 85.7%
#     con il cancello: 94.1% corretto sull'81.0% dei casi accettati
#
# PROGRESSIONE MISURATA (tutte CV a 5 fold, confronti appaiati su 63 casi):
#     legacy                 Dice 0.451 | isocentro 0.730 | FP/scan 0.95
#     + aug anatomica        Dice 0.470 | isocentro 0.810 | FP/scan 0.67
#     + pretrain LIDC        Dice 0.586 | isocentro 0.825 | FP/scan 0.70
#     + pretrain LIDC+NSCLC  Dice 0.623 | isocentro 0.857 | FP/scan 0.75
#
# Il guadagno di Dice e' schiacciante (0.470 -> 0.623, p = 1.1e-07) e la
# mediana 0.724 entra nella fascia di concordanza fra radiologi umani
# (0.70-0.85). L'ISOCENTRO invece non e' mai risultato significativo passo per
# passo (McNemar p = 0.125, 0.69): e' migliorato di 8 punti in totale ma
# restano 7 casi sbagliati da OGNI modello provato. Quelli non sono casi mal
# segmentati, sono casi dove la lesione primaria scelta e' un falso positivo:
# piu' dati migliorano il contorno di cio' che la rete trova, non la scelta
# dell'oggetto. Per quelli serve il cancello, o un ordinamento delle lesioni
# che usi priori anatomici invece del solo volume.
BEST_MODEL_PATH = os.path.join(OUTPUT_DIR, "model_final_all.pth")

#: Esperimento di cross-validation su cui si basa la VALIDAZIONE dell'intero
#: applicativo: mappe di incertezza (uncertainty.py / run_uncertainty.py),
#: calibrazione del margine adattivo (robust_target.py) e screening sui 63
#: casi (archivio/script/demo_screening.py). Ogni TAC e' segmentata dal modello del fold che
#: non l'ha vista. DEVE corrispondere alla ricetta di BEST_MODEL_PATH,
#: altrimenti i numeri dell'applicativo descrivono un modello diverso da
#: quello in produzione - e' esattamente cio' che era successo fra il 14 e il
#: 15 ottobre, quando il report dell'applicativo era ancora quello del modello
#: 'aug_anatomic' mentre la GUI usava gia' 'finetune_all'.
#: Prima questo nome era cablato in tre file diversi.
UNC_EXP = "finetune_all"

#: Modelli precedenti, conservati per la tabella di progressione nella
#: relazione. Non usarli in produzione.
PREV_MODEL_PATH = os.path.join(OUTPUT_DIR, "model_final_lidc.pth")
PREV2_MODEL_PATH = os.path.join(OUTPUT_DIR, "model_final_anatomic.pth")

#: Modello storico, conservato per il confronto nella relazione. NON usarlo:
#: e' addestrato con l'augmentation che ruota il torace di 90 gradi ed e'
#: selezionato sul massimo di una curva di validazione rumorosa.
LEGACY_MODEL_PATH = os.path.join(OUTPUT_DIR, "best_model_dice.pth")

# ===========================================================================
#  CONFIGURAZIONE DEL PLANNER (unica fonte di verita')
#
#  Prima questi valori vivevano nelle firme di compute_plan() E di
#  optimize_plan(): due default per lo stesso parametro, con quello a monte che
#  ombreggiava silenziosamente l'altro. Ora esistono solo qui.
#
#  Le funzioni ricevono un'istanza di PlannerConfig; non hanno piu' default
#  propri per questi parametri, cosi' e' impossibile che divergano.
# ===========================================================================
from dataclasses import dataclass, field, asdict


@dataclass
class PlannerConfig:
    """Parametri dell'inverse planning. Ogni campo e' definito una volta sola."""

    # --- Griglia di calcolo -------------------------------------------------
    # 1.5 mm invece di 2.5: le metriche di conformita' su una lesione di pochi
    # cm3 erano quantizzate da una griglia troppo grossolana (172 voxel di
    # bersaglio). Costa ~2x per fascio, recuperato con la parallelizzazione.
    spacing_mm: float = 1.5

    # --- Stadio 2: decomposizione del bersaglio ----------------------------
    # Molti piu' punti focali: e' cio' che permette a un piano di scolpire una
    # forma irregolare invece di approssimarla con poche sfere grandi.
    max_targets: int = 40
    target_coverage: float = 0.995
    min_sphere_radius_mm: float = 1.5

    # --- Stadio 3: candidati ------------------------------------------------
    n_dense_nodes: int = 420      # direzioni generate prima del filtraggio
    n_keep_nodes: int = 96        # nodi conservati dopo i vincoli geometrici
    keep_frac: float = 0.6        # quota di nodi per mira, i piu' superficiali
    cone_choices: int = 2         # diametri di cono offerti per ogni mira
    cone_margin_mm: float = 3.0

    # Candidati conservati per il solve globale. Il costo del NNLS cresce piu'
    # che linearmente col numero di variabili: con 4640 candidati il solve da
    # solo superava i 200 s. Lo screening tiene i migliori PER PUNTO DI MIRA,
    # quindi riduce il costo senza lasciare scoperta nessuna zona del bersaglio.
    max_candidates: int = 1600

    # --- Obiettivo di ottimizzazione ---------------------------------------
    max_tumor_pts: int = 3000     # punti bersaglio, campionati a risoluzione CT
    max_ring_vox: int = 5000      # voxel della corona peritumorale
    ring_mm: float = 15.0
    edge_boost: float = 10.0      # quanto pesano i punti di bordo

    # --- Termini del funzionale --------------------------------------------
    w_ring: float = 0.20          # penalita' sullo spill -> conformita'

    # Ridge L2 sui pesi. MISURA su lung_001, a parita' di tutto il resto:
    #   ridge 0.05 -> 127 fasci, CI 0.862, osso 0.30 cm
    #   ridge 0.20 -> 112 fasci, CI 0.858, osso 0.42 cm
    #   ridge 0.50 -> 133 fasci, CI 0.856, osso 0.42 cm
    # Il ridge NON e' la leva che moltiplica i fasci: a questo punto di lavoro
    # le differenze stanno nel rumore. Il numero di fasci e' salito da ~27 a
    # ~120 grazie alla griglia a 1.5 mm, alle 40 mire e al budget a 150.
    # Valore tenuto a 0.20 come richiesto; 0.05 misura leggermente meglio.
    lam_ridge: float = 0.20
    lam_path: float = 0.010       # penalita' sul percorso radiologico
    lam_bone: float = 0.80        # penalita' sull'osso attraversato

    # --- Selezione e rifinitura --------------------------------------------
    # Budget alto e ridge alto lavorano in coppia: il ridge impedisce al
    # solutore di concentrare tutto su pochi fasci intensi, il budget gli lascia
    # lo spazio per distribuirli. E' la firma di un piano CyberKnife reale:
    # molti fasci incrociati a bassa intensita'.
    beam_budget: int = 150
    min_weight_frac: float = 0.01
    # Nessun fascio oltre questa frazione delle unita' monitor totali (0 =
    # nessun limite). Robustezza al moto: vedi planner.cap_weights.
    max_beam_frac: float = 0.05
    sparsify_rounds: int = 3
    repair_rounds: int = 2

    # --- Clinica ------------------------------------------------------------
    prescription_pct: float = 80.0

    # --- Vincoli geometrici (zone proibite) --------------------------------
    couch_block_deg: float = 40.0
    max_axial_tilt_deg: float = 45.0
    # Quanto SOTTO il dorso passa il piano del lettino. Vale 0 perche' il
    # paziente sul lettino ci appoggia: il piano coincide con la sua schiena.
    # Con i 40 mm di prima restava una fessura da cui 4 fasci su 112 partivano
    # da sotto il lettino - dosimetricamente validi, ma inerogabili: nessun
    # braccio puo' attraversare il piano su cui giace il paziente.
    couch_clearance_mm: float = 0.0
    max_arm_mm: float = 5.0

    # --- Calcolo ------------------------------------------------------------
    # Processi e non thread: il ray-tracing sta in map_coordinates e
    # fftconvolve, CPU-bound, che trattengono il GIL.
    #
    # 6 non e' un valore arbitrario ne' il numero di core: e' l'ottimo MISURATO
    # su questa macchina (Ryzen 7 9700X, 16 thread). Oltre i 6 processi lo
    # speedup smette di crescere - 4:2.8x, 6:3.2x, 8:3.1x, 12:2.8x, 15:2.9x -
    # perche' il collo di bottiglia e' la banda di memoria: ogni processo legge
    # a caso da una griglia di densita' di ~45 MB. Aggiungere core non aggiunge
    # banda. 0 = automatico, min(6, core-1).
    n_workers: int = 6
    parallel_min_beams: int = 400   # sotto questa soglia l'avvio del pool non ripaga

    def replace(self, **kw):
        """Copia con alcuni campi sostituiti (per test e ablazioni)."""
        return PlannerConfig(**{**asdict(self), **kw})


#: Istanza di riferimento usata da tutto il progetto.
PLANNER = PlannerConfig()


# ===========================================================================
#  CONFIGURAZIONE DEL POST-PROCESSING (unica fonte di verita')
#
#  Stessa logica di PlannerConfig: prima la regola "tieni la componente piu'
#  grande" era scritta tre volte (inference.py, inference_worker.py,
#  tumor_3d.py) e non veniva applicata affatto in validazione, quindi il Dice
#  riportato non descriveva la pipeline realmente eseguita. Ora la definizione
#  sta in postproc.py e i parametri stanno qui.
#
#  I due valori sotto NON sono default da lasciare per inerzia: sono il punto
#  di lavoro del rilevatore, e vanno tarati sulle predizioni out-of-fold con
#  `python evaluate.py --sweep`, che riscrive questi numeri con quelli
#  misurati.
# ===========================================================================
from postproc import PostProcConfig

#: Punto di lavoro corrente. 0.50 equivale all'argmax usato finora (cioe' e' il
#: comportamento storico, non una scelta ottimizzata); 100 mm3 = sfera di
#: 5.8 mm = la piu' piccola lesione realmente annotata nel dataset.
POSTPROC = PostProcConfig(
    prob_threshold=0.50,
    min_lesion_mm3=100.0,
    lung_roi_filter=False,
    lung_dilate_mm=20.0,
)


# ===========================================================================
#  CANCELLO DI CONFIDENZA
#
#  Misurato sui 63 casi out-of-fold: nel 27% dei casi l'isocentro predetto cade
#  FUORI dal tumore, e l'errore e' bimodale - o entro pochi mm, o dell'ordine
#  dei 100 mm. Senza cancello quei bersagli arrivano al planner senza alcun
#  segnale, e il piano che ne discende irradia tessuto sano.
#
#  Nessuna taratura della soglia di probabilita' risolve il problema (54
#  combinazioni provate: isocentro corretto da 0.730 a 0.746 nel migliore dei
#  casi). Il cancello invece, validato in modo ANNIDATO sui 5 fold:
#      senza: bersaglio sempre consegnato, corretto nel 73.0% dei casi
#      con  : consegnato nel 74.6% dei casi, corretto nell'89.4%
#  Rifiuta 16 casi su 63, di cui 12 davvero sbagliati e 4 buoni.
#
#  Le soglie qui sotto sono la mediana di quelle scelte sui fold di training
#  (4 fold su 5 hanno scelto 0.8-1.0 cm3), non un valore tarato sull'insieme
#  completo: quest'ultimo sarebbe di nuovo un numero scelto per essere alto.
# ===========================================================================
from postproc import GateConfig

# Soglie ritarate sul modello pre-addestrato su LIDC. La validazione annidata
# (soglia scelta sui 4 fold di training, misurata sul quinto) ha scelto
# volume >= 1.0 cm3 E prob media >= 0.80 in 4 fold su 5: e' la combinazione
# piu' stabile vista su questo progetto.
#
# Su un modello migliore il cancello RENDE DI PIU', non di meno:
#     modello            consegnati   corretti   rifiutati (sbagliati/buoni)
#     anatomic            50/63         90.0%     13  (7 / 6)
#     + pretrain LIDC     57/63         89.5%      6  (5 / 1)
# Stessa affidabilita', ma il bersaglio viene consegnato nel 90.5% dei casi
# invece del 79.4%, e dei 6 rifiutati 5 erano davvero sbagliati. In pratica
# converte 5 bersagli errati in rifiuti espliciti al prezzo di uno buono.
#: Ritarate sul modello LIDC+NSCLC. La validazione annidata ha scelto il solo
#: criterio di VOLUME in tutti e 5 i fold (0.8, 2.0, 1.0, 1.0, 2.0 cm3),
#: abbandonando la soglia di confidenza che serviva al modello precedente: con
#: una rete piu' accurata la probabilita' media non discrimina piu', il volume
#: si'. Qui si usa la mediana delle cinque scelte.
#:
#: Su un modello migliore il cancello rende ancora di piu':
#:     modello              consegnati   corretti   p90 del centroide
#:     + LIDC                57/63        89.5%        42.3 mm
#:     + LIDC+NSCLC          51/63        94.1%         9.4 mm
#: Quel p90 e' il numero che conta per il planner: il 90% dei bersagli
#: consegnati cade entro 9.4 mm dal vero, contro 42.3 mm prima. Fra i casi
#: accettati i fallimenti catastrofici sono quasi scomparsi.
GATE = GateConfig(
    min_volume_cm3=1.0,
    min_prob_mean=0.0,
)

#: Se False il cancello viene solo SEGNALATO, non applicato: la maschera e il
#: bersaglio restano disponibili e l'interfaccia mostra un avviso. E' il
#: comportamento voluto in un simulatore didattico, dove nascondere il
#: risultato impedirebbe di studiare i casi di fallimento.
GATE_BLOCKS_TARGET = False

# ---------------------------------------------------------------------------
#  PARAMETRI IMMAGINE / PREPROCESSING
# ---------------------------------------------------------------------------
# Risoluzione isotropica target (mm) a cui ricampionare ogni volume.
# 1.5mm in plane e 2.0mm lungo z è un buon compromesso per il polmone:
# riduce il numero di slice (meno VRAM) mantenendo dettaglio sufficiente.
TARGET_SPACING = (1.5, 1.5, 2.0)

# Finestra Hounsfield Units (HU) per il polmone/tessuti molli.
# -1000 HU = aria, +400 HU circa = tessuti densi/ossa: tutto fuori range viene
# clampato. Questo elimina rumore irrilevante e normalizza l'intensità.
HU_MIN = -1000.0
HU_MAX = 400.0

# Dimensione delle patch 3D estratte durante il training (VRAM-friendly).
PATCH_SIZE = (96, 96, 96)

# Numero di patch campionate per ogni volume ad ogni iterazione.
SAMPLES_PER_VOLUME = 4

# ---------------------------------------------------------------------------
#  PARAMETRI MODELLO
# ---------------------------------------------------------------------------
IN_CHANNELS = 1      # CT = singolo canale (scala di grigi / HU)
OUT_CHANNELS = 2     # 0 = background, 1 = tumore

# ---------------------------------------------------------------------------
#  PARAMETRI TRAINING
# ---------------------------------------------------------------------------
MAX_EPOCHS = 300
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
TRAIN_BATCH_SIZE = 2     # con patch 96^3 e AMP sta comodo nei 12 GB della 5070
VAL_INTERVAL = 2         # valida ogni N epoche
VAL_SPLIT = 0.2          # 20% dei volumi come validation set
NUM_WORKERS = 4          # solo per riempire la CacheDataset; i DataLoader
                         # usano num_workers=0 (vedi dataset.py)
SEED = 42

# Dimensione finestra per la sliding-window inference in validazione/test.
SW_ROI_SIZE = PATCH_SIZE
SW_BATCH_SIZE = 4
SW_OVERLAP = 0.5

# Fusione delle finestre sovrapposte. NON e' un dettaglio cosmetico: era una
# DISCREPANZA fra validazione e produzione. Il vecchio train.py validava con
# sliding_window_inference() senza passare mode=, cioe' con blending
# 'constant'; inference.py e inference_worker.py usavano invece 'gaussian'.
# Misurato sullo stesso checkpoint e sugli stessi casi di validazione:
#
#     caso        GT    constant   gaussian   dice_cst   dice_gau
#     lung_001    289        449       1348      0.686      0.341
#     lung_003    306        465        862      0.750      0.384
#     lung_006    920        299       3605      0.386      0.007
#     lung_010   2649       3512       3693      0.806      0.794
#     lung_014   7086       9279      10250      0.842      0.810
#
# Sulle lesioni grandi le due modalita' coincidono quasi; sulle piccole il
# gaussian gonfia il volume predetto fino a 12 volte, e con esso i falsi
# positivi. Il Dice storico di 0.586 e' stato misurato in 'constant': il
# sistema che girava nella GUI non era quello misurato.
SW_BLEND_MODE = "constant"
