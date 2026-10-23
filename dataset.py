# ============================================================================
#  dataset.py - DataLoader e pipeline di Transforms MONAI.
#
#  Responsabilita':
#    1. Trovare e accoppiare i file image/label (.nii.gz).
#    2. Definire il preprocessing deterministico e la data-augmentation.
#    3. Restituire i DataLoader di un FOLD della cross-validation.
#
#  TRE CAMBIAMENTI RISPETTO ALLA VERSIONE PRECEDENTE, e il perche':
#
#  (a) Lo split non si calcola piu' qui. Prima erano "i primi 20% della lista
#      ordinata": 12 volumi fissi, nessuna cross-validation. Ora la partizione
#      vive in splits.py, e' scritta su disco e viene letta verbatim, cosi' due
#      esperimenti diversi sono confrontabili caso per caso.
#
#  (b) FgBgToIndicesd nello stadio deterministico. RandCropByPosNegLabeld deve
#      sapere dove sono i voxel di tumore per campionare patch centrate su di
#      essi; se non gli si passano gli indici, li ricalcola A OGNI CHIAMATA
#      scandendo l'intero volume. Misurato: 63-80 ms per volume, contro 4-6 ms
#      quando gli indici sono precalcolati - 13 volte piu' veloce. Essendo
#      deterministico, il calcolo finisce nella cache di CacheDataset e si paga
#      una volta sola.
#
#  (c) L'augmentation "legacy" e' conservata ma non e' piu' il default
#      consigliato. RandRotate90d e i flip sugli assi antero-posteriore e
#      cranio-caudale generano toraci ruotati di 90 gradi o capovolti: pose che
#      non esistono in nessuna CT e che la rete non incontrera' mai in
#      inferenza. La capacita' spesa per imparare quelle invarianze e' sottratta
#      al problema vero. La variante 'anatomic' tiene il solo specchiamento
#      destra-sinistra, l'unico plausibile su un torace, e aggiunge le
#      perturbazioni che modellano la variabilita' REALE (rumore, dose,
#      spessore di strato, contrasto).
# ============================================================================
import os
import glob

import numpy as np

from monai.data import (CacheDataset, PersistentDataset, DataLoader,
                        list_data_collate)
from monai.utils import set_determinism
from monai.transforms import (
    Compose,
    MapTransform,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
    ScaleIntensityRanged,
    CropForegroundd,
    CastToTyped,
    FgBgToIndicesd,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandShiftIntensityd,
    RandRotate90d,
    RandAffined,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandScaleIntensityd,
    RandAdjustContrastd,
    RandSimulateLowResolutiond,
    SpatialCrop,
    EnsureTyped,
)

import config


# ---------------------------------------------------------------------------
#  1) Lettura ed accoppiamento dei file del dataset
# ---------------------------------------------------------------------------
def build_data_dicts(images_dir: str, labels_dir: str):
    """Crea la lista [{'image': path, 'label': path}, ...].

    MONAI lavora con dizionari: ogni Transform 'd' opera sulle chiavi indicate
    mantenendole sincronizzate.
    """
    images = sorted(glob.glob(os.path.join(images_dir, "*.nii.gz")))
    labels = sorted(glob.glob(os.path.join(labels_dir, "*.nii.gz")))

    # Il dataset MSD contiene file nascosti '._lung_xxx' (resource fork macOS).
    images = [p for p in images if not os.path.basename(p).startswith("._")]
    labels = [p for p in labels if not os.path.basename(p).startswith("._")]

    assert len(images) == len(labels), (
        f"Numero immagini ({len(images)}) != numero label ({len(labels)})")
    assert len(images) > 0, f"Nessun file trovato in {images_dir}"

    return [{"image": img, "label": lbl} for img, lbl in zip(images, labels)]


# ---------------------------------------------------------------------------
#  2) Ritaglio sulla ROI polmonare
# ---------------------------------------------------------------------------
class LungRoiCropd(MapTransform):
    """Ritaglia sul bounding box polmonare precalcolato da preprocess.py.

    Sostituisce CropForegroundd, che su questi volumi e' un no-op misurato: la
    soglia "> 0" dopo il rescale HU include anche il lettino, quindi il ritaglio
    toglie due voxel. Il box polmonare toglie invece cupola epatica, spalle,
    braccia e lettino - cioe' le regioni dove nascono i falsi positivi - e
    dimezza il numero di finestre della sliding window.

    Si delega a SpatialCrop (versione array) invece di affettare l'array a
    mano: SpatialCrop aggiorna l'affine della MetaTensor, e senza quel
    aggiornamento ogni coordinata fisica calcolata a valle sarebbe traslata
    della quantita' ritagliata.
    """

    def __init__(self, keys, margin_mm: float | None = None,
                 rois: dict | None = None, allow_missing_keys: bool = False):
        super().__init__(keys, allow_missing_keys)
        import preprocess
        self._pp = preprocess
        self.rois = rois if rois is not None else preprocess.load_rois()
        self.margin_mm = (preprocess.ROI_MARGIN_MM if margin_mm is None
                          else margin_mm)

    def __call__(self, data):
        d = dict(data)
        cid = d.get("id")
        if cid is None:
            cid = os.path.basename(str(d["image_meta_dict"]["filename_or_obj"])
                                   ).replace(".nii.gz", "")

        img = d[self.keys[0]]
        affine = np.asarray(getattr(img, "affine", np.eye(4)), dtype=np.float64)
        sl = self._pp.roi_slices(cid, tuple(img.shape[1:]), affine,
                                 rois=self.rois, margin_mm=self.margin_mm)
        if sl is None:
            # ROI non disponibile: si procede col volume intero. Piu' lento, ma
            # non si rischia di ritagliare via il bersaglio.
            return d

        cropper = SpatialCrop(roi_start=[s.start for s in sl],
                              roi_end=[s.stop for s in sl])
        for k in self.key_iterator(d):
            d[k] = cropper(d[k])
        return d


# ---------------------------------------------------------------------------
#  3) Transforms
# ---------------------------------------------------------------------------
def _deterministic(patch_size, use_lung_roi: bool, rois=None):
    """Parte deterministica: viene eseguita una volta e finisce in cache.

    ORDINE: prima la GEOMETRIA (orientamento, spacing), poi l'INTENSITA'
    (finestra HU), poi il ritaglio, infine gli indici fg/bg. Invertire i primi
    due produrrebbe spacing e intensita' incoerenti fra i casi.
    """
    steps = [
        LoadImaged(keys=["image", "label"], image_only=False),
        EnsureChannelFirstd(keys=["image", "label"]),
        # RAS: X, Y, Z hanno lo stesso significato anatomico per tutti i
        # pazienti. E' il presupposto di ogni cosa che segue, augmentation
        # inclusa (lo specchiamento "destra-sinistra" ha senso solo se si sa
        # quale asse e' destra-sinistra).
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=config.TARGET_SPACING,
                 mode=("bilinear", "nearest")),
        ScaleIntensityRanged(keys=["image"],
                             a_min=config.HU_MIN, a_max=config.HU_MAX,
                             b_min=0.0, b_max=1.0, clip=True),
    ]

    if use_lung_roi:
        steps.append(LungRoiCropd(keys=["image", "label"], rois=rois))
    else:
        steps.append(CropForegroundd(keys=["image", "label"], source_key="image",
                                     allow_smaller=True))

    steps += [
        # Label in uint8: dimezza l'occupazione della cache rispetto a float32
        # (con 50 volumi la cache passa da ~7.4 a ~5.5 GB) senza perdere nulla,
        # dato che i valori sono 0/1.
        CastToTyped(keys=["label"], dtype=np.uint8),
        # Indici fg/bg precalcolati: vedi nota (b) in testa al file.
        FgBgToIndicesd(keys=["label"], fg_postfix="_fg", bg_postfix="_bg",
                       image_key="image", image_threshold=0.0),
    ]
    return steps


def _crop_patches(patch_size, samples_per_volume):
    """Campionamento delle patch, bilanciato tumore/sfondo.

    pos=1, neg=1 -> meta' delle patch e' centrata su un voxel di tumore. Senza
    questo bilanciamento la rete vedrebbe quasi solo sfondo: il tumore occupa
    tipicamente meno dello 0.1% del volume.
    """
    return RandCropByPosNegLabeld(
        keys=["image", "label"], label_key="label",
        spatial_size=patch_size, pos=1, neg=1,
        num_samples=samples_per_volume,
        image_key="image", image_threshold=0,
        fg_indices_key="label_fg", bg_indices_key="label_bg",
    )


def _augment_legacy():
    """Augmentation storica. Conservata per poter RIPRODURRE la baseline.

    Contiene RandRotate90d e i flip su tutti e tre gli assi: vedi la nota (c)
    in testa al file sul perche' non sia una buona scelta su CT toraciche.
    """
    return [
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
        RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3),
        RandShiftIntensityd(keys=["image"], offsets=0.10, prob=0.5),
    ]


def _augment_anatomic(all_flip: bool = False):
    """Augmentation coerente con l'anatomia (ricetta nnU-Net).

    Dopo Orientationd(RAS) l'asse 0 e' destra-sinistra del paziente: e' l'unico
    specchiamento che produca un torace verosimile (esiste il situs inversus, e
    comunque la variabilita' destra/sinistra delle lesioni e' reale). Gli assi
    1 e 2 sono antero-posteriore e cranio-caudale: specchiarli produce un
    paziente prono o a testa in giu'.

    La variante all_flip specchia comunque tutti e tre gli assi: nnU-Net lo fa
    e funziona, quindi la cosa onesta e' misurare quale delle due sia migliore
    su questo dataset invece di deciderlo a priori.

    Le trasformazioni di intensita' modellano variabilita' REALE fra
    acquisizioni: rumore del rivelatore, dose, kernel di ricostruzione,
    spessore di strato, calibrazione del contrasto.
    """
    axes = [0, 1, 2] if all_flip else [0]
    aug = [RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=a)
           for a in axes]
    aug += [
        # Geometria PRIMA dell'intensita': una rotazione interpola l'immagine,
        # e interpolare dopo aver aggiunto rumore lo attenuerebbe, rendendo
        # l'augmentation di rumore meno efficace di quanto si crede.
        # rotate_range 0.26 rad = +-15 gradi: copre le differenze di
        # posizionamento sul lettino senza inventare pose impossibili.
        RandAffined(
            keys=["image", "label"], prob=0.30,
            rotate_range=(0.26, 0.26, 0.26),
            scale_range=(0.2, 0.2, 0.2),
            mode=("bilinear", "nearest"),
            padding_mode=("border", "zeros"),
        ),
        RandGaussianNoised(keys=["image"], prob=0.15, std=0.01),
        RandGaussianSmoothd(keys=["image"], prob=0.20,
                            sigma_x=(0.5, 1.0), sigma_y=(0.5, 1.0),
                            sigma_z=(0.5, 1.0)),
        RandScaleIntensityd(keys=["image"], prob=0.15, factors=0.25),
        RandAdjustContrastd(keys=["image"], prob=0.15, gamma=(0.7, 1.5)),
        # Simula una ricostruzione a risoluzione piu' bassa: e' cio' che rende
        # la rete robusta a spessori di strato diversi da quelli di training.
        RandSimulateLowResolutiond(keys=["image"], prob=0.15,
                                   zoom_range=(0.5, 1.0)),
    ]
    return aug


AUGMENTATIONS = {
    "legacy": _augment_legacy,
    "anatomic": lambda: _augment_anatomic(all_flip=False),
    "anatomic_allflip": lambda: _augment_anatomic(all_flip=True),
}


def get_train_transforms(patch_size=None, samples_per_volume=None,
                         augment: str = "legacy", use_lung_roi: bool = False,
                         rois=None):
    patch_size = tuple(patch_size or config.PATCH_SIZE)
    samples_per_volume = samples_per_volume or config.SAMPLES_PER_VOLUME
    if augment not in AUGMENTATIONS:
        raise ValueError(f"augment '{augment}' sconosciuta. "
                         f"Disponibili: {', '.join(AUGMENTATIONS)}")

    return Compose(
        _deterministic(patch_size, use_lung_roi, rois)
        + [_crop_patches(patch_size, samples_per_volume)]
        + AUGMENTATIONS[augment]()
        + [EnsureTyped(keys=["image", "label"])]
    )


def get_val_transforms(use_lung_roi: bool = False, rois=None):
    """Validazione: SOLO preprocessing deterministico.

    Niente crop random ne' augmentation: si valuta il volume intero con la
    sliding window, che e' la condizione dell'inferenza reale.
    """
    return Compose(
        _deterministic(config.PATCH_SIZE, use_lung_roi, rois)
        + [EnsureTyped(keys=["image", "label"])]
    )


# ---------------------------------------------------------------------------
#  Compatibilita' PersistentDataset (MONAI 1.4) con torch >= 2.6
# ---------------------------------------------------------------------------
class CompatPersistentDataset(PersistentDataset):
    """PersistentDataset di MONAI 1.4 reso compatibile con torch >= 2.6.

    IL PROBLEMA, incontrato e misurato su questo progetto.
    `PersistentDataset._cachecheck` scrive la cache con torch.save e la rilegge
    con `torch.load(hashfile)` SENZA passare weights_only. Da torch 2.6 il
    default di quel parametro e' passato da False a True, e il
    deserializzatore "weights only" rifiuta tutto cio' che non e' un tensore
    puro: prima `monai.data.meta_tensor.MetaTensor`, poi
    `numpy._core.multiarray._reconstruct`, e cosi' via per ogni oggetto nei
    metadati.

    L'effetto e' insidioso: la cache si SCRIVE correttamente, quindi la prima
    epoca gira - lentamente, mentre costruisce ~100 GB - e il training sembra
    avviato. E' la SECONDA epoca a morire, con un UnpicklingError che non
    nomina MONAI.

    PERCHE' NON LE ALTRE DUE STRADE:
      - `add_safe_globals` va inseguita un oggetto alla volta (provato: dopo
        MetaTensor salta fuori numpy._reconstruct) ed e' fragile al variare
        dei metadati.
      - riscrivere `_cachecheck` significherebbe duplicare anche la scrittura
        atomica su directory temporanea e i rimedi per i PermissionError di
        Windows, che non c'entrano nulla col problema.
    Si sostituisce quindi `torch.load` per la sola durata della chiamata al
    metodo originale, ripristinandolo subito. Il DataLoader qui gira con
    num_workers=0, quindi non c'e' concorrenza su quella sostituzione.

    I file deserializzati sono scritti da noi in questa stessa sessione: il
    controllo che weights_only introduce non protegge da nulla, in questo caso.
    """

    def _cachecheck(self, item_transformed):
        import torch

        original = torch.load

        def _load_trusting(*args, **kwargs):
            kwargs.setdefault("weights_only", False)
            return original(*args, **kwargs)

        torch.load = _load_trusting
        try:
            return super()._cachecheck(item_transformed)
        finally:
            torch.load = original


# ---------------------------------------------------------------------------
#  4) DataLoader di un fold
# ---------------------------------------------------------------------------
def get_fold_dataloaders(fold: int, patch_size=None, batch_size=None,
                         samples_per_volume=None, num_workers: int = 0,
                         use_lung_roi: bool = False, augment: str = "legacy",
                         cache_rate: float = 1.0, source: str = "msd"):
    """(train_loader, val_loader, train_files, val_files) per un fold.

    num_workers=0 di default, e non e' una rinuncia: dopo il precalcolo degli
    indici fg/bg la pipeline CPU costa ~64 ms per volume, cioe' pochi secondi
    per epoca contro decine di secondi di GPU. Su Windows i worker vengono
    creati con 'spawn' e devono ripicklare la CacheDataset a ogni epoca, il che
    costava ~15 s per epoca - piu' del lavoro utile. In piu' con un solo
    processo la ri-semina per epoca delle transform (train.py::epoch_seed) e'
    banalmente corretta, mentre coi worker andrebbe propagata a ciascuno.
    """
    import splits as splits_mod

    set_determinism(seed=config.SEED * 1000 + fold)

    if source != "msd":
        # PRETRAINING su dati esterni. Nessun validation set e nessuna
        # cross-validation: questi dataset servono a far VEDERE alla rete piu'
        # lesioni, non a misurare. La misura resta la CV su MSD, invariata,
        # cosi' che il confronto col modello precedente resti appaiato caso
        # per caso.
        #
        # I due dataset sono COMPLEMENTARI per dimensione della lesione, ed e'
        # il motivo per cui 'lidc+nsclc' esiste come opzione:
        #     LIDC   mediana 1.8 cm3  -> il regime dove il modello fallisce
        #     MSD    mediana 5.2 cm3  -> il bersaglio della valutazione
        #     NSCLC  mediana 78 cm3   -> lesioni grandi, segmentate male
        #                                (Dice 0.595 con isocentro 0.952)
        # Insieme coprono l'intero spettro invece di sbilanciarlo da un lato.
        #
        # Il ritaglio sulla ROI polmonare NON e' disponibile qui: le ROI di
        # preprocess.py sono calcolate sui soli casi MSD. Si usa il ritaglio
        # di foreground, e use_lung_roi viene ignorato.
        train_files = []
        if "lidc" in source:
            import lidc
            f = lidc.case_files()
            train_files += f
            print(f"[dataset] LIDC: {len(f)} casi")
        if "nsclc" in source:
            import nsclc
            f = nsclc.case_files()
            train_files += f
            print(f"[dataset] NSCLC: {len(f)} casi")
        if not train_files:
            raise ValueError(
                f"source '{source}' non ha prodotto alcun caso: convertire "
                f"prima i dataset (lidc.py --convert / nsclc.py --convert)")
        val_files = []
        use_lung_roi = False
        print(f"[dataset] pretraining su {len(train_files)} casi totali")
    elif fold < 0:
        # fold = -1: MODELLO FINALE, addestrato su tutti i casi etichettati.
        #
        # Non ha validation set, e va bene: il checkpoint consegnato e'
        # final.pth, cioe' l'ultima epoca con lo scheduler portato a zero, che
        # non e' selezionato su alcuna metrica. Un validation set servirebbe
        # solo se si scegliesse il "migliore", ed e' proprio quella scelta ad
        # aver prodotto lo storico 0.586 - un massimo di rumore.
        #
        # La stima ONESTA delle prestazioni di questo modello resta quella
        # della cross-validation: 5 modelli addestrati con la stessa ricetta su
        # 50 casi ciascuno e valutati sui 13 mai visti. Questo modello vede piu'
        # dati, quindi non sara' peggiore; ma il numero da riportare e' quello
        # della CV, non uno misurato sui dati di addestramento.
        tr, va = splits_mod.fold_files(0)   # 50 + 13 = tutti i casi etichettati
        train_files, val_files = tr + va, []
    else:
        train_files, val_files = splits_mod.fold_files(fold)
    rois = None
    if use_lung_roi:
        import preprocess
        rois = preprocess.load_rois()

    train_tf = get_train_transforms(patch_size, samples_per_volume,
                                    augment, use_lung_roi, rois)

    # MISURATO: dopo Spacingd a 1.5 mm un caso LIDC occupa ~46 MB e uno NSCLC
    # ~91 MB (le CT di pianificazione hanno piu' slice). Il pool combinato di
    # 639 casi vuole 46.8 GB, contro ~24 GB di RAM disponibili: CacheDataset
    # non ci sta. PersistentDataset mette lo stadio deterministico su DISCO
    # (494 GB liberi) e lo mappa in memoria.
    #
    # Ha anche un vantaggio operativo: la cache su disco SOPRAVVIVE a un
    # riavvio, quindi riprendere un pretraining interrotto non ripaga il
    # preprocessing di 639 volumi.
    if len(train_files) > 300:
        cache_dir = os.path.join(config.OUTPUT_DIR, "cache", f"{source}_train")
        os.makedirs(cache_dir, exist_ok=True)
        print(f"[dataset] {len(train_files)} casi: cache su DISCO in {cache_dir}")
        train_ds = CompatPersistentDataset(data=train_files, transform=train_tf,
                                           cache_dir=cache_dir)
    else:
        train_ds = CacheDataset(
            data=train_files,
            transform=train_tf,
            cache_rate=cache_rate,
            num_workers=max(num_workers, 1),      # solo per riempire la cache
        )
    val_ds = CacheDataset(
        data=val_files,
        transform=get_val_transforms(use_lung_roi, rois),
        cache_rate=cache_rate,
        num_workers=max(num_workers, 1),
    ) if val_files else None

    train_loader = DataLoader(
        train_ds, batch_size=batch_size or config.TRAIN_BATCH_SIZE,
        shuffle=True, num_workers=num_workers,
        collate_fn=list_data_collate, pin_memory=True,
    )
    # batch_size=1 in validazione: ogni volume ha dimensioni diverse e si valuta
    # intero con la sliding window.
    val_loader = (DataLoader(val_ds, batch_size=1, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
                  if val_ds is not None else None)

    return train_loader, val_loader, train_files, val_files


def get_dataloaders(val_split: float = config.VAL_SPLIT):
    """Split storico (primi 20% della lista ordinata). DEPRECATO.

    Conservato solo per riprodurre esattamente la baseline legacy; ogni nuovo
    esperimento deve usare get_fold_dataloaders. Vedi la nota (a) in testa al
    file.
    """
    set_determinism(seed=config.SEED)
    data_dicts = build_data_dicts(config.IMAGES_DIR, config.LABELS_DIR)
    n_val = int(len(data_dicts) * val_split)
    val_files, train_files = data_dicts[:n_val], data_dicts[n_val:]
    print(f"[dataset] DEPRECATO - split storico | totali: {len(data_dicts)} | "
          f"train: {len(train_files)} | val: {len(val_files)}")

    train_ds = CacheDataset(data=train_files, transform=get_train_transforms(),
                            cache_rate=1.0, num_workers=config.NUM_WORKERS)
    val_ds = CacheDataset(data=val_files, transform=get_val_transforms(),
                          cache_rate=1.0, num_workers=config.NUM_WORKERS)
    train_loader = DataLoader(train_ds, batch_size=config.TRAIN_BATCH_SIZE,
                              shuffle=True, num_workers=0,
                              collate_fn=list_data_collate, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False,
                            num_workers=0, pin_memory=True)
    return train_loader, val_loader, train_files, val_files


# Test rapido: `python dataset.py --fold 0` stampa la forma di un batch.
if __name__ == "__main__":
    import argparse
    import time

    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--augment", default="anatomic", choices=list(AUGMENTATIONS))
    ap.add_argument("--lung-roi", action="store_true")
    ap.add_argument("--patch", type=int, nargs=3, default=list(config.PATCH_SIZE))
    args = ap.parse_args()

    t0 = time.time()
    tl, vl, tf, vf = get_fold_dataloaders(
        args.fold, patch_size=tuple(args.patch), augment=args.augment,
        use_lung_roi=args.lung_roi)
    print(f"[dataset] cache pronta in {time.time()-t0:.1f}s | "
          f"{len(tf)} train, {len(vf)} val")

    t0 = time.time()
    batch = next(iter(tl))
    print(f"[dataset] primo batch in {time.time()-t0:.2f}s")
    print("  image:", tuple(batch["image"].shape), batch["image"].dtype)
    print("  label:", tuple(batch["label"].shape), batch["label"].dtype)
    print(f"  intensita': {float(batch['image'].min()):.3f} -> "
          f"{float(batch['image'].max()):.3f}")
    frac = float((batch["label"] > 0).float().mean())
    print(f"  frazione di voxel tumorali nel batch: {frac*100:.3f}%")

    vb = next(iter(vl))
    print("  volume di validazione:", tuple(vb["image"].shape))
