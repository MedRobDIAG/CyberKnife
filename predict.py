# ============================================================================
#  predict.py - Inferenza -> mappa di PROBABILITA' nello spazio CT originale.
#
#  PERCHE' SALVARE LE PROBABILITA' E NON LE MASCHERE
#  La soglia (oggi fissa a 0.5, implicita nell'argmax) e il volume minimo di
#  una lesione sono i due parametri che scambiano sensibilita' contro falsi
#  positivi: sono LA manopola della detection. Se si salvano le maschere gia'
#  binarizzate, provare una soglia diversa costa un'altra passata di rete su
#  tutti i casi. Salvando le probabilita', l'intero sweep soglia x volume
#  diventa aritmetica su array e costa secondi invece di ore di GPU.
#
#  FORMATO DI SALVATAGGIO
#  Una mappa float16 a piena griglia pesa ~131 MB per caso (8 GB per il
#  dataset): insostenibile. Si salva invece il solo bounding box dove
#  prob > 0.05, quantizzato a uint8 (errore 1/255 = 0.004, due ordini di
#  grandezza sotto la sensibilita' di qualunque soglia che useremo), piu'
#  offset, forma piena e affine per ricostruire. Risultato: pochi MB per caso.
# ============================================================================
import os

import numpy as np
import torch
from scipy import ndimage

from monai.data import Dataset, DataLoader, decollate_batch
from monai.inferers import SlidingWindowInferer
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd,
    ScaleIntensityRanged, EnsureTyped, Invertd,
)

import config


#: Sotto questa probabilita' un voxel non e' salvato. Deve restare BEN SOTTO la
#: soglia piu' bassa che lo sweep esplorera' (0.10), altrimenti si taglierebbe
#: proprio l'informazione che serve a decidere la soglia.
STORE_FLOOR = 0.05


# ---------------------------------------------------------------------------
#  Transforms (identiche alla validazione: e' l'unico modo di avere numeri
#  confrontabili fra training e deployment)
# ---------------------------------------------------------------------------
def get_pre_transforms(target_spacing=None, hu_min=None, hu_max=None):
    return Compose([
        LoadImaged(keys=["image"], image_only=False),
        EnsureChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes="RAS"),
        Spacingd(keys=["image"], pixdim=target_spacing or config.TARGET_SPACING,
                 mode="bilinear"),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=config.HU_MIN if hu_min is None else hu_min,
            a_max=config.HU_MAX if hu_max is None else hu_max,
            b_min=0.0, b_max=1.0, clip=True,
        ),
        EnsureTyped(keys=["image"]),
    ])


def get_prob_post_transforms(pre_transforms):
    """Invertd verso lo spazio CT ORIGINALE.

    NIENTE Activationsd qui: la softmax e' gia' stata applicata da
    tta_softmax/ensemble_softmax, che DEVONO farlo prima di mediare (mediare i
    logit di piu' modelli o di piu' specchiamenti non equivale a mediarne le
    probabilita'). Applicarla una seconda volta comprimerebbe le probabilita'
    verso 0.5 e renderebbe insensata ogni soglia tarata dopo.

    nearest_interp=False, al contrario della maschera: qui si riporta indietro
    un campo CONTINUO di probabilita', e l'interpolazione nearest lo
    quantizzerebbe agli stessi valori dei voxel ricampionati, perdendo la
    risoluzione sub-voxel del bordo. La binarizzazione avviene DOPO, nello
    spazio originale, dentro postproc.
    """
    return Compose([
        Invertd(
            keys="pred",
            transform=pre_transforms,
            orig_keys="image",
            meta_keys="pred_meta_dict",
            orig_meta_keys="image_meta_dict",
            nearest_interp=False,
            to_tensor=True,
        ),
    ])


# ---------------------------------------------------------------------------
#  TTA
# ---------------------------------------------------------------------------
def _flip_combinations(axes):
    """Tutti i sottoinsiemi degli assi dati (incluso quello vuoto)."""
    combos = [()]
    for a in axes:
        combos += [c + (a,) for c in combos]
    return combos


def tta_softmax(model, x, inferer, axes=()):
    """Media della SOFTMAX sulle combinazioni di flip.

    Si media la softmax e non i logit: i logit di reti diverse (o della stessa
    rete su input specchiati) non sono su una scala comune, e la loro media non
    corrisponde alla media delle probabilita'. E' l'errore classico
    dell'ensembling, e produce un modello sistematicamente troppo sicuro.

    Si specchiano SOLO gli assi per cui la rete e' stata addestrata: applicare
    un flip che la rete non ha mai visto in training non e' data augmentation,
    e' distribuzione fuori campione.
    """
    acc = None
    combos = _flip_combinations(tuple(axes))
    for flip in combos:
        xi = torch.flip(x, dims=list(flip)) if flip else x
        with torch.amp.autocast("cuda", enabled=(x.device.type == "cuda")):
            logits = inferer(inputs=xi, network=model)
        p = torch.softmax(logits.float(), dim=1)
        if flip:
            p = torch.flip(p, dims=list(flip))
        acc = p if acc is None else acc + p
    return acc / len(combos)


def ensemble_softmax(models, x, inferer, axes=()):
    """Media della softmax su piu' modelli (e, per ciascuno, sulla TTA)."""
    acc = None
    for m in models:
        p = tta_softmax(m, x, inferer, axes)
        acc = p if acc is None else acc + p
    return acc / len(models)


# ---------------------------------------------------------------------------
#  Predizione di un caso
# ---------------------------------------------------------------------------
def predict_case(models, image_path, device=None, tta_axes=(),
                 roi_size=None, sw_batch_size=None, overlap=None,
                 target_spacing=None, blend_mode="constant"):
    """Probabilita' di tumore di un caso, nello spazio CT ORIGINALE.

    'models' puo' essere un singolo modello o una lista (ensemble).
    Restituisce {'prob': float32 (X,Y,Z), 'affine': 4x4, 'shape': ...}.

    blend_mode - COME SI FONDONO LE FINESTRE SOVRAPPOSTE. Non e' un dettaglio:
    e' una discrepanza misurata fra validazione e produzione in questo
    progetto. Il vecchio train.py validava con sliding_window_inference() senza
    passare mode=, cioe' con blending 'constant' (media uniforme); inference.py
    usa invece mode='gaussian' (pesi centrati sulla finestra). Misurato sullo
    stesso checkpoint e sugli stessi casi:

        caso        GT    constant   gaussian   dice_cst   dice_gau
        lung_001    289        449       1348     0.686      0.341
        lung_006    920        299       3605     0.386      0.007
        lung_014   7086       9279      10250     0.842      0.810

    Sulle lesioni grandi i due coincidono quasi; sulle piccole il gaussian
    gonfia il volume predetto fino a 12 volte, e con esso i falsi positivi.
    Il default qui e' 'constant' perche' e' la modalita' sotto cui il modello
    e' stato validato e il checkpoint selezionato: usarne un'altra in
    produzione significa eseguire un sistema diverso da quello misurato.
    """
    if not isinstance(models, (list, tuple)):
        models = [models]
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pre = get_pre_transforms(target_spacing)
    post = get_prob_post_transforms(pre)

    ds = Dataset(data=[{"image": image_path}], transform=pre)
    loader = DataLoader(ds, batch_size=1, num_workers=0)

    inferer = SlidingWindowInferer(
        roi_size=roi_size or config.SW_ROI_SIZE,
        sw_batch_size=sw_batch_size or config.SW_BATCH_SIZE,
        overlap=config.SW_OVERLAP if overlap is None else overlap,
        mode=blend_mode,
    )

    result = None
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            batch["pred"] = ensemble_softmax(models, images, inferer, tta_axes)
            result = [post(item) for item in decollate_batch(batch)][0]

    pred = result["pred"]                       # (2, X, Y, Z) nello spazio originale
    prob = pred[1].cpu().numpy().astype(np.float32)
    affine = np.asarray(pred.affine.cpu().numpy(), dtype=np.float64)
    return {"prob": prob, "affine": affine, "shape": prob.shape}


# ---------------------------------------------------------------------------
#  Persistenza compatta
# ---------------------------------------------------------------------------
def save_prob(path, prob, affine, floor: float = STORE_FLOOR):
    """Salva la mappa come bbox quantizzato uint8 (vedi nota in testa al file)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    prob = np.asarray(prob, dtype=np.float32)
    hot = prob > floor

    if not hot.any():
        np.savez_compressed(path, empty=True, shape=np.array(prob.shape),
                            affine=affine, floor=floor)
        return path

    sl = ndimage.find_objects(hot.astype(np.uint8))[0]
    sub = prob[sl]
    q = np.clip(np.rint(sub * 255.0), 0, 255).astype(np.uint8)
    np.savez_compressed(
        path, empty=False, q=q,
        offset=np.array([s.start for s in sl], dtype=np.int32),
        shape=np.array(prob.shape, dtype=np.int32),
        affine=affine, floor=floor,
    )
    return path


def load_prob(path):
    """Ricostruisce (prob float32 a piena griglia, affine).

    Fuori dal bounding box salvato la probabilita' e' < floor per costruzione:
    ricostruirla a zero e' esatto per qualunque soglia >= floor, cioe' per tutte
    quelle che lo sweep esplora.
    """
    with np.load(path) as z:
        shape = tuple(int(x) for x in z["shape"])
        affine = np.asarray(z["affine"], dtype=np.float64)
        prob = np.zeros(shape, dtype=np.float32)
        if not bool(z["empty"]):
            off = [int(x) for x in z["offset"]]
            q = z["q"]
            sl = tuple(slice(o, o + n) for o, n in zip(off, q.shape))
            prob[sl] = q.astype(np.float32) / 255.0
    return prob, affine


def prob_path(out_dir: str, case_id: str) -> str:
    return os.path.join(out_dir, f"{case_id}.npz")
