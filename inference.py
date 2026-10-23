# ============================================================================
#  inference.py - Fase 1 Medical Robotics (simulazione CyberKnife)
#
#  Obiettivo duplice:
#    (A) Processare una TAC intera -> maschera 3D del tumore (.nii.gz per ITK-SNAP)
#    (B) Tradurre la maschera in coordinate spaziali (mm) per il path-planning
#        del braccio robotico.
#
#  Punto chiave: usiamo Invertd per riportare la predizione dallo spazio
#  ricampionato (RAS, spacing isotropico) allo SPAZIO ORIGINALE della CT. Cosi':
#    - il .nii.gz salvato combacia voxel-per-voxel con la CT originale (overlay
#      perfetto in ITK-SNAP);
#    - il centro di massa viene proiettato con l'affine ORIGINALE del paziente.
#
#  USO:
#    python inference.py --image "Task06_Lung\Task06_Lung\imagesTr\lung_001.nii.gz"
#    python inference.py --image <ct.nii.gz> --model outputs\best_model_dice.pth
# ============================================================================
import os
import argparse

import numpy as np
import torch
import nibabel as nib
from scipy import ndimage

from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
    ScaleIntensityRanged,
    EnsureTyped,
)

import config
from model import get_model


# ---------------------------------------------------------------------------
#  STEP 2 - Transforms di inferenza (coerenti con la validazione)
# ---------------------------------------------------------------------------
def get_pre_transforms():
    """Pre-processing del volume di test.

    Identico alla pipeline di validazione (stesso orientamento, spacing e
    finestra HU) MA senza crop random/augmentation: si processa il volume intero.
    image_only=False mantiene i metadati (affine originale) necessari a Invertd
    e alla proiezione in coordinate fisiche.
    """
    return Compose([
        LoadImaged(keys=["image"], image_only=False),
        EnsureChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes="RAS"),
        Spacingd(keys=["image"], pixdim=config.TARGET_SPACING, mode="bilinear"),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=config.HU_MIN, a_max=config.HU_MAX,
            b_min=0.0, b_max=1.0, clip=True,
        ),
        EnsureTyped(keys=["image"]),
    ])


# ---------------------------------------------------------------------------
#  STEP 1 - Caricamento modello
# ---------------------------------------------------------------------------
def load_model(model_path, device, arch: str | None = None):
    """Inizializza l'architettura e ne carica i pesi.

    Gestisce sia il checkpoint completo salvato da train.py (dict con chiave
    'model_state_dict') sia un file contenente direttamente lo state_dict.

    'arch' si puo' omettere: i checkpoint scritti da train.py portano con se'
    la propria configurazione, quindi l'architettura si ricava dal file invece
    di doverla ripetere sulla riga di comando (e sbagliarla in silenzio,
    ottenendo un load_state_dict che fallisce o - peggio - riesce parzialmente).
    """
    checkpoint_peek = torch.load(model_path, map_location="cpu",
                                 weights_only=False)
    if arch is None and isinstance(checkpoint_peek, dict):
        arch = (checkpoint_peek.get("config") or {}).get("arch")
    model = get_model(device, arch or "unet")
    checkpoint = checkpoint_peek

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        best = checkpoint.get("best_dice", checkpoint.get("best_metric", "n/a"))
        # Il modello finale non ha validation set, quindi best_metric resta a
        # -inf: stamparlo cosi' com'e' sembrerebbe un errore.
        if isinstance(best, float) and best == float("-inf"):
            best = "n/d (nessuna validazione: modello finale)"
        ep = checkpoint.get("epoch", "?")
        print(f"[infer] Pesi da '{os.path.basename(model_path)}' "
              f"(arch={arch or 'unet'}, epoca={ep}, best={best})")
    else:
        model.load_state_dict(checkpoint)
        print(f"[infer] State_dict caricato da '{model_path}'")

    model.eval()   # modalita' valutazione: disattiva dropout/aggiorna BN stats
    return model


# ---------------------------------------------------------------------------
#  Pipeline completa
# ---------------------------------------------------------------------------
def run_inference(image_path, model_path=config.BEST_MODEL_PATH,
                  output_dir=config.OUTPUT_DIR, tta_axes=(),
                  postproc_cfg=None):
    """CT -> maschera del tumore + bersaglio in mm per il planner.

    La rete e il post-processing NON sono piu' implementati qui: si delega a
    predict.predict_case (inferenza + softmax + Invertd) e a
    postproc.probs_to_mask (soglia, filtro sul volume, lesioni). Prima la
    stessa logica era scritta tre volte - qui, in inference_worker.py e in
    tumor_3d.py - e in nessuna di queste veniva applicata in validazione, per
    cui il Dice riportato non descriveva questa pipeline. Con un'unica
    implementazione condivisa la coerenza e' garantita per costruzione.
    """
    import predict as P
    import postproc

    os.makedirs(output_dir, exist_ok=True)
    cfg = postproc_cfg or config.POSTPROC
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[infer] Device: {device}")

    model = load_model(model_path, device)

    # Probabilita' nello spazio CT ORIGINALE (predict.py applica Invertd).
    out = P.predict_case(model, image_path, device=device, tta_axes=tta_axes,
                         blend_mode=config.SW_BLEND_MODE)
    prob, affine = out["prob"], out["affine"]

    # Cross-check dell'affine con il file originale.
    try:
        nib_affine = nib.load(image_path).affine
        if not np.allclose(affine, nib_affine, atol=1e-3):
            print("[infer] (nota) affine della predizione e affine nibabel "
                  "differiscono leggermente; uso quello della predizione.")
    except Exception:
        pass

    mask, lesions = postproc.probs_to_mask(prob, cfg, affine=affine)

    # Salvataggio della maschera nello spazio originale (overlay in ITK-SNAP).
    base = os.path.basename(image_path).replace(".nii.gz", "").replace(".nii", "")
    seg_path = os.path.join(output_dir, f"{base}_seg.nii.gz")
    nib.save(nib.Nifti1Image(mask.astype(np.uint8), affine), seg_path)

    n_raw = int(ndimage.label(prob >= cfg.prob_threshold)[1])
    print("\n" + "=" * 60)
    print(f"[infer] Componenti sopra soglia {cfg.prob_threshold}: {n_raw}")
    print(f"[infer] Lesioni sopra {cfg.min_lesion_mm3:.0f} mm3: {len(lesions)}")

    if not lesions:
        # Esito legittimo, non un errore: con il vecchio "tieni la componente
        # piu' grande" il sistema non poteva mai arrivare qui, perche' un
        # singolo voxel acceso bastava a produrre un bersaglio.
        print("[infer] Nessun tumore rilevato: nessun bersaglio da fornire "
              "al planner.")
        print(f"[infer] Maschera (vuota) salvata in: {seg_path}")
        print("=" * 60)
        return None

    for i, l in enumerate(lesions):
        star = " <- primaria" if i == 0 else ""
        print(f"        [{i}] {l['volume_cm3']:7.3f} cm3  "
              f"prob_max {l['prob_max']:.2f}{star}")

    # La lesione PRIMARIA (la piu' voluminosa) definisce l'isocentro. Mai il
    # centroide dell'unione: fra due lesioni distinte cadrebbe nel tessuto sano
    # interposto, che e' il modo documentato di produrre un piano a copertura
    # nulla (archivio/documenti/STATO_PROGETTO.md, D95 ~ 0% con isocentro a 74 mm dal bersaglio).
    primary = postproc.primary_lesion(lesions)
    com_voxel = np.array(primary["com_voxel"], dtype=np.float64)
    com_world = np.array(primary["com_world"], dtype=np.float64)
    n_tumor = int(mask.sum())

    # Cancello di confidenza: non migliora il modello, rende visibile quando
    # non fidarsene. Sui 63 casi out-of-fold l'isocentro cade fuori dal tumore
    # nel 27% dei casi, con errore bimodale (pochi mm oppure ~100 mm).
    reliable = config.GATE.accepts(primary)
    if not reliable:
        print("-" * 60)
        print("  ATTENZIONE: bersaglio POCO AFFIDABILE")
        print(f"  {config.GATE.reason(primary)}")
        print("  Su casi con questo profilo l'isocentro cade fuori dal tumore")
        print("  molto piu' spesso. Verificare prima di pianificare.")

    # --- STEP 6 (output formattato) ---
    print("=" * 60)
    print("  RISULTATO FASE 1 - TARGET PER IL SIMULATORE CINEMATICO")
    print("=" * 60)
    print("  Coordinate Voxel (i, j, k):")
    print(f"     i = {com_voxel[0]:8.2f}")
    print(f"     j = {com_voxel[1]:8.2f}")
    print(f"     k = {com_voxel[2]:8.2f}")
    print("-" * 60)
    print("  Coordinate Reali (World, mm) - spazio fisico del paziente:")
    print(f"     X = {com_world[0]:+9.2f} mm")
    print(f"     Y = {com_world[1]:+9.2f} mm")
    print(f"     Z = {com_world[2]:+9.2f} mm")
    print("=" * 60)
    print(f"  Maschera salvata in: {output_dir} (sovrapponibile in ITK-SNAP)")
    print("=" * 60 + "\n")

    return {
        "mask": mask,
        "lesions": lesions,
        "n_lesions_kept": len(lesions),
        "target_reliable": reliable,
        "n_components": n_raw,
        "com_voxel": com_voxel,
        "com_world_mm": com_world,
        "affine": affine,
        "n_tumor_voxels": n_tumor,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Inferenza 3D U-Net + estrazione coordinate tumore (mm)."
    )
    parser.add_argument("--image", required=True,
                        help="Percorso della CT di test (.nii.gz).")
    parser.add_argument("--model", default=config.BEST_MODEL_PATH,
                        help="Percorso del checkpoint del modello.")
    parser.add_argument("--out", default=config.OUTPUT_DIR,
                        help="Cartella di output per la maschera .nii.gz.")
    args = parser.parse_args()

    assert os.path.exists(args.image), f"File non trovato: {args.image}"
    assert os.path.exists(args.model), f"Modello non trovato: {args.model}"
    run_inference(args.image, args.model, args.out)
