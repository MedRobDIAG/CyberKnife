# ============================================================================
#  inference_worker.py - Worker QThread per l'inferenza in background.
#
#  Perche' un thread separato: l'inferenza (caricamento volume + sliding window
#  su GPU + inversione spaziale) impiega diversi secondi. Eseguirla nel thread
#  della GUI congelerebbe la finestra ("Non risponde"). Il worker gira in
#  parallelo e comunica con la UI SOLO tramite segnali Qt (thread-safe).
#
#  Segnali emessi:
#    progress(int)      -> avanzamento 0-100 per la QProgressBar
#    status(str)        -> messaggio testuale di stato
#    resultReady(dict)  -> risultati finali (CT, maschera, coordinate, volume)
#    failed(str)        -> messaggio d'errore in caso di eccezione
#
#  NB: non si usa il nome 'finished' perche' e' gia' un segnale nativo di QThread.
# ============================================================================
import os
import traceback

import numpy as np
import nibabel as nib
from nibabel.affines import apply_affine
from scipy import ndimage

import torch
from PyQt6.QtCore import QThread, pyqtSignal

from monai.inferers import SlidingWindowInferer
from monai.transforms import Compose, Activationsd, Invertd

import config
import postproc
# Riuso dei componenti gia' collaudati nella pipeline da riga di comando.
from inference import load_model, get_pre_transforms
from tumor_3d import compute_volume


class InferenceWorker(QThread):
    """Esegue l'intera pipeline di percezione su un volume CT.

    Pipeline: NIfTI -> preprocessing MONAI -> sliding window inference ->
    inversione allo spazio originale -> maschera 3D -> centro di massa ->
    coordinate fisiche (mm) tramite matrice affine.
    """

    progress = pyqtSignal(int)
    status = pyqtSignal(str)
    resultReady = pyqtSignal(dict)
    failed = pyqtSignal(str)

    def __init__(self, image_path: str, model_path: str = config.BEST_MODEL_PATH,
                 with_uncertainty: bool = False, parent=None):
        super().__init__(parent)
        self.image_path = image_path
        self.model_path = model_path
        # Analisi di incertezza (uncertainty.py): 8 predizioni perturbate,
        # circa 10 s in piu' su GPU.
        self.with_uncertainty = with_uncertainty

    # ------------------------------------------------------------------
    #  Corpo del thread
    # ------------------------------------------------------------------
    def run(self):
        try:
            # --- 1) Device e modello -----------------------------------
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.status.emit(f"Inizializzazione modello su {str(device).upper()}...")
            self.progress.emit(5)

            if not os.path.exists(self.model_path):
                raise FileNotFoundError(
                    f"Pesi del modello non trovati:\n{self.model_path}"
                )
            model = load_model(self.model_path, device)
            self.progress.emit(15)

            # --- 2) Volume CT originale (per la visualizzazione) --------
            # Si legge con nibabel per avere l'affine ORIGINALE del paziente e i
            # dati grezzi in HU su cui disegnare le slice. float32 al posto del
            # float64 di get_fdata() per dimezzare l'occupazione di RAM.
            self.status.emit("Caricamento volume CT...")
            ct_nib = nib.load(self.image_path)
            ct_data = np.asanyarray(ct_nib.dataobj, dtype=np.float32)
            affine = ct_nib.affine.astype(np.float64)
            self.progress.emit(25)

            # --- 3) Preprocessing MONAI --------------------------------
            # Stesse transform della validazione: orientamento RAS, spacing
            # isotropico, finestra HU normalizzata.
            self.status.emit("Preprocessing (RAS, spacing, finestra HU)...")
            pre_transforms = get_pre_transforms()
            data = pre_transforms({"image": self.image_path})
            image_tensor = data["image"].unsqueeze(0).to(device)   # (1,1,H,W,D)
            self.progress.emit(40)

            # --- 4) Sliding Window Inference ---------------------------
            # Il volume intero non entra in VRAM: lo si percorre a finestre 96^3,
            # fondendo le sovrapposizioni secondo config.SW_BLEND_MODE.
            self.status.emit("Inferenza sliding window (96x96x96) in corso...")
            inferer = SlidingWindowInferer(
                roi_size=config.SW_ROI_SIZE,
                sw_batch_size=config.SW_BATCH_SIZE,
                overlap=config.SW_OVERLAP,
                # mode='constant' e NON 'gaussian': il modello e' stato validato
                # (e il checkpoint selezionato) con blending uniforme, perche'
                # train.py chiamava sliding_window_inference senza passare mode.
                # La differenza e' misurata e grande sulle lesioni piccole -
                # su lung_006 il gaussian predice 3605 voxel contro 299, con il
                # Dice che crolla da 0.386 a 0.007. Usare in produzione una
                # modalita' diversa da quella di validazione significa eseguire
                # un sistema diverso da quello misurato.
                mode=config.SW_BLEND_MODE,
            )
            with torch.no_grad():
                with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                    data["pred"] = inferer(inputs=image_tensor, network=model)[0]
            self.progress.emit(70)

            # --- 5) Post-processing + inversione allo spazio originale --
            # Invertd annulla Spacing/Orientation: la maschera torna nella
            # geometria esatta della CT originale, cosi' e' sovrapponibile
            # voxel-per-voxel alle slice mostrate nella GUI.
            self.status.emit("Ricostruzione maschera nello spazio originale...")
            # Si riporta indietro la PROBABILITA', non la maschera gia' binaria:
            # nearest_interp=False perche' e' un campo continuo, e la soglia si
            # applica dopo, nello spazio originale, dentro postproc. Cosi' la
            # soglia e' un parametro tarabile invece che un argmax nascosto in
            # mezzo alla catena.
            post_transforms = Compose([
                Activationsd(keys="pred", softmax=True),
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
            # I dati non sono in batch (un solo volume): le transform si
            # applicano direttamente al dizionario.
            result = post_transforms(data)
            prob = result["pred"][1].cpu().numpy().astype(np.float32)
            self.progress.emit(75)

            # --- 5b) Incertezza (opzionale) ---------------------------------
            # Con l'analisi attiva la maschera viene dalla probabilita' MEDIA
            # delle predizioni perturbate: e' il sistema validato in
            # run_uncertainty.py, e usare la predizione singola per il
            # bersaglio e la media per il semaforo significherebbe giudicare
            # una maschera con i segnali di un'altra.
            unc_maps = unc_signals = None
            if self.with_uncertainty:
                import uncertainty as U
                self.status.emit("Analisi di incertezza (8 predizioni "
                                 "perturbate)...")
                unc_maps = U.sample_case(model, self.image_path,
                                         U.tta_params(U.N_SAMPLES), device)
                # Subito nella forma compatta (8 bit, riquadro della lesione):
                # e' quella in cui il semaforo e' stato validato sui 63 casi,
                # ed e' quella che un progetto salvato conserva. Cosi' la
                # sessione dal vivo e quella riaperta sono identiche.
                unc_maps = U.expand_maps(U.compact_maps(unc_maps))
                if unc_maps["p_mean"].shape != prob.shape:
                    raise RuntimeError("mappe di incertezza non allineate "
                                       "alla CT")
                prob = unc_maps["p_mean"]
                unc_signals = U.case_signals(unc_maps)
            self.progress.emit(85)

            # --- 6) Estrazione del target ------------------------------
            # postproc.probs_to_mask sostituisce la vecchia regola "tieni la
            # componente piu' grande", che aveva due difetti gravi per un
            # sistema che alimenta un planner: non poteva MAI dire "nessun
            # tumore" (acceso un voxel, quel voxel diventava il bersaglio), e
            # se un falso positivo superava in volume la lesione vera
            # l'isocentro ci saltava sopra senza alcun segnale. Ora si tengono
            # tutte le componenti sopra una soglia di volume FISICA, e la
            # lesione primaria e' quella piu' voluminosa.
            self.status.emit("Analisi componenti connesse...")
            mask, lesions = postproc.probs_to_mask(
                prob, config.POSTPROC, affine=affine)
            n_raw = int(ndimage.label(prob >= config.POSTPROC.prob_threshold)[1])

            if not lesions:
                self.status.emit("Nessun tumore rilevato nella scansione.")
                self.resultReady.emit({
                    "ct_data": ct_data,
                    "mask": mask,
                    "affine": affine,
                    "tumor_found": False,
                    "n_components": n_raw,
                    "n_lesions_kept": 0,
                    "lesions": [],
                    "image_path": self.image_path,
                    "device": str(device),
                    "uncertainty": unc_maps,
                    "uncertainty_signals": unc_signals,
                    "traffic_light": ("rosso", ["nessuna lesione trovata"]),
                })
                self.progress.emit(100)
                return

            # La lesione PRIMARIA definisce l'isocentro. Mai il centroide
            # dell'unione: fra due lesioni distinte cadrebbe nel tessuto sano
            # che le separa.
            primary = postproc.primary_lesion(lesions)
            com_voxel = np.array(primary["com_voxel"], dtype=np.float64)
            com_world = np.array(primary["com_world"], dtype=np.float64)

            # --- Cancello di confidenza ---------------------------------
            # Misurato sui 63 casi out-of-fold: nel 27% dei casi l'isocentro
            # cade fuori dal tumore, e l'errore e' bimodale (o pochi mm, o
            # ~100 mm). Il cancello non migliora il modello: rende VISIBILE
            # quando non fidarsene, il che per un bersaglio che alimenta un
            # piano dosimetrico e' cio' che conta. Validato in modo annidato:
            # accetta il 74.6% dei casi, e su quelli l'isocentro e' corretto
            # nell'89.4% contro il 73.0% senza.
            reliable = config.GATE.accepts(primary)
            gate_reason = "" if reliable else config.GATE.reason(primary)
            if not reliable:
                self.status.emit(f"ATTENZIONE - bersaglio poco affidabile: "
                                 f"{gate_reason}")

            if unc_signals is not None:
                import uncertainty as U
                light = U.traffic_light(unc_signals, reliable,
                                        gate_reason)
            else:
                light = (("verde" if reliable else "giallo"),
                         [] if reliable else [gate_reason])

            # Volume della lesione (utile per la futura dosimetria).
            n_vox, voxel_vol, vol_mm3, vol_cm3 = compute_volume(mask, affine)
            self.progress.emit(95)

            self.status.emit("Inferenza completata")
            self.resultReady.emit({
                "ct_data": ct_data,
                "mask": mask,
                "affine": affine,
                "tumor_found": True,
                "com_voxel": com_voxel,
                "com_world": com_world,
                "n_tumor_voxels": n_vox,
                # n_components = componenti grezze sopra soglia (prima del
                # filtro sul volume); n_lesions_kept = quelle sopravvissute.
                # Vanno mostrate entrambe: la prima da' la misura del rumore,
                # la seconda quello su cui il planner lavorera' davvero.
                "n_components": n_raw,
                "n_lesions_kept": len(lesions),
                "lesions": lesions,
                # Il bersaglio viene comunque consegnato: config.GATE_BLOCKS_TARGET
                # e' False perche' in un simulatore didattico nascondere i casi di
                # fallimento impedirebbe di studiarli. Ma l'interfaccia li segnala.
                "target_reliable": reliable,
                "gate_reason": gate_reason,
                "volume_mm3": vol_mm3,
                "volume_cm3": vol_cm3,
                "image_path": self.image_path,
                "device": str(device),
                "uncertainty": unc_maps,
                "uncertainty_signals": unc_signals,
                "traffic_light": light,
            })
            self.progress.emit(100)

        except Exception as exc:                      # noqa: BLE001
            # L'intero traceback finisce nel messaggio d'errore mostrato in GUI.
            self.failed.emit(f"{type(exc).__name__}: {exc}\n\n{traceback.format_exc()}")
