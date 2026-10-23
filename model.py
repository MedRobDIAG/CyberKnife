# ============================================================================
#  model.py - Architetture di segmentazione 3D disponibili.
#
#  Ne esistono tre perche' il confronto fra loro E' uno dei risultati del
#  progetto, non un dettaglio implementativo. Vanno addestrate sugli stessi
#  fold, con la stessa augmentation e lo stesso schedule: altrimenti si
#  confrontano ricette, non architetture.
#
#    unet         la rete attuale. E' il riferimento da battere: qualunque
#                 alternativa che non la superi in modo misurabile non merita
#                 il costo aggiuntivo.
#    dynunet      il porting MONAI dell'architettura nnU-Net. E' la scelta
#                 coerente se la ricetta che si replica e' quella di nnU-Net:
#                 citare nnU-Net e poi usare tutt'altra rete sarebbe una
#                 pretesa difficile da difendere.
#    segresnetds  SegResNet con deep supervision. Piu' costosa (misurato ~7x
#                 per epoca rispetto a unet a parita' di patch), inclusa per
#                 avere una terza riga indipendente nel confronto.
#
#  DEEP SUPERVISION - nota che vale per entrambe le reti che la offrono:
#  le uscite ausiliarie a risoluzione ridotta esistono SOLO in .train(). In
#  .eval() la rete restituisce un tensore singolo, quindi sliding_window_inference,
#  inference.py e predict.py non richiedono alcuna modifica. La loss, invece,
#  deve saper gestire una lista di uscite (vedi train.py::deep_supervision_loss).
# ============================================================================
import torch
from monai.networks.nets import UNet, DynUNet, SegResNetDS
from monai.networks.layers import Norm

import config


ARCHITECTURES = ("unet", "dynunet", "segresnetds")


def _unet():
    """3D U-Net residuale MONAI - la configurazione storica del progetto.

      - channels (16..256) = 5 livelli; num_res_units=2 blocchi residuali per
        livello, che stabilizzano il gradiente su volumi 3D.
      - norm=INSTANCE e non BATCH: con batch da 2 le statistiche di batch sono
        rumore, mentre la Instance Norm normalizza per campione.
    """
    return UNet(
        spatial_dims=3,
        in_channels=config.IN_CHANNELS,
        out_channels=config.OUT_CHANNELS,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        norm=Norm.INSTANCE,
    )


def _dynunet(deep_supervision: bool = True, deep_supr_num: int = 2):
    """DynUNet: l'architettura nnU-Net.

    5 livelli, downsampling per 16 su tutti gli assi. Con patch (128,128,96) il
    livello piu' profondo lavora su 8x8x6, che e' abbastanza grande da restare
    informativo: scendere oltre produrrebbe feature map degeneri.

    I filtri sono espliciti invece che lasciati al default: il default di MONAI
    parte da 64 e su una scheda da 12 GB, con patch di questa dimensione,
    l'occupazione diventa difficile da prevedere.
    """
    strides = [[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]]
    kernel_size = [[3, 3, 3]] * len(strides)
    return DynUNet(
        spatial_dims=3,
        in_channels=config.IN_CHANNELS,
        out_channels=config.OUT_CHANNELS,
        kernel_size=kernel_size,
        strides=strides,
        upsample_kernel_size=strides[1:],
        filters=(32, 64, 128, 256, 320),
        norm_name="instance",
        deep_supervision=deep_supervision,
        # deep_supr_num dev'essere < len(strides)-1: con 5 livelli il massimo e'
        # 3. Se ne usano 2, come nella configurazione nnU-Net di riferimento.
        deep_supr_num=deep_supr_num,
        res_block=True,
    )


def _segresnetds(deep_supervision: bool = True):
    """SegResNet con deep supervision.

    ATTENZIONE al parametro norm: il default di SegResNetDS e' "batch". Con
    batch_size=2 la Batch Norm stima media e varianza su due campioni, il che
    e' attivamente dannoso (le statistiche di validazione non corrispondono a
    quelle di training). Va passata "instance" esplicitamente.
    """
    return SegResNetDS(
        spatial_dims=3,
        in_channels=config.IN_CHANNELS,
        out_channels=config.OUT_CHANNELS,
        init_filters=32,
        blocks_down=(1, 2, 2, 4),
        norm="instance",
        dsdepth=3 if deep_supervision else 1,
    )


def get_model(device: torch.device | None = None, arch: str = "unet",
              deep_supervision: bool = True) -> torch.nn.Module:
    """Crea l'architettura richiesta.

    'device' resta il PRIMO parametro per compatibilita' con i chiamanti
    esistenti (inference.py, predict.py), che invocano get_model(device).
    """
    arch = (arch or "unet").lower()
    if arch == "unet":
        model = _unet()                      # nessuna deep supervision
    elif arch == "dynunet":
        model = _dynunet(deep_supervision)
    elif arch == "segresnetds":
        model = _segresnetds(deep_supervision)
    else:
        raise ValueError(f"Architettura sconosciuta '{arch}'. "
                         f"Disponibili: {', '.join(ARCHITECTURES)}")

    if device is not None:
        model = model.to(device)
    return model


def has_deep_supervision(arch: str) -> bool:
    return (arch or "unet").lower() in ("dynunet", "segresnetds")


if __name__ == "__main__":
    # Confronto rapido: parametri, forma delle uscite in train e in eval, e
    # memoria occupata da un forward+backward alla patch corrente.
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--patch", type=int, nargs=3, default=list(config.PATCH_SIZE))
    ap.add_argument("--batch", type=int, default=config.TRAIN_BATCH_SIZE)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device} | patch {tuple(args.patch)} | batch {args.batch}\n")

    for arch in ARCHITECTURES:
        net = get_model(device, arch)
        n = sum(p.numel() for p in net.parameters())
        dummy = torch.randn(args.batch, config.IN_CHANNELS, *args.patch,
                            device=device)

        net.train()
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            out_tr = net(dummy)
        shapes = ([tuple(o.shape) for o in out_tr]
                  if isinstance(out_tr, (list, tuple))
                  else (tuple(out_tr.shape) if out_tr.ndim == 5
                        else [tuple(s) for s in out_tr]))

        net.eval()
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            out_ev = net(dummy)

        print(f"{arch:<14} {n/1e6:>6.1f}M parametri")
        print(f"{'':<14} train -> {shapes}")
        print(f"{'':<14} eval  -> {tuple(out_ev.shape)}")
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
            net.train()
            with torch.amp.autocast("cuda"):
                loss = net(dummy)
                loss = loss[0] if isinstance(loss, (list, tuple)) else loss
                (loss.float().mean()).backward()
            print(f"{'':<14} picco memoria fwd+bwd: "
                  f"{torch.cuda.max_memory_allocated()/2**30:.2f} GiB")
        del net, dummy
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print()
