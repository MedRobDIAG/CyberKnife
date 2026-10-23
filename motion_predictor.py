# ============================================================================
#  motion_predictor.py - Predittore di latenza a rete ricorrente (GRU).
#
#  Il nodo ROS non ha torch (Python 3.10 in WSL): la GRU vive solo qui, e si
#  registra fra i predittori di motion_model con il nome "gru".
#
#  COSA PREDICE
#  ------------
#  Dato l'ultimo secondo e mezzo di segnale del torace, l'incremento fra ora e
#  ora + 115 ms. Ingresso e uscita sono normalizzati per la deviazione
#  standard della finestra: la rete impara la FORMA del respiro, non la sua
#  ampiezza, e un paziente che respira piu' profondo non e' fuori distribuzione.
#
#  DATI
#  ----
#  Tracce sintetiche (motion_model.generate_trace) con parametri campionati a
#  caso: periodo 2.5-6 s, irregolarita', forma, rumore, curvatura. I semi di
#  addestramento partono da 10000, quelli di confronto (0-99) non vengono mai
#  visti: lo split e' PER TRACCIA, non per campione, altrimenti finestre
#  adiacenti della stessa traccia finirebbero in train e in test.
#
#  USO
#     python motion_predictor.py --train          addestra e salva
#     python motion_predictor.py --benchmark      confronta tutti i predittori
# ============================================================================
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn

import config
import motion_model as mm

WEIGHTS_PATH = os.path.join(config.OUTPUT_DIR, "motion_gru.pth")
WINDOW = 40            # 1.5 s a 26 Hz
TRAIN_SEED0 = 10000


class GruNet(nn.Module):
    def __init__(self, hidden=48):
        super().__init__()
        self.gru = nn.GRU(1, hidden, num_layers=2, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):                  # x: (B, W)
        y, _ = self.gru(x.unsqueeze(-1))
        return self.head(y[:, -1]).squeeze(-1)


def _windows(s, h, window=WINDOW):
    """Finestre normalizzate e bersagli. s: (N,). Restituisce X (M,W), y, scala.

    La finestra k termina al campione k (incluso); il bersaglio e' s[k+h].
    """
    from numpy.lib.stride_tricks import sliding_window_view
    W = sliding_window_view(s, window)               # W[i] termina in i+W-1
    last = W[:, -1:]
    scale = W.std(axis=1, keepdims=True) + 0.05
    X = (W - last) / scale
    return X.astype(np.float32), last[:, 0], scale[:, 0]


def random_config(rng):
    """Un paziente sintetico a caso."""
    c = mm.MotionConfig(
        period_s=float(rng.uniform(2.5, 6.0)),
        period_cv=float(rng.uniform(0.0, 0.25)),
        amplitude_cv=float(rng.uniform(0.0, 0.25)),
        shape_n=int(rng.integers(1, 4)),
        surrogate_gain_mm=float(rng.uniform(3.0, 15.0)),
        surrogate_curvature=float(rng.uniform(0.0, 0.4)),
        surrogate_noise_mm=float(rng.uniform(0.05, 0.3)),
        surrogate_lead_s=float(rng.uniform(-0.1, 0.3)),
        surrogate_drift_mm_per_min=float(rng.uniform(-1.0, 1.0)),
    )
    return c


def build_dataset(n_traces, seed0, duration_s=240.0, per_trace=1500):
    Xs, ys = [], []
    h = mm.MotionConfig().horizon_samples
    for i in range(n_traces):
        rng = np.random.default_rng(seed0 + i)
        c = random_config(rng).replace(seed=seed0 + i)
        tr = mm.generate_trace(c, duration_s, rng)
        s = tr.surrogate
        X, last, scale = _windows(s[:-h], h)
        y = ((s[WINDOW - 1 + h:] - last) / scale).astype(np.float32)
        sel = rng.choice(len(X), size=min(per_trace, len(X)), replace=False)
        Xs.append(X[sel])
        ys.append(y[sel])
    return np.concatenate(Xs), np.concatenate(ys)


def train(epochs=30, n_train=300, n_val=40, device=None, log=print):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    t0 = time.time()
    Xtr, ytr = build_dataset(n_train, TRAIN_SEED0)
    Xva, yva = build_dataset(n_val, TRAIN_SEED0 + 50000)
    log(f"dataset: {len(Xtr)} finestre di training, {len(Xva)} di "
        f"validazione ({time.time() - t0:.1f} s)")
    Xtr_t, ytr_t = torch.from_numpy(Xtr).to(device), torch.from_numpy(ytr).to(device)
    Xva_t, yva_t = torch.from_numpy(Xva).to(device), torch.from_numpy(yva).to(device)

    net = GruNet().to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    loss_fn = nn.SmoothL1Loss(beta=0.1)
    bs = 1024
    best, best_state = float("inf"), None
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(len(Xtr_t), device=device)
        tot = 0.0
        for i in range(0, len(perm), bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            loss = loss_fn(net(Xtr_t[idx]), ytr_t[idx])
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            tot += loss.item() * len(idx)
        sched.step()
        net.eval()
        with torch.no_grad():
            va = torch.sqrt(torch.mean((net(Xva_t) - yva_t) ** 2)).item()
        if va < best:
            best = va
            best_state = {k: v.detach().cpu().clone()
                          for k, v in net.state_dict().items()}
        log(f"epoca {ep + 1:2d}  loss {tot / len(perm):.4f}  "
            f"val RMSE (normalizzato) {va:.4f}")
    torch.save({"state": best_state, "window": WINDOW,
                "val_rmse_norm": best}, WEIGHTS_PATH)
    log(f"salvato {WEIGHTS_PATH} ({time.time() - t0:.0f} s)")
    return best


class GruPredictor:
    """Stessa interfaccia dei predittori di motion_model."""
    name = "gru"
    _net = None

    def __init__(self, cfg, weights=WEIGHTS_PATH):
        self.cfg = cfg
        self.h = cfg.horizon_samples
        if GruPredictor._net is None:
            if not os.path.exists(weights):
                raise FileNotFoundError(
                    f"pesi della GRU non trovati: {weights}\n"
                    "eseguire: python motion_predictor.py --train")
            ck = torch.load(weights, map_location="cpu")
            net = GruNet()
            net.load_state_dict(ck["state"])
            net.eval()
            GruPredictor._net = net
        self.net = GruPredictor._net
        self.reset()

    def reset(self):
        self.buf = []

    @torch.no_grad()
    def step(self, s):
        self.buf.append(float(s))
        if len(self.buf) > WINDOW:
            self.buf.pop(0)
        if len(self.buf) < WINDOW:
            return float(s)
        X, last, scale = _windows(np.asarray(self.buf), self.h)
        d = self.net(torch.from_numpy(X)).item()
        return float(last[0] + d * scale[0])

    @torch.no_grad()
    def series(self, s):
        """Tutta la serie in blocco: identico a step() ripetuto."""
        out = np.asarray(s, float).copy()
        if len(s) < WINDOW:
            return out
        X, last, scale = _windows(np.asarray(s, float), self.h)
        d = np.concatenate([
            self.net(torch.from_numpy(X[i:i + 8192])).numpy()
            for i in range(0, len(X), 8192)])
        out[WINDOW - 1:] = last + d * scale
        return out


def register():
    """Rende disponibile "gru" in motion_model.make_predictor."""
    mm.register_predictor("gru", GruPredictor)


def benchmark(n=20, duration_s=300.0, log=print):
    """Confronto dei predittori su tracce mai viste.

    Due popolazioni: il paziente di riferimento (MotionConfig di default, semi
    diversi) e pazienti a parametri casuali. Si riporta la mediana sulle
    tracce dell'RMSE di predizione, e il p95 dell'errore 3D di inseguimento.
    """
    register()
    names = mm.predictor_names()
    out = {}
    for pop in ("riferimento", "casuali"):
        pred = {k: [] for k in names}
        trk = {k: [] for k in names}
        for i in range(n):
            if pop == "riferimento":
                c = mm.MotionConfig(seed=i)
                rng = np.random.default_rng(i)
            else:
                rng = np.random.default_rng(500 + i)
                c = random_config(rng).replace(seed=500 + i)
            tr = mm.generate_trace(c, duration_s, rng)
            h = c.horizon_samples
            skip = int(20 * c.fs_hz)
            for k in names:
                p = mm.predict_series(mm.make_predictor(k, c), tr.surrogate)
                e = mm.prediction_error(tr.surrogate, p, h, skip)
                pred[k].append(float(np.sqrt(np.mean(e ** 2))))
                r = mm.simulate_tracking(tr, k, c)
                trk[k].append(mm.error_stats(r.residual)["p95_3d"])
        out[pop] = {k: {"pred_rmse_mm": float(np.median(pred[k])),
                        "track_p95_mm": float(np.median(trk[k]))}
                    for k in names}
        log(f"\n--- pazienti {pop} ({n} tracce da {duration_s:.0f} s) ---")
        log(f"{'predittore':12s} {'RMSE pred. 115 ms':>18s} "
            f"{'p95 inseguimento':>18s}")
        for k in names:
            log(f"{k:12s} {out[pop][k]['pred_rmse_mm']:15.3f} mm "
                f"{out[pop][k]['track_p95_mm']:15.2f} mm")
    path = os.path.join(config.OUTPUT_DIR, "motion_benchmark.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    log(f"\nsalvato {path}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--benchmark", action="store_true")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--n", type=int, default=20)
    a = ap.parse_args()
    if a.train:
        train(epochs=a.epochs)
    if a.benchmark:
        benchmark(n=a.n)
