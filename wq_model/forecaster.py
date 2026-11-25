"""LSTM forecaster - paper Section III-E.

Input: a 3-h window (36 steps) of the 7 smoothed, standardized sensor signals, the base
index W^b_t and a sine-cosine time-of-day encoding (10 inputs per step).
Output: the change W_{t+h} - W^b_t for h = 15, 30 and 60 min.
The early-warning forecast is the worst of the three, W^f_t = max_h W_{t+h}.
"""
import numpy as np
import torch
import torch.nn as nn

from .cwqi import SENSORS
from .simulator import STEPS_PER_HOUR

WINDOW = 36                                  # 3 h
HORIZONS = [3, 6, 12]                        # 15, 30, 60 min
HORIZON_MIN = [h * 5 for h in HORIZONS]


class LSTMNet(nn.Module):
    def __init__(self, n_inputs=10, hidden=64):
        super().__init__()
        self.lstm = nn.LSTM(n_inputs, hidden, num_layers=2, batch_first=True, dropout=0.2)
        self.head = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(), nn.Linear(32, len(HORIZONS)))

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1])


class CWQIForecaster:
    def __init__(self):
        self.net = LSTMNet()
        self.stats = {}

    # ---------------------------------------------------------------- inputs
    def _inputs(self, smoothed, base, step_index):
        s = (smoothed[SENSORS].values - self.stats["mu"]) / self.stats["sd"]
        b = (np.asarray(base) - self.stats["b_mu"]) / self.stats["b_sd"]
        tod = (np.asarray(step_index) % (24 * STEPS_PER_HOUR)) / (24 * STEPS_PER_HOUR)
        return np.column_stack([s, b, np.sin(2 * np.pi * tod), np.cos(2 * np.pi * tod)]).astype(np.float32)

    @staticmethod
    def _windows(Z, idx):
        return np.stack([Z[i - WINDOW + 1:i + 1] for i in idx])

    # ---------------------------------------------------------------- training
    def fit(self, smoothed, base, target, train_idx, val_idx, epochs=40, patience=5, seed=0, log=print):
        torch.manual_seed(seed)
        sm = smoothed[SENSORS].values
        self.stats = dict(mu=sm[train_idx].mean(0), sd=sm[train_idx].std(0) + 1e-9,
                          b_mu=float(base[train_idx].mean()), b_sd=float(base[train_idx].std()))
        Z = self._inputs(smoothed, base, np.arange(len(base)))
        n = len(base)
        Y = np.column_stack([np.r_[target[h:], np.full(h, np.nan)] for h in HORIZONS]) - base[:, None]
        ok = lambda idx: idx[(idx >= WINDOW - 1) & (idx < n - max(HORIZONS))]
        tr, va = ok(np.asarray(train_idx)), ok(np.asarray(val_idx))
        self.stats["y_mu"] = float(np.nanmean(Y[tr])); self.stats["y_sd"] = float(np.nanstd(Y[tr]))
        norm = lambda y: torch.tensor((y - self.stats["y_mu"]) / self.stats["y_sd"], dtype=torch.float32)
        Xtr, Ytr = torch.tensor(self._windows(Z, tr)), norm(Y[tr])
        Xva, Yva = torch.tensor(self._windows(Z, va)), norm(Y[va])

        opt = torch.optim.Adam(self.net.parameters(), lr=1e-3)
        loss_fn = nn.MSELoss()
        best, best_state, bad = np.inf, None, 0
        for epoch in range(epochs):
            self.net.train()
            perm = torch.randperm(len(Xtr))
            for j in range(0, len(perm), 256):
                b = perm[j:j + 256]
                opt.zero_grad()
                loss = loss_fn(self.net(Xtr[b]), Ytr[b])
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                opt.step()
            self.net.eval()
            with torch.no_grad():
                val = float(loss_fn(self.net(Xva), Yva))
            log(f"  LSTM epoch {epoch + 1:2d}  validation loss {val:.4f}")
            if val < best - 1e-4:
                best, bad = val, 0
                best_state = {k: v.clone() for k, v in self.net.state_dict().items()}
            else:
                bad += 1
                if bad >= patience:
                    break
        self.net.load_state_dict(best_state)
        return self

    # ---------------------------------------------------------------- prediction
    def predict(self, smoothed, base, step_index, rows):
        """Forecasts (n_rows x 3 horizons) for the given row positions (each needs 35 rows of history)."""
        Z = self._inputs(smoothed, base, step_index)
        rows = np.asarray(rows)
        self.net.eval()
        out = []
        with torch.no_grad():
            for j in range(0, len(rows), 2048):
                out.append(self.net(torch.tensor(self._windows(Z, rows[j:j + 2048]))).numpy())
        delta = np.vstack(out) * self.stats["y_sd"] + self.stats["y_mu"]
        return delta + np.asarray(base)[rows, None]
