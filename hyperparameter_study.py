#!/usr/bin/env python3
import os, sys, time, itertools
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

from splits import load_data, make_splits, verify_against_legacy

# ============================================================
# Fixed hyperparameters (from run_experiments.py)
# ============================================================
from splits import SEED
PRIOR_SIGMA = 1.0
TOTAL_EPOCHS = 500
STAGE_A_EPOCHS = 100
KL_BETA = 0.1
BATCH_SIZE = 64
LR = 5e-4
LR_MIN = 1e-6
GRAD_CLIP = 1.0
CV_FOLDS = 3
DEVICE = torch.device('cpu')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

np.random.seed(SEED)
torch.manual_seed(SEED)

# ============================================================
# Neuron configurations to test
# ============================================================
LAYER1_CANDIDATES = [64, 128, 256]
LAYER2_CANDIDATES = [32, 64, 128]
LAYER3_CANDIDATES = [16, 32, 64]

# Filter: enforce decreasing or equal widths (funnel shape)
CONFIGS = [
    (l1, l2, l3)
    for l1 in LAYER1_CANDIDATES
    for l2 in LAYER2_CANDIDATES
    for l3 in LAYER3_CANDIDATES
    if l1 >= l2 >= l3
]

print(f"Total configurations to evaluate: {len(CONFIGS)}")
for i, cfg in enumerate(CONFIGS):
    print(f"  [{i+1:2d}] {list(cfg)}")

# ============================================================
# Data Loading
# ============================================================
# load_data() is imported from splits.py

# ============================================================
# Model
# ============================================================
class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features, prior_sigma=1.0):
        super().__init__()
        self.prior_sigma = prior_sigma
        self.weight_mu  = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_rho = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu    = nn.Parameter(torch.empty(out_features))
        self.bias_rho   = nn.Parameter(torch.empty(out_features))
        self._init()

    def _init(self):
        nn.init.kaiming_normal_(self.weight_mu, nonlinearity='relu')
        nn.init.constant_(self.weight_rho, -3.0)
        nn.init.zeros_(self.bias_mu)
        nn.init.constant_(self.bias_rho, -3.0)

    def forward(self, x, sample=True):
        if sample:
            ws = F.softplus(self.weight_rho)
            bs = F.softplus(self.bias_rho)
            w  = self.weight_mu + ws * torch.randn_like(ws)
            b  = self.bias_mu  + bs * torch.randn_like(bs)
        else:
            w = self.weight_mu
            b = self.bias_mu
        return F.linear(x, w, b)

    def kl(self):
        ws = F.softplus(self.weight_rho)
        bs = F.softplus(self.bias_rho)
        ps2 = self.prior_sigma ** 2
        lps = np.log(self.prior_sigma)
        k  = (0.5*(ws**2 + self.weight_mu**2)/ps2 - 0.5 - torch.log(ws) + lps).sum()
        k += (0.5*(bs**2 + self.bias_mu**2)/ps2 - 0.5 - torch.log(bs) + lps).sum()
        return k


class BNN(nn.Module):
    def __init__(self, d_in, hidden, ps=PRIOR_SIGMA):
        super().__init__()
        layers = []
        prev = d_in
        for h in hidden:
            layers.append(BayesianLinear(prev, h, ps))
            prev = h
        self.hidden = nn.ModuleList(layers)
        self.head_y = BayesianLinear(prev, 1, ps)
        self.head_v = BayesianLinear(prev, 1, ps)

    def forward(self, x, sample=True):
        h = x
        for layer in self.hidden:
            h = F.relu(layer(h, sample=sample))
        y    = self.head_y(h, sample=sample).squeeze(-1)
        logv = self.head_v(h, sample=sample).squeeze(-1)
        v    = F.softplus(logv) + 1e-6
        return y, v

    def kl(self):
        return sum(l.kl() for l in self.hidden) + self.head_y.kl() + self.head_v.kl()

    def param_count(self):
        return sum(p.numel() for p in self.parameters())

# ============================================================
# Training
# ============================================================
def train_bnn(model, Xtr, ytr):
    n_train = len(ytr)
    ds = TensorDataset(torch.tensor(Xtr), torch.tensor(ytr))
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.0)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, TOTAL_EPOCHS, eta_min=LR_MIN)

    stage_b_len = TOTAL_EPOCHS - STAGE_A_EPOCHS
    kl_warmup_epochs = int(0.2 * stage_b_len)

    model.train()
    for ep in range(TOTAL_EPOCHS):
        for xb, yb in dl:
            opt.zero_grad()
            if ep < STAGE_A_EPOCHS:
                yp, _ = model(xb, sample=False)
                loss = F.mse_loss(yp, yb)
            else:
                yp, vp = model(xb, sample=True)
                nll = 0.5 * (torch.log(vp) + (yb - yp)**2 / vp).mean()
                stage_b_ep = ep - STAGE_A_EPOCHS
                beta = KL_BETA * min(1.0, stage_b_ep / max(kl_warmup_epochs, 1))
                kl = model.kl() / n_train
                loss = nll + beta * kl
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()
        sch.step()
    return model

# ============================================================
# Main study
# ============================================================
def main():
    t0 = time.time()
    print("\n" + "=" * 65)
    print(" BNN Hyperparameter Study: Neuron Count per Layer")
    print("=" * 65)

    X_raw, y_raw = load_data()
    n, p = X_raw.shape
    print(f"\nData: {n} specimens, {p} features")

    # Proper-training rows only (calibration/test rows are not used here)
    idx_fit, idx_cal, idx_te = make_splits(n, SEED)
    verify_against_legacy(X_raw, y_raw, idx_fit, idx_cal, idx_te, SEED)
    Xfit_raw, yfit = X_raw[idx_fit], y_raw[idx_fit]
    print(f"Proper-training rows used for the search: {len(yfit)}")

    kf = KFold(CV_FOLDS, shuffle=True, random_state=SEED + 999)

    results = []

    for ci, cfg in enumerate(CONFIGS):
        hidden = list(cfg)
        t1 = time.time()
        print(f"\n[{ci+1}/{len(CONFIGS)}] Hidden = {hidden}", end="", flush=True)

        fold_mse = []
        fold_mae = []

        for fold_i, (ti, vi) in enumerate(kf.split(Xfit_raw)):
            # Scaler fitted on this fold's training portion only
            sc = StandardScaler().fit(Xfit_raw[ti])
            Xti = sc.transform(Xfit_raw[ti]).astype(np.float32)
            Xvi = sc.transform(Xfit_raw[vi]).astype(np.float32)

            torch.manual_seed(SEED + fold_i)
            model = BNN(p, hidden)
            model = train_bnn(model, Xti, yfit[ti])

            model.eval()
            with torch.no_grad():
                yp, _ = model(torch.tensor(Xvi), sample=False)
                yp = yp.numpy()

            fold_mse.extend((yfit[vi] - yp) ** 2)
            fold_mae.extend(np.abs(yfit[vi] - yp))

        rmse = np.sqrt(np.mean(fold_mse))
        mae  = np.mean(fold_mae)
        r2   = 1 - np.sum(fold_mse) / np.sum((yfit - yfit.mean()) ** 2)

        # Count parameters for this config
        dummy = BNN(p, hidden)
        n_params = dummy.param_count()

        elapsed_cfg = time.time() - t1
        print(f"  →  RMSE={rmse:.3f}  MAE={mae:.3f}  R²={r2:.3f}"
              f"  params={n_params:,}  ({elapsed_cfg:.0f}s)")

        results.append({
            'layer_1': cfg[0],
            'layer_2': cfg[1],
            'layer_3': cfg[2],
            'config': str(hidden),
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'params': n_params,
            'time_s': elapsed_cfg,
        })

    # --------------------------------------------------------
    # Results table sorted by RMSE
    # --------------------------------------------------------
    df = pd.DataFrame(results).sort_values('rmse').reset_index(drop=True)
    df.index += 1  # 1-based ranking
    df.index.name = 'rank'

    print("\n" + "=" * 65)
    print(" RESULTS (sorted by CV-RMSE, ascending)")
    print("=" * 65)
    print(f"\n{'Rank':<6}{'Config':<20}{'RMSE':>8}{'MAE':>8}{'R²':>8}{'Params':>10}{'Time':>8}")
    print("-" * 68)
    for rank, row in df.iterrows():
        print(f"{rank:<6}{row['config']:<20}{row['rmse']:8.3f}{row['mae']:8.3f}"
              f"{row['r2']:8.3f}{row['params']:>10,}{row['time_s']:7.0f}s")

    # Best configuration
    best = df.iloc[0]
    print(f"\n{'=' * 65}")
    print(f" BEST CONFIGURATION")
    print(f"{'=' * 65}")
    print(f"  Hidden dims : {best['config']}")
    print(f"  CV-RMSE     : {best['rmse']:.3f} MPa")
    print(f"  CV-MAE      : {best['mae']:.3f} MPa")
    print(f"  CV-R²       : {best['r2']:.3f}")
    print(f"  Parameters  : {best['params']:,}")

    # No retraining / test evaluation here: the selected architecture is
    # read by run_experiments.py, which fits the final model on the
    # proper-training rows and touches the test rows exactly once.

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------
    csv_path = os.path.join(BASE_DIR, 'hyperparameter_study_results.csv')
    df.to_csv(csv_path)
    print(f"\n  Results saved to: {csv_path}")

    elapsed = time.time() - t0
    print(f"\n{'=' * 65}")
    print(f" Study completed in {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"{'=' * 65}")

    return df


if __name__ == '__main__':
    main()
