#!/usr/bin/env python3
import os, sys, time, json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
import warnings
warnings.filterwarnings('ignore')

from splits import load_data, make_splits, verify_against_legacy

# ============================================================
# Configuration (matching paper Table 1)
# ============================================================
from splits import SEED

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Architecture selected by hyperparameter_study.py (CV on proper-training rows only)
_HP_CSV = os.path.join(BASE_DIR, 'hyperparameter_study_results.csv')
if not os.path.exists(_HP_CSV):
    sys.exit("Run hyperparameter_study.py first (writes hyperparameter_study_results.csv).")
_hp_best = pd.read_csv(_HP_CSV).sort_values('rmse').iloc[0]
HIDDEN_DIMS = [int(_hp_best['layer_1']), int(_hp_best['layer_2']), int(_hp_best['layer_3'])]
PRIOR_SIGMA = 1.0
TOTAL_EPOCHS = 500
STAGE_A_EPOCHS = 100
M_SAMPLES = 50
ALPHA = 0.10                              # nominal target 1 - alpha = 0.90
BATCH_SIZE = 64
LR = 5e-4
WEIGHT_DECAY = 1e-3      # deterministic baselines only; BNN uses KL instead
LR_MIN = 1e-6
GRAD_CLIP = 1.0
KL_BETA_CANDIDATES = [0.01, 0.05, 0.1, 0.2, 0.5]   # sweep candidates
KL_BETA = 0.1            # will be overwritten by sweep
DEVICE = torch.device('cpu')

np.random.seed(SEED)
torch.manual_seed(SEED)

FEATURE_NAMES_SHORT = ['Cement', 'Slag', 'Fly ash', 'Water',
                       'SP', 'Coarse agg.', 'Fine agg.', 'Age']

# ============================================================
# 1. Data Loading
# ============================================================
# load_data() is imported from splits.py

# ============================================================
# 2. Model Definitions
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
    def __init__(self, d_in, hidden=HIDDEN_DIMS, ps=PRIOR_SIGMA):
        super().__init__()
        layers = []
        prev = d_in
        for h in hidden:
            layers.append(BayesianLinear(prev, h, ps)); prev = h
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

    @torch.no_grad()
    def mc_predict(self, x, M=M_SAMPLES):
        """Draw M posterior weight samples; each gives different (y, v)."""
        self.eval()
        ys, vs = [], []
        for _ in range(M):
            y, v = self.forward(x, sample=True)   # stochastic
            ys.append(y.cpu().numpy()); vs.append(v.cpu().numpy())
        return np.stack(ys), np.stack(vs)   # (M, n)


class DetNN(nn.Module):
    def __init__(self, d_in, hidden=HIDDEN_DIMS):
        super().__init__()
        layers = []
        prev = d_in
        for h in hidden:
            layers.append(nn.Linear(prev, h)); layers.append(nn.ReLU()); prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)

# ============================================================
# 3. Training
# ============================================================
def train_bnn(model, Xtr, ytr, verbose=False):
    """Staged BNN training:
    Stage A — deterministic forward (sample=False), MSE only, NO KL
              → learns good features like a standard NN
    Stage B — stochastic forward (sample=True), hetero-NLL + KL
              → KL scaled by KL_BETA / n_data  (cold posterior)
              → KL linearly warmed up over first 20 % of Stage B
    No early stopping — KL regularises in place of a val split,
    avoiding the loss of 10 % training data per fold.
    """
    n_train = len(ytr)
    ds  = TensorDataset(torch.tensor(Xtr), torch.tensor(ytr))
    dl  = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)
    # No weight decay for BNN — KL provides the regularisation
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.0)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, TOTAL_EPOCHS, eta_min=LR_MIN)

    stage_b_len = TOTAL_EPOCHS - STAGE_A_EPOCHS
    kl_warmup_epochs = int(0.2 * stage_b_len)

    model.train()
    for ep in range(TOTAL_EPOCHS):
        for xb, yb in dl:
            opt.zero_grad()

            if ep < STAGE_A_EPOCHS:
                # --- Stage A: deterministic, MSE only ---
                yp, _ = model(xb, sample=False)
                loss = F.mse_loss(yp, yb)
            else:
                # --- Stage B: stochastic, hetero-NLL + cold-posterior KL ---
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


def train_det(model, Xtr, ytr, epochs=TOTAL_EPOCHS):
    ds  = TensorDataset(torch.tensor(Xtr), torch.tensor(ytr))
    dl  = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs, eta_min=LR_MIN)
    model.train()
    for _ in range(epochs):
        for xb, yb in dl:
            opt.zero_grad(); F.mse_loss(model(xb), yb).backward(); opt.step()
        sch.step()
    return model

# ============================================================
# 4. Conformal helpers
# ============================================================
def conf_quantile(scores, alpha):
    """Split-conformal quantile: the k-th smallest score, k = ceil((n+1)(1-alpha))."""
    n = len(scores)
    k = int(np.ceil((n + 1) * (1 - alpha) - 1e-9))
    return np.inf if k > n else float(np.sort(scores)[k - 1])

def metrics(yt, yp, lo, up):
    rmse = np.sqrt(((yt-yp)**2).mean())
    r2   = 1 - ((yt-yp)**2).sum() / ((yt-yt.mean())**2).sum()
    mae  = np.abs(yt-yp).mean()
    ecr  = ((yt>=lo)&(yt<=up)).mean()
    mbw  = (up-lo).mean()
    return dict(RMSE=rmse, R2=r2, MAE=mae, ECR=ecr, MBW=mbw)

STRAT_BINS = [(0, 10), (10, 20), (20, 30), (30, 40), (40, 50), (50, 60), (60, 70), (70, 90)]
STRAT_KEYS   = [f'{lo}--{hi}' for lo, hi in STRAT_BINS]
STRAT_LABELS = [f'{lo}–{hi}' for lo, hi in STRAT_BINS]
STRAT_LABELS_TEX = [f'{lo}--{hi}' for lo, hi in STRAT_BINS]

def strat_metrics(yt, lo, up):
    out = {}
    for (blo, bhi), key in zip(STRAT_BINS, STRAT_KEYS):
        mask = (yt >= blo) & (yt < bhi) if bhi < 90 else (yt >= blo)
        if mask.sum():
            out[key] = dict(
                ECR=((yt[mask]>=lo[mask])&(yt[mask]<=up[mask])).mean(),
                MBW=(up[mask]-lo[mask]).mean(), n=int(mask.sum()))
    return out

# ============================================================
# 5. Main experiment pipeline
# ============================================================
def main():
    t0 = time.time()
    print("="*60)
    print(" BNN Conformal Prediction — Full Experiment Suite")
    print("="*60)

    # --- Load ---
    X_raw, y_raw = load_data()
    n, p = X_raw.shape
    print(f"\nData: {n} specimens, {p} features")
    print(f"Strength range: {y_raw.min():.1f} – {y_raw.max():.1f} MPa")

    # --- Fixed partition: proper-training / calibration / test (splits.py) ---
    idx_fit, idx_cal, idx_te = make_splits(n, SEED)
    print("\nVerifying split against the legacy implementation:")
    verify_against_legacy(X_raw, y_raw, idx_fit, idx_cal, idx_te, SEED)
    Xfit_raw, yfit = X_raw[idx_fit], y_raw[idx_fit]
    Xcal_raw, ycal = X_raw[idx_cal], y_raw[idx_cal]
    Xte_raw,  yte  = X_raw[idx_te],  y_raw[idx_te]
    print(f"Proper-training: {len(yfit)}, Calibration: {len(ycal)}, Test: {len(yte)}")
    print(f"Architecture (from hyperparameter_study.py): {HIDDEN_DIMS}")

    d = p  # input dim

    # ===========================================================
    # C0  KL_BETA sweep — 3-fold CV on proper-training rows only,
    #     scaler refitted inside each fold on its training portion
    # ===========================================================
    print("\n[C0] KL_BETA sweep on proper-training rows …")
    global KL_BETA
    best_beta, best_sweep_rmse = KL_BETA_CANDIDATES[0], 1e30
    beta_sweep_rmse = []
    kf_sweep = KFold(3, shuffle=True, random_state=SEED + 999)
    for beta_c in KL_BETA_CANDIDATES:
        KL_BETA = beta_c
        sweep_errs = []
        for s_ti, s_vi in kf_sweep.split(Xfit_raw):
            sc_f = StandardScaler().fit(Xfit_raw[s_ti])
            Xs_ti = sc_f.transform(Xfit_raw[s_ti]).astype(np.float32)
            Xs_vi = sc_f.transform(Xfit_raw[s_vi]).astype(np.float32)
            torch.manual_seed(SEED + 999)
            m_s = train_bnn(BNN(d), Xs_ti, yfit[s_ti], verbose=False)
            m_s.eval()
            with torch.no_grad():
                yp_s, _ = m_s(torch.tensor(Xs_vi), sample=False)
            sweep_errs.extend((yfit[s_vi] - yp_s.numpy())**2)
        rmse_c = np.sqrt(np.mean(sweep_errs))
        beta_sweep_rmse.append(rmse_c)
        print(f"  β={beta_c:.2f}  RMSE={rmse_c:.3f}")
        if rmse_c < best_sweep_rmse:
            best_sweep_rmse = rmse_c; best_beta = beta_c
    KL_BETA = best_beta
    print(f"  → selected KL_BETA = {KL_BETA}")

    # --- Final scaler: refitted on all proper-training rows only ---
    sc = StandardScaler().fit(Xfit_raw)
    Xfit = sc.transform(Xfit_raw).astype(np.float32)
    Xcal = sc.transform(Xcal_raw).astype(np.float32)
    Xte  = sc.transform(Xte_raw).astype(np.float32)

    # ===========================================================
    # A/B  Deterministic network on proper-training rows.
    #      Used both for the heteroscedasticity diagnostic (A) and as the
    #      constant-band split conformal baseline (B).
    # ===========================================================
    print("\n[A/B] Training deterministic network on proper-training rows …")
    torch.manual_seed(SEED)
    spl_m = train_det(DetNN(d), Xfit, yfit)
    spl_m.eval()
    with torch.no_grad():
        cal_p  = spl_m(torch.tensor(Xcal)).numpy()
        spl_te = spl_m(torch.tensor(Xte)).numpy()

    # A — heteroscedasticity diagnostic on test residuals
    det_yhat = spl_te
    det_res = np.abs(yte - det_yhat)
    det_rmse = np.sqrt(((yte-det_yhat)**2).mean())
    det_r2   = 1 - ((yte-det_yhat)**2).sum()/((yte-yte.mean())**2).sum()
    print(f"  RMSE={det_rmse:.2f}  R²={det_r2:.3f}")

    resid_raw = yte - det_yhat
    exog = sm.add_constant(Xte)
    bp_stat, bp_pval, _, _ = het_breuschpagan(resid_raw, exog)
    print(f"  Breusch–Pagan: χ²={bp_stat:.2f}, p={bp_pval:.2e}")

    # B — split conformal (constant band): calibration rows used once here
    delta_spl = conf_quantile(np.abs(ycal - cal_p), ALPHA)
    spl_lo, spl_up = spl_te - delta_spl, spl_te + delta_spl
    sm_spl = metrics(yte, spl_te, spl_lo, spl_up)
    print(f"  δ={delta_spl:.2f}  ECR={sm_spl['ECR']:.3f}  MBW={sm_spl['MBW']:.2f}")

    # ===========================================================
    # C  BNN split conformal (proposed, normalised by total uncertainty)
    # ===========================================================
    print(f"\n[C] BNN split conformal (proposed, β={KL_BETA}) …")
    torch.manual_seed(SEED + 100)
    # Train one BNN on the proper-training subset (same partition as Split)
    bnn = train_bnn(BNN(d), Xfit, yfit, verbose=False)
    bnn.eval()

    # --- same M posterior draws for every calibration and test row ---
    U_FLOOR = 1e-6       # pre-specified floor, identical for calibration and test rows
    ys, vs = bnn.mc_predict(torch.cat([torch.tensor(Xcal), torch.tensor(Xte)]), M_SAMPLES)
    ybar = ys.mean(0)                                   # MC posterior mean (Eq. avg_y)
    u_e = np.maximum(np.sqrt(ys.var(0, ddof=1)), U_FLOOR)
    u_a = np.maximum(np.sqrt(vs.mean(0)), U_FLOOR)
    u_t = np.maximum(np.sqrt(u_e**2 + u_a**2), U_FLOOR)
    nc = len(ycal)
    ybar_c, ybar_t = ybar[:nc], ybar[nc:]
    ue_c, ua_c, ut_c = u_e[:nc], u_a[:nc], u_t[:nc]
    ue_t, ua_t, ut_t = u_e[nc:], u_a[nc:], u_t[nc:]
    eta = conf_quantile(np.abs(ycal - ybar_c) / ut_c, ALPHA)

    bnn_yh = ybar_t
    bnn_ue, bnn_ua, bnn_ut = ue_t, ua_t, ut_t
    bnn_lo = bnn_yh - eta * bnn_ut
    bnn_up = bnn_yh + eta * bnn_ut
    sm_bnn = metrics(yte, bnn_yh, bnn_lo, bnn_up)
    print(f"\n  η={eta:.3f}  RMSE={sm_bnn['RMSE']:.2f}  R²={sm_bnn['R2']:.3f}")
    print(f"  ECR={sm_bnn['ECR']:.3f}  MBW={sm_bnn['MBW']:.2f}")

    # ===========================================================
    # D  Conformalized quantile regression (adaptive comparator)
    #    Same architecture, proper-training rows; calibration rows once.
    # ===========================================================
    from repeated_experiments import QuantNN, train_quant
    print("\n[D] Conformalized quantile regression …")
    torch.manual_seed(SEED + 200)
    qnn = train_quant(sys.modules[__name__], QuantNN(d, list(HIDDEN_DIMS)), Xfit, yfit, ALPHA).eval()
    with torch.no_grad():
        q_c = np.sort(qnn(torch.tensor(Xcal)).numpy(), 1)
        q_t = np.sort(qnn(torch.tensor(Xte)).numpy(), 1)
    Q_cqr = conf_quantile(np.maximum(q_c[:, 0] - ycal, ycal - q_c[:, 1]), ALPHA)
    cqr_lo, cqr_up = q_t[:, 0] - Q_cqr, q_t[:, 1] + Q_cqr
    sm_cqr = metrics(yte, q_t.mean(1), cqr_lo, cqr_up)
    print(f"  Q={Q_cqr:.2f}  ECR={sm_cqr['ECR']:.3f}  MBW={sm_cqr['MBW']:.2f}")

    # Calibration-set held-out predictions, used in the analyses below.
    cal_yt = np.asarray(ycal)
    cal_yp = np.asarray(ybar_c)
    cal_ue = np.asarray(ue_c)
    cal_ua = np.asarray(ua_c)
    cal_ut = np.asarray(ut_c)
    cal_xr = np.asarray(Xcal_raw)
    oof_yt, oof_yp = cal_yt, cal_yp
    oof_ua = cal_ua
    oof_xr = cal_xr

    # ===========================================================
    # Stratified metrics (2 methods, alpha = 0.10)
    # ===========================================================
    strat_spl = strat_metrics(yte, spl_lo, spl_up)
    strat_bnn = strat_metrics(yte, bnn_lo, bnn_up)

    # ===========================================================
    # Print summary table
    # ===========================================================
    print("\n" + "="*60)
    print(" SUMMARY OF RESULTS")
    print("="*60)
    print(f"\n{'Method':<30} {'RMSE':>6} {'R²':>6} {'MAE':>6} {'ECR':>6} {'MBW':>7}")
    print("-"*65)
    for name, m in [('Split conformal',     sm_spl),
                    ('BNN split conformal', sm_bnn)]:
        print(f"{name:<30} {m['RMSE']:6.2f} {m['R2']:6.3f} {m['MAE']:6.2f} {m['ECR']:6.3f} {m['MBW']:7.2f}")

    print(f"\n{'Stratified MBW':>30}", end='')
    for r in STRAT_LABELS:
        print(f" {r:>10}", end='')
    print()
    print("-"*(30 + 11*len(STRAT_LABELS)))
    for name, st in [('Split conformal',     strat_spl),
                     ('BNN split conformal', strat_bnn)]:
        print(f"{name:<30}", end='')
        for k in STRAT_KEYS:
            if k in st:
                print(f" {st[k]['MBW']:10.2f}", end='')
            else:
                print(f" {'—':>10}", end='')
        print()

    print(f"\n{'Stratified ECR':>30}", end='')
    for r in STRAT_LABELS:
        print(f" {r:>10}", end='')
    print()
    print("-"*(30 + 11*len(STRAT_LABELS)))
    for name, st in [('Split conformal',     strat_spl),
                     ('BNN split conformal', strat_bnn)]:
        print(f"{name:<30}", end='')
        for k in STRAT_KEYS:
            if k in st:
                print(f" {st[k]['ECR']:10.3f}", end='')
            else:
                print(f" {'—':>10}", end='')
        print()

    # Residual std by strength range
    print("\n  Residual std (deterministic baseline):")
    for (blo, bhi), lab in zip(STRAT_BINS, STRAT_LABELS):
        mask = (yte >= blo) & (yte < bhi) if bhi < 90 else (yte >= blo)
        if mask.sum():
            print(f"    {lab} MPa: σ = {det_res[mask].std():.2f} (n={mask.sum()})")

    # Spearman correlations between features and u_ale (OOF)
    print("\n  Spearman ρ (feature vs u_ale, out-of-fold):")
    for i, fn in enumerate(FEATURE_NAMES_SHORT):
        rho, pv = stats.spearmanr(oof_xr[:, i], oof_ua)
        print(f"    {fn:<14} ρ={rho:+.3f}  p={pv:.2e}")

    # Water-to-binder ratio analysis (Q4 supplement)
    # w/b = water / (cement + slag + fly ash)
    # Feature indices: 0=Cement, 1=Slag, 2=Fly ash, 3=Water
    oof_binder = oof_xr[:, 0] + oof_xr[:, 1] + oof_xr[:, 2]
    oof_wb = np.where(oof_binder > 0, oof_xr[:, 3] / oof_binder, np.nan)
    wb_valid = ~np.isnan(oof_wb)

    WB_BINS = [(0.0, 0.30), (0.30, 0.40), (0.40, 0.50), (0.50, 0.60), (0.60, 1.0)]
    WB_LABELS = ['<0.30', '0.30–0.40', '0.40–0.50', '0.50–0.60', '>0.60']
    print("\n  w/b ratio vs aleatoric uncertainty (out-of-fold):")
    print(f"    {'w/b range':<14} {'n':>5} {'mean u_ale':>10} {'med u_ale':>10} {'mean |err|':>10}")
    oof_abs_err = np.abs(oof_yt - oof_yp)
    for (blo, bhi), lab in zip(WB_BINS, WB_LABELS):
        mask = wb_valid & (oof_wb >= blo) & (oof_wb < bhi)
        if mask.sum():
            print(f"    {lab:<14} {mask.sum():5d} {oof_ua[mask].mean():10.2f} "
                  f"{np.median(oof_ua[mask]):10.2f} {oof_abs_err[mask].mean():10.2f}")
    rho_wb, pv_wb = stats.spearmanr(oof_wb[wb_valid], oof_ua[wb_valid])
    print(f"    Spearman ρ(w/b, u_ale) = {rho_wb:+.3f}  p = {pv_wb:.2e}")

    # ===========================================================
    # LaTeX TABLE code
    # ===========================================================
    print("\n" + "="*60)
    print(" LaTeX TABLE 2  (copy into paper)")
    print("="*60)
    print(r"""
\begin{table}[ht!]
\centering
\caption{Overall prediction and band performance on the test set (206 specimens, target coverage $1{-}\alpha = 0.90$).}
\label{tab:overall}
\begin{tabular}{lccccc}
\toprule
\textbf{Method} & \textbf{RMSE} & $\boldsymbol{R^2}$ & \textbf{MAE} & \textbf{ECR} & \textbf{MBW} \\
 & (MPa) & & (MPa) & & (MPa) \\
\midrule""")
    for name, m in [('Split conformal (constant)', sm_spl),
                    ('BNN split conformal (proposed)', sm_bnn)]:
        print(f"{name} & {m['RMSE']:.2f} & {m['R2']:.3f} & {m['MAE']:.2f} & {m['ECR']:.3f} & {m['MBW']:.2f} \\\\")
    print(r"""\bottomrule
\end{tabular}
\end{table}""")

    print("\n" + "="*60)
    print(" LaTeX TABLE 3  (copy into paper)")
    print("="*60)
    n_bins = len(STRAT_KEYS)
    col_spec = 'll' + 'c' * n_bins
    col_range = f'{{3-{n_bins+2}}}'
    header_cells = ' & '.join(f'\\textbf{{{l}}}' for l in STRAT_LABELS_TEX)
    print(f"""
\\begin{{table}}[ht!]
\\centering
\\caption{{Stratified band performance by compressive-strength range.}}
\\label{{tab:stratified}}
\\begin{{tabular}}{{{col_spec}}}
\\toprule
 & & \\multicolumn{{{n_bins}}}{{c}}{{\\textbf{{Strength range (MPa)}}}} \\\\
\\cmidrule(lr){col_range}
\\textbf{{Method}} & \\textbf{{Metric}} & {header_cells} \\\\
\\midrule""")
    for name, st in [('Split conformal', strat_spl),
                     ('BNN (proposed)', strat_bnn)]:
        ecr_vals = ' & '.join(f"{st[k]['ECR']:.3f}" if k in st else '---' for k in STRAT_KEYS)
        mbw_vals = ' & '.join(f"{st[k]['MBW']:.2f}" if k in st else '---' for k in STRAT_KEYS)
        n_vals   = ' & '.join(f"{st[k]['n']}" if k in st else '---' for k in STRAT_KEYS)
        print(f"\\multirow{{3}}{{*}}{{{name}}}")
        print(f"  & ECR & {ecr_vals} \\\\")
        print(f"  & MBW (MPa) & {mbw_vals} \\\\")
        print(f"  & $n$ & {n_vals} \\\\")
        print(r"\midrule")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")

    # ===========================================================
    # Save numeric results for paper population
    # ===========================================================
    np.savez(os.path.join(BASE_DIR, 'results.npz'),
             yte=yte, det_yhat=det_yhat, det_res=det_res,
             spl_te=spl_te, spl_lo=spl_lo, spl_up=spl_up,
             bnn_yh=bnn_yh, bnn_lo=bnn_lo, bnn_up=bnn_up,
             bnn_ue=bnn_ue, bnn_ua=bnn_ua, bnn_ut=bnn_ut,
             cal_yt=cal_yt, cal_yp=cal_yp, cal_ue=cal_ue,
             cal_ua=cal_ua, cal_ut=cal_ut, cal_xr=cal_xr,
             Xte_raw=Xte_raw,
             bp_stat=bp_stat, bp_pval=bp_pval, eta=eta,
             delta_spl=delta_spl,
             cqr_lo=cqr_lo, cqr_up=cqr_up, cqr_Q=Q_cqr,
             # provenance: split, selection and scaling
             seed=SEED, idx_fit=idx_fit, idx_cal=idx_cal, idx_te=idx_te,
             hidden_dims=np.array(HIDDEN_DIMS), kl_beta=KL_BETA,
             kl_beta_candidates=np.array(KL_BETA_CANDIDATES),
             kl_beta_cv_rmse=np.array(beta_sweep_rmse),
             scaler_mean=sc.mean_, scaler_scale=sc.scale_)
    np.savez(os.path.join(BASE_DIR, 'split_indices.npz'),
             seed=SEED, idx_fit=idx_fit, idx_cal=idx_cal, idx_te=idx_te)
    with open(os.path.join(BASE_DIR, 'run_config.json'), 'w') as fh:
        json.dump({'seed': SEED, 'n_fit': int(len(idx_fit)), 'n_cal': int(len(idx_cal)),
                   'n_test': int(len(idx_te)), 'hidden_dims': HIDDEN_DIMS,
                   'kl_beta': KL_BETA,
                   'kl_beta_cv_rmse': dict(zip(map(str, KL_BETA_CANDIDATES),
                                               map(float, beta_sweep_rmse))),
                   'scaler_mean': sc.mean_.tolist(), 'scaler_scale': sc.scale_.tolist(),
                   'feature_names': FEATURE_NAMES_SHORT}, fh, indent=2)

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f" All done in {elapsed:.0f} s ({elapsed/60:.1f} min)")
    print(f"{'='*60}")

    # Return everything for interactive use
    return dict(
        sm_spl=sm_spl, sm_bnn=sm_bnn,
        strat_spl=strat_spl, strat_bnn=strat_bnn,
        bp_pval=bp_pval, bp_stat=bp_stat, eta=eta,
        delta_spl=delta_spl,
        det_rmse=det_rmse, det_r2=det_r2,
    )


if __name__ == '__main__':
    main()
