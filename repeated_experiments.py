#!/usr/bin/env python3
import os, sys, time, json, argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold, GroupKFold, GroupShuffleSplit, train_test_split
from sklearn.preprocessing import StandardScaler
import multiprocessing as mp

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_CSV = os.path.join(BASE_DIR, 'repeated_results.csv')
OUT_SEL = os.path.join(BASE_DIR, 'repeated_selection.csv')

ARCH_CANDIDATES = [(l1, l2, l3) for l1 in [64, 128, 256] for l2 in [32, 64, 128]
                   for l3 in [16, 32, 64] if l1 >= l2 >= l3]
ARCH_SEARCH_BETA = 0.1
CLASSES = [('lt30', 0, 30), ('30to50', 30, 50), ('ge50', 50, 1e9)]
Z90 = 1.6448536
U_FLOOR = 1e-6          # pre-specified floor, applied identically to calibration and test rows
PRED_DIR = os.path.join(BASE_DIR, 'repeated_preds')


# ------------------------------------------------------------------
# Additional deterministic models (same architecture as the BNN)
# ------------------------------------------------------------------
class QuantNN(nn.Module):
    """Two-output network for the alpha/2 and 1-alpha/2 conditional quantiles."""
    def __init__(self, d_in, hidden):
        super().__init__()
        layers, prev = [], d_in
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]; prev = h
        layers.append(nn.Linear(prev, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def _loader(X, y, bs):
    return DataLoader(TensorDataset(torch.tensor(X), torch.tensor(y)), batch_size=bs, shuffle=True)


def train_quant(rx, model, X, y, alpha):
    taus = torch.tensor([alpha / 2, 1 - alpha / 2])
    dl = _loader(X, y, rx.BATCH_SIZE)
    opt = torch.optim.AdamW(model.parameters(), lr=rx.LR, weight_decay=rx.WEIGHT_DECAY)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, rx.TOTAL_EPOCHS, eta_min=rx.LR_MIN)
    model.train()
    for _ in range(rx.TOTAL_EPOCHS):
        for xb, yb in dl:
            opt.zero_grad()
            diff = yb.unsqueeze(1) - model(xb)
            loss = torch.maximum(taus * diff, (taus - 1) * diff).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), rx.GRAD_CLIP)
            opt.step()
        sch.step()
    return model


# ------------------------------------------------------------------
# Selection on proper-training rows (scaler refitted per fold)
# ------------------------------------------------------------------
def mix_groups(X):
    """Mix-design identifier: the seven composition columns (age excluded)."""
    return pd.factorize(pd.Series([tuple(np.round(r[:7], 3)) for r in X]))[0]


def make_mode_splits(X, seed, mode):
    """random: splits.make_splits; grouped: whole mix designs held out (all ages
    of a mix fall in one subset); age: test = all specimens older than 28 days,
    fit/calibration drawn from specimens of age <= 28 days."""
    n = len(X)
    if mode == 'random':
        from splits import make_splits
        return make_splits(n, seed)
    if mode == 'grouped':
        g = mix_groups(X)
        tr, te = next(GroupShuffleSplit(1, test_size=0.2, random_state=seed).split(X, groups=g))
        fi, ca = next(GroupShuffleSplit(1, test_size=0.3, random_state=seed).split(X[tr], groups=g[tr]))
        return tr[fi], tr[ca], te
    if mode == 'age':
        te = np.where(X[:, 7] > 28)[0]
        tr = np.where(X[:, 7] <= 28)[0]
        fi, ca = train_test_split(tr, test_size=0.3, random_state=seed)
        return fi, ca, te
    raise ValueError(mode)


def cv_rmse_bnn(rx, Xfit_raw, yfit, hidden, beta, seed, groups=None):
    rx.KL_BETA = beta
    if groups is None:
        folds = KFold(3, shuffle=True, random_state=seed + 999).split(Xfit_raw)
    else:   # grouped CV: a mix design never spans training and validation folds
        folds = GroupKFold(3, shuffle=True, random_state=seed + 999).split(Xfit_raw, groups=groups)
    errs = []
    for k, (ti, vi) in enumerate(folds):
        sc = StandardScaler().fit(Xfit_raw[ti])
        Xt = sc.transform(Xfit_raw[ti]).astype(np.float32)
        Xv = sc.transform(Xfit_raw[vi]).astype(np.float32)
        torch.manual_seed(seed * 1000 + k)
        m = rx.train_bnn(rx.BNN(Xt.shape[1], hidden=list(hidden)), Xt, yfit[ti])
        m.eval()
        with torch.no_grad():
            yp, _ = m(torch.tensor(Xv), sample=False)
        errs.extend((yfit[vi] - yp.numpy()) ** 2)
    return float(np.sqrt(np.mean(errs)))


# ------------------------------------------------------------------
# Metrics
# ------------------------------------------------------------------
def efficiency_score(ecr, r2, width, yp, alpha):
    """Efficiency score of Tamuly & Nava (2025), Constr. Build. Mater. 487, 141844:
    ES = C_score * R^2 * IW_score - C_penalty, with
    C_score = min(EC/EC_t, 1), IW_score = 1/(1 + mean(width_i / yhat_i)),
    C_penalty = max(0, (EC_t - EC)/EC_t), EC_t = 1 - alpha."""
    ec_t = 1 - alpha
    c_score = min(ecr / ec_t, 1.0)
    niw_mean = float(np.mean(width / yp))
    iw_score = 1.0 / (1.0 + niw_mean)
    c_pen = max(0.0, (ec_t - ecr) / ec_t)
    return dict(ES=c_score * r2 * iw_score - c_pen, C_score=c_score, NIW_mean=niw_mean,
                IW_score=iw_score, C_penalty=c_pen)


def interval_metrics(yt, lo, up, alpha, yp=None):
    """yp: point prediction (for CQR the interval midpoint is passed)."""
    cov = (yt >= lo) & (yt <= up)
    width = up - lo
    iscore = width + (2 / alpha) * np.clip(lo - yt, 0, None) + (2 / alpha) * np.clip(yt - up, 0, None)
    out = dict(ECR=cov.mean(), MIW=width.mean(), IS=iscore.mean(), n_covered=int(cov.sum()))
    out['RMSE'] = float(np.sqrt(((yt - yp) ** 2).mean()))
    out['R2'] = float(1 - ((yt - yp) ** 2).sum() / ((yt - yt.mean()) ** 2).sum())
    out['min_yhat'] = float(yp.min())
    out.update(efficiency_score(out['ECR'], out['R2'], width, yp, alpha))
    ecr_cls = []
    for name, a, b in CLASSES:
        m = (yt >= a) & (yt < b)
        out[f'n_{name}'] = int(m.sum())
        out[f'ECR_{name}'] = cov[m].mean() if m.any() else np.nan
        out[f'MIW_{name}'] = width[m].mean() if m.any() else np.nan
        if m.any():
            ecr_cls.append(cov[m].mean())
    out['ECR_worst_class'] = min(ecr_cls)
    return out


# ------------------------------------------------------------------
# One seed
# ------------------------------------------------------------------
def run_seed(task):
    seed, mode = task if isinstance(task, tuple) else (task, 'random')
    torch.set_num_threads(2)
    sys.path.insert(0, BASE_DIR)
    import run_experiments as rx
    from splits import load_data, make_splits, verify_against_legacy
    t0 = time.time()
    alpha = rx.ALPHA
    X, y = load_data()
    idx_fit, idx_cal, idx_te = make_mode_splits(X, seed, mode)
    if mode == 'random' and seed == 42:
        verify_against_legacy(X, y, idx_fit, idx_cal, idx_te, seed)
    Xfit_raw, yfit = X[idx_fit], y[idx_fit]
    Xcal_raw, ycal = X[idx_cal], y[idx_cal]
    Xte_raw, yte = X[idx_te], y[idx_te]

    # --- selection (proper-training rows only) ---
    groups = mix_groups(Xfit_raw) if mode == 'grouped' else None
    arch_rmse = {a: cv_rmse_bnn(rx, Xfit_raw, yfit, a, ARCH_SEARCH_BETA, seed, groups) for a in ARCH_CANDIDATES}
    arch = min(arch_rmse, key=arch_rmse.get)
    beta_rmse = {b: cv_rmse_bnn(rx, Xfit_raw, yfit, arch, b, seed, groups) for b in rx.KL_BETA_CANDIDATES}
    beta = min(beta_rmse, key=beta_rmse.get)

    # --- final scaler and models on proper-training rows ---
    sc = StandardScaler().fit(Xfit_raw)
    Xfit = sc.transform(Xfit_raw).astype(np.float32)
    Xcal = sc.transform(Xcal_raw).astype(np.float32)
    Xte = sc.transform(Xte_raw).astype(np.float32)
    d = Xfit.shape[1]
    tc, tt = torch.tensor(Xcal), torch.tensor(Xte)

    torch.manual_seed(seed * 1000 + 101)
    det = rx.train_det(rx.DetNN(d, hidden=list(arch)), Xfit, yfit).eval()
    rx.KL_BETA = beta
    torch.manual_seed(seed * 1000 + 102)
    bnn = rx.train_bnn(rx.BNN(d, hidden=list(arch)), Xfit, yfit).eval()
    torch.manual_seed(seed * 1000 + 104)
    qnn = train_quant(rx, QuantNN(d, list(arch)), Xfit, yfit, alpha).eval()

    with torch.no_grad():
        det_c, det_t = det(tc).numpy(), det(tt).numpy()
        q_c, q_t = np.sort(qnn(tc).numpy(), 1), np.sort(qnn(tt).numpy(), 1)

    # Same M posterior weight draws for every calibration and test row:
    # one stochastic pass per draw over the stacked calibration+test inputs.
    torch.manual_seed(seed * 1000 + 105)
    ys, vs = bnn.mc_predict(torch.cat([tc, tt]), rx.M_SAMPLES)       # (M, n_c + n_t)
    ybar = ys.mean(0)                                                # MC posterior mean
    u_epi = np.sqrt(ys.var(0, ddof=1))
    u_ale = np.sqrt(vs.mean(0))
    u_tot = np.sqrt(u_epi ** 2 + u_ale ** 2)
    u_epi, u_ale, u_tot = (np.maximum(u, U_FLOOR) for u in (u_epi, u_ale, u_tot))
    nc = len(ycal)
    bm_c, bm_t = ybar[:nc], ybar[nc:]
    ue_c, ue_t = u_epi[:nc], u_epi[nc:]
    ua_c, ua_t = u_ale[:nc], u_ale[nc:]
    ut_c, ut_t = u_tot[:nc], u_tot[nc:]

    q = rx.conf_quantile
    methods = {}
    dd = q(np.abs(ycal - det_c), alpha)
    methods['det_const'] = (det_t - dd, det_t + dd, det_t)
    db = q(np.abs(ycal - bm_c), alpha)
    methods['bnn_const'] = (bm_t - db, bm_t + db, bm_t)
    for name, sc_c, sc_t in [('bnn_total', ut_c, ut_t), ('bnn_ale', ua_c, ua_t), ('bnn_epi', ue_c, ue_t)]:
        eta = q(np.abs(ycal - bm_c) / sc_c, alpha)
        methods[name] = (bm_t - eta * sc_t, bm_t + eta * sc_t, bm_t)
    methods['bnn_vanilla'] = (bm_t - Z90 * ut_t, bm_t + Z90 * ut_t, bm_t)
    E = np.maximum(q_c[:, 0] - ycal, ycal - q_c[:, 1])
    Q = q(E, alpha)
    methods['cqr'] = (q_t[:, 0] - Q, q_t[:, 1] + Q, q_t.mean(1))   # midpoint as point prediction

    # per-specimen outputs (so later metrics never need retraining)
    os.makedirs(PRED_DIR, exist_ok=True)
    pdir = PRED_DIR + ('' if mode == 'random' else f'_{mode}')
    os.makedirs(pdir, exist_ok=True)
    np.savez_compressed(os.path.join(pdir, f'seed_{seed:02d}.npz'),
        idx_fit=idx_fit, idx_cal=idx_cal, idx_te=idx_te, ycal=ycal, yte=yte,
        arch=np.array(arch), beta=beta, alpha=alpha,
        bnn_ybar=ybar, bnn_u_epi=u_epi, bnn_u_ale=u_ale, bnn_u_tot=u_tot,
        det_cal=det_c, det_te=det_t, cqr_q_cal=q_c, cqr_q_te=q_t,
        **{f'{m}_lo': v[0] for m, v in methods.items()},
        **{f'{m}_up': v[1] for m, v in methods.items()})

    rows = []
    for name, (lo, up, yp) in methods.items():
        r = dict(seed=seed, method=name, mode=mode, n_fit=len(idx_fit), n_cal=len(idx_cal), n_te=len(idx_te))
        r.update(interval_metrics(yte, lo, up, alpha, yp))
        rows.append(r)
    sel = dict(seed=seed, arch=str(list(arch)), arch_cv_rmse=arch_rmse[arch], beta=beta,
               beta_cv_rmse=beta_rmse[beta], minutes=(time.time() - t0) / 60)
    return rows, sel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, nargs='*', default=list(range(20)))
    ap.add_argument('--workers', type=int, default=5)
    ap.add_argument('--mode', choices=['random', 'grouped', 'age'], default='random')
    args = ap.parse_args()
    global OUT_CSV, OUT_SEL
    if args.mode != 'random':
        OUT_CSV = OUT_CSV.replace('.csv', f'_{args.mode}.csv')
        OUT_SEL = OUT_SEL.replace('.csv', f'_{args.mode}.csv')

    done = set(pd.read_csv(OUT_SEL)['seed']) if os.path.exists(OUT_SEL) else set()
    todo = [s for s in args.seeds if s not in done]
    print(f"Seeds to run: {todo}  (already done: {sorted(done)})", flush=True)
    ctx = mp.get_context('spawn')
    with ctx.Pool(args.workers) as pool:
        for rows, sel in pool.imap_unordered(run_seed, [(s, args.mode) for s in todo]):
            pd.DataFrame(rows).to_csv(OUT_CSV, mode='a', header=not os.path.exists(OUT_CSV), index=False)
            pd.DataFrame([sel]).to_csv(OUT_SEL, mode='a', header=not os.path.exists(OUT_SEL), index=False)
            prop = [r for r in rows if r['method'] == 'bnn_total'][0]
            print(f"seed {sel['seed']:>3}  arch={sel['arch']:<16} beta={sel['beta']:<5} "
                  f"proposed ECR={prop['ECR']:.3f} MIW={prop['MIW']:.2f}  ({sel['minutes']:.1f} min)", flush=True)


if __name__ == '__main__':
    main()
