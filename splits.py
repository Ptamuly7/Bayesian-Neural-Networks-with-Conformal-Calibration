#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42
TEST_SIZE = 0.2
CAL_SIZE = 0.3

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LEGACY_RESULTS = os.path.join(BASE_DIR, 'legacy', 'results_legacy.npz')


def load_data():
    try:
        df = pd.read_excel(os.path.join(BASE_DIR, 'Datasets', 'concrete_strength.xlsx'))
    except Exception:
        df = pd.read_excel(os.path.join(BASE_DIR, 'Datasets', 'concrete_strength.xls'))
    X = df.iloc[:, :-1].values.astype(np.float32)
    y = df.iloc[:, -1].values.astype(np.float32)
    return X, y


def make_splits(n, seed=SEED):
    """Return (idx_fit, idx_cal, idx_te) as row-index arrays into the dataset."""
    idx_tr, idx_te = train_test_split(np.arange(n), test_size=TEST_SIZE, random_state=seed)
    idx_fit, idx_cal = train_test_split(idx_tr, test_size=CAL_SIZE, random_state=seed)
    assert len(set(idx_fit) & set(idx_cal)) == 0
    assert len(set(idx_fit) & set(idx_te)) == 0
    assert len(set(idx_cal) & set(idx_te)) == 0
    assert len(idx_fit) + len(idx_cal) + len(idx_te) == n
    return idx_fit, idx_cal, idx_te


def verify_against_legacy(X, y, idx_fit, idx_cal, idx_te, seed=SEED):
    """Check (row values AND row order) that the index split matches the
    legacy array-based split of the submitted manuscript. Raises on mismatch."""
    # (a) legacy implementation, re-run in this session
    Xtr_l, Xte_l, ytr_l, yte_l = train_test_split(X, y, test_size=TEST_SIZE, random_state=seed)
    Xfit_l, Xcal_l, yfit_l, ycal_l = train_test_split(Xtr_l, ytr_l, test_size=CAL_SIZE, random_state=seed)
    checks = {
        'test X  vs legacy code': np.array_equal(X[idx_te], Xte_l),
        'test y  vs legacy code': np.array_equal(y[idx_te], yte_l),
        'cal X   vs legacy code': np.array_equal(X[idx_cal], Xcal_l),
        'cal y   vs legacy code': np.array_equal(y[idx_cal], ycal_l),
        'fit X   vs legacy code': np.array_equal(X[idx_fit], Xfit_l),
        'fit y   vs legacy code': np.array_equal(y[idx_fit], yfit_l),
    }
    # (b) arrays saved by the submitted run
    if os.path.exists(LEGACY_RESULTS):
        old = np.load(LEGACY_RESULTS)
        checks.update({
            'test X  vs legacy results.npz': np.array_equal(X[idx_te], old['Xte_raw']),
            'test y  vs legacy results.npz': np.array_equal(y[idx_te], old['yte']),
            'cal X   vs legacy results.npz': np.array_equal(X[idx_cal], old['cal_xr']),
            'cal y   vs legacy results.npz': np.array_equal(y[idx_cal], old['cal_yt']),
        })
    else:
        print(f"  [warn] {LEGACY_RESULTS} not found; skipping saved-array check")
    for name, ok in checks.items():
        print(f"  {'OK  ' if ok else 'FAIL'} {name}")
    failed = [k for k, ok in checks.items() if not ok]
    if failed:
        raise AssertionError(f"Split differs from legacy implementation: {failed}")


if __name__ == '__main__':
    X, y = load_data()
    idx_fit, idx_cal, idx_te = make_splits(len(y))
    print(f"fit={len(idx_fit)}  cal={len(idx_cal)}  test={len(idx_te)}")
    verify_against_legacy(X, y, idx_fit, idx_cal, idx_te)
    print("Split reproduces the legacy calibration and test rows exactly.")
