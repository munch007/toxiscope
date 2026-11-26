"""Pure scikit-learn part: train one Random Forest and evaluate it honestly."""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score, recall_score, roc_auc_score)

BINS = [(0.0, 0.3, "< 0.30"), (0.3, 0.5, "0.30 – 0.50"), (0.5, 1.01, "≥ 0.50")]


def _auc(y, s):
    return float(roc_auc_score(y, s)) if len(set(y.tolist())) == 2 else None


def fit_and_evaluate(Xtr, ytr, Xte, yte, sim_te, seed=0):
    ytr, yte = ytr.astype(int), yte.astype(int)
    rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=2, max_features="sqrt",
                                class_weight="balanced_subsample", oob_score=True,
                                n_jobs=-1, random_state=seed).fit(Xtr, ytr)
    # Decision threshold picked on out-of-bag predictions of the TRAIN set (no test leakage).
    oob = np.nan_to_num(rf.oob_decision_function_[:, 1])
    p, r, t = precision_recall_curve(ytr, oob)
    f1 = 2 * p[:-1] * r[:-1] / np.maximum(p[:-1] + r[:-1], 1e-9)
    thr = float(t[int(np.argmax(f1))])

    proba = rf.predict_proba(Xte)[:, 1]
    pred = (proba >= thr).astype(int)
    by_sim = []
    for lo, hi, name in BINS:
        m = (sim_te >= lo) & (sim_te < hi)
        by_sim.append({"bin": name, "n": int(m.sum()),
                       "roc_auc": _auc(yte[m], proba[m]) if m.sum() >= 30 else None})
    metrics = {
        "n_train": int(len(ytr)), "n_test": int(len(yte)),
        "train_active_rate": float(ytr.mean()), "test_active_rate": float(yte.mean()),
        "threshold": thr,
        "roc_auc": _auc(yte, proba),
        "pr_auc": float(average_precision_score(yte, proba)) if yte.sum() else None,
        "precision": float(precision_score(yte, pred, zero_division=0)),
        "recall": float(recall_score(yte, pred, zero_division=0)),
        "f1": float(f1_score(yte, pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(yte, pred, labels=[0, 1]).tolist(),  # [[TN,FP],[FN,TP]]
        "auc_by_similarity": by_sim,
    }
    return rf, metrics
