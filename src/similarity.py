import numpy as np
from src.config import LOW_SIM, HIGH_SIM


def tanimoto_matrix(Q: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Tanimoto similarity between binary fingerprint matrices Q (n,d) and R (m,d)."""
    Qf, Rf = Q.astype(np.float32), R.astype(np.float32)
    inter = Qf @ Rf.T
    union = Qf.sum(1)[:, None] + Rf.sum(1)[None, :] - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


def reliability(max_sim: float):
    """(level, message). Heuristic applicability-domain check, not a calibrated confidence."""
    if max_sim >= HIGH_SIM:
        return "supported", (f"Close analogues exist in the training data (nearest Tanimoto {max_sim:.2f}). "
                             "The prediction has structural support, but is still only a screening estimate.")
    if max_sim >= LOW_SIM:
        return "limited", (f"Only moderately similar training molecules (nearest Tanimoto {max_sim:.2f}). "
                           "Treat this prediction with caution.")
    return "outside", (f"OUTSIDE the model's chemical range: nearest training molecule has Tanimoto "
                       f"{max_sim:.2f} (< {LOW_SIM}). Do not rely on this prediction.")
