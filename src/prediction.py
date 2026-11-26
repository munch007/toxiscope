import json
import joblib, numpy as np
from src.config import MODELS_DIR, NBITS
from src.fingerprints import parse_smiles, standardize, fingerprint, bit_environments
from src.similarity import tanimoto_matrix, reliability


class ModelsMissing(Exception):
    pass


class Screener:
    def __init__(self):
        mp = MODELS_DIR / "metrics.json"
        if not mp.exists():
            raise ModelsMissing("No trained models found. Run:  python -m src.training")
        self.metrics = json.loads(mp.read_text())
        ref = np.load(MODELS_DIR / "reference.npz")
        self.ref_fp = np.unpackbits(ref["fp"], axis=1)[:, :NBITS]
        self.ref_smiles, self.ref_labels = ref["smiles"], ref["labels"]
        self.all_endpoints = list(ref["endpoints"])
        self.models = {ep: joblib.load(MODELS_DIR / f"{ep}.joblib") for ep in self.metrics["endpoints"]}

    def screen(self, smiles, k=6):
        mol = parse_smiles(smiles)
        if mol is None:
            return {"error": "Invalid input: this is not a valid SMILES structure."}
        mol = standardize(mol)
        fp = fingerprint(mol)
        sims = tanimoto_matrix(fp[None, :], self.ref_fp)[0]
        order = np.argsort(-sims)[:k]
        preds = {}
        for ep, rf in self.models.items():
            s = float(rf.predict_proba(fp[None, :])[0, 1])
            thr = self.metrics["endpoints"][ep]["threshold"]
            preds[ep] = {"score": s, "threshold": thr, "active": s >= thr}
        level, msg = reliability(float(sims[order[0]]))
        return {"mol": mol, "fp": fp, "max_sim": float(sims[order[0]]), "level": level, "message": msg,
                "predictions": preds,
                "neighbors": [(float(sims[i]), str(self.ref_smiles[i]), self.ref_labels[i]) for i in order]}

    def explain(self, mol, fp, ep, top=4):
        """Occlusion: switch off each substructure bit and see how the score changes.
        delta > 0 => the substructure pushes the score toward 'active'."""
        rf = self.models[ep]
        envs = bit_environments(mol)
        bits = [b for b in envs if fp[b]]
        if not bits:
            return []
        base = rf.predict_proba(fp[None, :])[0, 1]
        X = np.repeat(fp[None, :], len(bits), axis=0)
        for r, b in enumerate(bits):
            X[r, b] = 0
        delta = base - rf.predict_proba(X)[:, 1]
        return [{"bit": bits[i], "delta": float(delta[i]), "env": envs[bits[i]][0]}
                for i in np.argsort(-np.abs(delta))[:top]]
