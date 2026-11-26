"""python -m src.training [--all]   ->  trains models and writes real metrics to models/"""
import argparse, json
from collections import defaultdict
import joblib, numpy as np
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from src.config import ALL_ENDPOINTS, DEFAULT_ENDPOINTS, MODELS_DIR, NBITS
from src.data import load_tox21
from src.fingerprints import parse_smiles, standardize, fingerprint
from src.modeling import fit_and_evaluate
from src.similarity import tanimoto_matrix


def scaffold_split(mols, frac_train=0.8):
    """Molecules sharing a Murcko scaffold never straddle train/test -> no near-duplicate leakage."""
    groups = defaultdict(list)
    for i, m in enumerate(mols):
        try:
            s = MurckoScaffold.MurckoScaffoldSmiles(mol=m, includeChirality=False)
        except Exception:
            s = ""
        groups[s].append(i)
    train, test = [], []
    for g in sorted(groups.values(), key=lambda g: (-len(g), g[0])):
        (train if len(train) + len(g) <= frac_train * len(mols) else test).extend(g)
    return np.array(sorted(train)), np.array(sorted(test))


def main(endpoints):
    df = load_tox21()
    mols, smiles, keep, seen = [], [], [], set()
    n_invalid = n_dup = 0
    for i, smi in enumerate(df["smiles"]):
        m = parse_smiles(smi)
        if m is None:
            n_invalid += 1; continue
        m = standardize(m)
        can = Chem.MolToSmiles(m)
        if can in seen:
            n_dup += 1; continue
        seen.add(can); mols.append(m); smiles.append(can); keep.append(i)
    Y = df.loc[keep, list(ALL_ENDPOINTS)].to_numpy(dtype=float)      # NaN = not tested
    X = np.stack([fingerprint(m) for m in mols])
    tr, te = scaffold_split(mols)
    sim_te = tanimoto_matrix(X[te], X[tr]).max(1)
    print(f"{len(mols)} molecules | train {len(tr)} | test {len(te)} | invalid {n_invalid} | duplicates {n_dup}")

    MODELS_DIR.mkdir(exist_ok=True)
    out = {"dataset": {"source": "Tox21 (MoleculeNet)", "n_raw": int(len(df)), "n_valid_unique": len(mols),
                       "n_invalid": n_invalid, "n_duplicates": n_dup, "n_train": int(len(tr)),
                       "n_test": int(len(te)), "split": "Murcko scaffold split 80/20",
                       "test_max_sim_median": float(np.median(sim_te))},
           "endpoints": {}}
    cols = list(ALL_ENDPOINTS)
    for ep in endpoints:
        j = cols.index(ep)
        mtr, mte = ~np.isnan(Y[tr, j]), ~np.isnan(Y[te, j])       # only use labelled rows
        rf, met = fit_and_evaluate(X[tr][mtr], Y[tr, j][mtr], X[te][mte], Y[te, j][mte], sim_te[mte])
        met["description"] = ALL_ENDPOINTS[ep]
        out["endpoints"][ep] = met
        joblib.dump(rf, MODELS_DIR / f"{ep}.joblib", compress=3)
        print(f"{ep}: ROC-AUC {met['roc_auc']:.3f}  PR-AUC {met['pr_auc']:.3f}  "
              f"P {met['precision']:.2f}  R {met['recall']:.2f}")
    np.savez_compressed(MODELS_DIR / "reference.npz", fp=np.packbits(X[tr], axis=1),
                        smiles=np.array(smiles)[tr], labels=np.nan_to_num(Y[tr], nan=-1).astype(np.int8),
                        endpoints=np.array(cols))
    (MODELS_DIR / "metrics.json").write_text(json.dumps(out, indent=2))
    print("Saved to", MODELS_DIR)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="train all 12 endpoints (default: 3)")
    a = ap.parse_args()
    main(list(ALL_ENDPOINTS) if a.all else DEFAULT_ENDPOINTS)
