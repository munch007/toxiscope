"""Load Tox21 (MoleculeNet version). Downloads once, caches to data/tox21.csv."""
import pandas as pd
from src.config import DATA_CSV, ALL_ENDPOINTS

URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz"


def load_tox21() -> pd.DataFrame:
    if not DATA_CSV.exists():
        print(f"Downloading Tox21 from {URL} ...")
        try:
            df = pd.read_csv(URL)
        except Exception as e:
            raise SystemExit(
                f"Download failed ({e}).\nManually save the MoleculeNet Tox21 CSV "
                f"(columns: {', '.join(ALL_ENDPOINTS)}, mol_id, smiles) to {DATA_CSV}"
            )
        DATA_CSV.parent.mkdir(exist_ok=True)
        df.to_csv(DATA_CSV, index=False)
    df = pd.read_csv(DATA_CSV)
    df.columns = [c if c in ALL_ENDPOINTS else c.lower() for c in df.columns]
    missing = [c for c in ["smiles", *ALL_ENDPOINTS] if c not in df.columns]
    if missing:
        raise SystemExit(f"{DATA_CSV} is missing columns: {missing}")
    return df


if __name__ == "__main__":
    d = load_tox21()
    print(d.shape)
