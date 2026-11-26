from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_CSV = ROOT / "data" / "tox21.csv"
MODELS_DIR = ROOT / "models"

RADIUS, NBITS = 2, 2048          # Morgan fingerprint (ECFP4-like)
LOW_SIM, HIGH_SIM = 0.30, 0.50   # heuristic applicability-domain cut-offs (NOT calibrated)

ALL_ENDPOINTS = {
    "NR-AR": "Androgen receptor (full-length)",
    "NR-AR-LBD": "Androgen receptor (ligand-binding domain)",
    "NR-AhR": "Aryl hydrocarbon receptor activation",
    "NR-Aromatase": "Aromatase inhibition",
    "NR-ER": "Estrogen receptor (full-length)",
    "NR-ER-LBD": "Estrogen receptor (ligand-binding domain)",
    "NR-PPAR-gamma": "PPAR-gamma activation",
    "SR-ARE": "Antioxidant response element (oxidative stress)",
    "SR-ATAD5": "ATAD5 (DNA damage response)",
    "SR-HSE": "Heat shock response",
    "SR-MMP": "Mitochondrial membrane potential disruption",
    "SR-p53": "p53 pathway activation (DNA damage)",
}
DEFAULT_ENDPOINTS = ["NR-AhR", "SR-MMP", "SR-ARE"]
