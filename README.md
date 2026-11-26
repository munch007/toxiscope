# ToxiScope — Explainable Chemical Bioactivity Screening

Research/education prototype. Predicts Tox21 **assay activity** (not human toxicity, never "safe").

## Run
```bash
pip install -r requirements.txt
python -m src.data          # downloads Tox21 -> data/tox21.csv (or place the MoleculeNet CSV there manually)
python -m src.training      # trains 3 endpoints (NR-AhR, SR-MMP, SR-ARE); add --all for all 12. ~2-5 min
streamlit run app.py
```
Metrics in the app come only from `models/metrics.json`, written by the training script.

## Structure
- `src/fingerprints.py` SMILES validation, salt stripping, Morgan fingerprints, substructure environments
- `src/modeling.py` Random Forest + honest evaluation (ROC-AUC, PR-AUC, P/R/F1, confusion matrix, AUC by similarity bin)
- `src/training.py` data cleaning, scaffold split, trains/saves models + reference set
- `src/similarity.py` Tanimoto search + applicability-domain warning
- `src/prediction.py` screening + occlusion-based substructure explanation
- `app.py` Streamlit UI (Home, Screening, Performance, Limitations)

## Deploy
Push to GitHub (commit `models/`), then deploy `app.py` on Streamlit Community Cloud.

## Project description (for judges)
ToxiScope is a computational chemistry application for preliminary screening of molecules using public Tox21
bioactivity data. It predicts assay activity with fingerprint-based Random Forest classifiers, explains which
substructures drive each prediction, retrieves structurally similar reference molecules, and warns when a query
molecule lies outside the chemical space of its training data. It is a research and educational aid, not a
replacement for laboratory testing or professional toxicological assessment.
