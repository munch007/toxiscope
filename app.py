import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from rdkit import Chem
from rdkit.Chem import Draw

from src.config import ALL_ENDPOINTS, LOW_SIM, HIGH_SIM
from src.fingerprints import environment_atoms_bonds
from src.prediction import Screener, ModelsMissing

st.set_page_config(page_title="ToxiScope", page_icon="🧪", layout="wide")

EXAMPLES = {
    "Caffeine": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
    "Bisphenol A": "CC(C)(c1ccc(O)cc1)c1ccc(O)cc1",
    "Benzo[a]pyrene": "c1ccc2c(c1)cc1ccc3cccc4ccc2c1c34",
    "Aspirin": "CC(=O)Oc1ccccc1C(=O)O",
    "Paracetamol": "CC(=O)Nc1ccc(O)cc1",
    "Invalid input (test)": "this_is_not_smiles",
}
DISCLAIMER = ("Research & education screening aid. Predicts **assay activity** from public Tox21 data — "
              "it does **not** determine whether a chemical is toxic or safe for humans.")


@st.cache_resource
def get_screener():
    return Screener()


try:
    S = get_screener()
except ModelsMissing as e:
    st.title("🧪 ToxiScope")
    st.error(str(e))
    st.stop()

page = st.sidebar.radio("Navigate", ["Home", "Molecule screening", "Model performance",
                                     "Dataset & limitations"])
st.sidebar.caption(DISCLAIMER)


def label_text(v):
    return {1: "Active", 0: "Inactive"}.get(int(v), "Not tested")


# ------------------------------------------------------------------ Home
if page == "Home":
    st.title("🧪 ToxiScope")
    st.subheader("Explainable chemical bioactivity screening")
    st.info(DISCLAIMER)
    st.markdown(
        "**How it works**\n"
        "1. Enter a molecule as SMILES and it is validated and converted to a Morgan fingerprint.\n"
        "2. A Random Forest per Tox21 endpoint predicts assay activity.\n"
        "3. The closest known molecules are retrieved (Tanimoto similarity) as supporting evidence.\n"
        "4. If the molecule is too unlike the training data, ToxiScope warns you instead of "
        "pretending to be confident.")
    d = S.metrics["dataset"]
    c = st.columns(4)
    c[0].metric("Training molecules", f"{d['n_train']:,}")
    c[1].metric("Held-out test molecules", f"{d['n_test']:,}")
    c[2].metric("Endpoints", len(S.models))
    c[3].metric("Split", "Scaffold")
    st.caption("Go to **Molecule screening** in the sidebar to try it.")

# ------------------------------------------------------------------ Screening
elif page == "Molecule screening":
    st.title("Molecule screening")
    st.warning(DISCLAIMER)
    ex = st.selectbox("Example molecules", ["(type your own)"] + list(EXAMPLES))
    smi = st.text_input("SMILES", EXAMPLES.get(ex, "CC(C)(c1ccc(O)cc1)c1ccc(O)cc1"))
    if st.button("Screen molecule", type="primary") or smi:
        res = S.screen(smi)
        if "error" in res:
            st.error(res["error"]); st.stop()

        left, right = st.columns([1, 2])
        left.image(Draw.MolToImage(res["mol"], size=(320, 320)), caption=Chem.MolToSmiles(res["mol"]))
        with right:
            {"supported": st.success, "limited": st.warning, "outside": st.error}[res["level"]](
                "**Evidence check:** " + res["message"])
            st.caption(f"Similarity is a heuristic applicability-domain check (cut-offs {LOW_SIM} / {HIGH_SIM}), "
                       "not a calibrated confidence score.")
            show = True
            if res["level"] == "outside":
                show = st.checkbox("Show model output anyway (unreliable)")
            if show:
                rows = [{"Endpoint": ep, "Assay": S.metrics["endpoints"][ep]["description"],
                         "Model score": round(p["score"], 3), "Decision threshold": round(p["threshold"], 3),
                         "Predicted": "Active-like" if p["active"] else "Inactive-like",
                         "Test ROC-AUC": round(S.metrics["endpoints"][ep]["roc_auc"], 3)}
                        for ep, p in res["predictions"].items()]
                st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
                st.caption("Model score is a Random Forest output, not a calibrated probability of toxicity.")

        t1, t2 = st.tabs(["Why this prediction? (substructures)", "Similar known molecules"])
        with t1:
            ep = st.selectbox("Endpoint", list(res["predictions"]))
            items = S.explain(res["mol"], res["fp"], ep)
            st.caption("Each substructure is switched off in turn and the score re-computed. "
                       "Red = pushes toward 'active', blue = pushes toward 'inactive'. "
                       "This shows what the model relies on, not proven chemical causality.")
            cols = st.columns(max(len(items), 1))
            for col, it in zip(cols, items):
                atoms, bonds = environment_atoms_bonds(res["mol"], *it["env"])
                color = (0.9, 0.2, 0.2) if it["delta"] > 0 else (0.2, 0.4, 0.9)
                img = Draw.MolToImage(res["mol"], size=(260, 260), highlightAtoms=atoms, highlightBonds=bonds,
                                      highlightColor=color)
                col.image(img, caption=f"Score change if removed: {-it['delta']:+.3f}")
        with t2:
            nb = res["neighbors"]
            st.image(Draw.MolsToGridImage([Chem.MolFromSmiles(s) for _, s, _ in nb], molsPerRow=3,
                                          subImgSize=(230, 200),
                                          legends=[f"Tanimoto {sim:.2f}" for sim, _, _ in nb]))
            eps = S.all_endpoints
            tbl = [{"Tanimoto": round(sim, 3), "SMILES": s,
                    **{e: label_text(lab[eps.index(e)]) for e in S.models}} for sim, s, lab in nb]
            st.dataframe(pd.DataFrame(tbl), hide_index=True, use_container_width=True)

# ------------------------------------------------------------------ Performance
elif page == "Model performance":
    st.title("Model performance")
    st.caption("All numbers below were computed by `python -m src.training` on a held-out, "
               "scaffold-split test set. The shipped models are exactly the ones evaluated.")
    E = S.metrics["endpoints"]
    df = pd.DataFrame([{"Endpoint": k, "Test n": v["n_test"], "Train n": v["n_train"],
                        "Active rate (test)": round(v["test_active_rate"], 3),
                        "ROC-AUC": round(v["roc_auc"], 3), "PR-AUC": round(v["pr_auc"], 3),
                        "Precision": round(v["precision"], 3), "Recall": round(v["recall"], 3),
                        "F1": round(v["f1"], 3)} for k, v in E.items()])
    st.dataframe(df, hide_index=True, use_container_width=True)
    st.plotly_chart(px.bar(df.melt("Endpoint", ["ROC-AUC", "PR-AUC"]), x="Endpoint", y="value",
                           color="variable", barmode="group", labels={"value": "score", "variable": ""}),
                    use_container_width=True)
    st.caption("PR-AUC should be read against the active rate: a random model scores about that value.")
    ep = st.selectbox("Endpoint details", list(E))
    cm = E[ep]["confusion_matrix"]
    c1, c2 = st.columns(2)
    fig = go.Figure(go.Heatmap(z=cm, x=["Pred inactive", "Pred active"], y=["True inactive", "True active"],
                               text=cm, texttemplate="%{text}", colorscale="Blues", showscale=False))
    fig.update_yaxes(autorange="reversed")
    c1.plotly_chart(fig, use_container_width=True)
    c2.markdown("**ROC-AUC by similarity to training data**")
    c2.dataframe(pd.DataFrame(E[ep]["auc_by_similarity"]).rename(
        columns={"bin": "Nearest-neighbour Tanimoto", "n": "Test molecules", "roc_auc": "ROC-AUC"}),
        hide_index=True)
    c2.caption("Empty = fewer than 30 molecules or only one class. This is the evidence behind the "
               "reliability warning: performance usually drops for dissimilar molecules.")

# ------------------------------------------------------------------ Limitations
else:
    st.title("Dataset & limitations")
    d = S.metrics["dataset"]
    st.markdown(f"""
**Data:** Tox21 (MoleculeNet release), {d['n_raw']:,} raw records → {d['n_valid_unique']:,} valid unique
molecules ({d['n_invalid']} invalid, {d['n_duplicates']} duplicates removed; salts stripped to the largest fragment).
Missing assay labels are left out per endpoint, never treated as inactive.

**Split:** {d['split']} — molecules sharing a core scaffold are never in both train and test, so the metrics are
not inflated by near-duplicates. Decision thresholds were chosen from out-of-bag predictions on the training set only.

**Limitations**
- Tox21 assay activity is **not** proven human toxicity. Activity can come from cytotoxicity or assay interference.
- Active/inactive is based on single-concentration-series screens; labels are noisy.
- Predictions are unreliable for molecules unlike the training set (see the evidence check and the AUC-by-similarity table).
- Only {len(S.models)} of 12 Tox21 endpoints are modelled in this build.
- Morgan-fingerprint QSAR, applicability-domain checks and toxicity prediction tools (e.g. EPA TEST, CompTox)
  already exist; ToxiScope is a transparent educational workflow, not a novel method or a certified tool.
- Never use this to declare a substance safe. Not a substitute for laboratory testing or toxicological assessment.
""")
