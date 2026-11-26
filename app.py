import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from rdkit import Chem
from rdkit.Chem import Draw

from src.config import ALL_ENDPOINTS, LOW_SIM, HIGH_SIM
from src.fingerprints import environment_atoms_bonds
from src.prediction import Screener, ModelsMissing

st.set_page_config(page_title="ToxiScope", page_icon="🧪", layout="centered")

st.markdown("""
<style>
#MainMenu, footer, header {visibility:hidden;}
.block-container {padding-top:1rem; max-width:760px;}
.hero {background:linear-gradient(135deg,#0f766e,#1d4ed8); color:#fff; padding:22px 20px;
       border-radius:18px; margin-bottom:14px;}
.hero h1 {margin:0; font-size:2rem; color:#fff;}
.hero p {margin:6px 0 0; opacity:.92; font-size:1rem;}
.card {background:rgba(128,128,128,.10); border-radius:14px; padding:14px 16px; margin:10px 0;}
.card .t {font-weight:700; font-size:1.05rem;}
.card .d {opacity:.75; font-size:.88rem; margin:2px 0 8px;}
.card .v {font-weight:700; margin-bottom:6px;}
.card .s {opacity:.7; font-size:.8rem; margin-top:4px;}
.bar {height:10px; background:rgba(128,128,128,.25); border-radius:6px; overflow:hidden;}
.bar span {display:block; height:100%; border-radius:6px;}
.badge {border-radius:14px; padding:14px 16px; margin:10px 0; color:#fff;}
.badge b {font-size:1.05rem;} .badge div {font-size:.9rem; margin-top:3px; opacity:.95;}
.stTabs [data-baseweb="tab"] {font-size:.95rem; padding:8px 10px;}
</style>
""", unsafe_allow_html=True)

EXAMPLES = {
    "☕ Caffeine": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
    "💊 Aspirin": "CC(=O)Oc1ccccc1C(=O)O",
    "💊 Paracetamol": "CC(=O)Nc1ccc(O)cc1",
    "💊 Ibuprofen": "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
    "🧴 Bisphenol A (plastics)": "CC(C)(c1ccc(O)cc1)c1ccc(O)cc1",
    "🧴 Triclosan (antibacterial soap)": "Oc1cc(Cl)ccc1Oc1ccc(Cl)cc1Cl",
    "🔥 Benzo[a]pyrene (smoke / grilled food)": "c1ccc2c(c1)cc1ccc3cccc4ccc2c1c34",
    "🚬 Nicotine": "CN1CCCC1c1cccnc1",
    "🛢️ Benzene": "c1ccccc1",
    "🧿 Naphthalene (mothballs)": "c1ccc2ccccc2c1",
    "🍺 Ethanol (alcohol)": "CCO",
    "🍬 Glucose (sugar)": "OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O",
}
PLAIN = {
    "NR-AhR": ("Pollutant alarm (AhR)", "Does it switch on the cell's alarm for pollutants like smoke and dioxins?"),
    "SR-MMP": ("Cell power-plant damage", "Does it disturb mitochondria, the parts of a cell that make energy?"),
    "SR-ARE": ("Oxidative stress", "Does it trigger the cell's 'rust' / damage-defence alarm?"),
}


@st.cache_resource
def get_screener():
    return Screener()


st.markdown('<div class="hero"><h1>🧪 ToxiScope</h1><p>Pick a molecule. A computer model that learned from '
            'thousands of real lab tests tells you which cell-stress alarms it might trigger — '
            'and how much to trust that answer.</p></div>', unsafe_allow_html=True)

try:
    S = get_screener()
except ModelsMissing as e:
    st.error(str(e)); st.stop()

tab_try, tab_learn, tab_acc, tab_lim = st.tabs(["🔬 Try it", "📖 Learn", "📊 Accuracy", "⚠️ Limits"])


def plain(ep):
    return PLAIN.get(ep, (ep, ALL_ENDPOINTS.get(ep, "")))


def card(ep, p):
    title, desc = plain(ep)
    flagged = p["active"]
    color = "#dc2626" if flagged else "#64748b"
    verdict = "🚩 Flagged: likely to trigger this test" if flagged else "Not flagged"
    pct = max(2, int(round(p["score"] * 100)))
    st.markdown(f'<div class="card" style="border-left:6px solid {color}"><div class="t">{title}</div>'
                f'<div class="d">{desc}</div><div class="v" style="color:{color}">{verdict}</div>'
                f'<div class="bar"><span style="width:{pct}%;background:{color}"></span></div>'
                f'<div class="s">Model score {p["score"]:.2f} · flagged at {p["threshold"]:.2f} or higher</div></div>',
                unsafe_allow_html=True)


# ------------------------------------------------------------ TRY IT
with tab_try:
    choice = st.selectbox("1️⃣  Pick a molecule", list(EXAMPLES))
    custom = st.text_input("…or paste your own SMILES code (optional — see the Learn tab)", "",
                           placeholder="example: CCO")
    smi = custom.strip() or EXAMPLES[choice]
    res = S.screen(smi)

    if "error" in res:
        st.error("That isn't a valid molecule code. Clear the box, or pick one from the list.")
    else:
        st.image(Draw.MolToImage(res["mol"], size=(380, 300)), caption="Structure drawn by the computer")
        look = {"supported": ("#16a34a", "Good evidence behind this result",
                              "Very similar molecules were in the lab data the model learned from."),
                "limited": ("#d97706", "Limited evidence — be careful",
                            "Only somewhat similar molecules were in the training data."),
                "outside": ("#dc2626", "Too unusual — don't trust this",
                            "Nothing like it was in the training data, so the model is basically guessing.")}
        col, head, sub = look[res["level"]]
        st.markdown(f'<div class="badge" style="background:{col}"><b>{head}</b><div>{sub} '
                    f'(closest known molecule: {res["max_sim"]*100:.0f}% similar)</div></div>',
                    unsafe_allow_html=True)

        show = True
        if res["level"] == "outside":
            show = st.checkbox("Show the model's guess anyway")
        if show:
            st.markdown("#### 2️⃣  What the model predicts")
            for ep, p in res["predictions"].items():
                card(ep, p)
            st.caption("This predicts lab-test reactions only. **Not flagged ≠ safe**, and **flagged ≠ poisonous**.")

            with st.expander("🧩 Which parts of the molecule drove this? (optional)"):
                ep = st.selectbox("Test", list(res["predictions"]), format_func=lambda e: plain(e)[0])
                items = S.explain(res["mol"], res["fp"], ep)
                st.caption("We hide each small piece of the molecule and see how the score changes. "
                           "🔴 red piece = pushes toward flagged, 🔵 blue = pushes away. "
                           "It shows what the model relies on, not proven chemistry.")
                for it in items:
                    atoms, bonds = environment_atoms_bonds(res["mol"], *it["env"])
                    color = (0.9, 0.2, 0.2) if it["delta"] > 0 else (0.2, 0.4, 0.9)
                    st.image(Draw.MolToImage(res["mol"], size=(300, 220), highlightAtoms=atoms,
                                             highlightBonds=bonds, highlightColor=color),
                             caption=("Pushes toward flagged" if it["delta"] > 0 else "Pushes away from flagged")
                             + f" ({it['delta']:+.2f})")

        with st.expander("👯 Similar molecules the lab already tested"):
            nb = res["neighbors"]
            st.image(Draw.MolsToGridImage([Chem.MolFromSmiles(s) for _, s, _ in nb], molsPerRow=2,
                                          subImgSize=(220, 190),
                                          legends=[f"{sim*100:.0f}% similar" for sim, _, _ in nb]))
            eps = S.all_endpoints
            lab = lambda v: {1: "Active", 0: "Inactive"}.get(int(v), "Not tested")
            st.dataframe(pd.DataFrame([{"Similarity": f"{sim*100:.0f}%", "SMILES": s,
                                        **{plain(e)[0]: lab(l[eps.index(e)]) for e in S.models}}
                                       for sim, s, l in nb]), hide_index=True)
        st.caption(f"SMILES used: `{smi}`")

# ------------------------------------------------------------ LEARN
with tab_learn:
    st.markdown("""
### The idea in 30 seconds
Scientists want to know early if a chemical might harm cells — but testing every chemical in a lab is slow and
expensive. So a US government programme (**Tox21**) used robots to test about 8,000 chemicals on living cells.
We trained a computer model on those results. Now it can *guess* what a new molecule would do — like a music app
recommending songs based on what similar songs you liked.

### "Since when does chemistry have models?"
Since the **1960s**. The field is called **QSAR** (*quantitative structure–activity relationship*): the shape of a
molecule predicts how it behaves. What's new is that we now use machine learning and big public datasets.

### What is a SMILES?
**A way to write a molecule as a line of text** so a computer can read it.
- Letters are atoms: **C** carbon, **O** oxygen, **N** nitrogen, **Cl** chlorine
- `=` is a double bond, digits like `1` open and close a ring, lowercase letters are ring atoms
- `CCO` = ethanol (drinking alcohol) · `c1ccccc1` = benzene (a 6-carbon ring)

You don't need to type one — the picker does it for you.

### What the app does, step by step
1. Reads the molecule and turns it into a **fingerprint** (a list of 2,048 yes/no flags: "has this small piece?").
2. A **Random Forest** (hundreds of simple decision trees voting) predicts each lab test.
3. Looks up the **most similar molecules** the lab really tested.
4. If nothing similar exists, it **tells you not to trust the result**. That honesty is the main feature.

### The 3 lab tests
- **Pollutant alarm (AhR):** a sensor cells use against smoke and dioxin-type pollutants.
- **Cell power-plant damage:** harm to mitochondria, the cell's energy factories.
- **Oxidative stress:** the cell's damage-defence alarm, a bit like rust.
""")

# ------------------------------------------------------------ ACCURACY
with tab_acc:
    E = S.metrics["endpoints"]
    st.markdown("Tested on molecules **the model never saw**, grouped so close relatives of training molecules "
                "couldn't leak in. These numbers come from the real training run.")
    for ep, v in E.items():
        title, _ = plain(ep)
        st.markdown(f'<div class="card"><div class="t">{title}</div>'
                    f'<div class="d">ROC-AUC {v["roc_auc"]:.2f} (0.5 = coin flip, 1.0 = perfect)</div>'
                    f'🎯 When it flags a molecule, it is right about <b>{v["precision"]*100:.0f}</b> times out of 100.<br>'
                    f'🔎 It catches about <b>{v["recall"]*100:.0f}</b> of every 100 truly active molecules.</div>',
                    unsafe_allow_html=True)
    df = pd.DataFrame([{"Test": plain(k)[0], "ROC-AUC": round(v["roc_auc"], 3), "PR-AUC": round(v["pr_auc"], 3),
                        "Precision": round(v["precision"], 3), "Recall": round(v["recall"], 3),
                        "Test molecules": v["n_test"]} for k, v in E.items()])
    st.dataframe(df, hide_index=True)
    st.plotly_chart(px.bar(df.melt("Test", ["ROC-AUC", "PR-AUC"]), x="Test", y="value", color="variable",
                           barmode="group", labels={"value": "score", "variable": ""}))
    st.markdown("#### Does 'unusual molecule' really mean worse accuracy? Yes:")
    rows = []
    for k, v in E.items():
        for b in v["auc_by_similarity"]:
            rows.append({"Test": plain(k)[0], "Similarity to training data": b["bin"],
                         "Molecules": b["n"], "ROC-AUC": None if b["roc_auc"] is None else round(b["roc_auc"], 3)})
    st.dataframe(pd.DataFrame(rows), hide_index=True)
    ep = st.selectbox("Confusion matrix for", list(E), format_func=lambda e: plain(e)[0])
    cm = E[ep]["confusion_matrix"]
    fig = go.Figure(go.Heatmap(z=cm, x=["Said no", "Said flagged"], y=["Truly inactive", "Truly active"],
                               text=cm, texttemplate="%{text}", colorscale="Blues", showscale=False))
    fig.update_yaxes(autorange="reversed")
    st.plotly_chart(fig)

# ------------------------------------------------------------ LIMITS
with tab_lim:
    d = S.metrics["dataset"]
    st.markdown(f"""
### Please read
- ⚠️ **This is a student research prototype, not a safety tool.** Never use it to decide a chemical is safe.
- A lab-test reaction is **not** the same as poisoning a human. Some reactions come from cell damage or test noise.
- Only **3 of the 12** Tox21 tests are modelled.
- Predictions get less reliable for unusual molecules (see Accuracy tab). The similarity cut-offs
  ({LOW_SIM:.2f} and {HIGH_SIM:.2f}) are rule-of-thumb values, not calibrated confidence.
- Tools like EPA TEST and CompTox already predict toxicity this way. ToxiScope is a transparent teaching workflow,
  not a new invention.

### Data
Tox21 (MoleculeNet release): {d['n_raw']:,} records → {d['n_valid_unique']:,} valid unique molecules
({d['n_invalid']} invalid and {d['n_duplicates']} duplicates removed). Missing results are skipped, never counted as
"inactive". Train/test split by molecular scaffold ({d['n_train']:,} / {d['n_test']:,}).
""")
