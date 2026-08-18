"""
SHIELD Prototype - Investigator Dashboard
------------------------------------------
Run with:  streamlit run app.py

Shows what Section 4.7 / 4.9 of the solution doc describes as the
"investigator workbench": ranked risk scores, SHAP-based plain-language
explanations, and a graph view for mule-ring detection.

NOTE on the network graph: this dataset is one row per account with no
account-to-account transaction pairs, so there is no real edge list to
build a transaction graph from. The network tab below SIMULATES a
plausible ring topology among the highest-risk accounts purely to
demonstrate what the Section 4.5.2 graph analytics layer (Neo4j +
Louvain community detection + PageRank) would surface once real
transaction-level data is wired in. It is clearly labeled as simulated
in the UI itself.
"""

import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx

st.set_page_config(page_title="SHIELD - Mule Detection Prototype", layout="wide")

ARTIFACTS = "artifacts"


@st.cache_data
def load_artifacts():
    scored = pd.read_parquet(f"{ARTIFACTS}/scored_accounts.parquet")
    X = pd.read_parquet(f"{ARTIFACTS}/feature_matrix.parquet")
    shap_values = np.load(f"{ARTIFACTS}/shap_values.npy")
    with open(f"{ARTIFACTS}/feature_names.json") as f:
        feature_names = json.load(f)
    with open(f"{ARTIFACTS}/metrics.json") as f:
        metrics = json.load(f)
    return scored, X, shap_values, feature_names, metrics


def friendly_feature_name(f: str) -> str:
    mapping = {
        "sentinel_count": "Count of 'not applicable' (-1) sentinel fields",
        "na_count": "Count of missing / not-collected fields",
        "sparsity_ratio": "Overall data sparsity (product coverage) ratio",
    }
    if f in mapping:
        return mapping[f]
    if any(f.startswith(p) for p in ["account_type", "segment", "occupation",
                                      "gender", "business_type", "tenure_bucket"]):
        return f.replace("_", ": ", 1).replace("_", " ")
    return f"Engineered signal {f} (velocity / ratio / balance feature)"


def build_synthetic_network(scored: pd.DataFrame, n_rings: int = 3, ring_size: int = 5,
                             n_background: int = 120, seed: int = 7):
    """
    SIMULATED transaction graph for demo purposes only (see module docstring).
    Clusters a handful of top-risk accounts into tight "rings" (mirroring
    Section 4.5.2's convergence pattern) and scatters background noise
    edges among random lower-risk accounts.
    """
    rng = np.random.RandomState(seed)
    top_risk = scored.sort_values("shield_score", ascending=False).head(n_rings * ring_size)
    background = scored[~scored["account_id"].isin(top_risk["account_id"])].sample(
        n=min(n_background, len(scored)), random_state=seed)

    G = nx.DiGraph()
    for _, row in pd.concat([top_risk, background]).iterrows():
        G.add_node(row["account_id"], score=row["shield_score"], flagged=row["flagged"])

    ring_ids = top_risk["account_id"].tolist()
    for i in range(n_rings):
        members = ring_ids[i * ring_size:(i + 1) * ring_size]
        if len(members) < 2:
            continue
        collector = members[0]  # simulate a "collector" account aggregating funds
        for m in members[1:]:
            G.add_edge(m, collector, weight=rng.uniform(0.5, 1.0))
        # collector forwards onward to an external-looking node
        G.add_edge(collector, f"EXT-{i}", weight=1.0)
        G.add_node(f"EXT-{i}", score=0, flagged=0)

    bg_ids = background["account_id"].tolist()
    for _ in range(len(bg_ids) // 3):
        a, b = rng.choice(bg_ids, 2, replace=False)
        G.add_edge(a, b, weight=rng.uniform(0.1, 0.4))

    return G, ring_ids


scored, X, shap_values, feature_names, metrics = load_artifacts()

st.title("🛡️ SHIELD — Mule Account Detection (Prototype)")
st.caption("Scaled-down demo of the SHIELD solution doc's ML + graph detection layers. "
           "Not the full production Kafka/Flink/Neo4j pipeline — see README for scope.")

# ---------------- KPI row ----------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Accounts scored", f"{metrics['n_accounts']:,}")
c2.metric("Flagged (score ≥ 650)", f"{metrics['flagged_count']:,}")
c3.metric("Features used", metrics["n_features"])
c4.metric("Model AUC (pseudo-labels)", f"{metrics['oof_auc']:.3f}", help=(
    "Trained against rule-derived pseudo-labels since this dataset has no "
    "confirmed SAR outcomes. In production this is replaced by real "
    "investigator-confirmed labels (Section 4.8 feedback loop) — treat this "
    "number as a pipeline sanity check, not a real-world accuracy claim."
))

st.warning(
    "⚠️ **Demo caveat:** this dataset has no ground-truth mule labels. Risk scores here "
    "are trained against rule-based pseudo-labels (KYC/occupation mismatch + anomaly "
    "ranking), not confirmed fraud outcomes. Good for demonstrating the pipeline end-to-end; "
    "not a validated production model.",
    icon="⚠️",
)

tab1, tab2, tab3 = st.tabs(["📋 Investigator Queue", "🔍 Account Explanation", "🕸️ Ring Detection (simulated)"])

# ---------------- Tab 1: queue ----------------
with tab1:
    st.subheader("Risk-ranked account queue")
    colf1, colf2, colf3 = st.columns(3)
    min_score = colf1.slider("Minimum SHIELD score", 0, 1000, 0, step=10)
    occ_filter = colf2.multiselect("Occupation", sorted(scored["occupation"].dropna().unique().tolist()))
    type_filter = colf3.multiselect("Account type", sorted(scored["account_type"].dropna().unique().tolist()))

    view = scored.sort_values("shield_score", ascending=False).copy()
    view = view[view["shield_score"] >= min_score]
    if occ_filter:
        view = view[view["occupation"].isin(occ_filter)]
    if type_filter:
        view = view[view["account_type"].isin(type_filter)]

    st.dataframe(
        view[["account_id", "shield_score", "flagged", "account_type", "segment",
              "occupation", "gender", "age", "iso_score", "xgb_score"]]
        .rename(columns={"shield_score": "SHIELD Score", "flagged": "Flagged",
                          "account_type": "Account Type", "occupation": "Occupation",
                          "gender": "Gender", "age": "Age", "iso_score": "Anomaly (0-1)",
                          "xgb_score": "XGB prob (0-1)"}),
        use_container_width=True, height=420,
    )

    fig = px.histogram(scored, x="shield_score", nbins=50, title="SHIELD Score distribution")
    fig.add_vline(x=650, line_dash="dash", line_color="red", annotation_text="alert threshold (650)")
    st.plotly_chart(fig, use_container_width=True)

# ---------------- Tab 2: explanation ----------------
with tab2:
    st.subheader("Per-account SHAP explanation")
    default_acct = int(scored.sort_values("shield_score", ascending=False)["account_id"].iloc[0])
    acct_id = st.selectbox("Select account", scored["account_id"].tolist(),
                            index=scored["account_id"].tolist().index(default_acct))

    row_idx = scored.index[scored["account_id"] == acct_id][0]
    row = scored.loc[row_idx]

    m1, m2, m3 = st.columns(3)
    m1.metric("SHIELD Score", f"{row['shield_score']:.0f} / 1000")
    m2.metric("Anomaly component", f"{row['iso_score']:.2f}")
    m3.metric("XGBoost probability", f"{row['xgb_score']:.2f}")

    sv = shap_values[row_idx]
    top_idx = np.argsort(-np.abs(sv))[:8]
    contrib = pd.DataFrame({
        "feature": [friendly_feature_name(feature_names[i]) for i in top_idx],
        "impact": sv[top_idx],
    }).sort_values("impact")

    fig2 = px.bar(contrib, x="impact", y="feature", orientation="h",
                  color="impact", color_continuous_scale=["#2b6cb0", "#e53e3e"],
                  title=f"Top contributing factors — Account {acct_id}")
    st.plotly_chart(fig2, use_container_width=True)

    direction = "increase" if row["shield_score"] >= 500 else "decrease"
    narrative = f"Account {acct_id} flagged because:\n\n"
    for rank, i in enumerate(top_idx[:4], start=1):
        sign = "+" if sv[i] > 0 else "-"
        narrative += f"{rank}. {friendly_feature_name(feature_names[i])} ({sign}{abs(sv[i]):.0f} pts contribution)\n"
    st.text(narrative)
    if row["flagged"]:
        st.error(f"🚩 FLAGGED — SHIELD Score {row['shield_score']:.0f} ≥ 650 threshold. "
                  "Routed to investigator queue per Section 4.7 (2hr SLA on any auto-freeze).")
    else:
        st.success("Not currently flagged.")

# ---------------- Tab 3: network ----------------
with tab3:
    st.subheader("Simulated mule-ring detection (graph analytics demo)")
    st.info(
        "This dataset has no real account-to-account transaction pairs, so the graph "
        "below is **simulated**: it clusters the top risk-scored accounts into ring "
        "topologies (funds converging to a collector account, per Section 4.5.2) purely "
        "to demonstrate what the Neo4j + Louvain + PageRank layer would surface with real "
        "transaction-level data.",
        icon="🕸️",
    )

    G, ring_ids = build_synthetic_network(scored)
    pos = nx.spring_layout(G, seed=7, k=0.4)

    edge_x, edge_y = [], []
    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
    edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=1, color="#aaaaaa"),
                             hoverinfo="none", mode="lines")

    node_x, node_y, node_color, node_text, node_size = [], [], [], [], []
    for n in G.nodes():
        x, y = pos[n]
        node_x.append(x)
        node_y.append(y)
        is_ring = n in ring_ids
        node_color.append("#e53e3e" if is_ring else "#4a5568")
        node_size.append(18 if is_ring else 8)
        node_text.append(f"Account {n}")
    node_trace = go.Scatter(x=node_x, y=node_y, mode="markers", text=node_text,
                             hoverinfo="text",
                             marker=dict(color=node_color, size=node_size, line_width=1))

    fig3 = go.Figure(data=[edge_trace, node_trace],
                      layout=go.Layout(showlegend=False, height=600,
                                        margin=dict(l=10, r=10, t=10, b=10),
                                        xaxis=dict(showgrid=False, zeroline=False, visible=False),
                                        yaxis=dict(showgrid=False, zeroline=False, visible=False)))
    st.plotly_chart(fig3, use_container_width=True)
    st.caption("Red = high-risk accounts clustered into simulated rings (with a collector node "
               "forwarding onward). Grey = background accounts with random low-weight noise edges.")

st.divider()
st.caption("SHIELD Prototype — hackathon demo build. See README.md for scope, limitations, "
           "and what would need to change for a production deployment.")
