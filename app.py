"""
app.py
======
Streamlit Web Application for the Hybrid Recommendation System.

Run with:
    streamlit run app.py

Features:
  - Beautiful dark-themed UI
  - Select items you like from the dropdown
  - Choose recommendation domain (movies / courses / all)
  - Adjust CB vs CF alpha slider
  - View recommendations with similarity score bars
  - View similarity heatmap (top 20 items)
  - Evaluation metrics dashboard
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sys
import os

# ── Make src importable ────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_loader         import load_unified_dataset, generate_interaction_matrix
from src.preprocessor        import preprocess, fill_interaction_matrix
from src.feature_engineering import TFIDFFeatureBuilder
from src.content_based       import ContentBasedRecommender
from src.collaborative       import CollaborativeRecommender
from src.hybrid              import HybridRecommender
from src.evaluator           import (train_test_split_matrix,
                                      evaluate_recommender,
                                      print_evaluation_report)

# ─────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="Hybrid Recommender System",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# Custom CSS — Dark Premium Theme
# ─────────────────────────────────────────────

st.markdown("""
<style>
    /* Import font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Dark background */
    .stApp {
        background: linear-gradient(135deg, #0d1117 0%, #161b22 100%);
        color: #e6edf3;
    }

    /* Header banner */
    .hero-banner {
        background: linear-gradient(135deg, #1a237e 0%, #4a148c 50%, #880e4f 100%);
        border-radius: 16px;
        padding: 2.5rem 2rem;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 8px 32px rgba(74, 20, 140, 0.4);
    }
    .hero-banner h1 {
        font-size: 2.4rem;
        font-weight: 700;
        color: #ffffff;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .hero-banner p {
        color: #ce93d8;
        font-size: 1.05rem;
        margin: 0.5rem 0 0;
    }

    /* Metric cards */
    .metric-card {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        text-align: center;
        backdrop-filter: blur(10px);
    }
    .metric-card .value {
        font-size: 2rem;
        font-weight: 700;
        color: #ce93d8;
    }
    .metric-card .label {
        font-size: 0.8rem;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* Section header */
    .section-header {
        font-size: 1.15rem;
        font-weight: 600;
        color: #e6edf3;
        border-left: 4px solid #7c4dff;
        padding-left: 0.8rem;
        margin: 1.5rem 0 1rem;
    }

    /* Rec card */
    .rec-card {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 12px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.7rem;
        transition: all 0.2s ease;
    }
    .rec-card:hover {
        border-color: rgba(124,77,255,0.4);
        background: rgba(124,77,255,0.08);
        transform: translateX(4px);
    }
    .rec-title {
        font-weight: 600;
        font-size: 1rem;
        color: #e6edf3;
    }
    .rec-meta {
        font-size: 0.8rem;
        color: #8b949e;
        margin-top: 0.2rem;
    }
    .rec-explanation {
        font-size: 0.75rem;
        color: #ce93d8;
        margin-top: 0.4rem;
        font-style: italic;
    }

    /* Domain badge */
    .badge-movie  { background: #1a237e; color: #90caf9; border-radius: 4px; padding: 2px 8px; font-size: 0.72rem; font-weight: 600; }
    .badge-course { background: #1b5e20; color: #a5d6a7; border-radius: 4px; padding: 2px 8px; font-size: 0.72rem; font-weight: 600; }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #161b22;
        border-right: 1px solid rgba(255,255,255,0.07);
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #7c4dff, #e040fb);
        color: white;
        border: none;
        border-radius: 10px;
        font-weight: 600;
        padding: 0.6rem 2rem;
        font-size: 0.95rem;
        transition: opacity 0.2s;
    }
    .stButton > button:hover { opacity: 0.85; }

    /* Hide streamlit branding */
    #MainMenu, footer { visibility: hidden; }
    .stDeployButton { display: none; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Cache System Build (expensive — run once)
# ─────────────────────────────────────────────

@st.cache_resource(show_spinner=True)
def build_cached_system():
    """Build and cache the full recommendation system."""
    items      = load_unified_dataset()
    items      = preprocess(items)
    raw_matrix = generate_interaction_matrix(items, n_users=50, sparsity=0.85)

    train_raw, test_raw = train_test_split_matrix(raw_matrix, test_ratio=0.2)
    train_filled        = fill_interaction_matrix(train_raw, strategy="user_mean")

    builder = TFIDFFeatureBuilder(max_features=500, ngram_range=(1, 2))
    builder.fit_transform(items["content"])
    sim_matrix = builder.compute_similarity()

    cb = ContentBasedRecommender(items, sim_matrix, builder)
    cf = CollaborativeRecommender(train_filled, n_neighbors=10)

    return {
        "items":        items,
        "raw_matrix":   raw_matrix,
        "train_matrix": train_filled,
        "test_matrix":  test_raw,
        "sim_matrix":   sim_matrix,
        "cb":           cb,
        "cf":           cf,
        "builder":      builder,
    }


# ─────────────────────────────────────────────
# Load System
# ─────────────────────────────────────────────

with st.spinner("⚙️ Building recommendation system..."):
    system = build_cached_system()

items      = system["items"]
sim_matrix = system["sim_matrix"]
cb         = system["cb"]
cf         = system["cf"]


# ─────────────────────────────────────────────
# Hero Banner
# ─────────────────────────────────────────────

st.markdown("""
<div class="hero-banner">
  <h1>🎯 Hybrid Recommendation Engine</h1>
  <p>TF-IDF Content-Based + KNN Collaborative Filtering · Movies &amp; Courses · Domain-Independent</p>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Sidebar Controls
# ─────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚙️ Controls")
    st.markdown("---")

    # User selection
    st.markdown("### 👤 User Profile")
    all_users  = list(system["train_matrix"].index)
    user_id    = st.selectbox("Select User ID", ["Cold Start (New User)"] + all_users)
    if user_id == "Cold Start (New User)":
        user_id = None

    # Item selection
    st.markdown("### ❤️ Liked Items")
    all_titles = items["title"].tolist()
    all_ids    = items["item_id"].tolist()
    title_to_id = dict(zip(all_titles, all_ids))

    selected_titles = st.multiselect(
        "Select items you like:",
        options=all_titles,
        default=["inception", "the dark knight", "machine learning a z"][:2]
            if "inception" in all_titles else all_titles[:2]
    )
    liked_ids = [title_to_id[t] for t in selected_titles if t in title_to_id]

    st.markdown("### 🌐 Domain Filter")
    domain_choice = st.radio(
        "Show recommendations from:",
        ["All", "Movies Only", "Courses Only"]
    )
    filter_domain = {"All": None, "Movies Only": "movie", "Courses Only": "course"}[domain_choice]

    st.markdown("### ⚖️ Algorithm Weights")
    alpha = st.slider(
        "Content-Based Weight (α)",
        min_value=0.0, max_value=1.0, value=0.5, step=0.05,
        help="α=1.0 → pure content-based | α=0.0 → pure collaborative"
    )
    st.caption(f"CB: **{int(alpha*100)}%** | CF: **{int((1-alpha)*100)}%**")

    top_n = st.slider("Number of Recommendations", 3, 20, 8)

    st.markdown("---")
    run_btn = st.button("🚀 Get Recommendations", use_container_width=True)


# ─────────────────────────────────────────────
# Dataset Overview Stats
# ─────────────────────────────────────────────

st.markdown('<div class="section-header">📊 Dataset Overview</div>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f'<div class="metric-card"><div class="value">{len(items)}</div><div class="label">Total Items</div></div>', unsafe_allow_html=True)
with col2:
    st.markdown(f'<div class="metric-card"><div class="value">{items[items.domain=="movie"].shape[0]}</div><div class="label">Movies</div></div>', unsafe_allow_html=True)
with col3:
    st.markdown(f'<div class="metric-card"><div class="value">{items[items.domain=="course"].shape[0]}</div><div class="label">Courses</div></div>', unsafe_allow_html=True)
with col4:
    density = system["raw_matrix"].notna().sum().sum() / (system["raw_matrix"].shape[0] * system["raw_matrix"].shape[1]) * 100
    st.markdown(f'<div class="metric-card"><div class="value">{density:.1f}%</div><div class="label">Matrix Density</div></div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Tabs: Recommendations | Similarity | Evaluation
# ─────────────────────────────────────────────

tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 Recommendations",
    "🔥 Similarity Map",
    "📈 Evaluation",
    "📋 Dataset Explorer"
])


# ──────────────────────────────────────────────────────────
# TAB 1: Recommendations
# ──────────────────────────────────────────────────────────

with tab1:
    if run_btn or True:   # auto-show on load
        hybrid = HybridRecommender(items, cb, cf, alpha=alpha)

        if not liked_ids and user_id is None:
            st.warning("⚠️ Select at least one liked item or a user ID to get recommendations.")
        else:
            recs = hybrid.recommend(
                user_id=user_id,
                liked_item_ids=liked_ids,
                top_n=top_n,
                filter_domain=filter_domain,
                exclude_seen=True
            )

            if recs.empty:
                st.info("No recommendations found. Try changing filters.")
            else:
                # Split into two columns
                col_left, col_right = st.columns([1.2, 1])

                with col_left:
                    st.markdown('<div class="section-header">🎯 Top Recommendations</div>',
                                unsafe_allow_html=True)

                    for _, row in recs.iterrows():
                        badge = (f'<span class="badge-movie">🎬 Movie</span>'
                                 if row.get("domain") == "movie"
                                 else f'<span class="badge-course">📚 Course</span>')
                        score_bar = "▰" * int(row["hybrid_score"] * 10) + "▱" * (10 - int(row["hybrid_score"] * 10))
                        explanation = row.get("explanation", "")
                        st.markdown(f"""
                        <div class="rec-card">
                            <div class="rec-title">{row['title'].title()}</div>
                            <div class="rec-meta">
                                {badge}
                                &nbsp;&nbsp;{row.get('category','').title()}
                                &nbsp;·&nbsp; Score: <b>{row['hybrid_score']:.3f}</b>
                                &nbsp;{score_bar}
                            </div>
                            <div class="rec-explanation">{explanation}</div>
                        </div>
                        """, unsafe_allow_html=True)

                with col_right:
                    st.markdown('<div class="section-header">📊 Score Breakdown</div>',
                                unsafe_allow_html=True)

                    fig = go.Figure()
                    y_labels = [r["title"].title()[:28] for _, r in recs.iterrows()]

                    if "cb_norm" in recs.columns:
                        fig.add_trace(go.Bar(
                            name="Content-Based",
                            y=y_labels,
                            x=recs["cb_norm"].tolist(),
                            orientation="h",
                            marker_color="rgba(124,77,255,0.7)",
                        ))
                    if "cf_norm" in recs.columns:
                        fig.add_trace(go.Bar(
                            name="Collaborative",
                            y=y_labels,
                            x=recs["cf_norm"].tolist(),
                            orientation="h",
                            marker_color="rgba(224,64,251,0.7)",
                        ))

                    fig.update_layout(
                        barmode="group",
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font=dict(color="#e6edf3", size=11),
                        legend=dict(bgcolor="rgba(0,0,0,0)"),
                        height=380,
                        margin=dict(l=0, r=10, t=10, b=10),
                        xaxis=dict(
                            title="Normalised Score",
                            gridcolor="rgba(255,255,255,0.05)",
                            range=[0, 1]
                        ),
                        yaxis=dict(autorange="reversed"),
                    )
                    st.plotly_chart(fig, use_container_width=True)


# ──────────────────────────────────────────────────────────
# TAB 2: Similarity Heatmap
# ──────────────────────────────────────────────────────────

with tab2:
    st.markdown('<div class="section-header">🔥 Content Similarity Heatmap (Top 20 Items)</div>',
                unsafe_allow_html=True)

    n_show = 20
    sub_items  = items.head(n_show)
    sub_matrix = sim_matrix[:n_show, :n_show]
    labels     = [t[:22] for t in sub_items["title"].str.title().tolist()]

    fig_heat = px.imshow(
        sub_matrix,
        x=labels, y=labels,
        color_continuous_scale="Purples",
        aspect="auto",
        title="",
        zmin=0, zmax=1
    )
    fig_heat.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e6edf3", size=10),
        height=560,
        margin=dict(l=10, r=10, t=20, b=10),
        coloraxis_colorbar=dict(title="Cosine Sim")
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    st.info("💡 Darker squares = higher content similarity. "
            "Items with similar genres/tags cluster together.")


# ──────────────────────────────────────────────────────────
# TAB 3: Evaluation Metrics
# ──────────────────────────────────────────────────────────

with tab3:
    st.markdown('<div class="section-header">📈 Model Evaluation (Precision@K & Recall@K)</div>',
                unsafe_allow_html=True)

    k_val = st.slider("Select K for evaluation", 5, 20, 10, key="eval_k")

    if st.button("▶ Run Evaluation", key="eval_btn"):
        with st.spinner("Running evaluation over all test users..."):

            test_mat = system["test_matrix"]
            train_mat = system["train_matrix"]

            def cb_fn(uid):
                ur = train_mat.loc[uid]
                li = ur[ur >= 3.5].index.tolist() or ur.nlargest(3).index.tolist()
                return cb.recommend(li, top_n=k_val)["item_id"].tolist()

            def cf_fn(uid):
                return cf.recommend(uid, top_n=k_val, items_df=items)["item_id"].tolist()

            hybrid_tmp = HybridRecommender(items, cb, cf, alpha=alpha)
            def hybrid_fn(uid):
                ur = train_mat.loc[uid]
                li = ur[ur >= 3.5].index.tolist() or ur.nlargest(3).index.tolist()
                return hybrid_tmp.recommend(uid, li, top_n=k_val)["item_id"].tolist()

            cb_res  = evaluate_recommender(cb_fn,     test_mat, k=k_val)
            cf_res  = evaluate_recommender(cf_fn,     test_mat, k=k_val)
            hyb_res = evaluate_recommender(hybrid_fn, test_mat, k=k_val)

        # Comparison table
        eval_data = {
            "Model":       ["Content-Based", "Collaborative (KNN)", f"Hybrid (α={alpha})"],
            f"Precision@{k_val}": [cb_res["precision_at_k"], cf_res["precision_at_k"], hyb_res["precision_at_k"]],
            f"Recall@{k_val}":    [cb_res["recall_at_k"],    cf_res["recall_at_k"],    hyb_res["recall_at_k"]],
            "Users Eval":  [cb_res["n_users_evaluated"], cf_res["n_users_evaluated"], hyb_res["n_users_evaluated"]],
        }
        eval_df = pd.DataFrame(eval_data)
        st.dataframe(eval_df.style.highlight_max(
            subset=[f"Precision@{k_val}", f"Recall@{k_val}"],
            color="#4a148c"
        ), use_container_width=True)

        # Bar chart comparison
        fig_eval = go.Figure()
        for model, pr, rec in zip(
            eval_data["Model"],
            eval_data[f"Precision@{k_val}"],
            eval_data[f"Recall@{k_val}"]
        ):
            fig_eval.add_trace(go.Bar(name=f"{model} P@K", x=[model], y=[pr],
                                       marker_color="rgba(124,77,255,0.8)"))
            fig_eval.add_trace(go.Bar(name=f"{model} R@K", x=[model], y=[rec],
                                       marker_color="rgba(224,64,251,0.6)"))

        fig_eval.update_layout(
            barmode="group",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e6edf3"),
            height=380,
            showlegend=False,
            yaxis=dict(title="Score", gridcolor="rgba(255,255,255,0.05)"),
            margin=dict(t=20, b=20),
        )
        st.plotly_chart(fig_eval, use_container_width=True)

        # Explanation
        st.markdown("""
        > **Precision@K** = Fraction of top-K recommendations that are actually relevant (higher = better).  
        > **Recall@K** = Fraction of all relevant items that appeared in top-K (higher = better).
        """)
    else:
        st.info("Click **Run Evaluation** above to compute metrics over test users.")


# ──────────────────────────────────────────────────────────
# TAB 4: Dataset Explorer
# ──────────────────────────────────────────────────────────

with tab4:
    st.markdown('<div class="section-header">📋 Unified Item Dataset</div>',
                unsafe_allow_html=True)

    domain_filter = st.selectbox("Filter by domain:", ["All", "movie", "course"], key="exp_domain")
    exp_items = items if domain_filter == "All" else items[items["domain"] == domain_filter]

    st.dataframe(
        exp_items[["item_id", "title", "category", "domain", "cost", "time", "tags"]]
        .rename(columns={"item_id": "ID", "title": "Title", "category": "Category",
                          "domain": "Domain", "cost": "Cost ($)", "time": "Duration (hrs)", "tags": "Tags"}),
        use_container_width=True,
        height=400
    )

    st.markdown('<div class="section-header">📊 Category Distribution</div>',
                unsafe_allow_html=True)

    cat_counts = exp_items["category"].value_counts().reset_index()
    cat_counts.columns = ["Category", "Count"]
    fig_cat = px.bar(
        cat_counts, x="Category", y="Count",
        color="Count",
        color_continuous_scale="Purples",
        title=""
    )
    fig_cat.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e6edf3"),
        height=350,
        showlegend=False,
        coloraxis_showscale=False,
        xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
        margin=dict(t=10)
    )
    st.plotly_chart(fig_cat, use_container_width=True)
