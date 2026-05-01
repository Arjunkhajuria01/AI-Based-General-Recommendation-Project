"""
recommender.py
==============
Main orchestration pipeline — ties all modules together.

This is the single entry point for:
  1. Building the hybrid recommendation system
  2. Running recommendations for any user
  3. Running full evaluation

Run this file directly for a terminal demo:
    python recommender.py
"""

import sys
import os

# Ensure src/ is on the path when running from project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np

from src.data_loader         import load_unified_dataset, generate_interaction_matrix, save_datasets
from src.preprocessor        import preprocess, fill_interaction_matrix
from src.feature_engineering import TFIDFFeatureBuilder
from src.content_based       import ContentBasedRecommender
from src.collaborative       import CollaborativeRecommender
from src.hybrid              import HybridRecommender
from src.evaluator           import (train_test_split_matrix, evaluate_recommender,
                                      print_evaluation_report)


# ─────────────────────────────────────────────
# SECTION 1: Build the full recommendation system
# ─────────────────────────────────────────────

def build_system(alpha: float = 0.5,
                 n_neighbors: int = 10,
                 n_users: int = 50,
                 sparsity: float = 0.85) -> dict:
    """
    Builds and returns the full hybrid recommendation system.
    
    Parameters:
        alpha       : Blend ratio — CB weight (0.5 = equal blend)
        n_neighbors : K for KNN collaborative filtering
        n_users     : Number of synthetic users to generate
        sparsity    : Fraction of missing ratings in interaction matrix
    
    Returns:
        Dict containing:
            items, raw_matrix, train_matrix, test_matrix,
            cb, cf, hybrid (all fitted components)
    """
    print("\n" + "="*55)
    print("  🚀 Building Hybrid Recommendation System")
    print("="*55)

    # ── 1. Load data ──────────────────────────────
    items     = load_unified_dataset()
    items     = preprocess(items)
    raw_matrix = generate_interaction_matrix(items, n_users=n_users, sparsity=sparsity)

    # ── 2. Train/Test Split (BEFORE filling) ──────
    train_raw, test_raw = train_test_split_matrix(raw_matrix, test_ratio=0.2)

    # ── 3. Fill training matrix ───────────────────
    train_filled = fill_interaction_matrix(train_raw, strategy="user_mean")

    # ── 4. Content-Based: TF-IDF + Similarity ─────
    builder = TFIDFFeatureBuilder(max_features=500, ngram_range=(1, 2))
    builder.fit_transform(items["content"])
    sim_matrix = builder.compute_similarity()
    cb = ContentBasedRecommender(items, sim_matrix, builder)

    # ── 5. Collaborative: KNN on train matrix ─────
    cf = CollaborativeRecommender(train_filled, n_neighbors=n_neighbors)

    # ── 6. Hybrid Engine ──────────────────────────
    hybrid = HybridRecommender(items, cb, cf, alpha=alpha)

    # ── 7. Save data to disk ──────────────────────
    save_datasets(items, raw_matrix, output_dir="data/processed")

    print("\n✅ System ready!\n")

    return {
        "items":        items,
        "raw_matrix":   raw_matrix,
        "train_matrix": train_filled,
        "test_matrix":  test_raw,
        "cb":           cb,
        "cf":           cf,
        "hybrid":       hybrid,
        "builder":      builder,
    }


# ─────────────────────────────────────────────
# SECTION 2: Get Recommendations
# ─────────────────────────────────────────────

def get_recommendations(system: dict,
                          user_id: str,
                          liked_item_ids: list,
                          top_n: int = 10,
                          filter_domain: str = None,
                          alpha: float = None) -> pd.DataFrame:
    """
    Public API for getting hybrid recommendations.
    
    Parameters:
        system         : Dict returned by build_system()
        user_id        : User ID (or None for cold-start)
        liked_item_ids : List of item_ids the user has liked/interacted with
        top_n          : Number of recommendations
        filter_domain  : 'movie', 'course', or None
        alpha          : Override default alpha (optional)
    
    Returns:
        DataFrame of recommended items with scores and explanations
    """
    hybrid = system["hybrid"]

    if alpha is not None:
        hybrid.set_alpha(alpha)

    recs = hybrid.recommend(
        user_id=user_id,
        liked_item_ids=liked_item_ids,
        top_n=top_n,
        filter_domain=filter_domain,
        exclude_seen=True
    )

    return recs


# ─────────────────────────────────────────────
# SECTION 3: Run Evaluation
# ─────────────────────────────────────────────

def run_evaluation(system: dict, k: int = 10) -> dict:
    """
    Evaluates all three models (CB, CF, Hybrid) and prints a comparison report.
    
    Parameters:
        system : Dict returned by build_system()
        k      : Cutoff rank for Precision@K and Recall@K
    
    Returns:
        Dict of {model_name: results}
    """
    items    = system["items"]
    test_mat = system["test_matrix"]
    cb       = system["cb"]
    cf       = system["cf"]
    hybrid   = system["hybrid"]

    print("\n" + "="*55)
    print("  📊 Running Evaluation (K={})".format(k))
    print("="*55)

    all_results = {}

    # ── Evaluate Content-Based ────────────────────
    def cb_fn(user_id):
        # For CB, use the user's train ratings as liked items
        user_row   = system["train_matrix"].loc[user_id]
        liked_ids  = user_row[user_row >= 3.5].index.tolist()
        if not liked_ids:
            liked_ids = user_row.nlargest(3).index.tolist()
        recs = cb.recommend(liked_ids, top_n=k, exclude_seen=True)
        return recs["item_id"].tolist()

    cb_results = evaluate_recommender(cb_fn, test_mat, k=k)
    print_evaluation_report(cb_results, "Content-Based (TF-IDF + Cosine)")
    all_results["Content-Based"] = cb_results

    # ── Evaluate Collaborative ────────────────────
    def cf_fn(user_id):
        recs = cf.recommend(user_id, top_n=k, items_df=items)
        return recs["item_id"].tolist()

    cf_results = evaluate_recommender(cf_fn, test_mat, k=k)
    print_evaluation_report(cf_results, "Collaborative (KNN)")
    all_results["Collaborative"] = cf_results

    # ── Evaluate Hybrid ───────────────────────────
    def hybrid_fn(user_id):
        user_row  = system["train_matrix"].loc[user_id]
        liked_ids = user_row[user_row >= 3.5].index.tolist()
        if not liked_ids:
            liked_ids = user_row.nlargest(3).index.tolist()
        recs = hybrid.recommend(user_id, liked_item_ids=liked_ids, top_n=k)
        return recs["item_id"].tolist()

    hybrid_results = evaluate_recommender(hybrid_fn, test_mat, k=k)
    print_evaluation_report(hybrid_results, "Hybrid (CB + CF)")
    all_results["Hybrid"] = hybrid_results

    # ── Comparison Summary ────────────────────────
    print("\n" + "="*55)
    print("  Comparison Summary @ K={}".format(k))
    print("="*55)
    print(f"  {'Model':<25} {'Precision@K':>12} {'Recall@K':>10}")
    print(f"  {'-'*25} {'-'*12} {'-'*10}")
    for name, res in all_results.items():
        print(f"  {name:<25} {res['precision_at_k']:>12.4f} {res['recall_at_k']:>10.4f}")
    print("="*55 + "\n")

    return all_results


# ─────────────────────────────────────────────
# SECTION 4: Terminal Demo
# ─────────────────────────────────────────────

def demo(system: dict) -> None:
    """
    Runs an interactive terminal demo showing recommendations for sample inputs.
    """
    items  = system["items"]
    hybrid = system["hybrid"]

    print("\n" + "="*55)
    print("  🎬 Demo: Hybrid Recommendations")
    print("="*55)

    # ── Demo 1: Movie fan ─────────────────────────
    print("\n[Demo 1] User likes Inception + The Dark Knight (movie fan)")
    recs = hybrid.recommend(
        user_id="U001",
        liked_item_ids=["M002", "M001"],   # Inception, Dark Knight
        top_n=5,
        filter_domain=None
    )
    _print_recs(recs)

    # ── Demo 2: Course learner ────────────────────
    print("\n[Demo 2] User likes 'Python for Beginners' + 'Data Science' (learner)")
    recs = hybrid.recommend(
        user_id="U003",
        liked_item_ids=["C001", "C005"],   # Python, Data Science
        top_n=5,
        filter_domain="course"
    )
    _print_recs(recs)

    # ── Demo 3: Cold Start (new user, unknown ID) ──
    print("\n[Demo 3] Cold Start — new user with no history")
    cb = system["cb"]
    recs = cb.recommend_by_tags(
        preference_text="I enjoy space exploration, science fiction and thrillers",
        top_n=5
    )
    print(recs[["item_id", "title", "category", "cb_score"]].to_string(index=False))

    # ── Demo 4: Explain a recommendation ──────────
    print("\n[Demo 4] Explaining why 'Arrival' (M025) was recommended")
    explanation = cb.explain("M025")
    print(f"  Title:    {explanation['title']}")
    print(f"  Category: {explanation['category']}")
    print(f"  Domain:   {explanation['domain']}")
    print(f"  Keywords: {explanation['keywords']}")


def _print_recs(recs: pd.DataFrame) -> None:
    """Helper to print recommendation DataFrame cleanly."""
    cols = [c for c in ["item_id", "title", "domain", "hybrid_score", "explanation"]
            if c in recs.columns]
    print(recs[cols].to_string(index=False))


# ─────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    system = build_system(alpha=0.5, n_neighbors=10)

    # Run demo
    demo(system)

    # Run evaluation
    run_evaluation(system, k=10)
