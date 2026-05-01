"""
hybrid.py
=========
Hybrid Recommendation Engine — combines Content-Based + Collaborative Filtering
using a weighted averaging strategy.

Why Hybrid?
  - Content-Based alone: doesn't discover novel items (echo chamber effect)
  - Collaborative alone: fails for cold-start users (no history)
  - Hybrid: best of both worlds — personalised + content-aware

Weighting Formula:
    hybrid_score = α × cb_score + (1 - α) × cf_score_normalised

Where:
  α (alpha) = weight given to content-based score (default: 0.5)
  Scores are normalised to [0, 1] before combining to ensure fair weighting.
"""

import numpy as np
import pandas as pd
from typing import List, Optional

from src.content_based  import ContentBasedRecommender
from src.collaborative  import CollaborativeRecommender


# ─────────────────────────────────────────────
# Helper: Min-Max Normalisation
# ─────────────────────────────────────────────

def _normalise(series: pd.Series) -> pd.Series:
    """
    Scales a Series to [0, 1] range.
    Formula: (x - min) / (max - min)
    If all values are the same, returns a series of 0.5 (mid-scale).
    """
    min_val = series.min()
    max_val = series.max()
    if max_val - min_val == 0:
        return pd.Series([0.5] * len(series), index=series.index)
    return (series - min_val) / (max_val - min_val)


# ─────────────────────────────────────────────
# Hybrid Recommender
# ─────────────────────────────────────────────

class HybridRecommender:
    """
    Combines Content-Based and Collaborative Filtering via weighted fusion.

    Parameters:
        items_df      : Preprocessed items DataFrame
        cb_recommender: Fitted ContentBasedRecommender instance
        cf_recommender: Fitted CollaborativeRecommender instance
        alpha         : Weight for content-based score (0.0 to 1.0)
                        alpha=1.0 → pure content-based
                        alpha=0.0 → pure collaborative
                        alpha=0.5 → equal blend (default)
    """

    def __init__(self,
                 items_df: pd.DataFrame,
                 cb_recommender: ContentBasedRecommender,
                 cf_recommender: CollaborativeRecommender,
                 alpha: float = 0.5):

        self.items_df    = items_df.reset_index(drop=True)
        self.cb          = cb_recommender
        self.cf          = cf_recommender
        self.alpha       = alpha    # content weight

        self.all_item_ids = list(self.items_df["item_id"])

    # ─────────────────────────────────────────────
    # Main Hybrid Recommend Function
    # ─────────────────────────────────────────────

    def recommend(self,
                  user_id: Optional[str],
                  liked_item_ids: List[str],
                  top_n: int = 10,
                  filter_domain: Optional[str] = None,
                  exclude_seen: bool = True) -> pd.DataFrame:
        """
        Generates hybrid recommendations by fusing CB and CF scores.

        Parameters:
            user_id        : User ID for collaborative filtering.
                             If None or unknown → falls back to content-based only.
            liked_item_ids : List of item_ids the user liked (for content-based).
            top_n          : Number of top items to return.
            filter_domain  : 'movie', 'course', or None (both).
            exclude_seen   : Whether to remove already-liked items from results.

        Returns:
            DataFrame sorted by hybrid_score with explanation columns.
        """

        candidate_ids = [
            iid for iid in self.all_item_ids
            if not (exclude_seen and iid in liked_item_ids)
        ]

        # ── Step 1: Get Content-Based Scores ─────────
        cb_scores = self.cb.get_scores_for_items(liked_item_ids, candidate_ids)

        # ── Step 2: Get Collaborative Scores ──────────
        cf_scores = self.cf.get_scores_for_items(user_id, candidate_ids)

        # ── Step 3: Build score DataFrame ─────────────
        score_df = pd.DataFrame({
            "item_id":  candidate_ids,
            "cb_score": [cb_scores.get(iid, 0.0) for iid in candidate_ids],
            "cf_score": [cf_scores.get(iid, 0.0) for iid in candidate_ids],
        })

        # ── Step 4: Normalise scores to [0, 1] ────────
        score_df["cb_norm"] = _normalise(score_df["cb_score"])
        score_df["cf_norm"] = _normalise(score_df["cf_score"])

        # ── Step 5: Compute weighted hybrid score ──────
        # If user_id is unknown (cold start), use only content-based
        alpha = self.alpha
        if user_id is None or user_id not in self.cf.user_to_idx:
            alpha = 1.0   # full content-based for cold-start
            print(f"[Hybrid] Cold-start detected → using pure content-based (α=1.0)")

        score_df["hybrid_score"] = (
            alpha       * score_df["cb_norm"] +
            (1 - alpha) * score_df["cf_norm"]
        )

        # ── Step 6: Merge with item metadata ──────────
        meta_cols = ["item_id", "title", "category", "domain", "cost", "time"]
        meta = self.items_df[[c for c in meta_cols if c in self.items_df.columns]]
        result = score_df.merge(meta, on="item_id", how="left")

        # ── Step 7: Optional domain filter ────────────
        if filter_domain and "domain" in result.columns:
            result = result[result["domain"] == filter_domain]

        # ── Step 8: Sort and return top-N ─────────────
        result = result.sort_values("hybrid_score", ascending=False)
        result = result.head(top_n).reset_index(drop=True)

        # ── Step 9: Add explanation column ────────────
        result["explanation"] = result.apply(
            lambda row: self._build_explanation(row, alpha), axis=1
        )

        return result

    # ─────────────────────────────────────────────
    # Explanation Generator
    # ─────────────────────────────────────────────

    def _build_explanation(self, row: pd.Series, alpha: float) -> str:
        """
        Builds a human-readable explanation for each recommendation.
        This makes the system 'explainable AI' — important for viva!
        """
        cb_pct = round(alpha * 100)
        cf_pct = round((1 - alpha) * 100)

        if row["cb_norm"] >= 0.7 and row["cf_norm"] >= 0.7:
            reason = "Both content match and user behaviour agree."
        elif row["cb_norm"] > row["cf_norm"]:
            reason = "Similar content to your liked items."
        elif row["cf_norm"] > row["cb_norm"]:
            reason = "Users with similar taste also enjoyed this."
        else:
            reason = "Balanced content and collaborative signals."

        return (f"{reason} "
                f"[CB:{row['cb_norm']:.2f}×{cb_pct}% + "
                f"CF:{row['cf_norm']:.2f}×{cf_pct}%]")

    # ─────────────────────────────────────────────
    # Dynamic Alpha Tuning
    # ─────────────────────────────────────────────

    def set_alpha(self, alpha: float) -> None:
        """
        Dynamically change the content vs. collaborative weighting.
        Useful for exploring trade-offs.
        """
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("Alpha must be between 0.0 and 1.0")
        self.alpha = alpha
        print(f"[Hybrid] Alpha updated to {alpha} "
              f"(CB={alpha*100:.0f}%, CF={(1-alpha)*100:.0f}%)")


# ─────────────────────────────────────────────
# Quick test
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

    from src.data_loader         import load_unified_dataset, generate_interaction_matrix
    from src.preprocessor        import preprocess, fill_interaction_matrix
    from src.feature_engineering import TFIDFFeatureBuilder
    from src.content_based       import ContentBasedRecommender
    from src.collaborative       import CollaborativeRecommender

    # Load & preprocess
    items   = load_unified_dataset()
    items   = preprocess(items)

    # Content-based
    builder = TFIDFFeatureBuilder()
    builder.fit_transform(items["content"])
    sim     = builder.compute_similarity()
    cb      = ContentBasedRecommender(items, sim, builder)

    # Collaborative
    raw_mat = generate_interaction_matrix(items)
    filled  = fill_interaction_matrix(raw_mat, strategy="user_mean")
    cf      = CollaborativeRecommender(filled, n_neighbors=10)

    # Hybrid
    hybrid = HybridRecommender(items, cb, cf, alpha=0.5)

    print("\n=== Hybrid Recs for U001 (liked Inception + Python ML) ===")
    recs = hybrid.recommend(
        user_id="U001",
        liked_item_ids=["M002", "C002"],
        top_n=8
    )
    print(recs[["item_id", "title", "hybrid_score", "explanation"]].to_string(index=False))
