"""
content_based.py
================
Content-Based Recommender using TF-IDF + Cosine Similarity.

How It Works:
  1. Each item is represented as a TF-IDF vector (based on its title, category, tags).
  2. Cosine similarity measures how "similar" two item vectors are.
  3. Given a user's liked items, we find items with the highest similarity scores.

Strengths:
  - Works for NEW users (cold start) — only needs item metadata, not user history.
  - Recommendations are explainable (based on matching keywords).

Limitations:
  - Cannot discover serendipitous recommendations (limited to similar content).
"""

import numpy as np
import pandas as pd
from typing import List, Dict


class ContentBasedRecommender:
    """
    Recommends items similar in content to items the user already likes.

    Parameters:
        items_df      : Preprocessed items DataFrame
        sim_matrix    : Precomputed cosine similarity matrix (n_items × n_items)
        feature_builder: TFIDFFeatureBuilder instance (for keyword explanations)
    """

    def __init__(self, items_df: pd.DataFrame,
                 sim_matrix: np.ndarray,
                 feature_builder=None):
        self.items_df       = items_df.reset_index(drop=True)
        self.sim_matrix     = sim_matrix
        self.feature_builder = feature_builder

        # Map item_id → row index for fast lookup
        self.id_to_idx = {item_id: idx
                          for idx, item_id in enumerate(self.items_df["item_id"])}
        self.idx_to_id = {v: k for k, v in self.id_to_idx.items()}

    # ─────────────────────────────────────────────
    # Core Recommendation Logic
    # ─────────────────────────────────────────────

    def recommend(self,
                  liked_item_ids: List[str],
                  top_n: int = 10,
                  filter_domain: str = None,
                  exclude_seen: bool = True) -> pd.DataFrame:
        """
        Recommend items based on a list of items the user already liked.

        Algorithm:
            1. Get the similarity row for each liked item.
            2. Average the similarity scores across all liked items.
               (Aggregating avoids over-fitting to one item.)
            3. Sort by average score and return top-N.

        Parameters:
            liked_item_ids : List of item_id strings the user has liked
            top_n          : Number of recommendations to return
            filter_domain  : 'movie', 'course', or None (return all)
            exclude_seen   : If True, won't recommend already-liked items

        Returns:
            DataFrame with columns: item_id, title, category, domain, cb_score
        """
        if not liked_item_ids:
            return pd.DataFrame()

        # Validate that provided IDs exist
        valid_ids = [iid for iid in liked_item_ids if iid in self.id_to_idx]
        if not valid_ids:
            print("[ContentBased] None of the liked_item_ids found in dataset.")
            return pd.DataFrame()

        n_items = len(self.items_df)

        # Step 1: Accumulate similarity scores from each liked item
        agg_scores = np.zeros(n_items)
        for iid in valid_ids:
            idx = self.id_to_idx[iid]
            agg_scores += self.sim_matrix[idx]  # add row from sim matrix

        # Step 2: Average across all liked items
        agg_scores /= len(valid_ids)

        # Step 3: Build result DataFrame
        result_df = self.items_df[["item_id", "title", "category", "domain",
                                    "cost", "time"]].copy()
        result_df["cb_score"] = agg_scores

        # Step 4: Optionally exclude items the user already liked
        if exclude_seen:
            result_df = result_df[~result_df["item_id"].isin(valid_ids)]

        # Step 5: Optionally filter by domain
        if filter_domain:
            result_df = result_df[result_df["domain"] == filter_domain]

        # Step 6: Sort by similarity score descending
        result_df = result_df.sort_values("cb_score", ascending=False)

        return result_df.head(top_n).reset_index(drop=True)

    # ─────────────────────────────────────────────
    # Cold Start: Recommend by Preference Tags
    # ─────────────────────────────────────────────

    def recommend_by_tags(self,
                          preference_text: str,
                          top_n: int = 10,
                          filter_domain: str = None) -> pd.DataFrame:
        """
        Cold-start fallback: User describes their preferences as free text.
        We compute similarity of that text to all items using existing TF-IDF model.

        Example:
            preference_text = "I like science fiction and space exploration"

        Parameters:
            preference_text : Free text describing user preferences
            top_n           : Number of items to return
            filter_domain   : Optional domain filter

        Returns:
            DataFrame with content-based recommendations
        """
        if self.feature_builder is None:
            raise RuntimeError("feature_builder required for tag-based recommendation")

        from sklearn.metrics.pairwise import cosine_similarity

        # Transform user preference text using the FITTED TF-IDF vectorizer
        # (we use transform, NOT fit_transform, to use the same vocabulary)
        user_vec = self.feature_builder.vectorizer.transform([preference_text])

        # Compute similarity between user preference vector and all items
        scores = cosine_similarity(user_vec, self.feature_builder.tfidf_matrix).flatten()

        result_df = self.items_df[["item_id", "title", "category", "domain",
                                    "cost", "time"]].copy()
        result_df["cb_score"] = scores

        if filter_domain:
            result_df = result_df[result_df["domain"] == filter_domain]

        result_df = result_df.sort_values("cb_score", ascending=False)
        return result_df.head(top_n).reset_index(drop=True)

    # ─────────────────────────────────────────────
    # Explanation Generator
    # ─────────────────────────────────────────────

    def explain(self, item_id: str) -> Dict:
        """
        Explains why an item was recommended by showing its top TF-IDF keywords.
        This is very useful for academic vivas!

        Parameters:
            item_id : The item to explain

        Returns:
            dict with item info and top keywords
        """
        if item_id not in self.id_to_idx:
            return {"error": f"Item '{item_id}' not found"}

        idx = self.id_to_idx[item_id]
        row = self.items_df.iloc[idx]

        explanation = {
            "item_id":   row["item_id"],
            "title":     row["title"],
            "category":  row["category"],
            "domain":    row["domain"],
            "keywords":  []
        }

        if self.feature_builder:
            explanation["keywords"] = self.feature_builder.get_top_keywords(idx, top_n=6)

        return explanation

    # ─────────────────────────────────────────────
    # Score for a Specific Item (used by Hybrid)
    # ─────────────────────────────────────────────

    def get_scores_for_items(self,
                              liked_item_ids: List[str],
                              candidate_ids: List[str]) -> Dict[str, float]:
        """
        Returns a dict of {item_id: cb_score} for a list of candidate items.
        Called by the Hybrid Engine to get content-based scores.
        """
        if not liked_item_ids:
            return {iid: 0.0 for iid in candidate_ids}

        valid_liked = [iid for iid in liked_item_ids if iid in self.id_to_idx]
        n_items = len(self.items_df)
        agg_scores = np.zeros(n_items)

        for iid in valid_liked:
            idx = self.id_to_idx[iid]
            agg_scores += self.sim_matrix[idx]

        if valid_liked:
            agg_scores /= len(valid_liked)

        return {iid: float(agg_scores[self.id_to_idx[iid]])
                for iid in candidate_ids if iid in self.id_to_idx}


# ─────────────────────────────────────────────
# Quick test
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

    from src.data_loader       import load_unified_dataset
    from src.preprocessor      import preprocess
    from src.feature_engineering import TFIDFFeatureBuilder

    items   = load_unified_dataset()
    items   = preprocess(items)
    builder = TFIDFFeatureBuilder()
    builder.fit_transform(items["content"])
    sim     = builder.compute_similarity()

    cb = ContentBasedRecommender(items, sim, builder)

    print("\n=== Content-Based Recs (liked: 'Inception', 'The Matrix') ===")
    recs = cb.recommend(["M002", "M011"], top_n=5)
    print(recs[["item_id", "title", "category", "cb_score"]].to_string(index=False))

    print("\n=== Explain item M002 ===")
    print(cb.explain("M002"))

    print("\n=== Cold Start (tag-based) ===")
    cold = cb.recommend_by_tags("I love space and science fiction movies", top_n=5)
    print(cold[["item_id", "title", "category", "cb_score"]].to_string(index=False))
