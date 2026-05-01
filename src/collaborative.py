"""
collaborative.py
================
Collaborative Filtering using K-Nearest Neighbours (KNN).

How It Works:
  1. Each USER is a vector of item ratings (e.g., [5, NaN, 3, NaN, 4, ...]).
  2. KNN finds the K users most similar in rating behaviour to the target user.
  3. We predict ratings for unseen items by averaging the neighbours' ratings.
  4. Items with the highest predicted ratings are recommended.

Why KNN (not SVD or ALS)?
  - No training loop needed — simple and explainable.
  - Easy to justify in a viva: "we find similar users and average their ratings."
  - Works well on moderate-sized datasets.

Similarity Metric Used: Cosine Similarity (on filled interaction vectors)
"""

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from typing import List, Dict


class CollaborativeRecommender:
    """
    User-based Collaborative Filtering using KNN.

    Parameters:
        interaction_df : Filled user-item rating matrix (no NaN)
                         Rows = users, Columns = item_ids
        n_neighbors    : Number of similar users to consider
    """

    def __init__(self, interaction_df: pd.DataFrame, n_neighbors: int = 10):
        self.interaction_df = interaction_df      # rows=users, cols=items
        self.n_neighbors    = min(n_neighbors, len(interaction_df))
        self.user_ids       = list(interaction_df.index)
        self.item_ids       = list(interaction_df.columns)

        # Map IDs to integer indices
        self.user_to_idx = {uid: i for i, uid in enumerate(self.user_ids)}
        self.item_to_idx = {iid: i for i, iid in enumerate(self.item_ids)}

        # Build numpy matrix for KNN
        self.matrix = interaction_df.values.astype(float)   # shape: (n_users, n_items)

        # ── Fit KNN Model ───────────────────────────
        # metric='cosine' compares user rating patterns (direction, not magnitude)
        # algorithm='brute' is fine for small datasets; use 'ball_tree' for larger ones
        self.knn = NearestNeighbors(
            n_neighbors=self.n_neighbors + 1,  # +1 because user is its own neighbour
            metric="cosine",
            algorithm="brute",
            n_jobs=-1   # use all CPU cores
        )
        self.knn.fit(self.matrix)
        print(f"[Collaborative] KNN fitted: {len(self.user_ids)} users × "
              f"{len(self.item_ids)} items | K={self.n_neighbors}")

    # ─────────────────────────────────────────────
    # Find Similar Users
    # ─────────────────────────────────────────────

    def get_similar_users(self, user_id: str) -> List[tuple]:
        """
        Returns the K most similar users to the given user.

        Returns:
            List of (user_id, similarity_score) sorted by similarity desc.
        """
        if user_id not in self.user_to_idx:
            return []

        user_idx   = self.user_to_idx[user_id]
        user_vec   = self.matrix[user_idx].reshape(1, -1)

        # distances are cosine distances (0=identical, 2=opposite)
        distances, indices = self.knn.kneighbors(user_vec)

        distances = distances.flatten()
        indices   = indices.flatten()

        similar = []
        for dist, idx in zip(distances, indices):
            uid = self.user_ids[idx]
            if uid == user_id:
                continue                    # skip self
            similarity = 1 - dist           # convert cosine distance → similarity
            similar.append((uid, round(similarity, 4)))

        return sorted(similar, key=lambda x: x[1], reverse=True)[:self.n_neighbors]

    # ─────────────────────────────────────────────
    # Predict Ratings for All Items
    # ─────────────────────────────────────────────

    def predict_ratings(self, user_id: str) -> pd.Series:
        """
        Predicts ratings for all items for a given user.
        Uses weighted average of neighbour ratings.

        Formula:
            predicted_rating(u, i) = Σ(sim(u, v) × rating(v, i)) / Σsim(u, v)
            where v ranges over K nearest neighbours

        Returns:
            pd.Series indexed by item_id with predicted ratings
        """
        similar_users = self.get_similar_users(user_id)

        if not similar_users:
            # Fallback: return global average per item
            return self.interaction_df.mean(axis=0)

        # Weighted average of neighbour ratings
        weighted_sum  = np.zeros(len(self.item_ids))
        similarity_sum = 0.0

        for (neighbour_id, sim_score) in similar_users:
            neighbour_idx = self.user_to_idx[neighbour_id]
            neighbour_ratings = self.matrix[neighbour_idx]
            weighted_sum   += sim_score * neighbour_ratings
            similarity_sum += sim_score

        if similarity_sum == 0:
            predicted = np.zeros(len(self.item_ids))
        else:
            predicted = weighted_sum / similarity_sum

        return pd.Series(predicted, index=self.item_ids)

    # ─────────────────────────────────────────────
    # Recommend Top-N Items for a User
    # ─────────────────────────────────────────────

    def recommend(self,
                  user_id: str,
                  top_n: int = 10,
                  items_df: pd.DataFrame = None,
                  exclude_rated: bool = True,
                  filter_domain: str = None) -> pd.DataFrame:
        """
        Recommends top-N items for a given user using collaborative filtering.

        Parameters:
            user_id       : Target user ID (must exist in interaction matrix)
            top_n         : Number of recommendations to return
            items_df      : Optional item metadata for joining (title, category, etc.)
            exclude_rated : If True, skip items the user already rated
            filter_domain : Optional domain filter ('movie' or 'course')

        Returns:
            DataFrame with columns: item_id, cf_score (and metadata if items_df given)
        """
        if user_id not in self.user_to_idx:
            print(f"[Collaborative] User '{user_id}' not found. "
                  f"Returning popular items as fallback.")
            return self._popular_items_fallback(top_n, items_df, filter_domain)

        predicted_ratings = self.predict_ratings(user_id)

        # Get items the user already rated (to optionally exclude)
        user_idx = self.user_to_idx[user_id]
        user_row = self.interaction_df.iloc[user_idx]
        rated_items = set(user_row[user_row.notna()].index.tolist())

        result = predicted_ratings.reset_index()
        result.columns = ["item_id", "cf_score"]

        if exclude_rated:
            result = result[~result["item_id"].isin(rated_items)]

        # Join metadata if provided
        if items_df is not None:
            meta_cols = ["item_id", "title", "category", "domain", "cost", "time"]
            meta = items_df[[c for c in meta_cols if c in items_df.columns]]
            result = result.merge(meta, on="item_id", how="left")

            if filter_domain and "domain" in result.columns:
                result = result[result["domain"] == filter_domain]

        result = result.sort_values("cf_score", ascending=False)
        return result.head(top_n).reset_index(drop=True)

    # ─────────────────────────────────────────────
    # Fallback: Popular Items (for unknown users)
    # ─────────────────────────────────────────────

    def _popular_items_fallback(self,
                                 top_n: int,
                                 items_df: pd.DataFrame = None,
                                 filter_domain: str = None) -> pd.DataFrame:
        """
        Returns the most-rated (popular) items as a fallback for unknown users.
        This is the simplest cold-start strategy for collaborative filtering.
        """
        # Average rating per item (higher avg = better)
        avg_ratings = self.interaction_df.mean(axis=0)
        result = avg_ratings.reset_index()
        result.columns = ["item_id", "cf_score"]

        if items_df is not None:
            meta_cols = ["item_id", "title", "category", "domain", "cost", "time"]
            meta = items_df[[c for c in meta_cols if c in items_df.columns]]
            result = result.merge(meta, on="item_id", how="left")

            if filter_domain and "domain" in result.columns:
                result = result[result["domain"] == filter_domain]

        return result.sort_values("cf_score", ascending=False).head(top_n).reset_index(drop=True)

    # ─────────────────────────────────────────────
    # Score Lookup (used by Hybrid Engine)
    # ─────────────────────────────────────────────

    def get_scores_for_items(self, user_id: str,
                              candidate_ids: List[str]) -> Dict[str, float]:
        """
        Returns {item_id: cf_score} for given candidates.
        Called by the Hybrid Engine.
        """
        if user_id not in self.user_to_idx:
            return {iid: 0.0 for iid in candidate_ids}

        predicted = self.predict_ratings(user_id)
        return {iid: float(predicted.get(iid, 0.0)) for iid in candidate_ids}


# ─────────────────────────────────────────────
# Quick test
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

    from src.data_loader  import load_unified_dataset, generate_interaction_matrix
    from src.preprocessor import preprocess, fill_interaction_matrix

    items   = load_unified_dataset()
    items   = preprocess(items)
    raw_mat = generate_interaction_matrix(items)
    filled  = fill_interaction_matrix(raw_mat, strategy="user_mean")

    cf = CollaborativeRecommender(filled, n_neighbors=10)

    print("\n=== CF Recs for user U001 ===")
    recs = cf.recommend("U001", top_n=5, items_df=items)
    print(recs[["item_id", "title", "cf_score"]].to_string(index=False))

    print("\n=== Similar users to U001 ===")
    similar = cf.get_similar_users("U001")
    for uid, sim in similar[:5]:
        print(f"  {uid}: similarity={sim}")
