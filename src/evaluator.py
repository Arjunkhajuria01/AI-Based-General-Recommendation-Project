"""
evaluator.py
============
Evaluates recommendation system performance using standard IR metrics.

Metrics Implemented:
  1. Precision@K  — Of the top-K recommended items, what fraction are relevant?
  2. Recall@K     — Of all relevant items, what fraction appear in top-K?

Evaluation Protocol:
  - Train-test split on the interaction matrix (hold out 20% ratings as test set)
  - For each test user, hide their test ratings and ask the model to predict
  - Compare predictions against actual ratings to compute metrics
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple


# ─────────────────────────────────────────────
# SECTION 1: Train-Test Split
# ─────────────────────────────────────────────

def train_test_split_matrix(interaction_df: pd.DataFrame,
                              test_ratio: float = 0.2,
                              seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits the user-item interaction matrix into train and test sets.
    
    Strategy: For each user, randomly hold out `test_ratio` of their rated items.
    The held-out ratings form the test set; the rest form the training set.
    
    Parameters:
        interaction_df : Full interaction matrix (may contain NaN)
        test_ratio     : Fraction of ratings to hold out for testing
        seed           : Random seed for reproducibility
    
    Returns:
        train_df : Matrix with test ratings replaced by NaN
        test_df  : Matrix with only test ratings (rest are NaN)
    """
    np.random.seed(seed)
    train_df = interaction_df.copy()
    test_df  = pd.DataFrame(np.nan,
                             index=interaction_df.index,
                             columns=interaction_df.columns)

    for user_id in interaction_df.index:
        # Find items the user has actually rated
        user_row   = interaction_df.loc[user_id]
        rated_mask = user_row.notna()
        rated_items = user_row[rated_mask].index.tolist()

        if len(rated_items) < 2:
            continue   # skip users with very few ratings

        # Randomly select test items
        n_test = max(1, int(len(rated_items) * test_ratio))
        test_items = np.random.choice(rated_items, size=n_test, replace=False)

        # Move them from train to test
        for item_id in test_items:
            test_df.loc[user_id, item_id]  = interaction_df.loc[user_id, item_id]
            train_df.loc[user_id, item_id] = np.nan   # hide from training

    train_rated = train_df.notna().sum().sum()
    test_rated  = test_df.notna().sum().sum()
    print(f"[Evaluator] Split complete: {train_rated} train ratings | "
          f"{test_rated} test ratings")

    return train_df, test_df


# ─────────────────────────────────────────────
# SECTION 2: Determine Relevant Items
# ─────────────────────────────────────────────

def get_relevant_items(test_df: pd.DataFrame,
                        user_id: str,
                        relevance_threshold: float = 3.5) -> List[str]:
    """
    Returns the list of items that the user liked in the test set.
    An item is considered "relevant" if its rating >= relevance_threshold.
    
    Parameters:
        test_df              : Test portion of the interaction matrix
        user_id              : Target user
        relevance_threshold  : Min rating to count as "liked"
    
    Returns:
        List of relevant item_ids
    """
    if user_id not in test_df.index:
        return []

    user_row = test_df.loc[user_id]
    return user_row[user_row >= relevance_threshold].index.tolist()


# ─────────────────────────────────────────────
# SECTION 3: Precision@K and Recall@K
# ─────────────────────────────────────────────

def precision_at_k(recommended: List[str],
                    relevant: List[str],
                    k: int) -> float:
    """
    Precision@K: fraction of top-K recommended items that are relevant.
    
    Formula:
        Precision@K = |recommended[:K] ∩ relevant| / K
    
    Example:
        recommended = [A, B, C, D, E]
        relevant    = [B, D, F]
        k           = 5
        → intersection = {B, D} → size = 2
        → Precision@5 = 2/5 = 0.40
    
    Parameters:
        recommended : List of recommended item_ids (ordered, top first)
        relevant    : List of ground-truth relevant item_ids
        k           : Cutoff rank
    
    Returns:
        Precision@K score in [0, 1]
    """
    if k == 0 or not recommended:
        return 0.0

    top_k = recommended[:k]
    hits  = len(set(top_k) & set(relevant))
    return hits / k


def recall_at_k(recommended: List[str],
                 relevant: List[str],
                 k: int) -> float:
    """
    Recall@K: fraction of all relevant items that appear in top-K.
    
    Formula:
        Recall@K = |recommended[:K] ∩ relevant| / |relevant|
    
    Example:
        recommended = [A, B, C, D, E]
        relevant    = [B, D, F]
        k           = 5
        → hits = {B, D} → 2
        → Recall@5 = 2/3 = 0.667
    
    Parameters:
        recommended : List of recommended item_ids (ordered, top first)
        relevant    : List of ground-truth relevant item_ids
        k           : Cutoff rank
    
    Returns:
        Recall@K score in [0, 1]
    """
    if not relevant:
        return 0.0

    top_k = recommended[:k]
    hits  = len(set(top_k) & set(relevant))
    return hits / len(relevant)


# ─────────────────────────────────────────────
# SECTION 4: Full Evaluation Loop
# ─────────────────────────────────────────────

def evaluate_recommender(recommender_fn,
                          test_df: pd.DataFrame,
                          k: int = 10,
                          relevance_threshold: float = 3.5,
                          min_test_ratings: int = 1) -> Dict:
    """
    Runs evaluation over all test users and averages Precision@K and Recall@K.
    
    Parameters:
        recommender_fn      : Callable(user_id) → List[item_id] (ordered recs)
        test_df             : Test portion of the interaction matrix
        k                   : Cutoff for evaluation
        relevance_threshold : Minimum rating to count as "liked"
        min_test_ratings    : Skip users with fewer than this many test ratings
    
    Returns:
        Dict with: precision_at_k, recall_at_k, n_users_evaluated
    """
    precisions = []
    recalls    = []
    n_skipped  = 0

    for user_id in test_df.index:
        relevant = get_relevant_items(test_df, user_id, relevance_threshold)

        if len(relevant) < min_test_ratings:
            n_skipped += 1
            continue

        # Get recommendations from the model
        try:
            recommended = recommender_fn(user_id)
        except Exception as e:
            print(f"[Evaluator] Error for {user_id}: {e}")
            n_skipped += 1
            continue

        precisions.append(precision_at_k(recommended, relevant, k))
        recalls.append(recall_at_k(recommended, relevant, k))

    n_evaluated = len(precisions)
    results = {
        "precision_at_k":    round(np.mean(precisions), 4) if precisions else 0.0,
        "recall_at_k":       round(np.mean(recalls), 4)    if recalls    else 0.0,
        "k":                 k,
        "n_users_evaluated": n_evaluated,
        "n_users_skipped":   n_skipped,
    }

    print(f"\n[Evaluator] Results @ K={k}")
    print(f"  Precision@{k}: {results['precision_at_k']:.4f}")
    print(f"  Recall@{k}:    {results['recall_at_k']:.4f}")
    print(f"  Users evaluated: {n_evaluated} | Skipped: {n_skipped}")

    return results


# ─────────────────────────────────────────────
# SECTION 5: Summary Report
# ─────────────────────────────────────────────

def print_evaluation_report(results: Dict, model_name: str = "Model") -> None:
    """Prints a nicely formatted evaluation summary."""
    k = results["k"]
    print(f"\n{'='*50}")
    print(f"  Evaluation Report — {model_name}")
    print(f"{'='*50}")
    print(f"  Top-K (K):          {k}")
    print(f"  Precision@{k}:      {results['precision_at_k']:.4f}  "
          f"({results['precision_at_k']*100:.1f}%)")
    print(f"  Recall@{k}:         {results['recall_at_k']:.4f}  "
          f"({results['recall_at_k']*100:.1f}%)")
    print(f"  Users Evaluated:    {results['n_users_evaluated']}")
    print(f"{'='*50}\n")


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

    # Build system
    items   = load_unified_dataset()
    items   = preprocess(items)
    raw_mat = generate_interaction_matrix(items)

    # Split BEFORE filling (use train set for training)
    train_raw, test_raw = train_test_split_matrix(raw_mat, test_ratio=0.2)
    train_filled = fill_interaction_matrix(train_raw, strategy="user_mean")

    # Build CB
    builder = TFIDFFeatureBuilder()
    builder.fit_transform(items["content"])
    sim = builder.compute_similarity()
    cb  = ContentBasedRecommender(items, sim, builder)

    # Build CF on TRAIN only
    cf = CollaborativeRecommender(train_filled, n_neighbors=10)

    # Evaluate CF
    def cf_rec_fn(user_id):
        recs = cf.recommend(user_id, top_n=10, items_df=items)
        return recs["item_id"].tolist()

    results = evaluate_recommender(cf_rec_fn, test_raw, k=10)
    print_evaluation_report(results, "Collaborative (KNN)")
