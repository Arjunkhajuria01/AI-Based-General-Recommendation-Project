"""
preprocessor.py
===============
Handles all data cleaning and preprocessing steps before feeding into ML models.

Steps:
  1. Lowercase and strip whitespace
  2. Fill missing values
  3. Normalize numeric columns (cost, time)
  4. Create a combined 'content' text field for TF-IDF vectorization
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


# ─────────────────────────────────────────────
# SECTION 1: Text Cleaning
# ─────────────────────────────────────────────

def clean_text(text: str) -> str:
    """
    Cleans a string by:
      - Lowercasing
      - Removing extra whitespace
      - Replacing hyphens with spaces (so 'sci-fi' becomes 'sci fi')
    
    Example:
        >>> clean_text("  Sci-Fi  THRILLER ")
        'sci fi thriller'
    """
    if not isinstance(text, str):
        return ""
    text = text.lower().strip()
    text = text.replace("-", " ")
    text = " ".join(text.split())  # collapse multiple spaces
    return text


# ─────────────────────────────────────────────
# SECTION 2: Main Preprocessing Pipeline
# ─────────────────────────────────────────────

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full preprocessing pipeline for the unified item dataset.
    
    Input columns: item_id | title | category | tags | cost | time | domain
    
    Returns:
        Cleaned DataFrame with an additional 'content' column
        (used for TF-IDF feature extraction)
    """
    df = df.copy()

    # ── Step 1: Fill missing values ──────────────
    df["title"]    = df["title"].fillna("Unknown")
    df["category"] = df["category"].fillna("General")
    df["tags"]     = df["tags"].fillna("")
    df["cost"]     = df["cost"].fillna(0.0)
    df["time"]     = df["time"].fillna(df["time"].median())

    # ── Step 2: Clean text fields ─────────────────
    df["title"]    = df["title"].apply(clean_text)
    df["category"] = df["category"].apply(clean_text)
    df["tags"]     = df["tags"].apply(clean_text)

    # ── Step 3: Build 'content' field for TF-IDF ──
    # We repeat category twice to give it more weight than individual tags
    df["content"] = (
        df["title"] + " " +
        df["category"] + " " +
        df["category"] + " " +   # intentional duplication = boosted weight
        df["tags"]
    )

    # ── Step 4: Normalize numeric columns ─────────
    # MinMaxScaler brings cost and time into [0, 1] range.
    # This is needed if we use numeric features in similarity computations.
    scaler = MinMaxScaler()
    df[["cost_norm", "time_norm"]] = scaler.fit_transform(df[["cost", "time"]])

    print(f"[Preprocessor] Preprocessing done. Shape: {df.shape}")
    return df


# ─────────────────────────────────────────────
# SECTION 3: Fill Missing Ratings in Interaction Matrix
# ─────────────────────────────────────────────

def fill_interaction_matrix(matrix: pd.DataFrame,
                              strategy: str = "user_mean") -> pd.DataFrame:
    """
    Fills NaN values in the user-item interaction matrix.
    
    Why fill? KNN-based collaborative filtering needs a numeric matrix.
    
    Strategies:
        'user_mean'  : Fill each row (user) with that user's average rating
        'item_mean'  : Fill each column (item) with that item's average rating
        'global_mean': Fill everything with the global average
        'zero'       : Fill with 0 (treats missing = no interest)
    
    Parameters:
        matrix   : Raw user-item matrix (rows=users, cols=items, values=ratings or NaN)
        strategy : One of 'user_mean', 'item_mean', 'global_mean', 'zero'
    
    Returns:
        Filled matrix (no NaN values)
    """
    matrix = matrix.copy()

    if strategy == "user_mean":
        # For each row (user), replace NaN with that user's mean rating
        row_means = matrix.mean(axis=1)  # Series: index=user_id, value=mean
        for user in matrix.index:
            mean_val = row_means[user]
            if np.isnan(mean_val):
                mean_val = 3.0  # fallback if user has rated nothing
            matrix.loc[user] = matrix.loc[user].fillna(mean_val)

    elif strategy == "item_mean":
        col_means = matrix.mean(axis=0)
        for item in matrix.columns:
            matrix[item] = matrix[item].fillna(col_means[item])

    elif strategy == "global_mean":
        global_mean = matrix.stack().mean()
        matrix = matrix.fillna(global_mean)

    elif strategy == "zero":
        matrix = matrix.fillna(0)

    else:
        raise ValueError(f"Unknown strategy '{strategy}'. "
                         f"Choose from: user_mean, item_mean, global_mean, zero")

    print(f"[Preprocessor] Interaction matrix filled using strategy='{strategy}'")
    return matrix


# ─────────────────────────────────────────────
# Quick test
# ─────────────────────────────────────────────

if __name__ == "__main__":
    # Simulate a small DataFrame for testing
    sample = pd.DataFrame({
        "item_id":  ["M001", "C001"],
        "title":    ["The Dark Knight", "Python for Beginners"],
        "category": ["Action", "Programming"],
        "tags":     ["batman superhero crime", "python basics loops"],
        "cost":     [0.0, 19.99],
        "time":     [2.5, 12.0],
        "domain":   ["movie", "course"],
    })

    result = preprocess(sample)
    print(result[["item_id", "title", "content", "cost_norm", "time_norm"]])
