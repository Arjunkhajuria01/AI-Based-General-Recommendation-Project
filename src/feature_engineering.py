"""
feature_engineering.py
=======================
Builds TF-IDF feature vectors from item content text.

TF-IDF (Term Frequency – Inverse Document Frequency):
  - TF: How often a word appears in THIS document (item)
  - IDF: How rare that word is across ALL documents (penalizes common words)
  - Result: Each item is represented as a numeric vector of word importances.

These vectors are then used to compute cosine similarity between items.
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ─────────────────────────────────────────────
# SECTION 1: TF-IDF Vectorizer
# ─────────────────────────────────────────────

class TFIDFFeatureBuilder:
    """
    Converts item 'content' text into TF-IDF feature vectors.
    
    Usage:
        builder = TFIDFFeatureBuilder()
        tfidf_matrix = builder.fit_transform(df["content"])
        sim_matrix   = builder.compute_similarity()
    """

    def __init__(self, max_features: int = 500, ngram_range: tuple = (1, 2)):
        """
        Parameters:
            max_features : Maximum number of unique words (vocabulary size)
            ngram_range  : (1,1) = single words; (1,2) = unigrams + bigrams
                           Bigrams help catch phrases like "machine learning"
        """
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words="english",   # ignore common English words (the, is, a, ...)
            sublinear_tf=True       # apply log(TF) smoothing to avoid outlier bias
        )
        self.tfidf_matrix = None   # shape: (n_items, max_features)
        self.feature_names = None  # vocabulary words

    def fit_transform(self, content_series: pd.Series) -> np.ndarray:
        """
        Fits the TF-IDF model and transforms the content text into vectors.
        
        Parameters:
            content_series : pd.Series of strings (one per item)
        
        Returns:
            Sparse matrix of shape (n_items, n_features)
        """
        self.tfidf_matrix = self.vectorizer.fit_transform(content_series)
        self.feature_names = self.vectorizer.get_feature_names_out()

        print(f"[FeatureEng] TF-IDF matrix shape: {self.tfidf_matrix.shape}")
        print(f"[FeatureEng] Vocabulary size: {len(self.feature_names)}")
        return self.tfidf_matrix

    def compute_similarity(self) -> np.ndarray:
        """
        Computes pairwise cosine similarity between all item TF-IDF vectors.
        
        Cosine Similarity: 
            sim(A, B) = (A · B) / (||A|| × ||B||)
            Range: [0, 1] where 1 = identical content, 0 = no overlap
        
        Returns:
            2D numpy array of shape (n_items, n_items)
        """
        if self.tfidf_matrix is None:
            raise RuntimeError("Call fit_transform() before compute_similarity()")

        sim_matrix = cosine_similarity(self.tfidf_matrix)
        print(f"[FeatureEng] Cosine similarity matrix computed: {sim_matrix.shape}")
        return sim_matrix

    def get_top_keywords(self, item_index: int, top_n: int = 5) -> list:
        """
        Returns the top TF-IDF keywords for a given item.
        Useful for explanation ("Why this was recommended").
        
        Parameters:
            item_index : Row index of the item in the dataset
            top_n      : Number of top keywords to return
        
        Returns:
            List of (keyword, score) tuples
        """
        if self.tfidf_matrix is None:
            raise RuntimeError("Call fit_transform() first")

        row = self.tfidf_matrix[item_index].toarray().flatten()
        top_indices = row.argsort()[::-1][:top_n]
        return [(self.feature_names[i], round(row[i], 4)) for i in top_indices if row[i] > 0]


# ─────────────────────────────────────────────
# Quick test
# ─────────────────────────────────────────────

if __name__ == "__main__":
    # Tiny test corpus
    content = pd.Series([
        "action batman superhero crime thriller dark",
        "scifi dream timeloop thriller mindwarp",
        "python machine learning sklearn supervised",
        "web html css javascript fullstack frontend",
    ])

    builder = TFIDFFeatureBuilder(max_features=50)
    mat = builder.fit_transform(content)
    sim = builder.compute_similarity()

    print("\nSimilarity Matrix:")
    print(np.round(sim, 3))
    print("\nTop keywords for item 0:", builder.get_top_keywords(0))
