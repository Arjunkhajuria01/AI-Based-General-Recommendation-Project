"""
data_loader.py
==============
Responsible for loading and merging datasets from multiple domains (movies + courses).
All data is unified into a standard schema: item_id | title | category | tags | cost | time

This module also generates a synthetic user-item interaction matrix for collaborative filtering.
"""

import pandas as pd
import numpy as np
import os

# ─────────────────────────────────────────────
# SECTION 1: Generate Synthetic Movie Dataset
# ─────────────────────────────────────────────

def generate_movie_dataset() -> pd.DataFrame:
    """
    Generates a synthetic movie dataset with realistic genres and tags.
    In a real project, replace this with: pd.read_csv('data/raw/movies.csv')
    """
    movies_raw = [
        ("The Dark Knight",        "Action",    "batman superhero crime thriller dark"),
        ("Inception",              "SciFi",     "dream timeloop mindwarp thriller scifi"),
        ("Interstellar",           "SciFi",     "space time blackhole epic science"),
        ("The Shawshank Redemption","Drama",    "prison freedom hope inspirational drama"),
        ("Forrest Gump",           "Drama",     "life journey history love inspirational"),
        ("The Avengers",           "Action",    "superhero marvel team action adventure"),
        ("Titanic",                "Romance",   "love ship tragedy history romance"),
        ("The Godfather",          "Crime",     "mafia family crime power classic"),
        ("Pulp Fiction",           "Crime",     "violence dialogue nonlinear crime cult"),
        ("Schindler's List",       "History",   "war holocaust survival history drama"),
        ("The Matrix",             "SciFi",     "simulation reality hacker cyberpunk action"),
        ("Goodfellas",             "Crime",     "gangster mafia true-story crime drama"),
        ("Fight Club",             "Thriller",  "identity violence twist dark psychological"),
        ("The Silence of the Lambs","Thriller", "serial-killer psychology FBI horror thriller"),
        ("Spirited Away",          "Animation", "fantasy anime journey magic adventure"),
        ("Parasite",               "Drama",     "class society dark twist korean drama"),
        ("Joker",                  "Drama",     "villain mental-health dark society origin"),
        ("Mad Max: Fury Road",     "Action",    "postapocalyptic action chase survival"),
        ("Whiplash",               "Drama",     "music ambition perfectionism jazz intense"),
        ("La La Land",             "Romance",   "music dreams love jazz romance musical"),
        ("Get Out",                "Horror",    "race psychology horror social-commentary thriller"),
        ("A Beautiful Mind",       "Drama",     "math genius mental-illness biography drama"),
        ("The Social Network",     "Drama",     "startup tech ambition betrayal biography"),
        ("Her",                    "Romance",   "AI future love loneliness technology"),
        ("Arrival",                "SciFi",     "aliens language time communication scifi"),
        ("1917",                   "History",   "war worldwar mission history drama"),
        ("Dunkirk",                "History",   "war evacuation survival history action"),
        ("Blade Runner 2049",      "SciFi",     "AI replicant future noir scifi"),
        ("The Prestige",           "Thriller",  "magic rivalry obsession twist mystery"),
        ("Memento",                "Thriller",  "memory nonlinear mystery dark psychological"),
    ]

    records = []
    for idx, (title, category, tags) in enumerate(movies_raw):
        records.append({
            "item_id":   f"M{idx+1:03d}",     # unique ID with M prefix
            "title":     title,
            "category":  category,
            "tags":      tags,
            "cost":      0.0,                  # movies are free/included
            "time":      round(np.random.uniform(1.5, 3.0), 1),  # hours
            "domain":    "movie"
        })

    return pd.DataFrame(records)


# ─────────────────────────────────────────────
# SECTION 2: Generate Synthetic Course Dataset
# ─────────────────────────────────────────────

def generate_course_dataset() -> pd.DataFrame:
    """
    Generates a synthetic online course dataset (Udemy-style).
    In a real project, replace this with: pd.read_csv('data/raw/courses.csv')
    """
    courses_raw = [
        ("Python for Beginners",           "Programming",  "python basics syntax loops beginner"),
        ("Machine Learning A-Z",           "AI",           "ml supervised unsupervised sklearn python"),
        ("Deep Learning Specialization",   "AI",           "neural networks deep learning tensorflow"),
        ("Web Development Bootcamp",       "Web",          "html css javascript fullstack bootcamp"),
        ("Data Science with Python",       "DataScience",  "pandas numpy visualization statistics python"),
        ("React JS Complete Guide",        "Web",          "react frontend javascript components hooks"),
        ("SQL Mastery",                    "Database",     "sql database queries joins relational"),
        ("Docker & Kubernetes",            "DevOps",       "containers kubernetes deployment devops cloud"),
        ("NLP with Python",                "AI",           "nlp text processing transformers python"),
        ("Computer Vision with OpenCV",    "AI",           "opencv image detection recognition python"),
        ("Django REST API",                "Web",          "django api backend python rest"),
        ("AWS Cloud Practitioner",         "Cloud",        "aws cloud services infrastructure devops"),
        ("Linear Algebra for ML",          "Math",         "linear algebra matrix vectors mathematics"),
        ("Statistics for Data Science",    "Math",         "statistics probability distributions inference"),
        ("Git & GitHub Masterclass",       "DevOps",       "git version-control github collaboration"),
        ("JavaScript Algorithms",          "Programming",  "javascript algorithms data-structures interview"),
        ("Flask Microservices",            "Web",          "flask microservices api python backend"),
        ("Tableau for Beginners",          "DataScience",  "tableau visualization business dashboard"),
        ("Ethical Hacking",                "Security",     "cybersecurity hacking penetration kali linux"),
        ("Blockchain Fundamentals",        "Blockchain",   "blockchain crypto ethereum smart-contracts"),
        ("iOS Development with Swift",     "Mobile",       "swift ios apple xcode mobile apps"),
        ("Android Development",            "Mobile",       "android kotlin java mobile apps"),
        ("Excel for Data Analysis",        "DataScience",  "excel pivot charts formulas data"),
        ("TensorFlow Lite for Mobile",     "AI",           "tensorflow mobile edge AI deployment"),
        ("Pandas Complete Tutorial",       "DataScience",  "pandas dataframe data wrangling python"),
        ("Agile & Scrum Masterclass",      "Management",   "agile scrum project-management sprints"),
        ("Cyber Security Fundamentals",    "Security",     "security firewall protocols network threats"),
        ("Power BI Dashboard",             "DataScience",  "powerbi visualization reports dashboard"),
        ("C++ for Game Development",       "Programming",  "cpp game-dev opengl rendering engine"),
        ("Unity 3D Game Design",           "Programming",  "unity 3d game design csharp"),
    ]

    records = []
    for idx, (title, category, tags) in enumerate(courses_raw):
        records.append({
            "item_id":  f"C{idx+1:03d}",
            "title":    title,
            "category": category,
            "tags":     tags,
            "cost":     round(np.random.choice([0, 9.99, 19.99, 49.99, 99.99]), 2),
            "time":     round(np.random.uniform(5, 40), 1),  # hours to complete
            "domain":   "course"
        })

    return pd.DataFrame(records)


# ─────────────────────────────────────────────
# SECTION 3: Merge into Unified Schema
# ─────────────────────────────────────────────

def load_unified_dataset() -> pd.DataFrame:
    """
    Merges movie and course datasets into one unified DataFrame.
    Schema: item_id | title | category | tags | cost | time | domain
    """
    movies  = generate_movie_dataset()
    courses = generate_course_dataset()

    # Stack both datasets
    unified = pd.concat([movies, courses], ignore_index=True)
    unified.reset_index(drop=True, inplace=True)

    print(f"[DataLoader] Unified dataset loaded: {len(unified)} items "
          f"({len(movies)} movies + {len(courses)} courses)")
    return unified


# ─────────────────────────────────────────────
# SECTION 4: Generate User-Item Interaction Matrix
# ─────────────────────────────────────────────

def generate_interaction_matrix(items_df: pd.DataFrame,
                                 n_users: int = 50,
                                 sparsity: float = 0.85,
                                 seed: int = 42) -> pd.DataFrame:
    """
    Creates a synthetic user-item rating matrix for collaborative filtering.

    Parameters:
        items_df  : Unified items DataFrame (must have 'item_id' column)
        n_users   : Number of synthetic users to simulate
        sparsity  : Fraction of missing ratings (0.85 = 85% empty cells)
        seed      : Random seed for reproducibility

    Returns:
        DataFrame where rows = users, columns = item_ids, values = ratings (1-5 or NaN)
    """
    np.random.seed(seed)

    item_ids = items_df["item_id"].tolist()
    user_ids = [f"U{i+1:03d}" for i in range(n_users)]

    # Build random rating matrix
    # Non-NaN ratings share (1 - sparsity) of probability equally across 1-5 stars
    # Weights below are proportional; we normalise them to sum exactly to (1-sparsity)
    raw_weights   = [0.06, 0.10, 0.14, 0.30, 0.40]          # relative rating weights
    rating_share  = 1.0 - sparsity                            # fraction of cells that ARE rated
    rating_probs  = [w / sum(raw_weights) * rating_share for w in raw_weights]
    all_probs     = rating_probs + [sparsity]                 # append NaN probability
    all_probs[-1] = round(1.0 - sum(all_probs[:-1]), 10)      # force exact sum = 1

    ratings = np.random.choice(
        [1, 2, 3, 4, 5, np.nan],
        size=(n_users, len(item_ids)),
        p=all_probs
    )

    interaction_df = pd.DataFrame(ratings, index=user_ids, columns=item_ids)

    rated_count = interaction_df.notna().sum().sum()
    total_cells = n_users * len(item_ids)
    density = rated_count / total_cells * 100
    print(f"[DataLoader] Interaction matrix: {n_users} users × {len(item_ids)} items | "
          f"Density: {density:.1f}%")

    return interaction_df


# ─────────────────────────────────────────────
# SECTION 5: Save processed data to disk
# ─────────────────────────────────────────────

def save_datasets(unified_df: pd.DataFrame,
                   interaction_df: pd.DataFrame,
                   output_dir: str = "data/processed") -> None:
    """Saves processed datasets to CSV for reuse."""
    os.makedirs(output_dir, exist_ok=True)
    unified_df.to_csv(f"{output_dir}/unified_items.csv", index=False)
    interaction_df.to_csv(f"{output_dir}/interaction_matrix.csv")
    print(f"[DataLoader] Saved datasets to '{output_dir}/'")


# ─────────────────────────────────────────────
# Quick test (run this file directly)
# ─────────────────────────────────────────────

if __name__ == "__main__":
    items = load_unified_dataset()
    matrix = generate_interaction_matrix(items)
    save_datasets(items, matrix)
    print("\nSample items:")
    print(items.head(5).to_string(index=False))
    print("\nInteraction matrix (first 5 users, first 8 items):")
    print(matrix.iloc[:5, :8])
