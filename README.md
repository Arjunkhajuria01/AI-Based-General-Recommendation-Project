# 🎯 Hybrid Recommendation System

> A **production-quality, domain-independent recommendation engine** combining **Content-Based Filtering** (TF-IDF + Cosine Similarity) and **Collaborative Filtering** (K-Nearest Neighbours) via weighted hybrid fusion.

Built for a university AI project — modular, explainable, and viva-ready.

---

## 📁 Project Structure

```
AI RECOMMENDOR PROJECT/
│
├── src/                          # Core AI modules
│   ├── __init__.py               # Package declaration
│   ├── data_loader.py            # Dataset generation & interaction matrix
│   ├── preprocessor.py           # Cleaning, normalisation, content field
│   ├── feature_engineering.py    # TF-IDF vectorisation + cosine similarity
│   ├── content_based.py          # Content-Based Recommender
│   ├── collaborative.py          # Collaborative Filtering (KNN)
│   ├── hybrid.py                 # Hybrid Ranking Engine (weighted fusion)
│   └── evaluator.py              # Precision@K, Recall@K evaluation
│
├── data/
│   ├── raw/                      # (placeholder for real CSV datasets)
│   └── processed/                # Auto-generated: unified_items.csv, interaction_matrix.csv
│
├── outputs/                      # Saved plots, reports
│
├── main.py                       # ✅ Interactive terminal CLI (primary entry point)
├── recommender.py                # 🔧 Programmatic demo + evaluation runner
├── app.py                        # 🌐 Streamlit web application
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip3 install -r requirements.txt
```

### 2. Run Interactive Terminal CLI ⭐

```bash
python3 main.py
```

The interactive menu lets you:

| Option | Feature |
|--------|---------|
| `1` | Get hybrid recommendations by typing item names |
| `2` | Cold-start — describe preferences in free text |
| `3` | Browse the full item catalogue |
| `4` | Explain an item (view top TF-IDF keywords) |
| `5` | Run evaluation (Precision@K, Recall@K for all 3 models) |
| `6` | Tune settings (alpha, top-N, domain filter) |

### 3. Run Programmatic Demo

```bash
python3 recommender.py
```

Shows preset demos and runs full evaluation automatically.

### 4. Launch the Web App

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## 🧠 Algorithm Overview

### Content-Based Filtering
| Step | What Happens |
|------|--------------|
| 1 | Combine `title + category + tags` into a `content` text field |
| 2 | Apply **TF-IDF** vectorisation (500 features, unigrams + bigrams) |
| 3 | Compute **cosine similarity** between all item pairs |
| 4 | For a user's liked items, average their similarity rows |
| 5 | Return top-N items with highest average similarity |

### Collaborative Filtering (KNN)
| Step | What Happens |
|------|--------------|
| 1 | Build user–item rating matrix (50 users × 60 items) |
| 2 | Fill NaN ratings with **user mean** (per-row imputation) |
| 3 | Fit **KNN** (K=10, cosine metric) on filled matrix |
| 4 | For a target user, find K most similar users |
| 5 | Predict ratings via **weighted average** of neighbours |

### Hybrid Fusion
```
hybrid_score = α × cb_score_normalised + (1-α) × cf_score_normalised
```
- Both scores are **min-max normalised** to [0,1] before blending
- Default **α = 0.5** (equal blend)
- **Cold-start** detected → α automatically set to 1.0 (pure content-based)

---

## 📊 Evaluation

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **Precision@K** | `\|top-K ∩ relevant\| / K` | % of recommendations that are good |
| **Recall@K** | `\|top-K ∩ relevant\| / \|relevant\|` | % of good items we found |

Evaluation uses a **train/test split** (80/20) on the interaction matrix.
Items rated ≥ 3.5 stars are considered **relevant**.

---

## 🔌 Extending to New Domains

The system is **fully domain-independent**. To add events, books, or products:

1. Add a new `generate_<domain>_dataset()` function in `data_loader.py`
2. Assign unique item IDs (e.g., `E001`, `B001`)
3. Populate the unified schema: `item_id | title | category | tags | cost | time | domain`
4. Include in `load_unified_dataset()` — **no changes needed in any other module**

---

## 💡 Viva Q&A Cheat Sheet

**Q: Why TF-IDF instead of bag-of-words?**  
A: TF-IDF penalises common words (like "the") that appear everywhere, giving more weight to distinctive keywords. This makes similarity scores more meaningful.

**Q: Why cosine similarity instead of Euclidean distance?**  
A: Cosine similarity measures the angle between vectors, not their magnitude. A long item description and a short one can still be similar in content.

**Q: Why KNN for collaborative filtering?**  
A: KNN is transparent and explainable — we can directly say "users similar to you also liked X." No black-box training like matrix factorisation.

**Q: How does cold-start work?**  
A: New users with no history use content-based filtering on their stated preferences (free-text tags). α is automatically set to 1.0 in this case.

**Q: Why min-max normalise before hybrid fusion?**  
A: CB and CF scores have different scales. Without normalisation, the higher-magnitude signal dominates the blend unfairly.

---

## 🧑‍💻 Tech Stack

| Library | Purpose |
|---------|---------|
| `pandas` | DataFrames, data manipulation |
| `numpy` | Matrix operations |
| `scikit-learn` | TF-IDF, KNN, cosine similarity, MinMaxScaler |
| `streamlit` | Web application UI |
| `plotly` | Interactive charts and heatmaps |

---

*Built as a university AI project · No deep learning · No external APIs · Fully explainable*
