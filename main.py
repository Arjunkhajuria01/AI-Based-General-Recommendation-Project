"""
main.py
=======
Interactive Terminal CLI for the Hybrid Recommendation System.

Run this file to start the interactive session:
    python3 main.py

Features:
  - Search items by name or describe preferences in free text
  - Get Top-N hybrid recommendations with explanations
  - Choose domain filter (movies / courses / both)
  - Adjust alpha (content vs. collaborative weight)
  - Run Precision@K and Recall@K evaluation
  - Cold-start fallback for new users
"""

import sys
import os
import textwrap

# ── Ensure src/ is importable from the project root ──────────────
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


# ═══════════════════════════════════════════════════════════════
# DISPLAY HELPERS
# ═══════════════════════════════════════════════════════════════

LINE  = "─" * 62
DLINE = "═" * 62

def banner():
    """Prints the ASCII welcome banner."""
    print()
    print(DLINE)
    print("  🎯  HYBRID RECOMMENDATION SYSTEM")
    print("       Content-Based (TF-IDF) + Collaborative (KNN)")
    print(DLINE)
    print("  Domains : Movies  |  Online Courses")
    print("  Metrics : Precision@K  |  Recall@K")
    print("  Authors : University AI Project")
    print(DLINE)
    print()


def section(title: str):
    print()
    print(f"  ── {title} {'─'*(55-len(title))}")


def print_items_table(df: pd.DataFrame, score_col: str = "hybrid_score") -> None:
    """
    Pretty-prints a recommendation DataFrame in the terminal.
    Shows rank, item ID, title, domain, score, and explanation.
    """
    if df is None or df.empty:
        print("  [!] No recommendations found.")
        return

    score_col = score_col if score_col in df.columns else \
                next((c for c in ["hybrid_score","cb_score","cf_score"] if c in df.columns), None)

    print()
    print(f"  {'#':<4} {'Item ID':<8} {'Title':<34} {'Domain':<8} {'Score':<7}  Explanation")
    print(f"  {'─'*4} {'─'*8} {'─'*34} {'─'*8} {'─'*7}  {'─'*30}")

    for rank, (_, row) in enumerate(df.iterrows(), start=1):
        title  = str(row.get("title", ""))[:33]
        domain = str(row.get("domain", ""))[:7]
        score  = float(row[score_col]) if score_col and score_col in row else 0.0
        expl   = str(row.get("explanation", ""))
        # Wrap long explanations
        expl_short = textwrap.shorten(expl, width=40, placeholder="...")
        print(f"  {rank:<4} {row['item_id']:<8} {title:<34} {domain:<8} {score:<7.4f}  {expl_short}")

    print()


def ask(prompt: str, default: str = "") -> str:
    """Reads user input, returning default if blank."""
    try:
        val = input(f"  {prompt}").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return default
    return val if val else default


def ask_int(prompt: str, default: int, lo: int = 1, hi: int = 100) -> int:
    """Reads an integer in [lo, hi]."""
    raw = ask(f"{prompt} [{default}]: ", str(default))
    try:
        val = int(raw)
        return max(lo, min(hi, val))
    except ValueError:
        return default


def ask_float(prompt: str, default: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """Reads a float in [lo, hi]."""
    raw = ask(f"{prompt} [{default}]: ", str(default))
    try:
        val = float(raw)
        return max(lo, min(hi, val))
    except ValueError:
        return default


# ═══════════════════════════════════════════════════════════════
# SYSTEM BUILDER
# ═══════════════════════════════════════════════════════════════

def build_system(alpha: float = 0.5,
                 n_neighbors: int = 10,
                 n_users: int = 50,
                 sparsity: float = 0.85) -> dict:
    """
    Builds and returns the complete hybrid recommendation system.

    Steps:
      1. Load + merge datasets (movies + courses)
      2. Preprocess text and normalise numerics
      3. TF-IDF vectorisation → cosine similarity matrix
      4. Fit KNN collaborative filtering on training ratings
      5. Wire up hybrid engine

    Returns a dict with all fitted components.
    """
    print(DLINE)
    print("  🔧  Initialising Recommendation Engine …")
    print(DLINE)

    # Step 1 — Data
    items      = load_unified_dataset()
    items      = preprocess(items)
    raw_matrix = generate_interaction_matrix(items, n_users=n_users, sparsity=sparsity)

    # Step 2 — Train/test split (split BEFORE filling NaNs)
    train_raw, test_raw = train_test_split_matrix(raw_matrix, test_ratio=0.2)
    train_filled        = fill_interaction_matrix(train_raw, strategy="user_mean")

    # Step 3 — Content-Based (TF-IDF + Cosine Similarity)
    builder    = TFIDFFeatureBuilder(max_features=500, ngram_range=(1, 2))
    builder.fit_transform(items["content"])
    sim_matrix = builder.compute_similarity()
    cb         = ContentBasedRecommender(items, sim_matrix, builder)

    # Step 4 — Collaborative (KNN on filled train matrix)
    cf = CollaborativeRecommender(train_filled, n_neighbors=n_neighbors)

    # Step 5 — Hybrid Engine
    hybrid = HybridRecommender(items, cb, cf, alpha=alpha)

    # Save processed files
    save_datasets(items, raw_matrix, output_dir="data/processed")

    print()
    print("  ✅  System ready! All components fitted.\n")

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


# ═══════════════════════════════════════════════════════════════
# ITEM SEARCH — resolves user-typed names to item_ids
# ═══════════════════════════════════════════════════════════════

def resolve_items(items_df: pd.DataFrame, query: str) -> list:
    """
    Matches a user's comma-separated query (titles or item_ids) to actual item_ids.

    Examples:
        "Inception, M003"      → ["M002", "M003"]
        "python, C001, data"   → ["C001", "C005", ...]

    Returns:
        List of matched item_ids (may be empty if nothing matches).
    """
    tokens = [t.strip().lower() for t in query.split(",") if t.strip()]
    matched_ids = []

    for token in tokens:
        # Exact item_id match
        exact = items_df[items_df["item_id"].str.lower() == token]
        if not exact.empty:
            matched_ids.extend(exact["item_id"].tolist())
            continue

        # Fuzzy title match (substring)
        fuzzy = items_df[items_df["title"].str.lower().str.contains(token, na=False)]
        if not fuzzy.empty:
            # Take the best match (shortest title = most specific)
            best = fuzzy.loc[fuzzy["title"].str.len().idxmin()]
            matched_ids.append(best["item_id"])
        else:
            print(f"  [!] Could not match '{token}' — skipped.")

    return list(dict.fromkeys(matched_ids))  # deduplicate, preserve order


def show_matched(items_df: pd.DataFrame, item_ids: list) -> None:
    """Prints a summary of matched items so user can confirm."""
    if not item_ids:
        return
    print()
    print("  Matched items:")
    for iid in item_ids:
        row = items_df[items_df["item_id"] == iid].iloc[0]
        print(f"    • [{iid}] {row['title']}  ({row['domain']} / {row['category']})")


# ═══════════════════════════════════════════════════════════════
# EVALUATION RUNNER
# ═══════════════════════════════════════════════════════════════

def run_evaluation(system: dict, k: int = 10) -> None:
    """
    Evaluates Content-Based, Collaborative, and Hybrid models.
    Prints Precision@K and Recall@K for each, then a comparison table.
    """
    items    = system["items"]
    test_mat = system["test_matrix"]
    cb       = system["cb"]
    cf       = system["cf"]
    hybrid   = system["hybrid"]

    print()
    print(DLINE)
    print(f"  📊  Evaluation — Precision@{k} and Recall@{k}")
    print(DLINE)

    all_results = {}

    # ── Content-Based ───────────────────────────────────────
    def cb_fn(user_id):
        user_row  = system["train_matrix"].loc[user_id]
        liked_ids = user_row[user_row >= 3.5].index.tolist()
        if not liked_ids:
            liked_ids = user_row.nlargest(3).index.tolist()
        recs = cb.recommend(liked_ids, top_n=k, exclude_seen=True)
        return recs["item_id"].tolist()

    cb_res = evaluate_recommender(cb_fn, test_mat, k=k)
    print_evaluation_report(cb_res, "Content-Based (TF-IDF + Cosine)")
    all_results["Content-Based"] = cb_res

    # ── Collaborative ───────────────────────────────────────
    def cf_fn(user_id):
        recs = cf.recommend(user_id, top_n=k, items_df=items)
        return recs["item_id"].tolist()

    cf_res = evaluate_recommender(cf_fn, test_mat, k=k)
    print_evaluation_report(cf_res, "Collaborative (KNN)")
    all_results["Collaborative"] = cf_res

    # ── Hybrid ──────────────────────────────────────────────
    def hybrid_fn(user_id):
        user_row  = system["train_matrix"].loc[user_id]
        liked_ids = user_row[user_row >= 3.5].index.tolist()
        if not liked_ids:
            liked_ids = user_row.nlargest(3).index.tolist()
        recs = hybrid.recommend(user_id, liked_item_ids=liked_ids, top_n=k)
        return recs["item_id"].tolist()

    h_res = evaluate_recommender(hybrid_fn, test_mat, k=k)
    print_evaluation_report(h_res, "Hybrid (CB + CF)")
    all_results["Hybrid"] = h_res

    # ── Comparison table ────────────────────────────────────
    print()
    print(DLINE)
    print(f"  Comparison Summary @ K={k}")
    print(DLINE)
    print(f"  {'Model':<26} {'Precision@K':>12} {'Recall@K':>10}")
    print(f"  {'─'*26} {'─'*12} {'─'*10}")
    for name, res in all_results.items():
        p = res["precision_at_k"]
        r = res["recall_at_k"]
        print(f"  {name:<26} {p:>12.4f} {r:>10.4f}")
    print(DLINE)


# ═══════════════════════════════════════════════════════════════
# MAIN MENU
# ═══════════════════════════════════════════════════════════════

def show_menu() -> None:
    print()
    print(LINE)
    print("  MAIN MENU")
    print(LINE)
    print("  [1]  Get Hybrid Recommendations  (by item name)")
    print("  [2]  Cold-Start Recommendation   (describe preferences)")
    print("  [3]  Browse All Items")
    print("  [4]  Explain an Item             (top keywords)")
    print("  [5]  Run Evaluation              (Precision@K, Recall@K)")
    print("  [6]  Settings                    (alpha, top-N, domain)")
    print("  [0]  Exit")
    print(LINE)


# ═══════════════════════════════════════════════════════════════
# MENU HANDLERS
# ═══════════════════════════════════════════════════════════════

def handle_hybrid_recs(system: dict, settings: dict) -> None:
    """
    Option 1: User types liked item names → hybrid recommendations.
    """
    section("Hybrid Recommendations")
    print("  Enter one or more items you like (comma-separated names or IDs).")
    print("  Example: Inception, The Matrix, Python for Beginners")
    print()

    query = ask("Your liked items: ")
    if not query:
        print("  [!] No input. Returning to menu.")
        return

    liked_ids = resolve_items(system["items"], query)
    if not liked_ids:
        print("  [!] No items matched. Try different names.")
        return

    show_matched(system["items"], liked_ids)

    # Optional user ID (for collaborative component)
    user_id = ask("\n  Enter your User ID (e.g. U001) or press Enter to skip: ", "COLD")
    if user_id == "COLD":
        print("  ℹ  No user ID — falling back to content-based only (cold-start mode).")

    alpha       = settings.get("alpha", 0.5)
    top_n       = settings.get("top_n", 10)
    domain      = settings.get("domain", None)

    system["hybrid"].set_alpha(alpha)

    recs = system["hybrid"].recommend(
        user_id        = user_id if user_id != "COLD" else None,
        liked_item_ids = liked_ids,
        top_n          = top_n,
        filter_domain  = domain,
        exclude_seen   = True,
    )

    section(f"Top-{top_n} Recommendations  (α={alpha})")
    print_items_table(recs, score_col="hybrid_score")


def handle_cold_start(system: dict, settings: dict) -> None:
    """
    Option 2: User describes preferences in free text → content-based recs.
    Suitable for brand-new users with no rating history.
    """
    section("Cold-Start Recommendation  (Free-Text Preferences)")
    print("  Describe what you enjoy — use keywords, genres, topics.")
    print("  Example: I love space exploration, machine learning, and thriller films")
    print()

    pref_text = ask("Your preferences: ")
    if not pref_text:
        print("  [!] No input. Returning to menu.")
        return

    top_n  = settings.get("top_n", 10)
    domain = settings.get("domain", None)

    recs = system["cb"].recommend_by_tags(
        preference_text = pref_text,
        top_n           = top_n,
        filter_domain   = domain,
    )

    # Add explanation column manually for cold-start
    recs["explanation"] = "Matches your stated preference keywords (content-based)."

    section(f"Top-{top_n} Cold-Start Recommendations")
    print_items_table(recs, score_col="cb_score")


def handle_browse(system: dict, settings: dict) -> None:
    """
    Option 3: Browse all items with optional domain filter.
    """
    section("Browse Items")
    domain = settings.get("domain", None)
    items  = system["items"]

    if domain:
        subset = items[items["domain"] == domain]
        print(f"  Showing {len(subset)} items  (domain: {domain})\n")
    else:
        subset = items
        print(f"  Showing all {len(subset)} items  (movies + courses)\n")

    print(f"  {'Item ID':<8} {'Domain':<8} {'Category':<14} {'Title'}")
    print(f"  {'─'*8} {'─'*8} {'─'*14} {'─'*36}")
    for _, row in subset.iterrows():
        print(f"  {row['item_id']:<8} {row['domain']:<8} {row['category']:<14} {row['title']}")


def handle_explain(system: dict) -> None:
    """
    Option 4: Explain why an item would be recommended (top TF-IDF keywords).
    """
    section("Explain Item")
    print("  Enter an item ID (e.g. M002, C005) to see its top keywords.\n")

    item_id = ask("Item ID: ").upper()
    if not item_id:
        return

    explanation = system["cb"].explain(item_id)

    if "error" in explanation:
        print(f"  [!] {explanation['error']}")
        return

    print()
    print(f"  Item ID  : {explanation['item_id']}")
    print(f"  Title    : {explanation['title']}")
    print(f"  Category : {explanation['category']}")
    print(f"  Domain   : {explanation['domain']}")
    print()
    print("  Top TF-IDF Keywords (word → score):")
    for word, score in explanation.get("keywords", []):
        bar = "█" * int(score * 60)
        print(f"    {word:<20}  {score:.4f}  {bar}")


def handle_settings(settings: dict) -> None:
    """
    Option 6: Interactively update alpha, top_n, and domain filter.
    """
    section("Settings")

    print(f"  Current settings:")
    print(f"    alpha  = {settings['alpha']}  (content weight; 0=full CF, 1=full CB)")
    print(f"    top_n  = {settings['top_n']}")
    print(f"    domain = {settings['domain'] or 'both'}")
    print()

    settings["alpha"]  = ask_float("  New alpha (0.0 – 1.0)", settings["alpha"])
    settings["top_n"]  = ask_int("  New top_n  (1 – 50)",   settings["top_n"], lo=1, hi=50)

    domain_choice = ask("  Domain filter  [movie / course / both]: ", "both").lower()
    if domain_choice in ("movie", "course"):
        settings["domain"] = domain_choice
    else:
        settings["domain"] = None

    print()
    print(f"  ✅  Settings saved: alpha={settings['alpha']}, "
          f"top_n={settings['top_n']}, "
          f"domain={settings['domain'] or 'both'}")


def handle_evaluation(system: dict) -> None:
    """
    Option 5: Run full evaluation suite.
    """
    k = ask_int("  Cutoff K for evaluation", 10, lo=1, hi=30)
    run_evaluation(system, k=k)


# ═══════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════

def main():
    banner()

    # ── Build system ─────────────────────────────────────────
    print("  Configure the engine (press Enter to use defaults):\n")
    alpha      = ask_float("  Alpha (content weight, 0.0–1.0)", 0.5)
    n_neighbors = ask_int("  KNN neighbours K",                 10, lo=2, hi=30)
    n_users    = ask_int("  Synthetic users for CF matrix",    50, lo=10, hi=200)

    system = build_system(alpha=alpha, n_neighbors=n_neighbors, n_users=n_users)

    # Session settings (mutable between menu choices)
    settings = {
        "alpha":  alpha,
        "top_n":  10,
        "domain": None,   # None = both movies and courses
    }

    # ── Interactive loop ──────────────────────────────────────
    while True:
        show_menu()
        choice = ask("Select option: ", "0")

        if choice == "1":
            handle_hybrid_recs(system, settings)

        elif choice == "2":
            handle_cold_start(system, settings)

        elif choice == "3":
            handle_browse(system, settings)

        elif choice == "4":
            handle_explain(system)

        elif choice == "5":
            handle_evaluation(system)

        elif choice == "6":
            handle_settings(settings)
            # Apply new alpha to hybrid engine
            system["hybrid"].set_alpha(settings["alpha"])

        elif choice == "0":
            print()
            print("  👋  Goodbye! Thanks for using the Hybrid Recommender.\n")
            break

        else:
            print("  [!] Invalid choice. Enter a number from the menu.")


if __name__ == "__main__":
    main()
