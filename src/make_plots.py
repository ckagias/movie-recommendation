import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer

import demo
from explain_mf import recommend_mf, score_parts, title_of
from movie_text import DATA, build_movie_texts

OUT = Path(__file__).resolve().parent.parent / "docs" / "img"

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
DEEMPHASIS = "#b9b7ae"
MAP_GRAY = "#d9d7cf"

ROLE_COLORS = {"built": BLUE, "library": AQUA, "reference": DEEMPHASIS, "target": ORANGE}
ROLE_NAMES = {
    "built": "built in this project",
    "library": "library model (implicit)",
    "reference": "baseline or reference list",
    "target": "what users actually liked",
}

# (mean, lowest shuffle, highest shuffle) over 10 shuffles, copied from results.md
RMSE = [
    ("1. Overall average", 1.043, 1.032, 1.050, "reference"),
    ("2. Movie average", 0.977, 0.963, 0.983, "reference"),
    ("3. Movie average + user bias", 0.894, 0.881, 0.899, "reference"),
    ("Matrix factorization", 0.856, 0.843, 0.862, "built"),
]
HIT_RATE = [
    ("ALS, liked only", 0.759, 0.743, 0.788, "built"),
    ("ALS, rated at all", 0.748, 0.720, 0.771, "built"),
    ("Item-item cosine, liked only", 0.722, 0.667, 0.745, "library"),
    ("Item-item cosine, rated at all", 0.692, 0.666, 0.716, "library"),
    ("BPR, liked only", 0.679, 0.637, 0.720, "library"),
    ("BPR, rated at all", 0.630, 0.580, 0.655, "library"),
    ("Hybrid (popularity + predicted rating)", 0.573, 0.548, 0.593, "built"),
    ("Most rated movies", 0.565, 0.549, 0.587, "reference"),
    ("Content-based (genres and decade)", 0.500, 0.465, 0.542, "built"),
    ("Matrix factorization", 0.319, 0.306, 0.339, "built"),
    ("Highest movie average", 0.262, 0.205, 0.310, "reference"),
    ("Random list", 0.113, 0.095, 0.134, "reference"),
]
BLEND = [
    (0.0, 0.319, 0.306, 0.339),
    (0.1, 0.477, 0.456, 0.518),
    (0.2, 0.505, 0.477, 0.542),
    (0.3, 0.521, 0.501, 0.556),
    (0.4, 0.533, 0.503, 0.573),
    (0.5, 0.546, 0.518, 0.572),
    (0.6, 0.556, 0.544, 0.584),
    (0.7, 0.563, 0.549, 0.584),
    (0.8, 0.567, 0.548, 0.585),
    (0.9, 0.575, 0.542, 0.593),
    (1.0, 0.565, 0.549, 0.587),
]
SIMILARITY = [
    ("Content-based top 5 (genres and decade)", 0.867, 0.840, 0.889, "built"),
    ("Random movies with 50 or more ratings", 0.954, 0.932, 0.978, "reference"),
    ("Random movies", 1.003, 0.951, 1.039, "reference"),
    ("The 5 most-rated movies", 1.017, 0.971, 1.043, "reference"),
]
POPULARITY = [
    ("Most rated movies", 100.0, 100.0, 100.0, "reference"),
    ("Content-based (genres and decade)", 64.7, 61.2, 67.5, "built"),
    ("Hidden liked movies (the target)", 31.0, 29.7, 32.0, "target"),
    ("Matrix factorization", 23.3, 20.3, 25.8, "built"),
    ("Highest movie average", 23.3, 7.2, 40.8, "reference"),
    ("Random list", 7.8, 7.2, 8.2, "reference"),
]
NEW_MOVIES = {
    "x": ["0", "1", "2 to 4", "5 to 9", "10 to 19", "20 to 49", "50+"],
    "Matrix factorization": [1.026, 0.958, 0.917, 0.882, 0.836, 0.842, 0.818],
    "Baseline 3, movie average + user bias": [1.077, 1.226, 1.009, 0.915, 0.854, 0.858, 0.841],
    "Baseline 2, movie average": [1.150, 1.294, 1.087, 1.005, 0.952, 0.948, 0.916],
}
NEW_USERS = {
    "x": ["0", "1", "2", "5", "10", "20", "all"],
    "Matrix factorization": [0.964, 0.967, 0.961, 0.935, 0.909, 0.893, 0.862],
    "Baseline 3, movie average + user bias": [0.986, 1.199, 1.061, 0.972, 0.936, 0.922, 0.899],
    "Baseline 2, movie average": [0.986, 0.986, 0.986, 0.986, 0.986, 0.986, 0.979],
}
SERIES_COLORS = {
    "Matrix factorization": BLUE,
    "Baseline 3, movie average + user bias": ORANGE,
    "Baseline 2, movie average": AQUA,
}
CLUSTER_NAMES = {
    "crime": "Crime and thriller",
    "action": "Action, adventure and sci-fi",
    "horror": "Horror and thriller",
    "children": "Children and animation",
    "documentary": "Documentary",
}


def set_style():
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "font.family": "sans-serif",
            "font.sans-serif": ["Segoe UI", "Helvetica", "Arial", "DejaVu Sans"],
            "text.color": INK,
            "axes.labelcolor": SECONDARY,
            "axes.edgecolor": BASELINE,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 1.0,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
        }
    )


def layout(fig, left, right, top_in, bottom_in, **kwargs):
    height = fig.get_figheight()
    fig.subplots_adjust(left=left, right=right, top=1 - top_in / height, bottom=bottom_in / height, **kwargs)


def headline(fig, title, subtitle):
    height = fig.get_figheight()
    fig.text(0.02, 1 - 0.15 / height, title, fontsize=14, fontweight="bold", va="top", ha="left")
    fig.text(0.02, 1 - 0.5 / height, subtitle, fontsize=9.5, color=SECONDARY, va="top", ha="left")


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=200)
    plt.close(fig)
    print("saved", OUT / name)


def role_legend(fig, roles):
    handles = [Patch(facecolor=ROLE_COLORS[role], label=ROLE_NAMES[role]) for role in ROLE_COLORS if role in roles]
    fig.legend(handles=handles, loc="lower left", ncol=len(handles), frameon=False, bbox_to_anchor=(0.02, 0.0), fontsize=9)


def bar_chart(name, title, subtitle, rows, xlim, value_format, xlabel, left=0.4):
    height = 0.55 * len(rows) + 1.9
    fig, ax = plt.subplots(figsize=(8, height))
    layout(fig, left, 0.96, 1.1, 1.0)
    ys = np.arange(len(rows))[::-1]
    pad = 0.012 * (xlim[1] - xlim[0])
    for y, (label, value, low, high, role) in zip(ys, rows):
        ax.barh(y, value, height=0.44, color=ROLE_COLORS[role], zorder=3)
        if high > low:
            ax.hlines(y, low, high, color=INK, linewidth=1.1, zorder=4)
            ax.vlines([low, high], y - 0.09, y + 0.09, color=INK, linewidth=1.1, zorder=4)
        ax.text(max(value, high) + pad, y, value_format(value), va="center", ha="left", fontsize=10)
    ax.set_yticks(ys)
    ax.set_yticklabels([row[0] for row in rows], fontsize=10, color=SECONDARY)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(*xlim)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.xaxis.grid(True)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    headline(fig, title, subtitle)
    role_legend(fig, {row[4] for row in rows})
    save(fig, name)


def dot_range_chart(name, title, subtitle, rows, xlim, xlabel, left=0.4):
    height = 0.55 * len(rows) + 1.9
    fig, ax = plt.subplots(figsize=(8, height))
    layout(fig, left, 0.96, 1.1, 1.0)
    ys = np.arange(len(rows))[::-1]
    pad = 0.012 * (xlim[1] - xlim[0])
    for y, (label, value, low, high, role) in zip(ys, rows):
        ax.hlines(y, low, high, color=ROLE_COLORS[role], linewidth=2, zorder=3)
        ax.plot(value, y, "o", markersize=9, color=ROLE_COLORS[role], markeredgecolor=SURFACE, markeredgewidth=2, zorder=4)
        ax.text(high + pad, y, f"{value:.3f}", va="center", ha="left", fontsize=10)
    ax.set_yticks(ys)
    ax.set_yticklabels([row[0] for row in rows], fontsize=10, color=SECONDARY)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(*xlim)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.xaxis.grid(True)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    headline(fig, title, subtitle)
    role_legend(fig, {row[4] for row in rows})
    save(fig, name)


def blend_curve():
    weights = np.array([row[0] for row in BLEND])
    mean = np.array([row[1] for row in BLEND])
    low = np.array([row[2] for row in BLEND])
    high = np.array([row[3] for row in BLEND])

    fig, ax = plt.subplots(figsize=(8, 4.6))
    layout(fig, 0.1, 0.96, 1.1, 0.8)
    ax.fill_between(weights, low, high, color=BLUE, alpha=0.1, linewidth=0, zorder=2)
    ax.plot(weights, mean, color=BLUE, linewidth=2, marker="o", markersize=7, markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=3)
    leader = {"arrowstyle": "-", "color": MUTED, "linewidth": 1, "shrinkA": 2, "shrinkB": 6}
    ax.annotate("matrix factorization 0.319", (0.0, 0.319), xytext=(0.13, 0.31), fontsize=9.5, va="center", arrowprops=leader)
    ax.annotate("best blend, weight 0.9: 0.575", (0.9, 0.575), xytext=(0.9, 0.615), fontsize=9.5, ha="center", va="bottom", arrowprops=leader)
    ax.annotate("most rated only: 0.565", (1.0, 0.565), xytext=(1.0, 0.505), fontsize=9.5, ha="right", va="top", arrowprops=leader)
    ax.set_xlim(-0.04, 1.04)
    ax.set_ylim(0.25, 0.65)
    ax.set_xticks(np.arange(0, 1.01, 0.1))
    ax.set_xlabel("weight on popularity (0 = only the model's prediction, 1 = only number of ratings)", fontsize=9)
    ax.set_ylabel("hit rate at 10", fontsize=9)
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    headline(
        fig,
        "Blending in popularity recovers the model's top 10 score",
        "Hit rate at 10 on the test pile. Line is the mean of 10 shuffles, the band is lowest to highest.",
    )
    save(fig, "hybrid_blend.png")


def cold_start():
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8), sharey=True)
    layout(fig, 0.07, 0.98, 1.3, 1.0, wspace=0.08)
    panels = [
        (axes[0], NEW_MOVIES, "New movies", "training ratings of the movie"),
        (axes[1], NEW_USERS, "New users", "ratings of the user kept for training"),
    ]
    for ax, data, panel_title, xlabel in panels:
        positions = np.arange(len(data["x"]))
        for name, color in SERIES_COLORS.items():
            ax.plot(positions, data[name], color=color, linewidth=2, marker="o", markersize=7, markeredgecolor=SURFACE, markeredgewidth=1.5, label=name, zorder=3)
        ax.set_xticks(positions)
        ax.set_xticklabels(data["x"], fontsize=9)
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_title(panel_title, loc="left", fontsize=11, color=INK, fontweight="bold")
        ax.yaxis.grid(True)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("RMSE in stars (lower is better)", fontsize=9)
    axes[0].set_ylim(0.78, 1.32)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower left", ncol=3, frameon=False, bbox_to_anchor=(0.07, 0.0), fontsize=9)
    headline(
        fig,
        "Few ratings hurt, and one rating can be worse than none",
        "Test RMSE, mean of 10 shuffles. The simple baselines take a single rating at face value, matrix factorization shrinks it.",
    )
    save(fig, "cold_start.png")


def pull_versus_taste(model, user_id=3):
    picks = recommend_mf(model, user_id, 10)
    titles = [title_of[movie_id] for movie_id, _, _ in picks]
    parts = [score_parts(model, user_id, m) for _, m, _ in picks]
    pull = np.array([p["movie"] for p in parts])
    taste = np.array([p["taste"] for p in parts])

    fig, ax = plt.subplots(figsize=(8.5, 6.4))
    layout(fig, 0.4, 0.96, 1.1, 1.1)
    ys = np.arange(len(picks))[::-1]
    ax.barh(ys + 0.19, pull, height=0.34, color=BLUE, label="movie pull (how well liked the movie is in general)", zorder=3)
    ax.barh(ys - 0.19, taste, height=0.34, color=ORANGE, label="taste match (how well it fits this user)", zorder=3)
    ax.axvline(0, color=BASELINE, linewidth=1, zorder=2)
    ax.set_yticks(ys)
    ax.set_yticklabels([re.sub(r" \([^)]*\)(?= \(\d{4}\))", "", t) for t in titles], fontsize=9.5, color=SECONDARY)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("stars added to the predicted rating", fontsize=9)
    ax.xaxis.grid(True)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    fig.legend(loc="lower left", ncol=1, frameon=False, bbox_to_anchor=(0.02, 0.0), fontsize=9)
    headline(
        fig,
        f"User {user_id}: where each of the top 10 picks comes from",
        f"Matrix factorization top 10, on top of the overall average and this user's generosity ({parts[0]['user']:+.2f} stars). Taste match is the larger part in every pick.",
    )
    save(fig, "pull_versus_taste.png")


def star_counts():
    counts = pd.read_csv(DATA / "ratings.csv")["rating"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(8, 4.4))
    layout(fig, 0.1, 0.96, 1.1, 0.75)
    positions = np.arange(len(counts))
    ax.bar(positions, counts.to_numpy(), width=0.6, color=BLUE, zorder=3)
    ax.set_xticks(positions)
    ax.set_xticklabels([f"{value:.1f}" for value in counts.index])
    ax.set_xlabel("stars", fontsize=9)
    ax.set_ylabel("number of ratings", fontsize=9)
    ax.yaxis.set_major_formatter(lambda value, _: f"{value:,.0f}")
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    headline(
        fig,
        "4.0 is the most common rating, and whole stars beat half stars",
        f"How often each star value is given, {int(counts.sum()):,} ratings from 610 users.",
    )
    save(fig, "star_counts.png")


def cluster_map(k=10):
    movies = build_movie_texts()
    movies["text"] = movies["genres_text"] + " " + movies["decade_text"]
    movies = movies[movies["text"].str.strip() != ""].reset_index(drop=True)

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(movies["text"]).toarray()
    words = vectorizer.get_feature_names_out()
    pca = PCA(n_components=2, random_state=0)
    coords = pca.fit_transform(X)
    coords = coords + np.random.default_rng(0).normal(0, 0.012, size=coords.shape)
    km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(X)

    top_word = {c: words[np.argmax(km.cluster_centers_[c])] for c in range(k)}
    genre_clusters = [c for c in np.argsort(-np.bincount(km.labels_)) if not top_word[c].startswith("decade")]
    panels = [(CLUSTER_NAMES[top_word[c]], km.labels_ == c) for c in genre_clusters]
    decade_mask = np.isin(km.labels_, [c for c in range(k) if top_word[c].startswith("decade")])
    panels.append(("The five decade clusters", decade_mask))

    fig, axes = plt.subplots(2, 3, figsize=(11, 7.4))
    layout(fig, 0.02, 0.98, 1.4, 0.3, wspace=0.06, hspace=0.22)
    for ax, (panel_title, mask) in zip(axes.ravel(), panels):
        ax.scatter(coords[~mask, 0], coords[~mask, 1], s=3, color=MAP_GRAY, linewidths=0, zorder=2)
        ax.scatter(coords[mask, 0], coords[mask, 1], s=4, color=BLUE, alpha=0.65, linewidths=0, zorder=3)
        ax.set_title(f"{panel_title} ({mask.sum():,})", loc="left", fontsize=10.5, color=INK, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])
        for side in ax.spines.values():
            side.set_visible(False)
    headline(
        fig,
        "On the 2D map every cluster is spread out and overlaps the others",
        f"Same 2D map in every panel: genre and decade vectors squeezed with PCA ({pca.explained_variance_ratio_.sum():.1%} of the variation kept).\n"
        f"Blue marks one K-Means cluster (k = {k}, movies counted in brackets). Points are jittered slightly.",
    )
    save(fig, "cluster_map.png")


def hit_rate():
    bar_chart(
        "hit_rate.png",
        "Models trained on who rated what lead the top 10 test",
        "Share of users with at least one hidden liked movie in their top 10, higher is better.\nLine is lowest to highest of 10 shuffles.",
        HIT_RATE,
        (0, 0.92),
        lambda v: f"{v:.3f}",
        "hit rate at 10",
    )


if __name__ == "__main__":
    set_style()
    star_counts()
    bar_chart(
        "baseline_ladder.png",
        "Matrix factorization has the lowest rating error",
        "RMSE in stars on the test pile, lower is better. Bar is the mean of 10 shuffles, line is lowest to highest.",
        RMSE,
        (0, 1.25),
        lambda v: f"{v:.3f}",
        "RMSE in stars",
    )
    hit_rate()
    blend_curve()
    dot_range_chart(
        "similarity_gap.png",
        "Content-based matches get more similar ratings than chance",
        "Average rating gap between a movie and its top 5 matches, lower is better.\nDot is the mean, line is lowest to highest of 10 shuffles.",
        SIMILARITY,
        (0.8, 1.12),
        "rating gap in stars (the axis does not start at zero)",
    )
    bar_chart(
        "popularity.png",
        "Matrix factorization picks lean less popular than what users liked",
        "Share of each list's top 10 picks that are among the 100 most-rated movies. Line is lowest to highest of 10 shuffles.",
        POPULARITY,
        (0, 118),
        lambda v: f"{v:.1f}%",
        "share of picks in the 100 most rated movies (%)",
    )
    cold_start()
    pull_versus_taste(demo.load_model(refit=False))
    cluster_map()
