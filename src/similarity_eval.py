from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from movie_text import build_movie_texts

DATA = Path(__file__).resolve().parent.parent / "data"

TOP_N = 5
MIN_SHARED = 5
MIN_RATINGS_QUERY = 50
N_QUERY = 100
N_SEEDS = 10

movies = build_movie_texts()
ratings = pd.read_csv(DATA / "ratings.csv")

n_ratings = ratings.groupby("movieId").size()
movies["n_ratings"] = movies["movieId"].map(n_ratings).fillna(0).astype(int)
popularity = movies["n_ratings"].to_numpy()

R = (
    ratings.pivot(index="userId", columns="movieId", values="rating")
    .reindex(columns=movies["movieId"])
    .to_numpy()
)
M = ~np.isnan(R)

X_genres = TfidfVectorizer().fit_transform(movies["genres_text"] + " " + movies["decade_text"])


def tag_matrix(tags_text):
    return TfidfVectorizer().fit_transform(tags_text)


def combine(X_tags, tag_weight):
    return hstack([X_genres, tag_weight * X_tags]).tocsr()


def top_similar(X, i):
    scores = cosine_similarity(X[i], X).ravel().round(6)
    scores[i] = -1
    order = np.lexsort((-popularity, -scores))
    return order[:TOP_N]


def rating_gap(i, j):
    shared = M[:, i] & M[:, j]
    if shared.sum() < MIN_SHARED:
        return None
    return np.abs(R[shared, i] - R[shared, j]).mean()


def score_queries(partners_for, queries):
    gaps_per_query = []
    usable, total = 0, 0
    for i in queries:
        gaps = [rating_gap(i, j) for j in partners_for(i)]
        total += len(gaps)
        gaps = [g for g in gaps if g is not None]
        usable += len(gaps)
        if gaps:
            gaps_per_query.append(np.mean(gaps))
    return np.mean(gaps_per_query), usable / total


def summarise(values):
    return f"{np.mean(values):.3f} ({np.min(values):.3f} to {np.max(values):.3f})"


def evaluate(configs):
    eligible = np.flatnonzero(popularity >= MIN_RATINGS_QUERY)
    most_popular = np.argsort(-popularity)
    results = {name: {"val": [], "test": [], "usable": []} for name in configs}
    baselines = {"random": {"val": [], "test": []}, "most rated": {"val": [], "test": []}}

    for seed in range(N_SEEDS):
        rng = np.random.default_rng(seed)
        picked = rng.choice(eligible, size=2 * N_QUERY, replace=False)
        splits = {"val": picked[:N_QUERY], "test": picked[N_QUERY:]}

        for part, queries in splits.items():
            for name, X in configs.items():
                gap, usable = score_queries(lambda i: top_similar(X, i), queries)
                results[name][part].append(gap)
                if part == "test":
                    results[name]["usable"].append(usable)

            gap, _ = score_queries(lambda i: rng.choice(len(movies), size=TOP_N), queries)
            baselines["random"][part].append(gap)

            gap, _ = score_queries(
                lambda i: [j for j in most_popular[: TOP_N + 1] if j != i][:TOP_N], queries
            )
            baselines["most rated"][part].append(gap)

    return results, baselines


def print_table(results, baselines, label_width=34):
    print(f"{'setting':<{label_width}}{'validation':<26}{'test':<26}{'pairs usable'}")
    for name, r in results.items():
        print(
            f"{name:<{label_width}}{summarise(r['val']):<26}{summarise(r['test']):<26}"
            f"{np.mean(r['usable']):.0%}"
        )
    for name, r in baselines.items():
        print(f"{name:<{label_width}}{summarise(r['val']):<26}{summarise(r['test']):<26}")
    best = min(results, key=lambda n: np.mean(results[n]["val"]))
    print("\nbest setting on the validation movies:", best)
    print("its score on the test movies:", summarise(results[best]["test"]))
    return best
