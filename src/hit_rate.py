import time

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from baselines import summarise
from explain_content import LIKED_FROM, X, row_of
from final_score import N_EPOCHS, N_FACTORS, REGULARIZATION
from mf import LEARNING_RATE, fit
from split import load_ratings, split_ratings

TOP_N = 10
MIN_TRAIN_RATINGS = 20
N_SEEDS = 10

METHODS = ["random", "most rated", "highest movie average", "content-based", "matrix factorization"]


def summarise_counts(values):
    return f"{np.mean(values):.0f} ({np.min(values)} to {np.max(values)})"


def top_n(scores, seen, popularity):
    scores = np.where(seen, -np.inf, scores)
    return np.lexsort((-popularity, -scores))[:TOP_N]


def evaluate_seed(ratings, seed, model=None):
    train, test = split_ratings(ratings, seed)

    counts = train.groupby("movieId").size()
    counts = counts[counts >= MIN_TRAIN_RATINGS]
    pool_ids = counts.index.to_numpy()
    popularity = counts.to_numpy()
    position = {movie_id: p for p, movie_id in enumerate(pool_ids)}

    movie_means = train.groupby("movieId")["rating"].mean().reindex(pool_ids).to_numpy()
    X_pool = X[[row_of[movie_id] for movie_id in pool_ids]]

    if model is None:
        model, _ = fit(train, N_FACTORS, LEARNING_RATE, N_EPOCHS, seed, regularization=REGULARIZATION)
    in_model = np.array([model["movie_index"][movie_id] for movie_id in pool_ids])
    pool_bias, pool_Q = model["movie_bias"][in_model], model["Q"][in_model]

    liked_test = test[test["rating"] >= LIKED_FROM]
    in_pool = liked_test["movieId"].isin(position)
    relevant = (
        liked_test[in_pool].groupby("userId")["movieId"].agg(lambda ids: {position[m] for m in ids})
    )
    train_by_user = dict(list(train.groupby("userId")))

    rng = np.random.default_rng(seed)
    hit_flags = {name: [] for name in METHODS}
    recalls = {name: [] for name in METHODS}

    for user_id, wanted in relevant.items():
        mine = train_by_user[user_id]
        seen = np.zeros(len(pool_ids), dtype=bool)
        seen[[position[m] for m in mine["movieId"] if m in position]] = True
        liked_rows = [row_of[m] for m in mine.loc[mine["rating"] >= LIKED_FROM, "movieId"]]
        u = model["user_index"][user_id]

        content = (
            cosine_similarity(X[liked_rows], X_pool).max(axis=0).round(6)
            if liked_rows
            else np.zeros(len(pool_ids))
        )
        scores = {
            "random": rng.random(len(pool_ids)),
            "most rated": popularity.astype(float),
            "highest movie average": movie_means,
            "content-based": content,
            "matrix factorization": model["mu"]
            + model["user_bias"][u]
            + pool_bias
            + pool_Q @ model["P"][u],
        }
        for name, score in scores.items():
            hits = len(set(top_n(score, seen, popularity)) & wanted)
            hit_flags[name].append(hits > 0)
            recalls[name].append(hits / len(wanted))

    return {
        "hit rate": {name: np.mean(hit_flags[name]) for name in METHODS},
        "recall": {name: np.mean(recalls[name]) for name in METHODS},
        "users scored": len(relevant),
        "pool size": len(pool_ids),
        "liked in pool": in_pool.mean(),
    }


if __name__ == "__main__":
    ratings = load_ratings()

    runs = []
    for seed in range(N_SEEDS):
        start = time.time()
        runs.append(evaluate_seed(ratings, seed))
        print(f"finished seed {seed} in {time.time() - start:.0f} s", flush=True)

    print(f"\nTop {TOP_N} from movies with at least {MIN_TRAIN_RATINGS} training ratings, {N_SEEDS} shuffles")
    print(f"{'method':<26}{'hit rate at 10':<26}{'recall at 10'}")
    for name in METHODS:
        hit_rates = [run["hit rate"][name] for run in runs]
        recalls = [run["recall"][name] for run in runs]
        print(f"{name:<26}{summarise(hit_rates):<26}{summarise(recalls)}")

    best_other = max(METHODS[:-1], key=lambda name: np.mean([run["hit rate"][name] for run in runs]))
    gains = np.array(
        [run["hit rate"]["matrix factorization"] - run["hit rate"][best_other] for run in runs]
    )
    print(f"\nhit rate gain of matrix factorization over {best_other}, per shuffle:")
    print(" ".join(f"{g:+.3f}" for g in gains))
    print("shuffles where it wins:", int((gains > 0).sum()), "of", N_SEEDS)

    print("\nusers scored per shuffle:", summarise_counts([run["users scored"] for run in runs]))
    print("movies in the candidate pool:", summarise_counts([run["pool size"] for run in runs]))
    print("share of hidden liked ratings that are in the pool:", summarise([run["liked in pool"] for run in runs]))
