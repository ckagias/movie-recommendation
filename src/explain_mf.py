import time

import numpy as np

from baselines import MAX_RATING, MIN_RATING
from explain_content import LIKED_FROM, genre_sets, join_words, movies, ratings, row_of
from final_score import N_EPOCHS, N_FACTORS, REGULARIZATION
from mf import LEARNING_RATE, fit

SEED = 0
MIN_RATINGS = 20
N_NEIGHBOURS = 3

title_of = dict(zip(movies["movieId"], movies["title"]))
n_ratings = ratings.groupby("movieId").size()


def fit_final_model():
    model, _ = fit(ratings, N_FACTORS, LEARNING_RATE, N_EPOCHS, SEED, regularization=REGULARIZATION)
    model["movie_ids"] = np.fromiter(model["movie_index"], dtype=int, count=len(model["movie_index"]))
    return model


def score_parts(model, user_id, m):
    u = model["user_index"][user_id]
    return {
        "average": model["mu"],
        "user": model["user_bias"][u],
        "movie": model["movie_bias"][m],
        "taste": model["P"][u] @ model["Q"][m],
    }


def recommend_mf(model, user_id, n=10):
    u = model["user_index"][user_id]
    scores = (
        model["mu"] + model["user_bias"][u] + model["movie_bias"] + model["Q"] @ model["P"][u]
    )
    seen = ratings.loc[ratings["userId"] == user_id, "movieId"]
    scores[[model["movie_index"][movie_id] for movie_id in seen]] = -np.inf
    scores[model["movie_ids"].searchsorted(n_ratings.index[n_ratings < MIN_RATINGS])] = -np.inf
    top = np.argsort(-scores)[:n]
    return [(model["movie_ids"][m], m, float(np.clip(scores[m], MIN_RATING, MAX_RATING))) for m in top]


def nearest_liked(model, user_id, m, k=N_NEIGHBOURS):
    mine = ratings[(ratings["userId"] == user_id) & (ratings["rating"] >= LIKED_FROM)]
    liked_ids = mine["movieId"].to_numpy()
    liked = np.array([model["movie_index"][movie_id] for movie_id in liked_ids])

    Q = model["Q"]
    similarity = (Q[liked] @ Q[m]) / (np.linalg.norm(Q[liked], axis=1) * np.linalg.norm(Q[m]))
    best = np.argsort(-similarity)[:k]
    return [(liked_ids[b], mine["rating"].iloc[b], similarity[b]) for b in best]


def explain_mf(model, user_id, movie_id):
    m = model["movie_index"][movie_id]
    parts = score_parts(model, user_id, m)
    lines = [
        "Prediction built from: "
        f"overall average {parts['average']:.2f}, your generosity {parts['user']:+.2f}, "
        f"this movie's pull {parts['movie']:+.2f}, taste match {parts['taste']:+.2f}"
    ]
    lines.append("Its hidden numbers are closest to these movies you rated highly:")
    for liked_id, rating, similarity in nearest_liked(model, user_id, m):
        shared = genre_sets[row_of[movie_id]] & genre_sets[row_of[liked_id]]
        shared_text = f", shares {join_words(sorted(shared))}" if shared else ", shares no genre"
        lines.append(f"    {title_of[liked_id]} (you rated {rating}, similarity {similarity:.2f}{shared_text})")
    return "\n".join(lines)


def nearest_movies(model, movie_id, k=5):
    Q = model["Q"]
    popular = n_ratings.reindex(model["movie_ids"]).to_numpy() >= 50
    m = model["movie_index"][movie_id]
    similarity = (Q @ Q[m]) / (np.linalg.norm(Q, axis=1) * np.linalg.norm(Q[m]))
    similarity[~popular] = -np.inf
    similarity[m] = -np.inf
    return [(title_of[model["movie_ids"][j]], similarity[j]) for j in np.argsort(-similarity)[:k]]


if __name__ == "__main__":
    start = time.time()
    model = fit_final_model()
    print(f"fitted on all {len(ratings)} ratings in {time.time() - start:.0f} seconds")

    print("\nsanity check: movies with the closest hidden numbers (at least 50 ratings)")
    for title in ["Toy Story (1995)", "Matrix, The (1999)", "Pulp Fiction (1994)"]:
        movie_id = int(movies.loc[movies["title"] == title, "movieId"].iloc[0])
        near = ", ".join(f"{t} ({s:.2f})" for t, s in nearest_movies(model, movie_id))
        print(f"  {title}: {near}")

    for user_id in [1, 3, 414]:
        print(f"\n=== user {user_id} ===")
        for movie_id, m, predicted in recommend_mf(model, user_id, 4):
            print(f"{title_of[movie_id]}  (predicted {predicted:.2f}, {n_ratings[movie_id]} ratings)")
            print("    " + explain_mf(model, user_id, movie_id).replace("\n", "\n    "))
