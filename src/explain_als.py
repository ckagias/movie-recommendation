import numpy as np

from als import CONFIDENCE, N_FACTORS, N_ITERATIONS, REGULARIZATION, fit, interaction_matrix
from explain_content import LIKED_FROM, genre_sets, join_words, ratings, row_of
from explain_mf import MIN_RATINGS, N_NEIGHBOURS, n_ratings, title_of
from mf import index_ids

SEED = 0


def fit_final_als():
    user_index, movie_index = index_ids(ratings["userId"]), index_ids(ratings["movieId"])
    liked = interaction_matrix(ratings, user_index, movie_index, LIKED_FROM)
    _, items, _ = fit(liked, N_FACTORS, REGULARIZATION, CONFIDENCE, N_ITERATIONS, SEED)
    return {
        "items": items,
        "items_gram": items.T @ items,
        "liked": liked,
        "user_index": user_index,
        "movie_index": movie_index,
        "movie_ids": np.fromiter(movie_index, dtype=int, count=len(movie_index)),
    }


def user_system(model, user_id):
    row = model["user_index"][user_id]
    liked = model["liked"]
    columns = liked.indices[liked.indptr[row] : liked.indptr[row + 1]]
    rated = model["items"][columns]
    matrix = model["items_gram"] + (CONFIDENCE - 1) * rated.T @ rated + REGULARIZATION * np.eye(N_FACTORS)
    return matrix, rated, columns


def user_vector(model, user_id):
    matrix, rated, _ = user_system(model, user_id)
    if len(rated) == 0:
        return np.zeros(N_FACTORS)
    return np.linalg.solve(matrix, CONFIDENCE * rated.sum(axis=0))


def recommend_als(model, user_id, n=10):
    scores = model["items"] @ user_vector(model, user_id)
    seen = ratings.loc[ratings["userId"] == user_id, "movieId"]
    scores[[model["movie_index"][movie_id] for movie_id in seen]] = -np.inf
    popularity = n_ratings.reindex(model["movie_ids"]).fillna(0).to_numpy()
    scores[popularity < MIN_RATINGS] = -np.inf
    top = np.lexsort((-popularity, -scores))[:n]
    return [(model["movie_ids"][m], m, float(scores[m])) for m in top]


def contributions(model, user_id, m):
    matrix, rated, columns = user_system(model, user_id)
    if len(rated) == 0:
        return np.array([]), columns
    return CONFIDENCE * (model["items"][m] @ np.linalg.solve(matrix, rated.T)), columns


def explain_als(model, user_id, movie_id):
    m = model["movie_index"][movie_id]
    parts, columns = contributions(model, user_id, m)
    if len(parts) == 0:
        return "This user has no rating of 4.0 or more, so there is nothing to build the score from."
    liked_ids = model["movie_ids"][columns]
    best = np.argsort(-parts)[:N_NEIGHBOURS]
    lines = [
        f"Score {parts.sum():.2f} is the sum of what each of the {len(parts)} movies you rated "
        f"{LIKED_FROM} or more adds. The biggest contributions:"
    ]
    for b in best:
        liked_id = liked_ids[b]
        shared = genre_sets[row_of[movie_id]] & genre_sets[row_of[liked_id]]
        shared_text = f", shares {join_words(sorted(shared))}" if shared else ", shares no genre"
        lines.append(f"    {title_of[liked_id]} (adds {parts[b]:+.3f}{shared_text})")
    rest = parts.sum() - parts[best].sum()
    lines.append(f"    the other {len(parts) - len(best)} together add {rest:+.3f}")
    return "\n".join(lines)


if __name__ == "__main__":
    model = fit_final_als()

    movie_id = recommend_als(model, 3, 1)[0][0]
    m = model["movie_index"][movie_id]
    parts, _ = contributions(model, 3, m)
    score = float(model["items"][m] @ user_vector(model, 3))
    print(f"contributions of the liked movies add up to {parts.sum():.6f}, the score is {score:.6f}")

    for user_id in [1, 3, 414]:
        print(f"\n=== user {user_id} ===")
        for movie_id, m, score in recommend_als(model, user_id, 4):
            print(f"{title_of[movie_id]}  (score {score:.2f}, {n_ratings[movie_id]} ratings)")
            print("    " + explain_als(model, user_id, movie_id).replace("\n", "\n    "))
