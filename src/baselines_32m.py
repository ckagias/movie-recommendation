import numpy as np
from scipy.sparse import csr_matrix

from scale_check import train_matrix
from split_32m import PILE_NAMES, TEST, TRAIN, VALIDATION, split

LIKED_FROM = 4.0
TOP_N = 10
MIN_TRAIN_RATINGS = 20
N_SAMPLE_USERS = 20000
N_BOOTSTRAP = 1000
MIN_RATING = 0.5
MAX_RATING = 5.0
SEED = 0

HIT_METHODS = ["random", "most rated", "highest movie average"]


def interval(low, high):
    return f"{low:.3f} to {high:.3f}"


def bootstrap_mean(values, rng):
    picks = rng.integers(0, len(values), (N_BOOTSTRAP, len(values)))
    means = values[picks].mean(axis=1)
    return np.percentile(means, [2.5, 97.5])


def bootstrap_rmse(per_user_squared_error, per_user_count, rng):
    n_users = len(per_user_count)
    low_high = []
    scores = np.empty(N_BOOTSTRAP)
    for b in range(N_BOOTSTRAP):
        weights = np.bincount(rng.integers(0, n_users, n_users), minlength=n_users)
        scores[b] = np.sqrt((weights * per_user_squared_error).sum() / (weights * per_user_count).sum())
    return np.percentile(scores, [2.5, 97.5])


def rating_error_baselines(user, movie, rating, pile, rng):
    user_ids, u = np.unique(user, return_inverse=True)
    movie_ids, m = np.unique(movie, return_inverse=True)
    rating = rating.astype(np.float64)
    train = pile == TRAIN

    overall = rating[train].mean()
    movie_count = np.bincount(m[train], minlength=len(movie_ids))
    movie_total = np.bincount(m[train], weights=rating[train], minlength=len(movie_ids))
    movie_mean = np.where(movie_count > 0, movie_total / np.maximum(movie_count, 1), overall)

    residual = rating[train] - movie_mean[m[train]]
    user_count = np.bincount(u[train], minlength=len(user_ids))
    user_bias = np.bincount(u[train], weights=residual, minlength=len(user_ids)) / user_count

    def predict(rows):
        by_movie = movie_mean[m[rows]]
        return {
            "baseline 1, overall average": np.full(rows.sum(), overall),
            "baseline 2, movie average": by_movie,
            "baseline 3, movie avg + user bias": np.clip(by_movie + user_bias[u[rows]], MIN_RATING, MAX_RATING),
        }

    print(f"{'model':<36}{'pile':<12}{'rmse':<10}{'95% interval over users'}")
    for code in (VALIDATION, TEST):
        rows = pile == code
        for name, predictions in predict(rows).items():
            squared = (predictions - rating[rows]) ** 2
            per_user_error = np.bincount(u[rows], weights=squared, minlength=len(user_ids))
            per_user_count = np.bincount(u[rows], minlength=len(user_ids))
            score = np.sqrt(squared.mean())
            low, high = bootstrap_rmse(per_user_error, per_user_count, rng)
            print(f"{name:<36}{PILE_NAMES[code]:<12}{score:<10.3f}{interval(low, high)}")

    unseen = movie_count[m[pile == TEST]] == 0
    print(f"\ntest ratings on movies with no train rating: {unseen.sum()} ({unseen.mean():.2%})")


def held_out_liked(user, movie, rating, pile, code, user_ids, movie_ids, position):
    rows = (pile == code) & (rating >= LIKED_FROM)
    users = np.searchsorted(user_ids, user[rows])
    movies = np.searchsorted(movie_ids, np.minimum(movie[rows], movie_ids[-1]))
    known = movie_ids[movies] == movie[rows]
    in_pool = np.zeros(len(users), dtype=bool)
    in_pool[known] = position[movies[known]] >= 0
    shape = (len(user_ids), int((position >= 0).sum()))
    matrix = csr_matrix(
        (np.ones(in_pool.sum(), dtype=np.int8), (users[in_pool], position[movies[in_pool]])), shape=shape
    )
    return matrix, in_pool.mean()


def hit_rate(user, movie, rating, pile, rng):
    matrix, user_ids, movie_ids = train_matrix(user, movie, pile)
    train = pile == TRAIN
    movie_of_train = np.searchsorted(movie_ids, movie[train])

    counts = matrix.getnnz(axis=0)
    pool_cols = np.flatnonzero(counts >= MIN_TRAIN_RATINGS)
    position = np.full(len(movie_ids), -1)
    position[pool_cols] = np.arange(len(pool_cols))
    popularity = counts[pool_cols]
    totals = np.bincount(movie_of_train, weights=rating[train].astype(np.float64), minlength=len(movie_ids))
    movie_means = (totals / np.maximum(counts, 1))[pool_cols]

    order = {
        "most rated": np.argsort(-popularity, kind="stable"),
        "highest movie average": np.lexsort((-popularity, -movie_means)),
    }
    print(f"\ncandidate pool: {len(pool_cols)} movies with at least {MIN_TRAIN_RATINGS} train ratings (of {len(movie_ids)})")

    sample = rng.choice(len(user_ids), N_SAMPLE_USERS, replace=False)
    seen_by_user = {}
    for row in sample:
        columns = matrix.indices[matrix.indptr[row] : matrix.indptr[row + 1]]
        seen_by_user[row] = position[columns][position[columns] >= 0]

    for code in (VALIDATION, TEST):
        liked, share_in_pool = held_out_liked(user, movie, rating, pile, code, user_ids, movie_ids, position)
        flags = {name: [] for name in HIT_METHODS}
        recalls = {name: [] for name in HIT_METHODS}
        for row in sample:
            wanted = liked.indices[liked.indptr[row] : liked.indptr[row + 1]]
            if len(wanted) == 0:
                continue
            seen = np.zeros(len(pool_cols), dtype=bool)
            seen[seen_by_user[row]] = True
            for name in HIT_METHODS:
                if name == "random":
                    shown = rng.choice(np.flatnonzero(~seen), TOP_N, replace=False)
                else:
                    ranked = order[name]
                    shown = ranked[~seen[ranked]][:TOP_N]
                hits = np.isin(shown, wanted).sum()
                flags[name].append(hits > 0)
                recalls[name].append(hits / len(wanted))

        print(f"\n{PILE_NAMES[code]} pile: {len(flags['random'])} of {N_SAMPLE_USERS} sampled users have a liked rating in the pool")
        print(f"share of held-out liked ratings that are in the pool: {share_in_pool:.1%}")
        print(f"{'method':<26}{'hit rate at 10':<16}{'95% interval over users':<28}{'recall at 10'}")
        for name in HIT_METHODS:
            values = np.array(flags[name], dtype=float)
            low, high = bootstrap_mean(values, rng)
            print(f"{name:<26}{values.mean():<16.3f}{interval(low, high):<28}{np.mean(recalls[name]):.3f}")


if __name__ == "__main__":
    user, movie, rating, pile = split()
    rng = np.random.default_rng(SEED)
    rating_error_baselines(user, movie, rating, pile, rng)
    hit_rate(user, movie, rating, pile, rng)
