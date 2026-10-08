import os

for name in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ.setdefault(name, "1")

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from baselines import rmse, summarise
from final_score import N_EPOCHS, N_FACTORS, REGULARIZATION
from mf import LEARNING_RATE, fit, predict
from movie_text import build_movie_texts
from split import load_ratings, split_ratings, split_validation

N_SEEDS = 10
N_WORKERS = 10
RELIABLE = 20
POWERS = [2, 8, 32]
STRENGTHS = [1, 3, 10, 30]
GRID = [(power, strength) for power in POWERS for strength in STRENGTHS]
MOVIE_BUCKETS = [(0, 0), (1, 1), (2, 4), (5, 9), (10, 19), (20, 10**9)]

MF_NAME = "matrix factorization"
VARIANTS = {
    "content prior, bias only": False,
    "content prior, bias and hidden numbers": True,
}

movies = build_movie_texts()
movie_ids = movies["movieId"].to_numpy()
X = TfidfVectorizer().fit_transform(movies["genres_text"] + " " + movies["decade_text"])


def bucket_label(lo, hi):
    if lo == hi:
        return str(lo)
    return f"{lo}+" if hi >= 10**9 else f"{lo} to {hi}"


def with_content_prior(model, train, power, strength, use_factors):
    counts = train.groupby("movieId").size().reindex(movie_ids).fillna(0).to_numpy()
    reliable = np.flatnonzero(counts >= RELIABLE)
    target = np.flatnonzero(counts < RELIABLE)

    reliable_pos = np.array([model["movie_index"][i] for i in movie_ids[reliable]])
    weights = cosine_similarity(X[target], X[reliable]) ** power
    totals = weights.sum(axis=1, keepdims=True)
    empty = totals[:, 0] == 0
    weights[empty] = 1.0
    weights /= weights.sum(axis=1, keepdims=True)
    prior_bias = weights @ model["movie_bias"][reliable_pos]
    prior_q = weights @ model["Q"][reliable_pos]

    n = counts[target]
    own_pos = np.array([model["movie_index"].get(i, -1) for i in movie_ids[target]])
    has_own = own_pos >= 0
    own_bias = np.zeros(len(target))
    own_q = np.zeros((len(target), model["Q"].shape[1]))
    own_bias[has_own] = model["movie_bias"][own_pos[has_own]]
    own_q[has_own] = model["Q"][own_pos[has_own]]

    own_share = n / (n + strength)
    new_bias = own_share * own_bias + (1 - own_share) * prior_bias
    new_q = own_share[:, None] * own_q + (1 - own_share[:, None]) * prior_q if use_factors else own_q

    movie_index = dict(model["movie_index"])
    movie_bias, Q = model["movie_bias"].copy(), model["Q"].copy()
    new_rows = np.flatnonzero(~has_own)
    movie_bias = np.concatenate([movie_bias, np.zeros(len(new_rows))])
    Q = np.vstack([Q, np.zeros((len(new_rows), Q.shape[1]))])
    for offset, row in enumerate(new_rows):
        own_pos[row] = len(model["movie_bias"]) + offset
        movie_index[movie_ids[target[row]]] = own_pos[row]
    movie_bias[own_pos] = new_bias
    Q[own_pos] = new_q
    return {**model, "movie_index": movie_index, "movie_bias": movie_bias, "Q": Q}


def fit_final(train, seed):
    model, _ = fit(train, N_FACTORS, LEARNING_RATE, N_EPOCHS, seed, regularization=REGULARIZATION)
    return model


def validation_rmse(model, fit_pile, validation, power, strength, use_factors):
    counts = validation["movieId"].map(fit_pile.groupby("movieId").size()).fillna(0)
    rare = (counts < RELIABLE).to_numpy()
    patched = with_content_prior(model, fit_pile, power, strength, use_factors)
    return rmse(predict(patched, validation[rare]), validation[rare]["rating"].to_numpy())


def run_seed(seed):
    train, test = split_ratings(load_ratings(), seed)
    inner, validation = split_validation(train, seed)

    inner_model = fit_final(inner, seed)
    chosen = {}
    for name, use_factors in VARIANTS.items():
        scores = {cfg: validation_rmse(inner_model, inner, validation, *cfg, use_factors) for cfg in GRID}
        chosen[name] = min(scores, key=scores.get)

    model = fit_final(train, seed)
    predictions = {MF_NAME: predict(model, test)}
    for name, use_factors in VARIANTS.items():
        patched = with_content_prior(model, train, *chosen[name], use_factors)
        predictions[name] = predict(patched, test)

    actual = test["rating"].to_numpy()
    movie_n = test["movieId"].map(train.groupby("movieId").size()).fillna(0).to_numpy()
    buckets = {}
    for lo, hi in MOVIE_BUCKETS:
        mask = (movie_n >= lo) & (movie_n <= hi)
        buckets[(lo, hi)] = {"share": mask.mean()}
        buckets[(lo, hi)].update({name: rmse(p[mask], actual[mask]) for name, p in predictions.items()})
    overall = {name: rmse(p, actual) for name, p in predictions.items()}
    return seed, chosen, buckets, overall


def diagnose(seed):
    train, test = split_ratings(load_ratings(), seed)
    model = fit_final(train, seed)
    patched = with_content_prior(model, train, 8, 1, False)
    movie_n = test["movieId"].map(train.groupby("movieId").size()).fillna(0).to_numpy()
    new = test[movie_n == 0]

    users = new["userId"].map(model["user_index"]).to_numpy()
    residual = new["rating"].to_numpy() - (model["mu"] + model["user_bias"][users])
    prior = new["movieId"].map(patched["movie_index"]).map(lambda i: patched["movie_bias"][i]).to_numpy()
    reliable_bias = model["movie_bias"][[model["movie_index"][i] for i in movie_ids[movies_reliable(train)]]]

    print(f"shuffle {seed}: {len(new)} test ratings of {new['movieId'].nunique()} movies with no training rating")
    print(f"mean of rating minus (average + user bias): {residual.mean():.3f}")
    print(f"mean content prior bias given to those movies (power 8, strength 1): {prior.mean():.3f}")
    print(f"mean movie bias of the movies with {RELIABLE}+ training ratings: {reliable_bias.mean():.3f}")
    print(f"correlation of the prior with the rating residual: {np.corrcoef(prior, residual)[0, 1]:.3f}")
    print(f"rmse of the residual around 0 / around its mean: {np.sqrt(np.mean(residual ** 2)):.3f} / {residual.std():.3f}")


def movies_reliable(train):
    counts = train.groupby("movieId").size().reindex(movie_ids).fillna(0).to_numpy()
    return np.flatnonzero(counts >= RELIABLE)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--diagnose", type=int, metavar="SEED", help="check the offset of the prior on one shuffle")
    args = parser.parse_args()
    if args.diagnose is not None:
        diagnose(args.diagnose)
        raise SystemExit

    runs = {}
    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        futures = [pool.submit(run_seed, seed) for seed in range(N_SEEDS)]
        for done, future in enumerate(as_completed(futures), start=1):
            seed, chosen, buckets, overall = future.result()
            runs[seed] = (chosen, buckets, overall)
            print(f"finished {done} of {N_SEEDS}", flush=True)

    names = [MF_NAME] + list(VARIANTS)
    print(f"\nRMSE by how many training ratings the test movie has, {N_SEEDS} shuffles")
    print(f"{'training ratings':<18}{'share':<8}" + "".join(f"{name:<42}" for name in names))
    for bucket in MOVIE_BUCKETS:
        share = np.mean([run[1][bucket]["share"] for run in runs.values()])
        cells = "".join(f"{summarise([run[1][bucket][name] for run in runs.values()]):<42}" for name in names)
        print(f"{bucket_label(*bucket):<18}{share:<8.1%}{cells}")
    cells = "".join(f"{summarise([run[2][name] for run in runs.values()]):<42}" for name in names)
    print(f"{'all test ratings':<26}{cells}")

    print("\nper bucket, shuffles where the content prior beats plain matrix factorization:")
    for name in VARIANTS:
        wins = {
            bucket_label(*b): sum(runs[s][1][b][name] < runs[s][1][b][MF_NAME] for s in runs)
            for b in MOVIE_BUCKETS
        }
        print(f"  {name}: {wins}")

    print("\nchosen (power, strength) per shuffle:")
    for name in VARIANTS:
        print(f"  {name}: {[runs[s][0][name] for s in sorted(runs)]}")
