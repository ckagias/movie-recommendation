import os

for name in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ.setdefault(name, "1")

from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.stats import rankdata

from baselines import summarise
from explain_content import LIKED_FROM
from final_score import N_EPOCHS, N_FACTORS, REGULARIZATION
from hit_rate import MIN_TRAIN_RATINGS, N_SEEDS, top_n
from mf import LEARNING_RATE, fit
from split import load_ratings, split_ratings, split_validation

WEIGHTS = [round(0.1 * w, 1) for w in range(11)]
FILTER_SIZES = [25, 50, 100, 200, 400]
N_WORKERS = 10

CONFIGS = [("blend", w) for w in WEIGHTS] + [("filter", m) for m in FILTER_SIZES]


def config_name(kind, value):
    if kind == "blend":
        return f"blend, popularity weight {value}"
    return f"top {value} popular, then predicted rating"


def combined_score(kind, value, popularity_part, prediction_part, popularity_rank):
    if kind == "blend":
        return value * popularity_part + (1 - value) * prediction_part
    return np.where(popularity_rank <= value, prediction_part, prediction_part - 10.0)


def hit_rates(train, held_out, model):
    counts = train.groupby("movieId").size()
    counts = counts[counts >= MIN_TRAIN_RATINGS]
    pool_ids = counts.index.to_numpy()
    popularity = counts.to_numpy()
    position = {movie_id: p for p, movie_id in enumerate(pool_ids)}

    in_model = np.array([model["movie_index"][movie_id] for movie_id in pool_ids])
    pool_bias, pool_Q = model["movie_bias"][in_model], model["Q"][in_model]

    liked = held_out[held_out["rating"] >= LIKED_FROM]
    relevant = (
        liked[liked["movieId"].isin(position)]
        .groupby("userId")["movieId"]
        .agg(lambda ids: {position[m] for m in ids})
    )
    train_by_user = dict(list(train.groupby("userId")))

    flags = {config: [] for config in CONFIGS}
    recalls = {config: [] for config in CONFIGS}
    for user_id, wanted in relevant.items():
        mine = train_by_user[user_id]
        seen = np.zeros(len(pool_ids), dtype=bool)
        seen[[position[m] for m in mine["movieId"] if m in position]] = True
        u = model["user_index"][user_id]
        predicted = model["mu"] + model["user_bias"][u] + pool_bias + pool_Q @ model["P"][u]

        candidates = ~seen
        popularity_part = np.zeros(len(pool_ids))
        prediction_part = np.zeros(len(pool_ids))
        popularity_rank = np.full(len(pool_ids), np.inf)
        n_candidates = candidates.sum()
        popularity_part[candidates] = rankdata(popularity[candidates]) / n_candidates
        prediction_part[candidates] = rankdata(predicted[candidates]) / n_candidates
        popularity_rank[candidates] = rankdata(-popularity[candidates], method="min")

        for kind, value in CONFIGS:
            score = combined_score(kind, value, popularity_part, prediction_part, popularity_rank)
            hits = len(set(top_n(score, seen, popularity)) & wanted)
            flags[(kind, value)].append(hits > 0)
            recalls[(kind, value)].append(hits / len(wanted))

    return (
        {config: np.mean(values) for config, values in flags.items()},
        {config: np.mean(values) for config, values in recalls.items()},
    )


def fit_final(train, seed):
    model, _ = fit(train, N_FACTORS, LEARNING_RATE, N_EPOCHS, seed, regularization=REGULARIZATION)
    return model


def run_seed(seed):
    train, test = split_ratings(load_ratings(), seed)
    inner, validation = split_validation(train, seed)

    validation_hits, _ = hit_rates(inner, validation, fit_final(inner, seed))
    chosen = max(CONFIGS, key=lambda config: validation_hits[config])

    test_hits, test_recalls = hit_rates(train, test, fit_final(train, seed))
    return seed, chosen, validation_hits, test_hits, test_recalls


if __name__ == "__main__":
    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        runs = sorted(pool.map(run_seed, range(N_SEEDS)))

    chosen = [run[1] for run in runs]
    validation_hits = [run[2] for run in runs]
    test_hits = [run[3] for run in runs]
    test_recalls = [run[4] for run in runs]

    print(f"Hit rate at 10, {N_SEEDS} shuffles, mean (lowest to highest)")
    print(f"{'setting':<44}{'validation':<26}{'test'}")
    for config in CONFIGS:
        validation_values = [hits[config] for hits in validation_hits]
        test_values = [hits[config] for hits in test_hits]
        print(f"{config_name(*config):<44}{summarise(validation_values):<26}{summarise(test_values)}")

    print("\nsetting chosen on the validation pile in each shuffle:")
    for seed, config in enumerate(chosen):
        print(f"  shuffle {seed}: {config_name(*config)}")

    hybrid = np.array([test_hits[i][config] for i, config in enumerate(chosen)])
    hybrid_recall = np.array([test_recalls[i][config] for i, config in enumerate(chosen)])
    most_rated = np.array([hits[("blend", 1.0)] for hits in test_hits])
    matrix_factorization = np.array([hits[("blend", 0.0)] for hits in test_hits])

    print("\nfinal test scores, setting chosen on validation")
    print(f"{'':<28}{'hit rate at 10':<26}{'recall at 10'}")
    print(f"{'hybrid':<28}{summarise(hybrid):<26}{summarise(hybrid_recall)}")
    print(f"{'most rated':<28}{summarise(most_rated)}")
    print(f"{'matrix factorization':<28}{summarise(matrix_factorization)}")
    print("\nhybrid minus most rated, per shuffle:", " ".join(f"{g:+.3f}" for g in hybrid - most_rated))
    print("shuffles where the hybrid is higher than most rated:", int((hybrid > most_rated).sum()), "of", N_SEEDS)
    print("hybrid minus matrix factorization, wins:", int((hybrid > matrix_factorization).sum()), "of", N_SEEDS)
