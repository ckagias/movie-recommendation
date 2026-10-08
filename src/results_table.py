import numpy as np

from baselines import MODELS, rmse, summarise
from final_score import MF_NAME, N_EPOCHS, N_FACTORS, REGULARIZATION
from hit_rate import N_SEEDS, evaluate_seed
from mf import LEARNING_RATE, fit, predict
from similarity_eval import combine, evaluate, movies, tag_matrix
from split import load_ratings, split_ratings

BASELINE_1, BASELINE_2, BASELINE_3 = MODELS
SAME_AS_2 = "same as baseline 2"
NOT_APPLICABLE = "n/a"


def per_seed(runs, kind, name):
    return [run[kind][name] for run in runs]


def hit_columns(runs, name):
    return summarise(per_seed(runs, "hit rate", name)), summarise(per_seed(runs, "recall", name))


def print_row(*cells):
    print("| " + " | ".join(cells) + " |")


if __name__ == "__main__":
    ratings = load_ratings()

    rmse_scores = {name: [] for name in list(MODELS) + [MF_NAME]}
    hit_runs = []
    for seed in range(N_SEEDS):
        train, test = split_ratings(ratings, seed)
        actual = test["rating"].to_numpy()

        model, _ = fit(train, N_FACTORS, LEARNING_RATE, N_EPOCHS, seed, regularization=REGULARIZATION)
        predictions = {name: predict_fn(train, test) for name, predict_fn in MODELS.items()}
        predictions[MF_NAME] = predict(model, test)
        for name, values in predictions.items():
            rmse_scores[name].append(rmse(values, actual))

        hit_runs.append(evaluate_seed(ratings, seed, model))
        print("finished seed", seed, flush=True)

    X_final = combine(tag_matrix(movies["tags_text"]), 0)
    similarity, similarity_baselines = evaluate({"content-based": X_final})
    content_gap = np.array(similarity["content-based"]["test"])

    print(f"\nPrediction and ranking, test pile, {N_SEEDS} shuffles, mean (lowest to highest)\n")
    print_row("Model", "RMSE in stars (lower is better)", "Hit rate at 10 (higher is better)", "Recall at 10")
    print_row("---", "---", "---", "---")
    print_row("1. Overall average", summarise(rmse_scores[BASELINE_1]), NOT_APPLICABLE, NOT_APPLICABLE)
    print_row("2. Movie average", summarise(rmse_scores[BASELINE_2]), *hit_columns(hit_runs, "highest movie average"))
    print_row("3. Movie average + user bias", summarise(rmse_scores[BASELINE_3]), SAME_AS_2, SAME_AS_2)
    print_row("Content-based (genres and decade)", NOT_APPLICABLE, *hit_columns(hit_runs, "content-based"))
    print_row("Matrix factorization", summarise(rmse_scores[MF_NAME]), *hit_columns(hit_runs, "matrix factorization"))
    print_row("Reference: random list", NOT_APPLICABLE, *hit_columns(hit_runs, "random"))
    print_row("Reference: most rated movies", NOT_APPLICABLE, *hit_columns(hit_runs, "most rated"))

    print(f"\nSimilarity test, rating gap of the top 5 matches, {N_SEEDS} shuffles (lower is better)\n")
    print_row("Partners", "Gap in stars", "Content-based is lower in")
    print_row("---", "---", "---")
    print_row("Content-based (genres and decade)", summarise(content_gap), "")
    for name, scores in similarity_baselines.items():
        other = np.array(scores["test"])
        print_row(name.capitalize(), summarise(other), f"{int((content_gap < other).sum())} of {N_SEEDS} shuffles")

    rmse_gain = np.array(rmse_scores[BASELINE_3]) - np.array(rmse_scores[MF_NAME])
    hit_gain = np.array(per_seed(hit_runs, "hit rate", "matrix factorization")) - np.array(
        per_seed(hit_runs, "hit rate", "most rated")
    )
    print("\nmatrix factorization against baseline 3 on RMSE, wins:", int((rmse_gain > 0).sum()), "of", N_SEEDS)
    print("matrix factorization against most rated on hit rate, wins:", int((hit_gain > 0).sum()), "of", N_SEEDS)
