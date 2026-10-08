import numpy as np

from baselines import summarise
from hit_rate import METHODS, N_SEEDS, evaluate_seed
from split import load_ratings

N_MOST_RATED = 100
HIDDEN = "hidden liked movies"


def seed_stats(run):
    popularity = run["popularity"]
    cutoff = np.sort(popularity)[-N_MOST_RATED]

    stats = {}
    for name in METHODS:
        picks = run["picks"][name]
        counts = popularity[picks]
        stats[name] = {
            "median": np.median(counts),
            "share": np.mean(counts >= cutoff),
            "coverage": len(np.unique(picks)) / len(popularity),
        }
    wanted = run["wanted popularity"]
    stats[HIDDEN] = {"median": np.median(wanted), "share": np.mean(wanted >= cutoff), "coverage": np.nan}
    return stats, cutoff


def summarise_counts(values):
    return f"{np.mean(values):.0f} ({np.min(values):.0f} to {np.max(values):.0f})"


def summarise_share(values):
    return f"{np.mean(values):.1%} ({np.min(values):.1%} to {np.max(values):.1%})"


if __name__ == "__main__":
    ratings = load_ratings()

    all_stats, cutoffs = [], []
    for seed in range(N_SEEDS):
        stats, cutoff = seed_stats(evaluate_seed(ratings, seed))
        all_stats.append(stats)
        cutoffs.append(cutoff)
        print("finished seed", seed, flush=True)

    print(f"\nHow well known are the top 10 movies? {N_SEEDS} shuffles, mean (lowest to highest)")
    print(f"'most rated' means the {N_MOST_RATED} movies with the most training ratings "
          f"(at least {summarise_counts(cutoffs)} ratings)\n")
    print(f"{'list':<24}{'median ratings per movie':<28}{'share in the 100 most rated':<32}{'share of pool ever recommended'}")
    for name in METHODS + [HIDDEN]:
        medians = [stats[name]["median"] for stats in all_stats]
        shares = [stats[name]["share"] for stats in all_stats]
        coverage = [stats[name]["coverage"] for stats in all_stats]
        coverage_text = "n/a" if name == HIDDEN else summarise_share(coverage)
        print(f"{name:<24}{summarise_counts(medians):<28}{summarise_share(shares):<32}{coverage_text}")
