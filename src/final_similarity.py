import numpy as np

from similarity_eval import N_SEEDS, combine, evaluate, movies, summarise, tag_matrix

FINAL = "genres and decade"

if __name__ == "__main__":
    X_final = combine(tag_matrix(movies["tags_text"]), 0)
    results, baselines = evaluate({FINAL: X_final})

    mine = np.array(results[FINAL]["test"])
    print(f"Average rating gap between a movie and its top 5 matches, test movies, {N_SEEDS} shuffles")
    print("Lower is better. Shown as mean (lowest to highest shuffle).\n")
    print(f"{'partners':<24}{'gap in stars':<26}{'my gap is lower in'}")
    print(f"{FINAL:<24}{summarise(mine):<26}")
    for name, scores in baselines.items():
        other = np.array(scores["test"])
        print(f"{name:<24}{summarise(other):<26}{int((mine < other).sum())} of {N_SEEDS} shuffles")

    print("\nsame test movies, average difference in gap (baseline minus mine):")
    for name, scores in baselines.items():
        print(f"  {name:<22}{summarise(np.array(scores['test']) - mine)}")

    print("\nshare of match pairs with at least 5 shared raters:", summarise(results[FINAL]["usable"]))
