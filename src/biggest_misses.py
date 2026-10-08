import numpy as np
import pandas as pd

from baselines import summarise
from final_score import N_EPOCHS, N_FACTORS, REGULARIZATION
from mf import LEARNING_RATE, fit, predict
from split import DATA, load_ratings, split_ratings

N_SEEDS = 10
N_LISTED = 10
WORST_SHARE = 0.01

PROFILE_ROWS = [
    "model guessed too high",
    "actual rating 1.5 or lower",
    "actual rating 4.5 or higher",
    "movie has no training rating",
    "movie has fewer than 5 training ratings",
    "median training ratings of the movie",
    "median training ratings of the user",
    "gap between actual and the user's own average",
]


def build_table(train, test, predictions):
    table = test.copy()
    table["predicted"] = predictions
    table["error"] = predictions - test["rating"].to_numpy()
    table["miss"] = table["error"].abs()

    movie_n = train.groupby("movieId").size()
    user_n = train.groupby("userId").size()
    user_mean = train.groupby("userId")["rating"].mean()
    table["movie_n"] = table["movieId"].map(movie_n).fillna(0)
    table["user_n"] = table["userId"].map(user_n)
    table["user_mean"] = table["userId"].map(user_mean)
    return table


def profile(table):
    return {
        PROFILE_ROWS[0]: (table["error"] > 0).mean(),
        PROFILE_ROWS[1]: (table["rating"] <= 1.5).mean(),
        PROFILE_ROWS[2]: (table["rating"] >= 4.5).mean(),
        PROFILE_ROWS[3]: (table["movie_n"] == 0).mean(),
        PROFILE_ROWS[4]: (table["movie_n"] < 5).mean(),
        PROFILE_ROWS[5]: table["movie_n"].median(),
        PROFILE_ROWS[6]: table["user_n"].median(),
        PROFILE_ROWS[7]: (table["rating"] - table["user_mean"]).abs().mean(),
    }


def print_worst(table, movies):
    worst = table.sort_values("miss", ascending=False).head(N_LISTED)
    worst = worst.merge(movies, on="movieId")
    worst = worst.sort_values("miss", ascending=False)
    print(f"\nThe {N_LISTED} biggest misses of matrix factorization on shuffle 0\n")
    print(f"{'user':<6}{'movie':<46}{'actual':<8}{'guess':<7}{'movie train n':<15}{'user avg':<10}{'user n'}")
    for row in worst.itertuples():
        print(
            f"{row.userId:<6}{row.title[:44]:<46}{row.rating:<8}{row.predicted:<7.2f}"
            f"{row.movie_n:<15.0f}{row.user_mean:<10.2f}{row.user_n:.0f}"
        )
        print(f"      genres: {row.genres}")


if __name__ == "__main__":
    ratings = load_ratings()
    movies = pd.read_csv(DATA / "movies.csv")

    worst_profiles, all_profiles, thresholds, error_shares = [], [], [], []
    for seed in range(N_SEEDS):
        train, test = split_ratings(ratings, seed)
        model, _ = fit(train, N_FACTORS, LEARNING_RATE, N_EPOCHS, seed, regularization=REGULARIZATION)
        table = build_table(train, test, predict(model, test))

        n_worst = int(np.ceil(WORST_SHARE * len(table)))
        worst = table.sort_values("miss", ascending=False).head(n_worst)
        thresholds.append(worst["miss"].min())
        error_shares.append((worst["miss"] ** 2).sum() / (table["miss"] ** 2).sum())
        worst_profiles.append(profile(worst))
        all_profiles.append(profile(table))

        if seed == 0:
            print_worst(table, movies)
        print("finished seed", seed, flush=True)

    print(f"\nWorst {WORST_SHARE:.0%} of misses against all test ratings, {N_SEEDS} shuffles, mean (lowest to highest)")
    print(f"the worst 1% are misses of at least {summarise(thresholds)} stars\n")
    print(f"the worst 1% make up {summarise(error_shares)} of the total squared error\n")
    print(f"{'':<50}{'worst 1%':<26}{'all test ratings'}")
    for name in PROFILE_ROWS:
        worst_values = [p[name] for p in worst_profiles]
        all_values = [p[name] for p in all_profiles]
        print(f"{name:<50}{summarise(worst_values):<26}{summarise(all_values)}")
