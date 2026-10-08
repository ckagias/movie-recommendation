import argparse
import pickle

from explain_content import ratings, recommend
from explain_mf import explain_mf, fit_final_model, n_ratings, recommend_mf, title_of
from movie_text import DATA

MODEL_PATH = DATA / "mf_model.pkl"
N_SHOWN_FAVOURITES = 5


def load_model(refit):
    if MODEL_PATH.exists() and not refit:
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f)
    print("Fitting the matrix factorization model (about 45 seconds, saved for next time)...")
    model = fit_final_model()
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    return model


def indent(text, spaces=5):
    return text.replace("\n", "\n" + " " * spaces)


def print_profile(user_id):
    mine = ratings[ratings["userId"] == user_id]
    favourites = mine.sort_values("rating", ascending=False, kind="stable").head(N_SHOWN_FAVOURITES)
    print(f"User {user_id}: {len(mine)} ratings, average {mine['rating'].mean():.2f}")
    print("Some of the movies this user rated highest:")
    for movie_id, rating in zip(favourites["movieId"], favourites["rating"]):
        print(f"  {rating}  {title_of[movie_id]}")


def print_content_based(user_id, n):
    print(f"\n=== Top {n}, content-based: movies that look like ones this user liked ===")
    for rank, row in enumerate(recommend(user_id, n).itertuples(), start=1):
        print(f"{rank:>2}. {row.title}\n     {row.why}")


def print_matrix_factorization(model, user_id, n):
    print(f"\n=== Top {n}, matrix factorization: learned from everyone's ratings ===")
    for rank, (movie_id, _, predicted) in enumerate(recommend_mf(model, user_id, n), start=1):
        print(f"{rank:>2}. {title_of[movie_id]}  (predicted {predicted:.2f}, {n_ratings[movie_id]} ratings)")
        print("     " + indent(explain_mf(model, user_id, movie_id)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Top movies for one MovieLens user, with the reasons.")
    parser.add_argument("user", type=int, help="user id, 1 to 610")
    parser.add_argument("-n", type=int, default=10, help="how many movies to show per method")
    parser.add_argument("--refit", action="store_true", help="train the model again instead of loading the saved one")
    args = parser.parse_args()

    if args.user not in set(ratings["userId"]):
        parser.error(f"user {args.user} not found, use an id from 1 to {ratings['userId'].max()}")

    model = load_model(args.refit)
    print_profile(args.user)
    print_content_based(args.user, args.n)
    print_matrix_factorization(model, args.user, args.n)
