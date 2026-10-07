from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
PLOTS = ROOT / "plots"
PLOTS.mkdir(exist_ok=True)

ratings = pd.read_csv(DATA / "ratings.csv")
movies = pd.read_csv(DATA / "movies.csv")
tags = pd.read_csv(DATA / "tags.csv")

print("--- first look ---")
print(ratings.head())
print("ratings shape:", ratings.shape)
print(movies.head())
print("movies shape:", movies.shape)

n_ratings = len(ratings)
n_users = ratings["userId"].nunique()
n_movies_rated = ratings["movieId"].nunique()
n_movies_total = len(movies)

print("\n--- basics ---")
print("ratings:", n_ratings)
print("users:", n_users)
print("movies in movies.csv:", n_movies_total)
print("movies with at least one rating:", n_movies_rated)
print("average rating:", round(ratings["rating"].mean(), 3))

star_counts = ratings["rating"].value_counts().sort_index()
print("\n--- how often each star value appears ---")
print(star_counts)

fig, ax = plt.subplots()
ax.bar(star_counts.index.astype(str), star_counts.values)
ax.set_xlabel("Stars")
ax.set_ylabel("Number of ratings")
ax.set_title("How often each star value is given")
fig.savefig(PLOTS / "star_counts.png", dpi=150, bbox_inches="tight")

per_movie = ratings.groupby("movieId").size().rename("n_ratings")
top_movies = (
    per_movie.sort_values(ascending=False)
    .head(10)
    .reset_index()
    .merge(movies[["movieId", "title"]], on="movieId")
)
print("\n--- 10 most rated movies ---")
print(top_movies[["title", "n_ratings"]].to_string(index=False))

per_user = ratings.groupby("userId").size().rename("n_ratings")
print("\n--- 10 most active users ---")
print(per_user.sort_values(ascending=False).head(10))
print("fewest ratings by any user:", per_user.min())
print("median ratings per user:", per_user.median())
print("median ratings per movie:", per_movie.median())
print("movies with only 1 rating:", (per_movie == 1).sum())

possible = n_users * n_movies_rated
print("\n--- sparsity ---")
print("possible user-movie pairs:", possible)
print("filled in: {:.2f}%".format(100 * n_ratings / possible))

tagged_movies = tags["movieId"].nunique()
print("\n--- tags ---")
print("tag rows:", len(tags))
print("movies with at least one tag:", tagged_movies)
print("share of all movies: {:.1f}%".format(100 * tagged_movies / n_movies_total))
