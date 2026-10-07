from pathlib import Path

import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "data"


def clean_genres(genres):
    if genres == "(no genres listed)":
        return ""
    return " ".join(g.replace("-", "").lower() for g in genres.split("|"))


def decade_token(title):
    year = title.strip()[-5:-1]
    if not year.isdigit():
        return ""
    return "decade" + year[:3] + "0s"


def build_movie_texts():
    movies = pd.read_csv(DATA / "movies.csv")
    tags = pd.read_csv(DATA / "tags.csv")

    movies["genres_text"] = movies["genres"].map(clean_genres)
    movies["decade_text"] = movies["title"].map(decade_token)

    tags["tag"] = tags["tag"].astype(str).str.lower().str.strip()
    tags_text = tags.groupby("movieId")["tag"].agg(" ".join).rename("tags_text")
    movies = movies.merge(tags_text, on="movieId", how="left")
    movies["tags_text"] = movies["tags_text"].fillna("")

    movies["text"] = (
        movies["genres_text"] + " " + movies["decade_text"] + " " + movies["tags_text"]
    ).str.split().str.join(" ")
    return movies


if __name__ == "__main__":
    texts = build_movie_texts()
    print(texts[["title", "text"]].head(8).to_string(index=False))
    print("movies with an empty text:", (texts["text"] == "").sum())
    print("movies with no year in the title:", (texts["decade_text"] == "").sum())
