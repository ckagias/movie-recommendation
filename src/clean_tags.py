import numpy as np
import pandas as pd

from similarity_eval import DATA, combine, evaluate, movies, print_table, summarise, tag_matrix

TAG_WEIGHTS = [0.5, 1, 2, 4]
JUNK_TAGS = {"in netflix queue"}

tags = pd.read_csv(DATA / "tags.csv")
tags["tag"] = tags["tag"].astype(str).str.lower().str.strip()
tags["clean"] = (
    tags["tag"].str.replace(r"[^a-z0-9 ]+", " ", regex=True).str.split().str.join(" ")
)
tags = tags[tags["clean"] != ""]


def to_text(df, token_col):
    joined = df.groupby("movieId")[token_col].agg(" ".join)
    return movies["movieId"].map(joined).fillna("")


def as_phrase(df):
    return df.assign(token=df["clean"].str.replace(" ", "_"))


words = tags.assign(token=tags["clean"])
phrases = as_phrase(tags)
one_vote = phrases.drop_duplicates(["movieId", "token"])
movies_per_tag = one_vote.groupby("token")["movieId"].nunique()
not_rare = one_vote[one_vote["token"].map(movies_per_tag) >= 2]
no_junk = not_rare[~not_rare["clean"].isin(JUNK_TAGS)]

variants = {
    "1 current (words, repeats count)": movies["tags_text"],
    "2 + merge punctuation variants": to_text(words, "token"),
    "3 + keep phrases together": to_text(phrases, "token"),
    "4 + one vote per movie": to_text(one_vote, "token"),
    "5 + drop tags on a single movie": to_text(not_rare, "token"),
    "6 + drop 'in netflix queue'": to_text(no_junk, "token"),
}

print("movies with at least one tag after each step:")
for name, text in variants.items():
    print(f"  {name:<36}{(text != '').sum():>5}")
print()

configs = {"no tags": combine(tag_matrix(variants["1 current (words, repeats count)"]), 0)}
for name, text in variants.items():
    X_tags = tag_matrix(text)
    for w in TAG_WEIGHTS:
        configs[f"{name[:1]} weight {w}"] = combine(X_tags, w)

results, baselines = evaluate(configs)
print("Average rating gap between a movie and its top 5 matches, over 10 shuffles")
print("Lower is better. Shown as mean (lowest to highest shuffle).\n")
best = print_table(results, baselines, label_width=18)

print("\nbest weight for each cleaning step (picked on validation movies, scored on test movies)")
for name in variants:
    step = name[:1]
    names = [n for n in configs if n.startswith(step + " ")]
    pick = min(names, key=lambda n: np.mean(results[n]["val"]))
    print(f"  {name:<36}{pick:<12}test {summarise(results[pick]['test'])}")

base = np.array(results["no tags"]["test"])
mine = np.array(results[best]["test"])
print("\nbest setting versus no tags, same test movies in each shuffle:")
print("  average improvement in rating gap:", round(float((base - mine).mean()), 3), "stars")
print("  better in", int((mine < base).sum()), "of", len(base), "shuffles")
