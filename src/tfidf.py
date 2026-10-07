import math

import pandas as pd

from movie_text import build_movie_texts

PICKS = ["Toy Story (1995)", "Toy Story 2 (1999)", "Heat (1995)", "Fargo (1996)", "Jumanji (1995)"]

movies = build_movie_texts()
docs = movies[movies["title"].isin(PICKS)].set_index("title").loc[PICKS]
words_per_doc = {title: row["genres_text"].split() + [row["decade_text"]] for title, row in docs.iterrows()}

n_docs = len(words_per_doc)
vocab = sorted({w for words in words_per_doc.values() for w in words})

doc_freq = {w: sum(w in words for words in words_per_doc.values()) for w in vocab}
idf = {w: math.log(n_docs / doc_freq[w]) for w in vocab}

tf = pd.DataFrame(
    {t: {w: words.count(w) / len(words) for w in vocab} for t, words in words_per_doc.items()}
)
tfidf = tf.mul(pd.Series(idf), axis=0)

if __name__ == "__main__":
    pd.set_option("display.width", 200)
    print("--- words in each movie ---")
    for title, words in words_per_doc.items():
        print(title, "->", words)

    print("\n--- document frequency and idf ---")
    print(pd.DataFrame({"in_how_many_movies": doc_freq, "idf": idf}).round(3))

    print("\n--- tf-idf table (rows are words, columns are movies) ---")
    print(tfidf.round(3))
