# Results log

A running record of every test, what it found and what it means. Raw material for the README. Update it after each step.

Dataset: MovieLens ml-latest-small, 100,836 ratings, 610 users, 9,742 movies. Harper and Konstan (2015), "The MovieLens Datasets: History and Context", ACM TiiS 5(4).

## Part 1: Looking at the data (`src/explore.py`)

| Measure | Value |
|---|---|
| Ratings | 100,836 |
| Users | 610 |
| Movies in movies.csv | 9,742 |
| Movies with at least one rating | 9,724 |
| Average rating | 3.502 |
| Share of the user-movie table that is filled in | 1.70% |
| Fewest ratings by one user | 20 |
| Median ratings per user | 70.5 |
| Median ratings per movie | 3 |
| Movies with only 1 rating | 3,446 |
| Movies with at least one tag | 1,572 (16.1%) |

What it shows:
- Ratings lean positive. 4.0 is the most common value and whole stars are used far more than half stars.
- A few movies take a large share of the ratings. Forrest Gump has 329 of a possible 610. The median movie has 3.
- The most active user gave 2,698 ratings.
- 98% of the user-movie table is empty. Predicting those cells is the whole task.
- Tags are thin: only 58 users ever tagged, and one of them (user 474) gave 41% of the 3,683 tag rows.

Consequences for later parts: a popularity baseline will look strong, new or rarely rated movies will be hard to predict, and content features will rely mostly on genres and decade.

## Part A: Content-based similarity

### How a movie becomes a text (`src/movie_text.py`)
Genres become lowercase words (hyphens removed, so "Sci-Fi" is `scifi`). The release year becomes a decade token such as `decade1990s`. User tags are lowercased and appended. Seven movies end up with an empty text because they have no genre and no year.

### TF-IDF and cosine similarity, built by hand (`src/tfidf.py`, `src/cosine.py`)
- TF is a word's share of the movie's words. IDF is `log(movies / movies containing the word)`.
- Cosine similarity is `(a . b) / (length a * length b)`. It compares direction, so a movie with many words can match one with few. Scaling a vector by 10 left the score unchanged (0.6447).
- The hand-written NumPy version matches scikit-learn's `cosine_similarity` on five test movies.

### Full model (`src/similar.py`)
scikit-learn TF-IDF over all movies gives a 9,742 by 1,759 table. `similar_to(title, n=5)` compares one movie against all others, with ties broken by number of ratings.

Examples:
- Toy Story: A Bug's Life (0.86), Toy Story 2 (0.65), Antz. Good matches.
- Pulp Fiction: Reservoir Dogs, The Big Lebowski, Django Unchained, Fight Club. Good, and tags helped.
- Heat: every result scores exactly 1.0 because Heat has no tags and many movies share its genres. The tie-break by popularity decides the order.
- The Matrix: Sliding Doors, House of Flying Daggers, Hero, Karate Kid. A failure. The Matrix has the tags `alternate universe` and `martial arts`, which are rare, so Sliding Doors and Karate Kid match on those and ignore that the rest of the movie does not match.

### The test: do "similar" movies get similar ratings? (`src/similarity_eval.py`)
- For a query movie, take its top 5 matches. For each match, take the users who rated both, and average the absolute difference between their two ratings. Lower is better.
- Query movies are those with at least 50 ratings (450 movies). A pair needs at least 5 shared raters to count.
- 100 validation movies pick the setting and 100 different test movies score it. This is repeated over 10 random shuffles and reported as mean (lowest to highest).
- Baselines: random partner movies score 1.003, and always using the 5 most-rated movies scores 1.017.

### Step 5: how much weight should tags get? (`src/tag_weights.py`)
Tag weight multiplies the tag part of each movie's vector. 0 means tags are ignored.

| Tag weight | Test gap in stars |
|---|---|
| 0 (genres and decade only) | 0.867 (0.840 to 0.889) |
| 0.25 | 0.899 |
| 0.5 | 0.913 |
| 1 | 0.911 |
| 2 | 0.889 |
| Random / most rated | 1.003 / 1.017 |

Finding: every setting beats both baselines by about 0.1 to 0.15 stars, so "similar" movies do get more similar ratings than chance. Genres and decade alone were best. Adding the raw tags made it slightly worse.

### Step 6: cleaning the tags (`src/clean_tags.py`)
What the tag data looked like: 3,683 tag rows, 1,589 distinct raw tags, 1,475 after lowercasing, 1,465 after removing punctuation. 923 of the 1,465 tags sit on only one movie. 38% of tag rows are multi-word phrases. `in netflix queue` is on 131 movies and describes a user's habit, not the movie.

Each step was added on top of the previous one, and each was scored at tag weights 0.5, 1, 2 and 4. The weight was picked on the validation movies and scored on the test movies.

| Cleaning step | Movies with a tag | Test gap at best weight (4) |
|---|---|---|
| 1 Current: words, repeats count | 1,572 | 0.882 |
| 2 Merge punctuation variants | 1,572 | 0.882 |
| 3 Keep phrases together (`martial_arts`) | 1,572 | 0.881 |
| 4 One vote per movie | 1,572 | 0.884 |
| 5 Drop tags on a single movie | 1,345 | 0.870 |
| 6 Drop `in netflix queue` | 1,231 | 0.869 |
| No tags at all | | 0.867 |

What helped:
- Steps 2, 3 and 4 changed nothing measurable. They touch only a handful of tags.
- Step 5, dropping tags that sit on a single movie, was the one change that helped, by about 0.014 stars. Such tags cannot link two movies, and they only dilute the weight of the other words.
- Step 6 changed almost nothing.

What did not help: even the fully cleaned tags did not beat using no tags. The best cleaned setting scored 0.869 against 0.867 for no tags, an average difference of -0.002 stars (it was better in 7 of 10 shuffles, but the average is slightly worse). That is well inside the shuffle-to-shuffle noise of about 0.05. Cleaning stopped the tags from hurting, but it did not make them useful.

### Limits of this test
- The best weight in step 6 was 4, the largest value tried, so a still larger weight was not tested.
- Only 16% of movies have tags (12% after cleaning), so tags can affect only a small part of the results.
- Different settings are partly scored on different pairs, because the share of pairs with enough shared raters varies (67% to 74%).
- Movie quality is mixed into the gap. Two well-liked movies have a small gap even if they are not alike.
- Tags come from users who also rated, which could leak a little into the test.
- The test uses all ratings, so it is a measure of similarity quality and not a held-out prediction test. Part C does the held-out test.
