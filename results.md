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

## Part B: A map of the movies (`src/clusters.py`, plot in `plots/pca_map.png`)

Setup: K-Means on the genres and decade vectors, since Part A showed that is the best base. The 8 movies with no genre and no year are left out, which leaves 9,734. K-Means was run with k = 5, 10, 15 and 20, and 5 different random starts for each k to see how much luck matters. PCA squeezes the vectors down to 2 dimensions for the plot.

What the data looks like:
- The vocabulary is only 31 words (the genres plus one token per decade).
- The 9,734 movies have only 2,216 distinct vectors, so many movies are exact copies of each other as far as the model can tell. On the map they stack into tight blobs, so the plot adds a tiny jitter to make them visible.
- The 2D map keeps only 19.9% of the variation, so it is a rough picture and not a faithful one.

| k | 5 | 10 | 15 | 20 |
|---|---|---|---|---|
| Silhouette (higher means cleaner clusters) | 0.200 | 0.222 | 0.220 | 0.245 |
| Agreement between random starts (ARI, 1 is identical) | 0.83 | 0.70 | 0.69 | 0.74 |
| Smallest cluster | 1,177 | 439 | 377 | 135 |
| Largest cluster | 2,850 | 1,444 | 1,300 | 837 |
| Match with first-listed genre (ARI) | 0.005 | 0.171 | 0.150 | 0.108 |

What the clusters are:
- k = 5: all five clusters are decades (2000s, 1990s, 2010s, 1960s and 70s, 1980s). Genres barely matter.
- k = 10: genres start to appear. Crime and thriller, action and sci-fi, horror, children and animation, and documentary each get their own cluster. The rest are still decade and drama or comedy mixes.
- k = 15 and 20: more genre clusters (mystery, fantasy, romance) and the old decades split into their own clusters (1930s, 1940s, 1950s). Silhouette keeps rising slightly, but the extra clusters are mostly decades, not new kinds of movie.
- Clusters that have a decade among their top 3 words: 5 of 5, 7 of 10, 11 of 15 and 15 of 20.

Cluster labels from tags were weak. A few made sense (`zombies`, `ghosts` for horror, `disney` and `animation` for children, `organized`, `mafia` and `drugs` for crime, `politics` and `business` for documentaries, `astaire` and `rogers` for the 1930s to 50s). Many were noise: `queue` and `netflix` (one user's habit), actor and character names such as `ferrell` and `jason`, and the word `and`.

What the clusters say about ratings (k = 10): average stars range from 2.93 (horror) to 3.78 (documentary), and old movies (1960s and 70s) average about 3.6. Action and sci-fi clusters get the most ratings per movie (about 21) and documentaries the fewest (about 3). I have not tested why. A likely reason is that only well-liked old movies and documentaries get watched and rated at all, but that is a guess.

Honest check: is it just genres again?
- Mostly yes, by construction. The only inputs are genres and decade, so the clusters can only be combinations of those two things. 90% to 97% of the movies in a cluster carry that cluster's top word.
- It is not only genres. The decade token is as strong as a genre, so the clusters at small k group by era first and genre second. The low match with first-listed genre (0.005 to 0.17) comes from that.
- The structure is weak. Silhouette is 0.20 to 0.25, so clusters overlap a lot, and the map shows one long band where the colors mix. Random restarts agree with each other only 69% to 83%, so the exact cluster boundaries depend on luck.
- Inertia falls steadily as k grows, so there is no clear elbow and no obviously right k. For the README map, k = 10 is the most readable choice. That is a judgment call and not a metric result.

Limits: the plot is a 2D squeeze of 31 dimensions that keeps about a fifth of the variation. The clusters use no rating information. Tags were left out of the features because Part A showed they did not help.

### Searching for a better k (`src/choose_k.py`, plot in `plots/choose_k.png`)

Instead of the four values from the plan, every k from 2 to 30 plus 35 and 40 was scored, each with 5 random starts. Three scores: inertia (how tight the clusters are), silhouette (how cleanly separated they are) and stability (how much random starts agree).

| k | Silhouette | Stability |
|---|---|---|
| 3 | 0.138 | 1.00 |
| 4 | 0.175 | 1.00 |
| 5 | 0.192 | 0.83 |
| 9 | 0.196 | 0.59 |
| 10 | 0.215 | 0.70 |
| 20 | 0.251 | 0.74 |
| 30 | 0.298 | 0.74 |
| 40 | 0.329 | 0.72 |

Findings:
- No k is clearly best. Inertia falls smoothly with no elbow. Silhouette rises all the way to k = 40 without a peak.
- The rising silhouette is not trustworthy here. The 9,734 movies have only 2,216 distinct vectors, so with more clusters each cluster becomes one repeated genre and decade combination, which scores well trivially. Taken to the extreme, one cluster per distinct vector would look perfect and tell us nothing.
- Stability is perfect at k = 3 and 4 but those clusters are too coarse to be useful (silhouette 0.14 and 0.18). It dips to 0.59 at k = 9, jumps back to 0.70 at k = 10, and sits on a plateau of about 0.70 to 0.74 from k = 20 to 30.
- The only visible structure is a small bump at k = 10 (silhouette 0.196 to 0.215 and stability 0.59 to 0.70 compared with k = 9) and the plateau from about 20. So the original guesses of 10 and 20 were reasonable, but the sweep did not find anything better.
- From about k = 24 onward some clusters become very small (as few as 22 movies at k = 40).

Decision: keep k = 10 for the README map because it is the most readable. A finer k of about 20 to 22 is a defensible alternative. Further tuning would not be worth it, since no score picks a clear winner.

## Part C: Predicting ratings (`src/split.py`, `src/baselines.py`, `src/mf.py`, `src/final_score.py`)

Setup: 20% of the ratings are hidden at random as the test pile and every model learns from the other 80%. Everything is repeated on 10 different shuffles. The score is RMSE, the typical miss in stars, and lower is better. The settings of the matrix factorization were chosen on a separate validation pile (10% of all ratings, cut out of the training pile), never on the test pile. About 4% of test ratings are for movies with no training rating. The movie-based models fall back to the overall average (plus the user's bias) for those.

Final test scores, average with range over 10 shuffles:

| Model | RMSE, all test ratings | RMSE, movies with no training rating |
|---|---|---|
| 1. Overall average | 1.043 (1.032 to 1.050) | 1.150 (1.112 to 1.203) |
| 2. Movie average | 0.977 (0.963 to 0.983) | 1.150 (1.112 to 1.203) |
| 3. Movie average + user bias | 0.894 (0.881 to 0.899) | 1.077 (1.041 to 1.137) |
| Matrix factorization | 0.856 (0.843 to 0.862) | 1.026 (0.994 to 1.073) |

Matrix factorization beats the best baseline on every one of the 10 shuffles, by 0.038 on average (0.037 to 0.040). That is 4% less error than baseline 3 and 18% less than guessing the average.

How the settings were chosen (all on the validation pile):
- Without regularization the model overfits quickly. With 10 hidden numbers the training RMSE fell from 0.78 to 0.57 while the score on unseen ratings got worse after about 10 epochs (0.872 up to 0.933).
- Without regularization the number of hidden numbers barely mattered. 5, 10 and 20 tied at about 0.884 and 50 was a little worse (0.890), because bigger lists overfit sooner.
- Regularization removed the overfitting. A strength of 0.1 was the sweet spot. Stronger penalties underfit, and at 0.3 all list sizes landed on 0.882. With regularization bigger lists stopped hurting, and 20 and 50 tied at 0.866 and 0.865.
- Final settings: 20 hidden numbers, strength 0.1, 55 epochs. 50 would have been as good, but 20 is simpler and cheaper.

What this does and does not show:
- The 10 out of 10 wins are less independent than they sound. Each shuffle hides a different random 20%, so the test piles overlap a lot, and the gain looking so steady (0.037 to 0.040) is partly because of that. The ranges show the effect of the shuffle, not of fresh data.
- The learning rate (0.01), the starting size of the hidden numbers and the 55 epochs were fixed by hand or by a coarse grid. A finer search might gain a little more.
- Matrix factorization is still bad on movies it has never seen (RMSE 1.03). It only knows the user's bias there. Part F looks at cold start properly, and the hybrid with content features is a stretch item.
- Only the rating miss is measured here. How good the top 10 lists are is a different question (hit rate in Part E).

## Part D: Explaining recommendations

### Step 1: content-based reasons (`src/explain_content.py`)
- A user's liked movies are those rated 4.0 or more. Every movie the user has not rated gets a score: its highest cosine similarity to any liked movie. The vectors use genres and decade only, because Part A found tags did not help.
- The reason names the liked movie that is closest, then lists what the two share: genres, decade, and cleaned tags (tags on at least 2 movies, no `in netflix queue`). Example: "Shares genre Comedy and the 1990s decade with My Cousin Vinny (1992), which you rated 5.0."
- Only 1,231 of 9,742 movies have a cleaned tag, so tags show up in the reason rarely (1 of 15 examples).
- Limits seen in the demo: all 15 examples scored 1.00. A heavy user has liked a movie with exactly the same genres and decade as many unseen movies, so the score cannot rank them and the tie-break by number of ratings decides. Among several tied liked movies, the highest rated one is named, which is a choice of the code, not evidence that it influenced the pick. The reason is truthful about what matches, but it is a weak reason for a heavy user.

### Step 2: matrix factorization reasons (`src/explain_mf.py`)
- The final model (20 hidden numbers, regularization 0.1, 55 epochs) is fitted on all 100,836 ratings, which takes about 42 seconds. The top 10 for a user are the unseen movies with the highest predicted rating. Movies with fewer than 20 ratings are skipped, because their hidden numbers rest on too little data.
- Reason, part 1: the prediction split into its pieces: overall average + the user's generosity + the movie's pull + taste match (the dot product of the two hidden-number lists).
- Reason, part 2: the 3 movies the user rated 4.0 or more whose hidden numbers have the highest cosine similarity to the recommended movie. Shared genres are added for readability. The model itself never sees genres.
- Sanity check on the hidden numbers: the closest popular movies to Toy Story are Aladdin (0.84) and Wallace & Gromit: The Wrong Trousers (0.73), to The Matrix Inception (0.81), to Pulp Fiction The Usual Suspects (0.77), True Romance (0.76), Fight Club and Reservoir Dogs. The model found sensible structure from ratings alone.
- The split shows when the neighbours are a real reason. User 1 is generous (+0.77) and the taste match is near zero (+0.00 to +0.11) for all their top picks, so those movies are recommended mostly because they are well liked by everyone, and the neighbour list adds little. User 3 is strict (-1.11) but the taste match is large (+0.78 to +1.29), and the neighbours (Saturn 3, Thing, Galaxy of Terror) make sense as a reason.
- Predictions above 5 are cut to 5.00 for display (user 1's top four all show 5.00), but ranking uses the uncut score.
- This is an explanation of the model, not a test. The model has seen every rating, so these numbers say nothing about accuracy.

### Step 3: command line demo (`src/demo.py`)
- Run from `src/`: `python demo.py USER [-n 10] [--refit]`. It prints a short profile of the user (number of ratings, average, the movies they rated highest), then the top N from the content-based method and the top N from matrix factorization, each with its reason.
- The fitted matrix factorization model is saved to `data/mf_model.pkl` (gitignored). The first run takes about 43 seconds to fit, later runs about 1.4 seconds. `--refit` trains it again.
- An unknown user id gives a clear error instead of a crash.
- Both methods use all of the user's ratings, so this is a demo of the explanations, not a test of quality. Quality is measured in Part E.
- Observed in the demo: the two methods often disagree. For user 3 the content-based top 3 are Jurassic Park, Independence Day and Alien (famous, matching genres), while matrix factorization gives Serenity, Waiting for Guffman and Payback. The content-based list is more popular and more obvious, and the learned one is more tied to this user's taste but harder to justify for a movie like Waiting for Guffman.

### Step 3b: notebook demo (`notebooks/demo.ipynb`)
- Same two methods as the command line demo, as a notebook: pick `USER_ID`, see the user's profile and a chart of their star ratings, the content-based top 10 with reasons, the matrix factorization top 10 as a table, the reason for the first pick, and a chart splitting each pick into movie pull and taste match.
- The chart makes the Step 2 finding visible. For user 3, Payback and Pleasantville are almost pure taste picks (movie pull about 0.0), and Wyatt Earp is recommended even though its movie pull is negative. The two methods share only one movie in their top 10 (Blade Runner).
- The notebook was run top to bottom with no errors, and its outputs are saved in the file.
- It reuses the functions in `src/` and the saved model in `data/mf_model.pkl`, so it adds no new modelling.

## Part E: Evaluation

### Step 2: hit rate at 10 (`src/hit_rate.py`)
Setup: the same 10 shuffles as Part C, so each shuffle hides the same 20% of the ratings. Every method learns from the other 80% and ranks the movies a user has not rated in that 80%. A hit is a hidden rating of 4.0 or more that lands in the user's top 10. The candidates are the movies with at least 20 training ratings (about 1,058), the same pool for every method. Users with no hidden liked movie in the pool are skipped (about 585 of 610 are scored per shuffle). Matrix factorization uses the final settings from Part C (20 hidden numbers, regularization 0.1, 55 epochs), and nothing was tuned for this test.

| Method | Hit rate at 10 (at least one hit) | Recall at 10 |
|---|---|---|
| Random | 0.113 (0.095 to 0.134) | 0.010 (0.007 to 0.014) |
| Most rated | 0.565 (0.549 to 0.587) | 0.118 (0.110 to 0.127) |
| Highest movie average | 0.262 (0.205 to 0.310) | 0.034 (0.022 to 0.040) |
| Content-based (genres and decade) | 0.500 (0.465 to 0.542) | 0.091 (0.082 to 0.105) |
| Matrix factorization | 0.319 (0.306 to 0.339) | 0.047 (0.039 to 0.053) |

What it shows:
- Matrix factorization beats random and the highest movie average, so its ranking carries signal. It loses to the non-personal "most rated" list on all 10 shuffles (by 0.21 to 0.27), and to the content-based list.
- The RMSE win in Part C does not carry over to top-10 lists. A model trained to shrink the rating error on rated movies is not trained to rank movies the user will go on to watch and like.
- The content-based list is close to the most-rated list. Many unseen movies tie at a similarity of 1.00 with some liked movie, and ties are broken by number of ratings, so the content-based list is largely popular movies that share genres. That is also why it beats the learned model here.
- "Highest movie average" ranks by one number per movie and has no personal part. Baseline 3 (average plus user bias) gives the same ranking, because a user bias shifts every movie by the same amount.

Diagnosis on shuffle 0 only (a check on the cause, not a tuning step, and nothing was chosen from it). The hit rate of matrix factorization split into its parts, with the candidate pool made stricter:

| Min training ratings | Pool size | Most rated | Matrix factorization | Taste part only | Movie bias only |
|---|---|---|---|---|---|
| 20 (used) | 1,058 | 0.583 | 0.334 | 0.175 | 0.264 |
| 50 | 326 | 0.615 | 0.459 | 0.282 | 0.438 |
| 100 | 80 | 0.708 | 0.629 | 0.602 | 0.565 |
| 200 | 6 | 1.000 | 1.000 | 1.000 | 1.000 |

- The matrix factorization picks are obscure. The median pick has 44 ratings at the 20-rating cutoff, so the model favours well-liked movies that few people have rated, while the hidden liked movies are mostly well known.
- The personal part adds something. The full score (0.334) beats movie bias alone (0.264), which is the non-personal part. The taste part on its own is the weakest (0.175), so it only helps on top of the bias.
- Raising the cutoff narrows the gap only because the pool shrinks until every user gets nearly the same list. At 200 the pool has 6 movies and every method scores 1.000, which says nothing. A stricter cutoff does not fix the model.
- No sign of a coding error. Random lands close to its expected rate (about 10 hidden liked movies among 1,058 candidates gives roughly 0.1), and the loss to the most-rated list shows on all 10 shuffles.

Limits of the test:
- The hidden ratings are only for movies the user chose to watch, so the test rewards guessing what people watch, which favours popular movies. A liked movie the user never watched counts as a miss. The result says little about how pleasing the lists are to the user, and a high hit rate for "most rated" does not make it a good recommender.
- 32% of the hidden liked ratings (68.4% are in the pool) are for movies with fewer than 20 training ratings and cannot be hit by any method.
- The 20-rating cutoff, the 4.0 threshold for "liked" and the top 10 are fixed choices. The test piles overlap across shuffles, so the ranges show the effect of the shuffle, not of fresh data.
- Part F checks the popularity side directly (how many of each method's top 10 are among the most-rated movies).

### Step 3: does the content-based similarity work? (`src/final_similarity.py`)
Rerun of the Part A test on the final content setup (genres and decade, no tags, ties broken by number of ratings). Same method as before: for 100 test movies per shuffle (at least 50 ratings each), take the top 5 matches and average the rating gap between users who rated both, over 10 shuffles. No setting was chosen in this run. The setting was fixed in Part A.

| Partners | Rating gap in stars | Content-based is lower in |
|---|---|---|
| Content-based top 5 (genres and decade) | 0.867 (0.840 to 0.889) | |
| Random movies | 1.003 (0.951 to 1.039) | 10 of 10 shuffles |
| The 5 most-rated movies | 1.017 (0.971 to 1.043) | 10 of 10 shuffles |
| Random movies with at least 50 ratings | 0.954 (0.932 to 0.978) | 10 of 10 shuffles |

Average advantage over each baseline, same test movies in each shuffle: random 0.135 (0.080 to 0.170), most rated 0.150, random popular 0.087 (0.065 to 0.121). 74% of the match pairs (69% to 79%) had at least 5 shared raters and were scorable.

What it shows:
- The match lists are better than chance on every shuffle. Part A's numbers reproduce exactly (0.867, 1.003, 1.017).
- The new control tests the limit that movie quality is mixed into the gap. Random partners from movies with 50 or more ratings already score 0.954 instead of 1.003, so about a third of the advantage over plain random (0.049 of 0.135 stars) comes from picking well-known, well-liked movies. The remaining 0.087 stars is the part that genre and decade explain.
- The control is not a perfect match. Ties are broken by number of ratings, so the actual matches are probably more popular than a random movie with 50 ratings. The true effect of similarity could be somewhat smaller than 0.087.

Limits:
- The gap is a rating gap, not a quality score. It says users rate similar movies similarly. It does not say they would enjoy the match.
- Only pairs with at least 5 shared raters count (74%), and these are skewed towards popular movies.
- Only the 450 movies with 50 or more ratings are used as queries, so the result says nothing about rarely rated movies, which is where content features would matter most.
- The test uses all ratings, not a held-out pile. It measures how well genres and decade describe similarity, and was not used to fit anything beyond the tag weight in Part A.
- Hidden-number (matrix factorization) similarity is not scored on this test. The model learned from the same ratings that the gap is computed on, so a low gap would be close to guaranteed and prove little. A fair version needs held-out raters, and there are too few shared raters in 20% of the data to score pairs.
- Test movies overlap across shuffles, so the ranges show the effect of the shuffle and not of fresh data.

### Step 5: results table (`src/results_table.py`)
One script reruns every test and prints the final tables, with one matrix factorization fit per shuffle shared by the RMSE and hit rate scores (about 6 minutes). All numbers match the earlier runs (Part C, step 2 and step 3), which checks that nothing drifted. 10 shuffles, mean with lowest to highest in brackets.

| Model | RMSE in stars (lower is better) | Hit rate at 10 (higher is better) | Recall at 10 |
|---|---|---|---|
| 1. Overall average | 1.043 (1.032 to 1.050) | n/a | n/a |
| 2. Movie average | 0.977 (0.963 to 0.983) | 0.262 (0.205 to 0.310) | 0.034 (0.022 to 0.040) |
| 3. Movie average + user bias | 0.894 (0.881 to 0.899) | same as baseline 2 | same as baseline 2 |
| Content-based (genres and decade) | n/a | 0.500 (0.465 to 0.542) | 0.091 (0.082 to 0.105) |
| Matrix factorization | **0.856** (0.843 to 0.862) | 0.319 (0.306 to 0.339) | 0.047 (0.039 to 0.053) |
| Reference: random list | n/a | 0.113 (0.095 to 0.134) | 0.010 (0.007 to 0.014) |
| Reference: most rated movies | n/a | **0.565** (0.549 to 0.587) | **0.118** (0.110 to 0.127) |

| Similarity test: partners for 100 test movies | Rating gap in stars (lower is better) | Content-based is lower in |
|---|---|---|
| Content-based top 5 (genres and decade) | 0.867 (0.840 to 0.889) | |
| Random movies | 1.003 (0.951 to 1.039) | 10 of 10 shuffles |
| The 5 most-rated movies | 1.017 (0.971 to 1.043) | 10 of 10 shuffles |
| Random movies with at least 50 ratings | 0.954 (0.932 to 0.978) | 10 of 10 shuffles |

Reading the tables:
- Matrix factorization is best at predicting ratings and beats baseline 3 on 10 of 10 shuffles. It is not best at the top 10 lists. The non-personal most-rated list beats it on 10 of 10 shuffles, and the content-based list beats it as well. The two jobs give different winners.
- Baseline 3 has no separate hit rate. A user bias adds the same amount to every movie for that user, so it cannot change the order, and its list is the same as baseline 2.
- The content-based method cannot be scored by RMSE because it ranks movies and predicts no ratings. Its only scores are the hit rate and the similarity test.
- The hit rate numbers use the pool of movies with at least 20 training ratings and the limits listed in step 2.
- Baseline 1 gives every movie the same score, so it has no ranking.
