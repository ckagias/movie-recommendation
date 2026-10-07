from similarity_eval import combine, evaluate, print_table, tag_matrix, movies

TAG_WEIGHTS = [0, 0.25, 0.5, 1, 2]

X_tags = tag_matrix(movies["tags_text"])
configs = {f"tag weight {w}": combine(X_tags, w) for w in TAG_WEIGHTS}

results, baselines = evaluate(configs)
print("Average rating gap between a movie and its top 5 matches, over 10 shuffles")
print("Lower is better. Shown as mean (lowest to highest shuffle).\n")
print_table(results, baselines)
