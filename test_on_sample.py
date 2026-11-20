# extracts random sample from 7M articles and outputs the percentage covered with the keywords in optimal_greedy_filters.csv
# then runs qwen on them to check for how many false negatives
import pandas as pd
import re

parts = []
source_file = "data/sample_from_large.csv"
columns = ["id","title", "abstract"]
table = 'results/optimal_greedy_filters.csv'
filter = pd.read_csv(table)
keywords = filter["ngram"].tolist()
df = pd.read_csv(source_file, usecols=columns)
pattern = "|".join(re.escape(key) for key in keywords)

df["filter"] = (
    df["title"].fillna("").str.contains(pattern, case=False, regex=True)
    | df["abstract"].fillna("").str.contains(pattern, case=False, regex=True)
).astype(int)

count = int(df["filter"].sum())

df.to_csv('results/filtered_sample.csv', index=False)
n = sum(1 for line in open(source_file)) - 1
# print percentage reduction by the filter
print(f"Percentage of article reduction:{((n - count)/n)*100}%!" )
    
    