# takes the classified filtered sample set of articles and appends a column verdict
# should have in the column TN TP FN FP 

import pandas as pd
 

df = pd.read_csv("results/classified_filtered_by_keyword_test.csv")

#df['verdict'] = 'TP'
#df.loc[(df["filter"] == 1) & (df["qwen2.5:14b_label"] <= 4), 'verdict'] = 'TP'
#df.loc[(df["filter"] == 0) & (df["qwen2.5:14b_label"] > 4), 'verdict'] = 'TN'
#df.loc[(df["filter"] == 0) & (df["qwen2.5:14b_label"] <= 4), 'verdict'] = 'FN'
#df.loc[(df["filter"] == 1) & (df["qwen2.5:14b_label"] > 4), 'verdict'] = 'FP'

#df.to_csv("results/classified_filtered_by_keyword_test.csv", index=False)

# creating confusion matrix
counts = df["verdict"].value_counts()

confusion = {
    "TP" : int(counts.get("TP")),
    "FP" : int(counts.get("FP")),
    "FN" : int(counts.get("FN")),
    "TN" : int(counts.get("TN"))
}


header = ["Actual \\ Predicted", "Positive", "Negative"]
row1 = ["Positive", confusion["TP"], confusion["FP"]]
row2 = ["Negative", confusion["FN"], confusion["TN"]]
data = [header, row1, row2]
output = pd.DataFrame(data)

output.to_csv("metrics/confusion_matrix.csv", index=False)