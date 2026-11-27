# samples out of the big data 
import random
import pandas as pd

file = "reseau_large.csv"
n = sum(1 for line in open(filename)) - 1
s = 4000 # sample size
skip = sorted(random.sample(range(1,n+1), n-s))
df = pd.read_csv(filename, skiprows=skip)
df.to_csv("data/sample_from_alrge.csv", index=False)