import csv
import json
import pandas as pd
from pathlib import Path

json_dir = "output_files/"
csv_file = "all_classified.csv"

for i, file in Path(json_dir).iterdir:
    df = pd.read_json(file)
    if i == 1:
        df.to_csv(csv_file, index=False)
    else:
        df.to_csv(csv_file, mode='a', header=False, index=False)

