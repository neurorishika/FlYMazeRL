import pandas as pd
import numpy as np
import os
import json
from tqdm import tqdm

print("Loading data...")

full_data = pd.DataFrame()

# find all folders
folders = [f for f in os.listdir(".") if os.path.isdir(f)]

print("Found {} folders".format(len(folders)))

print("Loading data from folders...")

for folder in folders:
    # load params.json from each folder
    with open(os.path.join(folder, "params.json")) as f:
        params = json.load(f)
    # load data from each folder
    try:
        data = pd.read_csv(os.path.join(folder, "results.csv.gz"), compression="gzip")
    except FileNotFoundError:
        continue
    # att remaining params to data
    keys = [
        i
        for i in params.keys()
        if i not in ["save_path", "action_set_data", "reward_set_data", "n_ensemble", "print_every"]
    ]
    for key in keys:
        if key == "hidden_state_sizes":
            data[key] = "x".join(map(str, params[key]))
        else:
            data[key] = params[key]
    # append to full_data
    full_data = pd.concat([full_data, data], ignore_index=True)

print("Found {} data points".format(len(full_data)))
print("Memory usage: {}".format(full_data.memory_usage(index=True, deep=True)))

print("Saving data...")

# save to compressed dataframe
full_data.to_csv("results.csv.gz", compression="gzip")

print("Done!")
