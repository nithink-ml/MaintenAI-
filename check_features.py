import os

import numpy as np
import pandas as pd

datasets = ["cmapss", "ims", "ai4i"]
splits = ["train", "val", "test"]

for dataset in datasets:
    print(f"\n--- {dataset.upper()} ---")
    feat_dir = f"data/processed/{dataset}_features"
    if not os.path.exists(feat_dir):
        print(f"Directory {feat_dir} does not exist.")
        continue
    for split in splits:
        file_path = f"{feat_dir}/{split}.parquet"
        if not os.path.exists(file_path):
            file_path = f"{feat_dir}/{split}.csv"
        if os.path.exists(file_path):
            df = (
                pd.read_parquet(file_path)
                if file_path.endswith(".parquet")
                else pd.read_csv(file_path)
            )
            has_nan = df.isna().any().any()
            has_inf = np.isinf(df.select_dtypes(include=[np.number])).any().any()
            print(f"{split}: {df.shape} (NaNs: {has_nan}, Infs: {has_inf})")
        else:
            print(f"{split}: File not found")
