# src/run_pipeline.py

from preprocessing import clean_agmarknet_csv
from labeling import apply_severity_labels
from model import train_model
import pandas as pd


def run_local_pipeline():

    crops = ["Onion", "Tomato", "Potato"]

    all_dfs = []

    for crop in crops:
        file_path = f"data/raw/{crop.lower()}.csv"
        df = clean_agmarknet_csv(file_path, crop)
        all_dfs.append(df)

    combined_df = pd.concat(all_dfs, ignore_index=True)

    labeled_df = apply_severity_labels(combined_df)

    model = train_model(labeled_df)

    return model


if __name__ == "__main__":
    run_local_pipeline()