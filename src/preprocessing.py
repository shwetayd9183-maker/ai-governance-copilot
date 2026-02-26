# src/preprocessing.py

import pandas as pd

def clean_agmarknet_csv(file_path, crop_name):
    """
    Clean and standardize Agmarknet CSV file.
    """

    df = pd.read_csv(file_path)

    # Clean column names
    df.columns = df.columns.str.strip()

    # Rename dynamic columns
    rename_map = {}
    for col in df.columns:
        if "Arrival Quantity" in col:
            rename_map[col] = "Arrival_Quantity"
        if "Modal Price" in col:
            rename_map[col] = "Modal_Price"

    df = df.rename(columns=rename_map)

    # Convert data types
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df['Arrival_Quantity'] = pd.to_numeric(df['Arrival_Quantity'], errors='coerce')
    df['Modal_Price'] = pd.to_numeric(df['Modal_Price'], errors='coerce')

    # Add crop column
    df['Crop'] = crop_name

    # Drop rows with missing critical values
    df = df.dropna(subset=['Date', 'Modal_Price'])

    return df