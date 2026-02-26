# src/model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import xgboost as xgb


def prepare_features(df):
    """
    Prepare features and target variable.
    """

    # Basic feature set (can expand later)
    feature_cols = [
        "Modal_Price",
        "Arrival_Quantity"
    ]

    # One-hot encode categorical variables
    df_encoded = pd.get_dummies(df, columns=["Crop", "District"], drop_first=True)

    X = df_encoded.drop(columns=["severity_class", "future_avg_min", "drop_pct"], errors="ignore")
    y = df_encoded["severity_class"]

    return X, y


def train_model(df):
    """
    Train multi-class XGBoost model.
    """

    X, y = prepare_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    # Handle class imbalance
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_train),
        y=y_train
    )

    weight_dict = {i: class_weights[i] for i in range(len(class_weights))}
    sample_weights = y_train.map(weight_dict)

    model = xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05
    )

    model.fit(X_train, y_train, sample_weight=sample_weights)

    y_pred = model.predict(X_test)

    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))

    return model