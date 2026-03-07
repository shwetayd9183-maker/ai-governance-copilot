import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
from sklearn.model_selection import TimeSeriesSplit
from sklearn.utils.class_weight import compute_class_weight
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, roc_auc_score
import os

crops = ['onion', 'tomato', 'potato']
all_metrics = []

for crop in crops:
    print(f"\n{'='*50}")
    print(f"Processing Crop: {crop.upper()}")
    print(f"{'='*50}\n")
    
    file_path = f"data/maharashtra_{crop}.csv"
    if not os.path.exists(file_path):
        print(f"Skipping {crop}: {file_path} not found.")
        continue

    print("Loading data...")
    if crop == 'tomato':
        df = pd.read_csv(file_path)
        df = df.rename(columns={
            "arrival_quantity": "Arrival_MT",
            "modal_price": "Modal_Price",
            "date": "Date",
            "district": "District"
        })
    else:
        df = pd.read_csv(file_path, skiprows=1) # The first row is an empty title in the original CSV
        cost_col = [c for c in df.columns if "Modal Price" in c][0]
        arr_col = [c for c in df.columns if "Arrival Quantity" in c][0]
        df = df.rename(columns={
            arr_col: 'Arrival_MT',
            cost_col: 'Modal_Price'
        })


    df['Date'] = pd.to_datetime(df['Date'], format='mixed', dayfirst=True)
    df = df.sort_values(['District', 'Date'])

    df['Arrival_Qtl'] = df['Arrival_MT'] * 10
    df['ret_1'] = df.groupby('District')['Modal_Price'].pct_change(1)
    df['ret_3'] = df.groupby('District')['Modal_Price'].pct_change(3)
    df['ret_7'] = df.groupby('District')['Modal_Price'].pct_change(7)

    df['ma_7'] = df.groupby('District')['Modal_Price'].transform(lambda x: x.rolling(7).mean())
    df['vol_14'] = df.groupby('District')['Modal_Price'].transform(lambda x: x.pct_change().rolling(14).std())

    df['arrival_3pct'] = df.groupby('District')['Arrival_Qtl'].pct_change(3)
    df['arrival_7mean'] = df.groupby('District')['Arrival_Qtl'].transform(lambda x: x.rolling(7).mean())

    df['arrival_spike'] = df.groupby('District')['Arrival_Qtl'].transform(
        lambda x: (x > x.quantile(0.90)).astype(int)
    )

    df['month'] = df['Date'].dt.month
    df['district_code'] = df['District'].astype('category').cat.codes
    df['rain_anomaly_30d'] = 0 # Dummy for training or compute it properly if needed.

    horizon = 7
    threshold_drop = 0.15

    df['future_min_7d'] = df.groupby('District')['Modal_Price'].transform(
        lambda x: x.shift(-1).rolling(horizon).min()
    )

    df['crash_label_7d'] = (
        df['future_min_7d'] < df['Modal_Price'] * (1 - threshold_drop)
    ).astype(int)

    df = df.dropna()

    feature_cols = [
        'ret_1','ret_3','ret_7',
        'ma_7','vol_14',
        'arrival_3pct','arrival_7mean',
        'arrival_spike',
        'month',
        'district_code',
        'rain_anomaly_30d'
    ]

    X = df[feature_cols]
    y = df['crash_label_7d']

    print("Splitting data chronologically...")
    dates = df['Date'].sort_values().unique()
    train_cutoff = dates[int(len(dates) * 0.70)]
    val_cutoff = dates[int(len(dates) * 0.85)]

    train_mask = df['Date'] < train_cutoff
    val_mask = (df['Date'] >= train_cutoff) & (df['Date'] < val_cutoff)
    test_mask = df['Date'] >= val_cutoff

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    print(f"Train size: {len(X_train)}, Val size: {len(X_val)}, Test size: {len(X_test)}")

    print("Training model...")
    classes = np.unique(y_train)
    weights = compute_class_weight(
        class_weight='balanced',
        classes=classes,
        y=y_train
    )

    class_weights = dict(zip(classes, weights))
    scale_pos_weight = class_weights[1] / class_weights[0]

    base_model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        eval_metric='auc',
        scale_pos_weight=scale_pos_weight
    )

    base_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    calibrated_model = CalibratedClassifierCV(base_model, method='isotonic', cv=3)
    calibrated_model.fit(X_train, y_train)

    def print_metrics(model, X_data, y_data, split_name):
        preds = model.predict(X_data)
        probs = model.predict_proba(X_data)[:, 1]
        auc = roc_auc_score(y_data, probs)
        report = classification_report(y_data, preds, output_dict=True)
        print(f"\n--- {split_name} Metrics ---")
        print(f"AUC: {auc:.4f}")
        print(classification_report(y_data, preds))
        
        return {
            "Crop": crop.upper(),
            "Split": split_name,
            "AUC": round(auc, 4),
            "Accuracy": round(report['accuracy'], 4),
            "F1_Class_0": round(report['0']['f1-score'], 4),
            "F1_Class_1": round(report['1']['f1-score'], 4)
        }

    print("\n" + "="*40)
    all_metrics.append(print_metrics(calibrated_model, X_train, y_train, "TRAIN"))
    all_metrics.append(print_metrics(calibrated_model, X_val, y_val, "VALIDATION"))
    all_metrics.append(print_metrics(calibrated_model, X_test, y_test, "TEST"))
    print("="*40 + "\n")

    print(f"Saving models for {crop}...")
    os.makedirs("models", exist_ok=True)
    joblib.dump(calibrated_model, f"models/xgb_crash_model_{crop}.joblib")

    # Save base XGBoost in AWS compatible formats
    base_model.save_model(f"models/xgb_crash_model_{crop}_aws.json")
    base_model.save_model(f"models/xgb_crash_model_{crop}_aws.bin")

    print(f"Done! Models for {crop} saved in models/\n")

metrics_df = pd.DataFrame(all_metrics)
metrics_csv_path = "models/model_performance_metrics.csv"
metrics_df.to_csv(metrics_csv_path, index=False)
print(f"All model performance metrics have been securely saved to: {metrics_csv_path}")

