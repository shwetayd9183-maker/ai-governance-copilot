import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import requests
from datetime import date, timedelta
import boto3
import json

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(layout="wide")
st.title("Phoenix AI — Climate-Aware Horticulture Stabilization Engine")

# ---------------------------------------------------
# DISTRICT GEO MAPPING
# ---------------------------------------------------
district_geo = {
    "Nashik": (19.9975, 73.7898),
    "Pune": (18.5204, 73.8567),
    "Dhule": (20.9042, 74.7740),
    "Kolhapur": (16.7050, 74.2433),
    "Satara": (17.6805, 74.0183),
    "Raigad": (18.5158, 73.1822),
    "Sholapur": (17.6599, 75.9064),
    "Amarawati": (20.9320, 77.7523),
    "Chandrapur": (19.9615, 79.2961),
    "Nandurbar": (21.3655, 74.2400),
    "Thane": (19.2183, 72.9781),
    "Dharashiv(Usmanabad)": (18.1860, 76.0419),
    "Sangli": (16.8524, 74.5815),
    "Jalana": (19.8410, 75.8860),
    "Wardha": (20.7453, 78.6022),
    "Akola": (20.7096, 76.9981),
    "Buldhana": (20.5292, 76.1840),
    "Jalgaon": (21.0077, 75.5626),
    "Beed": (18.9901, 75.7531),
    "Nagpur": (21.1458, 79.0882),
    "Mumbai": (19.0760, 72.8777),
    "Ahmednagar": (19.0948, 74.7480),
    "Chattrapati Sambhajinagar": (19.8762, 75.3433)
}

# ---------------------------------------------------
# CLIMATE FUNCTIONS
# ---------------------------------------------------
def fetch_rainfall(lat, lon, days=30):
    end_date = date.today()
    start_date = end_date - timedelta(days=days)

    url = (
        "https://power.larc.nasa.gov/api/temporal/daily/point"
        f"?parameters=PRECTOTCORR"
        f"&community=AG"
        f"&longitude={lon}"
        f"&latitude={lat}"
        f"&start={start_date.strftime('%Y%m%d')}"
        f"&end={end_date.strftime('%Y%m%d')}"
        f"&format=JSON"
    )

    try:
        response = requests.get(url)
        data = response.json()
        rainfall_data = data["properties"]["parameter"]["PRECTOTCORR"]

        df_rain = pd.DataFrame.from_dict(
            rainfall_data, orient="index", columns=["rainfall_mm"]
        )

        df_rain["rainfall_mm"] = df_rain["rainfall_mm"].replace(-999, np.nan)
        df_rain = df_rain.dropna()
        return df_rain

    except:
        return None


def compute_rain_anomaly(district):
    if district not in district_geo:
        return 0

    lat, lon = district_geo[district]

    df_recent = fetch_rainfall(lat, lon, 30)
    df_year = fetch_rainfall(lat, lon, 365)

    if df_recent is None or df_year is None:
        return 0

    return df_recent["rainfall_mm"].mean() - df_year["rainfall_mm"].mean()


# ---------------------------------------------------
# AMAZON BEDROCK COPILOT
# ---------------------------------------------------
def get_bedrock_recommendation(district, crash_prob, rain_anomaly):
    try:
        client = boto3.client('bedrock-runtime', region_name='us-east-1')
        prompt = f"Given a {crash_prob*100:.1f}% price crash risk in {district} and current rainfall anomaly of {rain_anomaly:.1f}mm, provide a concise governance and procurement recommendation."
        
        request_body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 200,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        }
        
        response = client.invoke_model(
            modelId='anthropic.claude-3-sonnet-20240229-v1:0',
            body=json.dumps(request_body)
        )
        
        response_body = json.loads(response.get('body').read())
        return response_body['content'][0]['text']
    except Exception as e:
        return f"Bedrock API Error: Please ensure AWS credentials are configured. ({str(e)})"

# ---------------------------------------------------
# LOAD MODEL & DATA
# ---------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

crop = st.sidebar.selectbox("Select Crop", ["Onion", "Tomato", "Potato"])
crop_lower = crop.lower()

model = joblib.load(os.path.join(BASE_DIR, "models", f"xgb_crash_model_{crop_lower}.joblib"))

file_path = os.path.join(BASE_DIR, "data", f"maharashtra_{crop_lower}.csv")

if crop_lower == 'tomato':
    df = pd.read_csv(file_path)
    df = df.rename(columns={
        "arrival_quantity": "Arrival_MT",
        "modal_price": "Modal_Price",
        "date": "Date",
        "district": "District"
    })
else:
    df = pd.read_csv(file_path, skiprows=1)
    df["Date"] = pd.to_datetime(df["Date"], format='mixed', dayfirst=True)
    
    cost_col = [c for c in df.columns if "Modal Price" in c][0]
    arr_col = [c for c in df.columns if "Arrival Quantity" in c][0]
    
    df = df.rename(columns={
        arr_col: "Arrival_MT",
        cost_col: "Modal_Price"
    })

df["Date"] = pd.to_datetime(df["Date"], format='mixed', dayfirst=True)

df["District"] = df["District"].astype(str)
df = df[df["District"] != "nan"]

districts = sorted(df["District"].unique())

# ---------------------------------------------------
# FEATURE ENGINEERING FUNCTION (REUSABLE)
# ---------------------------------------------------
def build_features(data, district):
    temp = data[data["District"] == district].copy()
    temp = temp.sort_values("Date")

    if len(temp) < 30:
        return None, None

    temp["Arrival_Qtl"] = temp["Arrival_MT"] * 10

    temp["ret_1"] = temp["Modal_Price"].pct_change(1)
    temp["ret_3"] = temp["Modal_Price"].pct_change(3)
    temp["ret_7"] = temp["Modal_Price"].pct_change(7)

    temp["ma_7"] = temp["Modal_Price"].rolling(7).mean()
    temp["vol_14"] = temp["Modal_Price"].pct_change().rolling(14).std()

    temp["arrival_3pct"] = temp["Arrival_Qtl"].pct_change(3)
    temp["arrival_7mean"] = temp["Arrival_Qtl"].rolling(7).mean()

    arrival_threshold = temp["Arrival_Qtl"].quantile(0.90)
    temp["arrival_spike"] = (temp["Arrival_Qtl"] > arrival_threshold).astype(int)

    temp["month"] = temp["Date"].dt.month
    temp["district_code"] = temp["District"].astype("category").cat.codes
    temp["rain_anomaly_30d"] = compute_rain_anomaly(district)

    temp = temp.dropna()

    if len(temp) == 0:
        return None, None

    latest = temp.iloc[-1]

    feature_cols = [
        "ret_1","ret_3","ret_7",
        "ma_7","vol_14",
        "arrival_3pct","arrival_7mean",
        "arrival_spike",
        "month",
        "district_code",
        "rain_anomaly_30d"
    ]

    X = pd.DataFrame([latest[feature_cols]], columns=feature_cols)
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0)

    return latest, X


# ---------------------------------------------------
# SINGLE DISTRICT VIEW
# ---------------------------------------------------
st.header("District-Level Risk Analysis")

district_name = st.selectbox("Select District", districts)

# Sidebar Policy Controls
st.sidebar.header("Policy Assumptions")

expected_drop_pct = st.sidebar.slider(
    "Expected Price Drop (%)", 5, 40, 20
) / 100

procurement_pct = st.sidebar.slider(
    "Procurement Percentage (%)", 5, 50, 30
) / 100

storage_cost_per_qtl_per_day = st.sidebar.slider(
    "Storage Cost (₹ per quintal per day)", 1, 20, 6
)

days = 30

latest_row, X_latest = build_features(df, district_name)

if latest_row is not None:

    # Charts
    df_plot = df[df["District"] == district_name].sort_values("Date").copy()
    df_plot["Arrival_Qtl"] = df_plot["Arrival_MT"] * 10

    st.subheader("Price Trend")
    st.line_chart(df_plot.set_index("Date")["Modal_Price"])

    st.subheader("Arrival Trend")
    st.line_chart(df_plot.set_index("Date")["Arrival_Qtl"])

    # Prediction
    crash_prob = model.predict_proba(X_latest)[0,1]

    st.subheader("Crash Probability (Next 7 Days)")
    st.metric("Crash Probability", f"{crash_prob:.2%}")

    threshold = 0.20

    if crash_prob > threshold:
        st.error("⚠ High Crash Risk — Consider Early Procurement")
    else:
        st.success("No Immediate Crash Risk")
        
    # Bedrock Policy Advisory
    if st.button("Generate AI Policy Advisory"):
        st.info("🤖 **Consulting Amazon Bedrock Copilot...**")
        with st.spinner("Generating policy advisory..."):
            rain_anomaly = latest_row["rain_anomaly_30d"]
            recommendation = get_bedrock_recommendation(district_name, crash_prob, rain_anomaly)
            st.markdown(f"> {recommendation}")

    # Economic Simulation
    current_price = latest_row["Modal_Price"]
    recent_arrival = latest_row["Arrival_MT"] * 10

    expected_price_after = current_price * (1 - expected_drop_pct)
    loss_per_quintal = current_price - expected_price_after

    procurement_qty = recent_arrival * procurement_pct
    procurement_price = current_price * 0.95
    resale_price = expected_price_after

    fiscal_loss = (procurement_price - resale_price) * procurement_qty
    storage_cost = procurement_qty * storage_cost_per_qtl_per_day * days

    total_cost = fiscal_loss + storage_cost
    expected_farmer_loss = loss_per_quintal * recent_arrival * crash_prob
    net_benefit = expected_farmer_loss - total_cost

    st.subheader("Procurement Simulation")

    st.write(f"Current Price: ₹{current_price:,.0f}")
    st.write(f"Expected Post-Crash Price: ₹{expected_price_after:,.0f}")
    st.write(f"Suggested Procurement Qty: {procurement_qty:,.0f} quintals")

    st.write(f"Estimated Farmer Loss (Weighted): ₹{expected_farmer_loss:,.0f}")
    st.write(f"Total Fiscal Cost: ₹{total_cost:,.0f}")
    st.write(f"Net Economic Benefit: ₹{net_benefit:,.0f}")

    if crash_prob <= threshold:
        st.info("Low Risk — No Intervention Needed")
    elif net_benefit > 0:
        st.success("High Risk and Economically Justified")
    else:
        st.warning("High Risk but Fiscal Cost May Exceed Benefit")

# ---------------------------------------------------
# STATE-WIDE RANKING
# ---------------------------------------------------
st.header("State-Wide District Risk Ranking")

results = []

for district in districts:

    latest, X = build_features(df, district)

    if X is None:
        continue

    crash_prob = model.predict_proba(X)[0,1]

    current_price = latest["Modal_Price"]
    recent_arrival = latest["Arrival_MT"] * 10

    expected_drop_pct = 0.20
    procurement_pct = 0.30
    storage_cost = 6
    days = 30

    expected_price_after = current_price * (1 - expected_drop_pct)
    loss_per_qtl = current_price - expected_price_after

    procurement_qty = recent_arrival * procurement_pct
    procurement_price = current_price * 0.95

    fiscal_loss = (procurement_price - expected_price_after) * procurement_qty
    storage_total = procurement_qty * storage_cost * days

    total_cost = fiscal_loss + storage_total
    expected_farmer_loss = loss_per_qtl * recent_arrival * crash_prob
    net_benefit = expected_farmer_loss - total_cost

    results.append({
        "District": district,
        "Crash Probability": round(crash_prob,3),
        "Fiscal Cost": round(total_cost,0),
        "Net Benefit": round(net_benefit,0),
        "Procurement Qty": round(procurement_qty,0)
    })

results_df = pd.DataFrame(results).sort_values("Net Benefit", ascending=False)

st.dataframe(results_df)

# ---------------------------------------------------
# BUDGET OPTIMIZER
# ---------------------------------------------------
st.header("Budget Allocation Optimizer")

total_budget = st.slider("Total Government Budget (₹)",100000,10000000,2000000,100000)

allocated = 0
selected = []

for _, row in results_df.iterrows():
    if row["Net Benefit"] > 0 and allocated + row["Fiscal Cost"] <= total_budget:
        allocated += row["Fiscal Cost"]
        selected.append(row["District"])

st.write(f"Budget Allocated: ₹{allocated:,.0f}")

for d in selected:
    st.write(f"✅ {d}")

# ---------------------------------------------------
# STORAGE CONSTRAINT
# ---------------------------------------------------
st.header("Storage Capacity Constraint")

storage_limit = st.slider("Total Storage Capacity (Quintals)",1000,50000,10000,1000)

used_storage = 0
storage_selected = []

for _, row in results_df.iterrows():
    if row["Net Benefit"] > 0 and used_storage + row["Procurement Qty"] <= storage_limit:
        used_storage += row["Procurement Qty"]
        storage_selected.append(row["District"])

st.write(f"Storage Used: {used_storage:,.0f} quintals")

for d in storage_selected:
    st.write(f"📦 {d}")
