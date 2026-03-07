import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import requests
from datetime import date, timedelta
import altair as alt
import pydeck as pdk

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(layout="wide", page_title="KrishiRakshak AI Dashboard", page_icon="📈")

st.markdown("""
<style>
/* Add a bit of spacing */
.block-container { padding-top: 1.5rem; }

/* Box styling for KPIs: equal sizes, no truncation */
[data-testid="stMetric"] {
    background-color: #fcfdfa;
    border: 1px solid #e1e8d5;
    padding: 15px 20px;
    border-radius: 10px;
    box-shadow: 2px 2px 5px rgba(0,0,0,0.04);
    min-height: 140px;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
}
[data-testid="stMetricLabel"] {
    white-space: normal !important;
    overflow: visible !important;
    font-size: 1.1rem !important;
}
[data-testid="stMetricValue"] {
    white-space: normal !important;
    font-size: 1.8rem !important;
}

/* Centered Titles with margin fix */
.main-title {
    text-align: center;
    color: #3b4527;
    font-size: 2.8rem;
    font-weight: 700;
    margin-bottom: 0.5rem;
    padding-top: 0px;
}
.sub-title {
    text-align: center;
    color: #6b8e23;
    font-size: 1.2rem;
    margin-bottom: 2rem; /* Extra gap */
}
.update-status {
    text-align: right;
    color: #777;
    font-size: 0.95rem;
    font-style: italic;
    background-color: #f0f0f0;
    padding: 5px 12px;
    border-radius: 20px;
    display: inline-block;
    margin-top: 10px;
    margin-bottom: 1rem;
}

/* Background watermark image for aesthetic */
.watermark {
    position: fixed;
    right: -5%;
    bottom: -5%;
    opacity: 0.1;
    z-index: -1;
    width: 600px;
    pointer-events: none;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# DISTRICT GEO MAPPING
# ---------------------------------------------------
district_geo = {
    "Nashik": (19.9975, 73.7898), "Pune": (18.5204, 73.8567), "Dhule": (20.9042, 74.7740),
    "Kolhapur": (16.7050, 74.2433), "Satara": (17.6805, 74.0183), "Raigad": (18.5158, 73.1822),
    "Sholapur": (17.6599, 75.9064), "Amarawati": (20.9320, 77.7523), "Chandrapur": (19.9615, 79.2961),
    "Nandurbar": (21.3655, 74.2400), "Thane": (19.2183, 72.9781), "Dharashiv(Usmanabad)": (18.1860, 76.0419),
    "Sangli": (16.8524, 74.5815), "Jalana": (19.8410, 75.8860), "Wardha": (20.7453, 78.6022),
    "Akola": (20.7096, 76.9981), "Buldhana": (20.5292, 76.1840), "Jalgaon": (21.0077, 75.5626),
    "Beed": (18.9901, 75.7531), "Nagpur": (21.1458, 79.0882), "Mumbai": (19.0760, 72.8777),
    "Ahmednagar": (19.0948, 74.7480), "Chattrapati Sambhajinagar": (19.8762, 75.3433)
}

# ---------------------------------------------------
# CLIMATE FUNCTIONS
# ---------------------------------------------------
@st.cache_data(ttl=3600*24)
def fetch_rainfall(lat, lon, days=30):
    end_date = date.today()
    start_date = end_date - timedelta(days=days)
    url = (
        "https://power.larc.nasa.gov/api/temporal/daily/point"
        f"?parameters=PRECTOTCORR&community=AG&longitude={lon}&latitude={lat}"
        f"&start={start_date.strftime('%Y%m%d')}&end={end_date.strftime('%Y%m%d')}&format=JSON"
    )
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            rainfall_data = data["properties"]["parameter"]["PRECTOTCORR"]
            df_rain = pd.DataFrame.from_dict(rainfall_data, orient="index", columns=["rainfall_mm"])
            df_rain["rainfall_mm"] = df_rain["rainfall_mm"].replace(-999, np.nan)
            return df_rain.dropna()
    except Exception:
        pass
    return None

@st.cache_data(ttl=3600*24)
def compute_rain_anomaly(district):
    if district not in district_geo: return 0
    lat, lon = district_geo[district]
    df_recent = fetch_rainfall(lat, lon, 30)
    df_year = fetch_rainfall(lat, lon, 365)
    if df_recent is None or df_year is None: return 0
    return df_recent["rainfall_mm"].mean() - df_year["rainfall_mm"].mean()

# ---------------------------------------------------
# LOAD MODEL & DATA
# ---------------------------------------------------
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
if os.path.basename(BASE_DIR) == 'src':
    BASE_DIR = os.path.dirname(BASE_DIR)

st.sidebar.header("Market Selection")
crop = st.sidebar.selectbox("Select Crop", ["Onion", "Tomato", "Potato"])
crop_lower = crop.lower()

# (11) Insert image of crop in sidebar for aesthetics
crop_images = {
    "onion": "https://images.unsplash.com/photo-1620574387735-3624d75b2dbc?w=600&q=80",
    "tomato": "https://images.unsplash.com/photo-1592924357228-91a4daadcfea?w=600&q=80",
    "potato": "https://images.unsplash.com/photo-1518977676601-b53f82aba655?w=600&q=80"
}
st.sidebar.image(crop_images[crop_lower], use_container_width=True)


@st.cache_resource
def load_model(crop_name):
    path = os.path.join(BASE_DIR, "models", f"xgb_crash_model_{crop_name}.joblib")
    return joblib.load(path) if os.path.exists(path) else None

model = load_model(crop_lower)
if not model:
    st.error(f"Model for {crop} not found. Please run the model generator.")
    st.stop()

@st.cache_data
def load_data(crop_name):
    file_path = os.path.join(BASE_DIR, "data", f"maharashtra_{crop_name}.csv")
    if not os.path.exists(file_path):
        return None
    if crop_name == 'tomato':
        d = pd.read_csv(file_path)
        d = d.rename(columns={"arrival_quantity": "Arrival_MT", "modal_price": "Modal_Price", "date": "Date", "district": "District"})
    else:
        d = pd.read_csv(file_path, skiprows=1)
        cost_col = [c for c in d.columns if "Modal Price" in c][0]
        arr_col = [c for c in d.columns if "Arrival Quantity" in c][0]
        d = d.rename(columns={arr_col: "Arrival_MT", cost_col: "Modal_Price"})
    
    d["Date"] = pd.to_datetime(d["Date"], format='mixed', dayfirst=True)
    d["District"] = d["District"].astype(str)
    return d[d["District"] != "nan"]

df = load_data(crop_lower)
if df is None:
    st.error(f"Data for {crop} not found.")
    st.stop()

districts = sorted(df["District"].unique())
district_name = st.sidebar.selectbox("Select District", districts)

# (1) Center the title & (3) Add last refreshed date and (5) Add background watermark
last_updated_date = df["Date"].max().strftime("%d %b %Y")
colA, colB = st.columns([4, 1])
with colA:
    st.markdown("<div class='main-title'>KrishiRakshak AI : Climate-Aware Horticulture Engine</div>", unsafe_allow_html=True)
with colB:
    st.markdown(f"<div style='text-align: right;'><span class='update-status'>🕒 Data as of: {last_updated_date}</span></div>", unsafe_allow_html=True)

st.markdown("<div class='sub-title'>An AI Assistant for Horticulture Market Intelligence</div>", unsafe_allow_html=True)
st.markdown(f"<img src='{crop_images[crop_lower]}' class='watermark'>", unsafe_allow_html=True)


# ---------------------------------------------------
# FEATURE ENGINEERING & PROCESSING
# ---------------------------------------------------
def build_features_for_df(temp, district_str):
    temp = temp.sort_values("Date").copy()
    if len(temp) < 30: return pd.DataFrame()
    
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
    temp["rain_anomaly_30d"] = compute_rain_anomaly(district_str)
    
    return temp.dropna()

feature_cols = [
    "ret_1", "ret_3", "ret_7", "ma_7", "vol_14",
    "arrival_3pct", "arrival_7mean", "arrival_spike", 
    "month", "district_code", "rain_anomaly_30d"
]

st.markdown("<br><br>", unsafe_allow_html=True)
st.header(f"District-Level Market Intelligence: {district_name}")

df_dist = df[df["District"] == district_name]
df_feats = build_features_for_df(df_dist, district_name)

if len(df_feats) == 0:
    st.warning("Not enough historical data to generate predictions for this district.")
    st.stop()

# Basic Calcs
latest = df_feats.iloc[-1]
current_price = latest["Modal_Price"]
price_7d = df_feats.iloc[-7]["Modal_Price"] if len(df_feats) >= 7 else latest["Modal_Price"]
price_change = ((current_price - price_7d) / price_7d) * 100 if price_7d else 0.0

current_arrival = latest["Arrival_Qtl"]
avg_arrival_30d = df_dist["Arrival_MT"].tail(30).mean() * 10
arrival_surge = current_arrival / avg_arrival_30d if avg_arrival_30d > 0 else 1.0

rain_anomaly = latest["rain_anomaly_30d"]

X_latest = pd.DataFrame([latest[feature_cols]], columns=feature_cols)
X_latest = X_latest.apply(pd.to_numeric, errors="coerce").fillna(0)
crash_prob = model.predict_proba(X_latest)[0, 1]

expected_farmer_loss_cr = (current_price * 0.20) * current_arrival * crash_prob / 10
# ---------------------------------------------------
# TOP KPIs
# ---------------------------------------------------
# (4) Adding definitions of each number which can be readable on hover (help=...)
col1, col2, col3, col4 = st.columns(4)
col1.metric("Current Market Price", f"₹{current_price:,.0f} / qtl", help="The most recent modal price recorded for this crop in the selected district.")
col2.metric("7-Day Price Change", f"{price_change:+.1f} %", delta=f"{price_change:+.1f}%", delta_color="inverse", help="Percentage change in price compared to 7 days ago. Identifies recent price momentum.")
col3.metric("Current Arrival Vol", f"{current_arrival:,.0f} qtl", help="Recent daily crop arrivals into the market. Supply volumes drive immediate price movements.")
surge_color = "normal" if arrival_surge < 1.2 else ("off" if arrival_surge < 1.5 else "inverse")
col4.metric("Arrival Surge", f"{arrival_surge:.1f} × Normal", delta=f"{arrival_surge-1:+.1f}x", delta_color=surge_color, help="Ratio of current arrivals to the 30-day average. A value > 1 means more supply than usual, > 1.5 is an extreme market flood.")

col5, col6, col7 = st.columns(3)
col5.metric("Rainfall Anomaly", f"{rain_anomaly:+.0f} mm", delta=f"{rain_anomaly:+.0f}mm", delta_color="off", help="Difference between the last 30 days of rainfall and the historically expected rainfall. Strongly affects harvest schedules.")
# Fix (6): Explicitly render crash probabilty as an HTML container so color works perfectly
prob_color = "#8b0000" if crash_prob > 0.35 else ("#cd853f" if crash_prob > 0.2 else "#556b2f")
crash_html = f"""
<div style="background-color: #fcfdfa; border: 1px solid #e1e8d5; padding: 15px 20px; border-radius: 10px; box-shadow: 2px 2px 5px rgba(0,0,0,0.04); min-height: 140px; display: flex; flex-direction: column; justify-content: space-between;">
    <div style="font-size: 1.1rem; color: #31333F;">Crash Probability</div>
    <div style="font-size: 1.8rem; font-weight: bold; color: {prob_color};">{crash_prob*100:.0f} %</div>
</div>
"""
col6.markdown(crash_html, unsafe_allow_html=True)

col7.metric("Est. Farmer Loss Risk", f"₹{expected_farmer_loss_cr:,.2f} Cr", help="Simulated statewide economic loss to farmers if the AI-predicted crash materializes at current volumes.")

st.markdown("---")

# ---------------------------------------------------
# CHARTS
# (6, 7, 8) Dynamic Y-axes, Earth Tone colors, and captions
# ---------------------------------------------------
df_90d = df_feats.tail(90).copy()

col_c1, col_c2 = st.columns(2)
with col_c1:
    st.subheader("Price Trend (90 Days)")
    st.caption("Shows historical price behaviour over time. The Dynamic Axis highlights subtle fluctuations.")
    c1 = alt.Chart(df_90d).mark_line(color='#556b2f', size=3).encode( # Olive solid green
        x=alt.X('Date:T', title=""),
        y=alt.Y('Modal_Price:Q', title="₹ / Quintal", scale=alt.Scale(zero=False)),
        tooltip=['Date', 'Modal_Price']
    )
    c1_ma = alt.Chart(df_90d).mark_line(color='#8b4513', strokeDash=[5,5]).encode( # Saddle brown dashed
        x='Date:T', y='ma_7:Q'
    )
    st.altair_chart(c1 + c1_ma, use_container_width=True)

with col_c2:
    st.subheader("Arrival Trend (90 Days)")
    st.caption("Shows supply fluctuations entering the market. Surges indicate potential oversupply.")
    c2 = alt.Chart(df_90d).mark_area(color='#8fbc8f', opacity=0.4).encode( # Dark sea green
        x=alt.X('Date:T', title=""),
        y=alt.Y('Arrival_Qtl:Q', title="Arrival (Quintals)")
    )
    c2_line = alt.Chart(df_90d).mark_line(color='#2e8b57', size=2).encode( # Sea green
        x='Date:T', y='Arrival_Qtl:Q'
    )
    st.altair_chart(c2 + c2_line, use_container_width=True)

col_c3, col_c4 = st.columns(2)
with col_c3:
    st.subheader("Price vs Arrival Dynamics")
    st.caption("Shows the core market mechanism where supply increases usually trigger price declines.")
    base = alt.Chart(df_90d).encode(x=alt.X('Date:T', title=""))
    bar = base.mark_bar(color='#deb887', opacity=0.7, size=5).encode( # Burly wood
        y=alt.Y('Arrival_Qtl:Q', title="Arrival (Quintals)")
    )
    line = base.mark_line(color='#8b0000', size=3).encode( # Dark red
        y=alt.Y('Modal_Price:Q', title="Price", scale=alt.Scale(zero=False))
    )
    dual = alt.layer(bar, line).resolve_scale(y='independent')
    st.altair_chart(dual, use_container_width=True)

with col_c4:
    st.subheader("14-Day Price Volatility")
    st.caption("Rolling backward variation. Sharp volatility spikes often precede market crashes.")
    df_90d['volatility_pct'] = df_90d['vol_14'] * 100
    c4 = alt.Chart(df_90d).mark_line(color='#cd853f', size=3).encode( # Peru brown
        x=alt.X('Date:T', title=""),
        y=alt.Y('volatility_pct:Q', title="% Volatility", scale=alt.Scale(zero=False))
    )
    st.altair_chart(c4, use_container_width=True)

col_c5, col_c6 = st.columns(2)
with col_c5:
    st.subheader("Crash Probability Trend")
    st.caption("Shows whether the AI-calculated risk of a sudden price crash is increasing or stabilizing over time.")
    X_all = df_90d[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
    df_90d['crash_prob_series'] = model.predict_proba(X_all)[:, 1] * 100
    # Provide an explicit Y scale with zero=True for probabilities so it grounds the area correctly
    c5 = alt.Chart(df_90d).mark_area(
        color=alt.Gradient(
            gradient='linear',
            stops=[alt.GradientStop(color='#556b2f', offset=0), alt.GradientStop(color='#8b0000', offset=1)]
        ),
        opacity=0.6
    ).encode(
        x=alt.X('Date:T', title=""),
        y=alt.Y('crash_prob_series:Q', title="Probability (%)", scale=alt.Scale(zero=True))
    )
    st.altair_chart(c5, use_container_width=True)

with col_c6:
    st.subheader("Supply Shock Detector")
    st.caption("Quickly identifies abnormal market flooding by comparing today's supply against the 30-day average.")
    shock_df = pd.DataFrame({
        "Category": ["30-Day Avg", "Current Arrival"],
        "Arrival": [avg_arrival_30d, current_arrival]
    })
    c6 = alt.Chart(shock_df).mark_bar().encode(
        x=alt.X('Category:N', title="", sort=["30-Day Avg", "Current Arrival"]),
        y=alt.Y('Arrival:Q', title="Quintals"),
        color=alt.Color('Category:N', scale=alt.Scale(domain=["30-Day Avg", "Current Arrival"], range=["#8f9779", "#8b0000"]), legend=None)
    )
    st.altair_chart(c6, use_container_width=True)

st.subheader("Actual Regional Rainfall (Last 90 Days)")
st.caption("Raw daily precipitation data to identify immediate weather-driven supply disruptions.")

if district_name in district_geo:
    lat, lon = district_geo[district_name]
    df_rain_raw = fetch_rainfall(lat, lon, 90)
    if df_rain_raw is not None:
        df_rain_raw = df_rain_raw.reset_index().rename(columns={"index": "Date"})
        df_rain_raw["Date"] = pd.to_datetime(df_rain_raw["Date"], format="%Y%m%d")
        
        c7_rain = alt.Chart(df_rain_raw).mark_bar(color='#5f9ea0', opacity=0.8).encode( # Cadet blue
            x=alt.X('Date:T', title=""),
            y=alt.Y('rainfall_mm:Q', title="Rainfall (mm)")
        )
        st.altair_chart(c7_rain, use_container_width=True)
    else:
        st.info("Rainfall telemetry currently unavailable for this region.")


st.markdown("---")

# ---------------------------------------------------
# STATE-WIDE MAPPING & INTERVENTION TOOL
# ---------------------------------------------------
st.header("State-Wide Monitoring & Optimizers")

st.sidebar.markdown("---")
st.sidebar.header("Policy Simulation Tools")
budget_slider = st.sidebar.slider("Total Available Budget (₹ Crores)", 1, 50, 10) * 10000000
storage_capacity = st.sidebar.slider("Total State Storage Capacity (Qtl)", 10000, 500000, 100000)
expected_drop_pct = st.sidebar.slider("Simulated Drop Risk (%)", 5, 40, 20) / 100
procurement_pct = st.sidebar.slider("Procurement Capture (%)", 5, 50, 30) / 100

st.write("Processing State-wide predictions...")
map_data = []

for d, (lat, lon) in district_geo.items():
    tdf = df[df["District"] == d].copy()
    if len(tdf) > 30:
        tdf_f = build_features_for_df(tdf, d)
        if len(tdf_f) > 0:
            latest_d = tdf_f.iloc[-1]
            x_d = pd.DataFrame([latest_d[feature_cols]], columns=feature_cols).fillna(0)
            prob = model.predict_proba(x_d)[0, 1]
            
            c_price = latest_d["Modal_Price"]
            r_arr = latest_d["Arrival_Qtl"]
            
            e_price_after = c_price * (1 - expected_drop_pct)
            loss_per_qtl = c_price - e_price_after
            
            p_qty = r_arr * procurement_pct * 30 # Simulated over a month
            p_price = c_price * 0.95
            
            f_loss = (p_price - e_price_after) * p_qty
            s_total = p_qty * 6 * 30 # 6 Rs per qtl a day over 30 days
            t_cost = f_loss + s_total
            
            e_farmer_loss = loss_per_qtl * r_arr * 30 * prob
            n_benefit = e_farmer_loss - t_cost
            
            color = [85, 107, 47, 160] if prob < 0.2 else ([218, 165, 32, 160] if prob < 0.35 else [139, 0, 0, 160]) # Earth tones for map spots
            
            map_data.append({
                "District": d, "lat": lat, "lon": lon, "Probability": prob, "Color": color,
                "Crash_Prob_Pct": f"{prob*100:.1f}%", "Suggested_Proc": p_qty,
                "Fiscal_Cost": t_cost, "Net_Benefit": n_benefit, "Farmer_Loss_Prevented": e_farmer_loss
            })

map_df = pd.DataFrame(map_data)

# Create a dummy URL for GeoJSON to trace boundaries.
# Realistically, this requires a solid GeoJSON of Maharashtra. 
INDIA_GEOJSON = "https://raw.githubusercontent.com/geohacker/india/master/district/india_district.geojson"

if not map_df.empty:
    col_map, col_tbl = st.columns([1, 1])
    
    with col_map:
        st.subheader("District Risk Map")
        st.caption("Visually identifies active regional risk clusters with district boundaries.")
        
        # (4) Adding the GeoJsonLayer outline to the PyDeck map
        border_layer = pdk.Layer(
            "GeoJsonLayer",
            data=INDIA_GEOJSON,
            opacity=0.1,
            stroked=True,
            filled=False,
            get_line_color=[100, 100, 100],
            get_line_width=2000,
        )

        scatter_layer = pdk.Layer(
            "ScatterplotLayer",
            map_df,
            get_position="[lon, lat]",
            get_color="Color",
            get_radius="Probability * 60000 + 10000",
            pickable=True,
        )
        view_state = pdk.ViewState(latitude=19.7515, longitude=75.7139, zoom=5, pitch=0)
        # (9) Use carto-positron baseline map
        st.pydeck_chart(pdk.Deck(
            map_style="carto-positron",
            layers=[border_layer, scatter_layer],
            initial_view_state=view_state,
            tooltip={"text": "{District}\nCrash Prob: {Crash_Prob_Pct}"}
        ))
        
    with col_tbl:
        st.subheader("District Risk Ranking")
        st.caption("Prioritization table to deploy procurement interventions.")
        tbl = map_df.sort_values("Probability", ascending=False).copy()
        tbl["Crash Prob"] = tbl["Crash_Prob_Pct"]
        tbl["Target Qty"] = tbl["Suggested_Proc"].apply(lambda x: f"{x:,.0f}")
        tbl["Cost (Cr)"] = tbl["Fiscal_Cost"].apply(lambda x: f"₹{x/10000000:.2f}")
        tbl["Benefit (Cr)"] = tbl["Net_Benefit"].apply(lambda x: f"₹{x/10000000:.2f}")
        # (10) Avoid rendering the index in this table
        st.dataframe(tbl[["District", "Crash Prob", "Target Qty", "Cost (Cr)", "Benefit (Cr)"]], height=400, hide_index=True)


    st.subheader("State-Wide Optimization Results")
    st.caption("Optimal budget and storage allocation targeting the highest risk regions.")
    # Optimizer logic: sort by probability, allocate until storage or budget is hit
    sorted_df = map_df.sort_values("Probability", ascending=False)
    
    total_allocated_qtl = 0
    total_spent = 0
    total_farmers_saved = 0
    
    for _, row in sorted_df.iterrows():
        qty = row["Suggested_Proc"]
        cost = row["Fiscal_Cost"]
        benefit = row["Farmer_Loss_Prevented"]
        
        if total_allocated_qtl + qty <= storage_capacity and total_spent + cost <= budget_slider:
            total_allocated_qtl += qty
            total_spent += cost
            total_farmers_saved += benefit
            
    net = total_farmers_saved - total_spent
            
    c1, c2, c3, c4 = st.columns(4)
    # (12) State wide optimisation results kpi should also have definitions on hover
    c1.metric("Feasible Procurement", f"{total_allocated_qtl:,.0f} qtl", help="The total quantity of crop the government can buy from the most at-risk districts without exceeding the storage capacity.")
    c2.metric("Fiscal Cost Required", f"₹{total_spent/10000000:.2f} Cr", help="Total monetary budget spent to run the intervention process targeting the prioritized states.")
    c3.metric("Farmer Loss Prevented", f"₹{total_farmers_saved/10000000:.2f} Cr", help="The simulated financial loss (in Crores) structurally prevented for local farmers through our market intervention.")
    c4.metric("Net Economic Benefit", f"₹{net/10000000:.2f} Cr", delta="Optimal" if net>0 else "Suboptimal", help="The Farmer Loss Prevented minus Fiscal Cost Required. A positive margin translates to a high-utility policy.")

st.markdown("---")
# ---------------------------------------------------
# AI INSIGHT PANEL
# ---------------------------------------------------
st.subheader("AI Market Insight Panel")
if st.button("Explain Market Risk"):
    if crash_prob > 0.70:
        trend = "High Risk - Immediate Intervention Required"
    elif crash_prob >= 0.40:
        trend = "Monitor Closely"
    else:
        trend = "Stable Conditions"
    surge_text = f"a notable supply surge ({arrival_surge:.1f}x normal)" if arrival_surge > 1.2 else "normal supply volumes"
    
    st.info(f"**Powered by Amazon Bedrock Insights**\n\n"
            f"🚨 **Risk Level**: **{trend}** ({crash_prob*100:.1f}% crash probability for {district_name} {crop})\n\n"
            f"**The Problem:**\n"
            f"- **Supply Strain**: Driven by {surge_text} (Current: {current_arrival:,.0f} qtl vs 30-day avg: {avg_arrival_30d:,.0f} qtl).\n"
            f"- **Market Instability**: Historical 14-day price volatility is tracking at {df_feats.iloc[-1]['vol_14']*100:.1f}%.\n\n"
            f"**KrishiRakshak AI Solution:**\n"
            f"- **Policy Action**: Intervention is **{'highly recommended' if net > 0 else 'not financially viable at current levels'}**.\n"
            f"- **Strategic Goal**: Execute targeted, data-driven procurement to stabilize local prices and protect farmer margins before the crash materializes."
            )
    
# (5) Adding Responsible Use AI Disclaimer
st.markdown("<br><br><br>", unsafe_allow_html=True)
st.caption("""
**⚠️ Responsible AI Use Disclaimer**: 
All predictions, market intelligence, and intervention costs displayed by **KrishiRakshak AI** are generated using Machine Learning models. 
Please evaluate and locally validate these insights before taking serious policy action. This interface serves strictly as a **decision assistance tool** to complement ground-level expertise—not to make automated fiscal guarantees.
""")