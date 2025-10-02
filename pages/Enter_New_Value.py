import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime


# ===============================
# FIXED NAVBAR
# ===============================
# ===============================
# HIDE STREAMLIT TOP BAR & PADDING
# ===============================
st.markdown("""
<style>
    header[data-testid="stHeader"] { display: none; }
    footer {visibility: hidden;}
    div.block-container { padding-top: 90px !important; }
    
    /* Navbar styling */
    .navbar {
        position: fixed; 
        top: 0; 
        left: 0; 
        right: 0; 
        z-index: 9999;
        display: flex; 
        justify-content: space-between; 
        align-items: center;
        background: linear-gradient(90deg, #2e7d32, #4caf50, #66bb6a);
        padding: 10px 40px; 
        box-shadow: 0 4px 15px rgba(0,0,0,0.4);
        backdrop-filter: blur(10px);
    }
    
    .navbar .logo { 
        display: flex; 
        align-items: center; 
    }
    
    .navbar .logo img { 
        height: 40px; 
        width: auto;
        object-fit: contain;
        cursor: pointer;
        filter: drop-shadow(0 2px 4px rgba(0,0,0,0.3));
    }
    
    .nav-links { 
        display: flex; 
        align-items: center; 
        gap: 30px;
    }
    
    .nav-item { 
        position: relative;
    }
    
    .nav-item > a {
        color: white; 
        text-decoration: none; 
        font-weight: 600;
        font-size: 17px; 
        padding: 10px 16px; 
        transition: all 0.3s ease;
        border-radius: 8px;
        display: inline-block;
    }
    
    .nav-item > a:hover { 
        background: rgba(255, 255, 255, 0.2);
        transform: translateY(-2px);
    }
    
    .dropdown-content {
        visibility: hidden;
        opacity: 0;
        position: absolute; 
        top: 50px; 
        left: 0;
        background: white; 
        min-width: 200px; 
        border-radius: 8px;
        box-shadow: 0 8px 16px rgba(0,0,0,0.3);
        transform: translateY(-10px);
        transition: all 0.3s ease;
        pointer-events: none;
    }
    
    .dropdown-content a {
        display: block; 
        padding: 12px 20px; 
        text-decoration: none;
        color: #333; 
        font-weight: 500; 
        transition: all 0.3s ease;
        border-bottom: 1px solid #f0f0f0;
    }
    
    .dropdown-content a:last-child {
        border-bottom: none;
    }
    
    .dropdown-content a:hover { 
        background: linear-gradient(90deg, #4caf50, #66bb6a);
        color: white; 
        padding-left: 25px;
    }
    
    .nav-item:hover .dropdown-content { 
        visibility: visible;
        opacity: 1;
        transform: translateY(0);
        pointer-events: auto;
    }
    
    .main { 
        margin-top: 80px; 
    }
</style>

<div class="navbar">
    <div class="logo">
        <div style="width: 40px; height: 40px; background: white; border-radius: 50%; display: flex; align-items: center; justify-content: center; box-shadow: 0 2px 4px rgba(0,0,0,0.3);">
            <span style="color: #2e7d32; font-size: 18px; font-weight: bold;">AT</span>
        </div>
    </div>
    <div class="nav-links">
        <div class="nav-item"><a href="/">Home</a></div>
        <div class="nav-item">
            <a href="/Analyze_Previous">Analyze Data ▾</a>
            <div class="dropdown-content">
                <a href="/analyze_internet">Analyze Internet</a>
                <a href="/analyze_fixe">Analyze Fixe</a>
                <a href="/analyze_mobile">Analyze Mobile</a>
            </div>
        </div>
        <div class="nav-item"><a href="/Enter_New_Value">New Value</a></div>
        <div class="nav-item"><a href="/Forecast_Coming">Forecast Coming</a></div>
    </div>
</div>
<div class="main">
""", unsafe_allow_html=True)

# =========================
# Paths to data
# =========================
DATA_PATHS = {
    "Internet": "data/internet_data.csv",
    "Fixe": "data/fixe_data.csv",
    "Mobile": "data/mobile_data.csv",
}

# =========================
# Page Config
# =========================
st.set_page_config(page_title="Enter New Value", layout="wide")

# =========================
# Custom CSS for style - DARK THEME
# =========================
st.markdown(
    """
    <style>
        .stApp {
            background: linear-gradient(135deg, #0c0c0c, #1a1a2e, #16213e);
            color: #ffffff;
        }
        .main-title {
            font-size: 36px;
            color: #4CAF50;
            font-weight: bold;
        }
        .sub-header {
            font-size: 18px;
            color: #81C784;
        }
        .card {
            background: rgba(30, 30, 46, 0.9);
            padding: 25px;
            border-radius: 15px;
            border: 1px solid #2e7d32;
            margin-bottom: 25px;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
        }
        .stButton>button {
            background: linear-gradient(45deg, #2e7d32, #4caf50);
            color: white;
            border: none;
            padding: 12px 30px;
            border-radius: 8px;
            font-size: 16px;
            font-weight: bold;
            cursor: pointer;
            transition: all 0.3s ease;
            width: 100%;
        }
        .stButton>button:hover {
            background: linear-gradient(45deg, #1b5e20, #2e7d32);
            transform: translateY(-2px);
        }
        .success-box {
            background: rgba(46, 125, 50, 0.2);
            border: 1px solid #4CAF50;
            border-radius: 10px;
            padding: 15px;
            margin: 10px 0;
        }
        .warning-box {
            background: rgba(255, 152, 0, 0.2);
            border: 1px solid #FF9800;
            border-radius: 10px;
            padding: 15px;
            margin: 10px 0;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# =========================
# Header
# =========================
st.markdown("<div class='main-title'>📊 Enter New Data Values</div>", unsafe_allow_html=True)
st.markdown("<p class='sub-header'>Add new quarterly data to keep your datasets up-to-date with future values.</p>", unsafe_allow_html=True)
st.markdown("---")

# =========================
# Data Loading and Processing Functions
# =========================
@st.cache_data
def load_data(domain):
    try:
        return pd.read_csv(DATA_PATHS[domain])
    except Exception as e:
        st.error(f"Error loading {domain} data: {e}")
        return pd.DataFrame()

def get_next_period(df):
    """Determine the next period (year and quarter) to add data for"""
    time_cols = [col for col in df.columns if col not in [df.columns[0]]]
    if not time_cols:
        return "2025", "S1"

    periods = []
    for col in time_cols:
        try:
            if "_" in col:
                year, quarter = col.split("_")
                periods.append((int(year), quarter))
        except:
            continue

    if not periods:
        return "2025", "S1"

    latest_year = max([p[0] for p in periods])
    latest_quarters = [p[1] for p in periods if p[0] == latest_year]

    quarter_order = {"S1": 1, "S2": 2, "S3": 3, "S4": 4,
                     "T1": 1, "T2": 2, "T3": 3, "T4": 4}

    if latest_quarters:
        latest_quarter = max(latest_quarters, key=lambda x: quarter_order.get(x, 0))
        latest_quarter_num = quarter_order.get(latest_quarter, 0)

        if latest_quarter_num < 4:
            next_quarter_num = latest_quarter_num + 1
            next_quarter = [k for k, v in quarter_order.items() if v == next_quarter_num][0]
            next_year = latest_year
        else:
            next_quarter_num = 1
            next_quarter = [k for k, v in quarter_order.items() if v == next_quarter_num][0]
            next_year = latest_year + 1
    else:
        next_year = latest_year
        next_quarter = "S1"

    return str(next_year), next_quarter

def get_period_format(df):
    time_cols = [col for col in df.columns if col not in [df.columns[0]]]
    for col in time_cols:
        if "_" in col:
            _, period = col.split("_")
            return "S" if "S" in period else "T"
    return "S"

# =========================
# Domain Selection
# =========================
st.subheader("🌐 Step 1: Choose the domain you want to update")
domain = st.selectbox("Select Domain", list(DATA_PATHS.keys()))

df = load_data(domain)
if df.empty:
    st.error("❌ Unable to load data. Please check if the data files exist.")
    st.stop()

# =========================
# Detect domain type
# =========================
id_column = df.columns[0]  # "Metric" for Mobile/Fixe, "Value" for Internet

# =========================
# Metric Selection
# =========================
st.subheader("📊 Step 2: Select the metric you want to update")
metrics = df[id_column].dropna().unique()

if len(metrics) == 0:
    st.error("No metrics found in the dataset.")
    st.stop()

metric = st.selectbox("Select Metric", metrics)

# =========================
# Determine Next Period
# =========================
st.subheader("📅 Step 3: Next Available Period")
next_year, next_quarter = get_next_period(df)
period_format = get_period_format(df)

col1, col2 = st.columns(2)
with col1:
    selected_year = st.selectbox(
        "Year",
        options=[str(int(next_year) + 1), next_year, str(int(next_year) + 2)],
        index=1
    )
with col2:
    quarters = ["S1", "S2", "S3", "S4"] if period_format == "S" else ["T1", "T2", "T3", "T4"]
    selected_quarter = st.selectbox(
        "Quarter",
        options=quarters,
        index=quarters.index(next_quarter) if next_quarter in quarters else 0
    )

next_period = f"{selected_year}_{selected_quarter}"

# =========================
# Input Field
# =========================
st.subheader("💾 Step 4: Enter the new value")

# Show current value trend
metric_data = df[df[id_column] == metric]
if not metric_data.empty:
    time_cols = [col for col in df.columns if col not in [id_column]]
    recent_cols = sorted(time_cols)[-4:]
    recent_values = []
    for col in recent_cols:
        val = metric_data[col].iloc[0]
        if pd.notna(val):
            recent_values.append((col, val))

    if recent_values:
        st.write("**Recent values for context:**")
        for period, value in recent_values[-3:]:
            try:
                numeric_val = float(value)
                st.write(f"- {period}: {numeric_val:,.0f}")
            except (ValueError, TypeError):
                st.write(f"- {period}: {value}")

new_value = st.text_input(
    f"Enter value for **{metric}** at **{next_period}**",
    placeholder="Enter numeric value..."
)

if new_value:
    try:
        float(new_value.replace(" ", "").replace(",", ""))
        st.success("✅ Valid numeric format")
    except ValueError:
        st.error("❌ Please enter a valid number")

# =========================
# Save Functionality
# =========================
st.subheader("🚀 Step 5: Save the new value")

if st.button("💾 Save New Data Value", use_container_width=True):
    if not new_value.strip():
        st.error("⚠️ Please enter a value before saving.")
    else:
        try:
            clean_value = float(new_value.replace(" ", "").replace(",", ""))
            df_updated = df.copy()

            if next_period not in df_updated.columns:
                time_cols = [col for col in df_updated.columns if col not in [id_column]]
                time_cols.append(next_period)

                def sort_periods(period):
                    try:
                        year, quarter = period.split("_")
                        quarter_order = {"S1": 1, "S2": 2, "S3": 3, "S4": 4,
                                         "T1": 1, "T2": 2, "T3": 3, "T4": 4}
                        return (int(year), quarter_order.get(quarter, 0))
                    except:
                        return (0, 0)

                time_cols_sorted = sorted(time_cols, key=sort_periods)
                new_columns = [id_column] + time_cols_sorted
                df_updated = df_updated.reindex(columns=new_columns)

            mask = df_updated[id_column] == metric
            df_updated.loc[mask, next_period] = clean_value
            df_updated.to_csv(DATA_PATHS[domain], index=False)

            st.markdown(f"""
            <div class="success-box">
            <h4>✅ Successfully Saved!</h4>
            <p><b>Metric:</b> {metric}</p>
            <p><b>Period:</b> {next_period}</p>
            <p><b>Value:</b> {clean_value:,.0f}</p>
            <p>The dataset has been updated with the new value.</p>
            </div>
            """, unsafe_allow_html=True)

            st.info("🔄 **Next steps:** The page will refresh to allow adding the next period.")
            st.rerun()

        except ValueError:
            st.error("❌ Invalid number format. Please enter a valid numeric value.")
        except Exception as e:
            st.error(f"❌ Error saving data: {str(e)}")

# =========================
# Dataset Information
# =========================
with st.expander("📋 Dataset Overview"):
    st.write(f"**Domain:** {domain}")
    st.write(f"**Total Metrics:** {len(metrics)}")
    st.write(f"**Time Periods:** {len([col for col in df.columns if col not in [id_column]])}")
    st.write(f"**Period Format:** {period_format} format")
    st.write("**Data Preview:**")
    st.dataframe(df.head(), use_container_width=True)

st.markdown("---")
st.info("💡 **Tip:** This tool is designed for adding new future data. Use consistent numeric formatting for accurate results.")
