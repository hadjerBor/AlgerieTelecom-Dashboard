import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
from io import BytesIO
import warnings
warnings.filterwarnings('ignore')

# Advanced ML imports with error handling
HAS_STATSMODELS = True
HAS_PMDARIMA = True
HAS_TENSORFLOW = True
HAS_XGBOOST = True
HAS_SKLEARN = True

try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from statsmodels.tsa.stattools import adfuller
    import statsmodels.api as sm
except Exception:
    HAS_STATSMODELS = False

try:
    from pmdarima import auto_arima
except Exception:
    HAS_PMDARIMA = False

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, LSTM, Dropout
    from tensorflow.keras.optimizers import Adam
except Exception:
    HAS_TENSORFLOW = False

try:
    import xgboost as xgb
except Exception:
    HAS_XGBOOST = False

try:
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import LinearRegression, Ridge
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
except Exception:
    HAS_SKLEARN = False

# ===============================
# PAGE CONFIGURATION
# ===============================
st.set_page_config(
    page_title="Advanced Forecast Data",
    page_icon="📈",
    layout="wide"
)

# ===============================
# HIDE STREAMLIT TOP BAR & STYLING
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

# ===============================
# DARK THEME CUSTOM CSS
# ===============================
st.markdown(
    """
    <style>
        .stApp {
            background: linear-gradient(135deg, #0c0c0c, #1a1a2e, #16213e);
            color: #ffffff;
        }
        h1, h2, h3, h4 {
            color: #ffffff;
            border-bottom: 1px solid #2e7d32;
            padding-bottom: 10px;
        }
        .control-card {
            background: rgba(30, 30, 46, 0.9);
            padding: 25px;
            border-radius: 15px;
            border: 1px solid #2e7d32;
            margin-bottom: 25px;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
        }
        .metric-card {
            background: rgba(30, 30, 46, 0.8);
            padding: 20px;
            border-radius: 12px;
            border: 1px solid #2e7d32;
            margin-bottom: 25px;
        }
        .stButton button {
            background: linear-gradient(45deg, #2e7d32, #4caf50);
            color: white;
            font-weight: bold;
            border: none;
            padding: 12px 30px;
            border-radius: 8px;
            transition: all 0.3s ease;
            width: 100%;
        }
        .stButton button:hover {
            background: linear-gradient(45deg, #1b5e20, #2e7d32);
            transform: translateY(-2px);
        }
    </style>
    """,
    unsafe_allow_html=True
)

# ===============================
# TITLE & USER GUIDE
# ===============================
st.title("📈 Advanced Time Series Forecasting - Algeria Telecom")
st.write("             ")
st.markdown(
    """<div class="control-card"><h4>ℹ️ Advanced Forecasting Guide</h4>
    Welcome to the <b>Advanced Time Series Forecasting Dashboard</b>.<br>
    Select a domain, metric, and forecasting models. The system will automatically:
    <ul>
        <li>📊 Train multiple ML models and compare their performance</li>
        <li>🤖 Generate forecasts with confidence intervals</li>
        <li>📈 Create comprehensive visualizations and analysis</li>
        <li>📋 Provide detailed evaluation metrics and model insights</li>
    </ul></div>""",
    unsafe_allow_html=True
)

# ===============================
# DATA LOADING FUNCTION
# ===============================
@st.cache_data
def load_data(domain):
    try:
        if domain == "Internet":
            df = pd.read_csv("data/internet_data.csv")
        elif domain == "Mobile":
            df = pd.read_csv("data/mobile_data.csv")
        else:  # Fixe
            df = pd.read_csv("data/fixe_data.csv")
        return df
    except Exception as e:
        st.error(f"Error loading {domain} data: {e}")
        return pd.DataFrame()

# ===============================
# DATA PROCESSING FUNCTION WITH NULL HANDLING
# ===============================
def process_data_advanced(df, domain):
    """Enhanced data processing with proper null value handling"""
    try:
        id_col = df.columns[0]
        metrics = df[id_col].tolist()

        # Transpose and clean
        df_t = df.set_index(id_col).T.reset_index()
        df_t.columns = ['Period'] + metrics

        # Melt to long format
        melted = pd.melt(df_t, id_vars=['Period'], value_vars=metrics,
                         var_name='Metric', value_name='Value')

        # Enhanced period parsing
        melted['Year'] = melted['Period'].str.extract(r'(\d{4})')
        melted['Period_Type'] = melted['Period'].str.extract(r'_(\D\d)') \
                                    .fillna(melted['Period'].str.extract(r'([A-Z]\d)'))

        # Clean and convert values with better null handling
        melted['Value'] = melted['Value'].astype(str).str.replace(' ', '', regex=False)
        melted['Value'] = melted['Value'].replace(['', 'nan', 'None', 'null', '--', 'N/A'], np.nan)
        melted['Value'] = pd.to_numeric(melted['Value'], errors='coerce')

        # Create proper dates
        def create_date(row):
            try:
                year = int(row['Year'])
                p = row['Period_Type']
                if pd.isna(p):
                    return pd.Timestamp(year=year, month=1, day=1)
                p = p.strip()
                mapping = {'T1': 1, 'S1': 1, 'T2': 4, 'S2': 4, 'T3': 7, 'S3': 7, 'T4': 10, 'S4': 10}
                month = mapping.get(p, 1)
                return pd.Timestamp(year=year, month=month, day=1)
            except Exception:
                return pd.NaT

        melted['Date'] = melted.apply(create_date, axis=1)
        melted = melted.dropna(subset=['Date']).sort_values(['Metric', 'Date'])
        melted['Year'] = melted['Date'].dt.year

        return melted, metrics
    except Exception as e:
        st.error(f"Error processing {domain} data: {e}")
        return pd.DataFrame(), []

# ===============================
# FORECASTING MODELS WITH BETTER NULL HANDLING
# ===============================

def create_forecast_model(data, dates, model_type, forecast_periods=8):
    """Create forecast using different models with proper error handling and null value management"""
    try:
        # Handle null values by forward filling and interpolation
        data_clean = data.copy()
        if data_clean.isnull().any():
            data_clean = data_clean.fillna(method='ffill').fillna(method='bfill')
            if data_clean.isnull().any():
                data_clean = data_clean.interpolate()
            if data_clean.isnull().any():
                data_clean = data_clean.fillna(data_clean.mean())

        if model_type == "ARIMA":
            if not HAS_STATSMODELS:
                return None
            model = ARIMA(data_clean, order=(1, 1, 1))
            fitted_model = model.fit()
            forecast = fitted_model.forecast(steps=forecast_periods)
            confidence_int = fitted_model.get_forecast(steps=forecast_periods).conf_int()
            
            return {
                'forecast': forecast,
                'lower_bound': confidence_int.iloc[:, 0],
                'upper_bound': confidence_int.iloc[:, 1],
                'model': fitted_model,
                'mae': None
            }
            
        elif model_type == "SARIMAX":
            if not HAS_STATSMODELS:
                return None
            model = SARIMAX(data_clean, order=(1, 1, 1), seasonal_order=(1, 1, 1, 4))
            fitted_model = model.fit(disp=False)
            forecast = fitted_model.forecast(steps=forecast_periods)
            confidence_int = fitted_model.get_forecast(steps=forecast_periods).conf_int()
            
            return {
                'forecast': forecast,
                'lower_bound': confidence_int.iloc[:, 0],
                'upper_bound': confidence_int.iloc[:, 1],
                'model': fitted_model,
                'mae': None
            }
            
        elif model_type == "Exponential Smoothing":
            if not HAS_STATSMODELS:
                return None
            model = ExponentialSmoothing(data_clean, seasonal='add', seasonal_periods=4)
            fitted_model = model.fit()
            forecast = fitted_model.forecast(forecast_periods)
            
            # Simple confidence interval estimation
            residuals = fitted_model.resid
            std_error = np.std(residuals)
            
            return {
                'forecast': forecast,
                'lower_bound': forecast - 1.96 * std_error,
                'upper_bound': forecast + 1.96 * std_error,
                'model': fitted_model,
                'mae': None
            }

        elif model_type == "Prophet":
            # Prepare Prophet data
            prophet_df = pd.DataFrame({
                'ds': dates,
                'y': data_clean.values
            })
            
            m = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=False,
                daily_seasonality=False,
                changepoint_prior_scale=0.05
            )
            m.fit(prophet_df)
            
            future = m.make_future_dataframe(periods=forecast_periods, freq='Q')
            forecast_df = m.predict(future)
            
            # Extract forecast portion
            forecast = forecast_df.tail(forecast_periods)['yhat']
            lower_bound = forecast_df.tail(forecast_periods)['yhat_lower']
            upper_bound = forecast_df.tail(forecast_periods)['yhat_upper']
            
            return {
                'forecast': forecast,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'model': m,
                'full_forecast': forecast_df,
                'mae': None
            }
            
        elif model_type == "Linear Regression":
            if not HAS_SKLEARN:
                return None
            X = np.array(range(len(data_clean))).reshape(-1, 1)
            y = data_clean.values
            
            model = LinearRegression()
            model.fit(X, y)
            
            future_X = np.array(range(len(data_clean), len(data_clean) + forecast_periods)).reshape(-1, 1)
            forecast = model.predict(future_X)
            
            # Calculate residuals for confidence interval
            y_pred = model.predict(X)
            residuals = y - y_pred
            std_error = np.std(residuals)
            
            return {
                'forecast': pd.Series(forecast),
                'lower_bound': pd.Series(forecast - 1.96 * std_error),
                'upper_bound': pd.Series(forecast + 1.96 * std_error),
                'model': model,
                'mae': mean_absolute_error(y, y_pred)
            }
        elif model_type == "XGBoost":
            if not HAS_XGBOOST:
                return None

            # Create lag features - Fixed approach
            n_lags = min(4, len(data_clean) - 1)
            if n_lags < 1:
                return None
                
            # Create feature matrix
            features = []
            targets = []

            for i in range(n_lags, len(data_clean)):
                # Get lag features as a list
                lag_vals = [data_clean.iloc[i-j] for j in range(1, n_lags + 1)]
                features.append(lag_vals)
                targets.append(data_clean.iloc[i])

            if len(features) < 5:  # Need minimum data points
                return None

            X = np.array(features)
            y = np.array(targets)

            from xgboost import XGBRegressor
            model = XGBRegressor(n_estimators=100, random_state=42)
            model.fit(X, y)

            # Generate forecast
            forecast_values = []
            last_values = data_clean.tail(n_lags).values.tolist()

            for _ in range(forecast_periods):
                # Use the last n_lags values as features
                current_features = last_values[-n_lags:]
                pred = model.predict([current_features])[0]
                forecast_values.append(pred)
                last_values.append(pred)

            # Calculate confidence interval
            y_pred = model.predict(X)
            residuals = y - y_pred
            std_error = np.std(residuals)

            forecast_series = pd.Series(forecast_values)

            return {
                'forecast': forecast_series,
                'lower_bound': forecast_series - 1.96 * std_error,
                'upper_bound': forecast_series + 1.96 * std_error,
                'model': model,
                'mae': mean_absolute_error(y, y_pred)
            }
    
    except Exception as e:
        st.error(f"Error in {model_type} model: {str(e)}")
        return None

# ===============================
# DARK PLOTLY TEMPLATE
# ===============================
dark_template = go.layout.Template(
    layout=go.Layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        xaxis=dict(gridcolor='rgba(255,255,255,0.08)'),
        yaxis=dict(gridcolor='rgba(255,255,255,0.08)')
    )
)

# ===============================
# MAIN APPLICATION LOGIC
# ===============================

# Domain selection
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("📊 Internet Data", use_container_width=True, key="internet_btn_advanced"):
        st.session_state.selected_domain = "Internet"
        st.session_state.data_loaded = False
with col2:
    if st.button("📱 Mobile Data", use_container_width=True, key="mobile_btn_advanced"):
        st.session_state.selected_domain = "Mobile"
        st.session_state.data_loaded = False
with col3:
    if st.button("📞 Fixed Line Data", use_container_width=True, key="fixe_btn_advanced"):
        st.session_state.selected_domain = "Fixe"
        st.session_state.data_loaded = False

# Initialize session state
if 'selected_domain' not in st.session_state:
    st.session_state.selected_domain = "Internet"
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = pd.DataFrame()
if 'available_metrics' not in st.session_state:
    st.session_state.available_metrics = []

st.markdown("</div>", unsafe_allow_html=True)

# Load and process data
if st.session_state.selected_domain and not st.session_state.data_loaded:
    with st.spinner(f"Loading {st.session_state.selected_domain} data..."):
        raw_data = load_data(st.session_state.selected_domain)
        if not raw_data.empty:
            processed_data, metrics = process_data_advanced(raw_data, st.session_state.selected_domain)
            if not processed_data.empty:
                st.session_state.processed_data = processed_data
                st.session_state.available_metrics = metrics
                st.session_state.data_loaded = True
                st.success(f"✅ {st.session_state.selected_domain} data loaded successfully!")
            else:
                st.error(f"❌ Failed to process {st.session_state.selected_domain} data")

# ===============================
# FIXED FORECASTING INTERFACE
# ===============================
if st.session_state.data_loaded and not st.session_state.processed_data.empty:
    
    # Control panel - Fixed metric selection
    c1, c2, c3 = st.columns(3)
    
    with c1:
        st.subheader("📊 Select Metric")
        st.write("             ")
        selected_metric = st.selectbox(
            "Choose metric to forecast:", 
            st.session_state.available_metrics, 
            key="metric_select_forecast"
        )
    
    with c2:
        st.subheader("📅 Training Period")
        st.write("             ")
        available_years = sorted(st.session_state.processed_data['Year'].unique())
        if available_years:
            training_year = st.slider(
                "Train model using data up to year:", 
                min_value=int(min(available_years)),
                max_value=int(max(available_years)), 
                value=int(max(available_years)) - 1,
                key="year_slider_forecast"
            )
        else:
            st.warning("No years available in data")
            training_year = datetime.now().year
    
    with c3:
        st.subheader("🔮 Forecast Horizon")
        st.write("             ")
        forecast_periods = st.slider(
            "Number of periods to forecast:", 
            min_value=4, 
            max_value=20, 
            value=8, 
            key="periods_slider_forecast"
        )

    # Model selection
    st.subheader("🤖 Model Selection")
    available_models = ["Prophet", "Linear Regression"]
    if HAS_STATSMODELS:
        available_models.extend(["ARIMA", "SARIMAX", "Exponential Smoothing"])
    if HAS_XGBOOST:
        available_models.append("XGBoost")
    st.write("    ")
    selected_models = st.multiselect(
        "Select models to run:",
        available_models,
        default=["Prophet", "SARIMAX", "Linear Regression"] if HAS_STATSMODELS else ["Prophet", "Linear Regression"],
        key="model_select_advanced"
    )

    if not selected_models:
        st.warning("Please select at least one forecasting model")
    elif not selected_metric:
        st.warning("Please select a metric to forecast")
    else:
        # Generate forecast button
        if st.button("🚀 Generate Advanced Forecast", use_container_width=True, key="generate_forecast"):
            try:
                # Get training data with proper filtering
                df_all = st.session_state.processed_data.copy()
                metric_data = df_all[df_all['Metric'] == selected_metric].dropna(subset=['Date']).sort_values('Date')
                train_data = metric_data[metric_data['Year'] <= int(training_year)].copy()
                
                if train_data.empty or len(train_data) < 8:
                    st.error("❌ Not enough training data points (minimum 8 required)")
                else:
                    data = train_data['Value']
                    dates = train_data['Date']
                    
                    # Handle null values in training data
                    if data.isnull().any():
                        st.warning(f"⚠️ Found {data.isnull().sum()} null values in data. Applying interpolation...")
                        data = data.fillna(method='ffill').fillna(method='bfill')
                        if data.isnull().any():
                            data = data.interpolate()
                        if data.isnull().any():
                            data = data.fillna(data.mean())
                    
                    # Create future dates
                    last_date = dates.iloc[-1]
                    future_dates = pd.date_range(
                        start=last_date + pd.DateOffset(months=3),
                        periods=forecast_periods,
                        freq='Q'
                    )
                    
                    # Generate forecasts
                    forecasts = {}
                    progress_bar = st.progress(0)
                    
                    for idx, model_name in enumerate(selected_models):
                        progress_bar.progress((idx + 1) / len(selected_models))
                        with st.spinner(f"Training {model_name}..."):
                            forecast_result = create_forecast_model(data, dates, model_name, forecast_periods)
                            if forecast_result:
                                forecasts[model_name] = forecast_result
                    
                    progress_bar.progress(1.0)
                    
                    if not forecasts:
                        st.error("❌ All models failed to train. Please check your data and try again.")
                    else:
                        st.success(f"✅ Successfully trained {len(forecasts)} models!")
                        
                        # Scale factor for display
                        scale_factor = 1e6 if 'Subscribers' in selected_metric else 1e12 if 'Traffic' in selected_metric else 1
                        scale_label = 'Millions' if 'Subscribers' in selected_metric else 'Téraoctets' if 'Traffic' in selected_metric else 'Units'
                        
                        # ===============================
                        # MAIN FORECAST VISUALIZATION
                        # ===============================
                        st.subheader("📊 Multi-model Forecast Comparison")
                        
                        fig = go.Figure()
                        
                        # Historical data
                        fig.add_trace(go.Scatter(
                            x=dates,
                            y=data / scale_factor,
                            mode='lines+markers',
                            name='Historical Data',
                            line=dict(color='#4CAF50', width=3),
                            marker=dict(size=6)
                        ))
                        
                        # Forecasts with different colors
                        colors = ['#2196F3', '#E91E63', '#FF9800', '#9C27B0', '#00BCD4', '#795548']
                        
                        for i, (model_name, forecast_data) in enumerate(forecasts.items()):
                            color = colors[i % len(colors)]
                            
                            # Main forecast line
                            fig.add_trace(go.Scatter(
                                x=future_dates,
                                y=forecast_data['forecast'] / scale_factor,
                                mode='lines+markers',
                                name=f'{model_name}',
                                line=dict(color=color, width=2, dash='dash'),
                                marker=dict(size=6)
                            ))
                            
                            # Confidence interval
                            if 'upper_bound' in forecast_data and 'lower_bound' in forecast_data:
                                fig.add_trace(go.Scatter(
                                    x=list(future_dates) + list(future_dates[::-1]),
                                    y=list(forecast_data['upper_bound'] / scale_factor) + 
                                      list(forecast_data['lower_bound'][::-1] / scale_factor),
                                    fill='toself',
                                    fillcolor=f'rgba(33,150,243,0.15)',
                                    line=dict(color='rgba(255,255,255,0)'),
                                    name=f'{model_name} - Confidence Interval',
                                    showlegend=False
                                ))
                        
                        fig.update_layout(
                            template=dark_template,
                            title=f'Advanced Forecast: {selected_metric.replace("_", " ").title()}',
                            xaxis_title='Date',
                            yaxis_title=f'{selected_metric.replace("_", " ").title()} ({scale_label})',
                            hovermode='x unified',
                            height=600,
                            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # ===============================
                        # ERROR ANALYSIS PLOTS (REQUIRED)
                        # ===============================
                        
                        # Prepare residuals for Prophet model (most comprehensive)
                        if 'Prophet' in forecasts and 'full_forecast' in forecasts['Prophet']:
                            prophet_forecast = forecasts['Prophet']['full_forecast']
                            prophet_df = pd.DataFrame({'ds': dates, 'y': data.values})
                            
                            # Get fitted values for the training period
                            fitted = prophet_forecast[prophet_forecast['ds'].isin(dates)].merge(
                                prophet_df, on='ds', how='inner'
                            )
                            fitted['error'] = fitted['y'] - fitted['yhat']
                            fitted['year'] = fitted['ds'].dt.year
                            fitted['month'] = fitted['ds'].dt.month
                            
                            # Error Analysis Section
                            st.subheader("- Model Error Analysis")
                            
                            row1_col1, row1_col2 = st.columns(2)
                            
                            with row1_col1:
                                st.subheader("📊 Error Distribution (Residuals)")
                                fig_err = px.histogram(fitted, x='error', nbins=20, marginal='box', 
                                                     title="Residual Distribution")
                                fig_err.update_layout(template=dark_template, height=380)
                                st.plotly_chart(fig_err, use_container_width=True)
                                
                                st.subheader("🔍 Actual vs Predicted")
                                fig_scatter = go.Figure()
                                fig_scatter.add_trace(go.Scatter(
                                    x=fitted['y'], y=fitted['yhat'],
                                    mode='markers',
                                    marker=dict(size=8, color=fitted['error'], colorscale='RdBu', showscale=True),
                                    name='Actual vs Predicted'
                                ))
                                minv = min(fitted['y'].min(), fitted['yhat'].min())
                                maxv = max(fitted['y'].max(), fitted['yhat'].max())
                                fig_scatter.add_trace(go.Scatter(
                                    x=[minv, maxv], y=[minv, maxv],
                                    mode='lines', line=dict(color='lime', dash='dash'), name='Perfect'))
                                fig_scatter.update_layout(template=dark_template, height=380, xaxis_title='Actual', yaxis_title='Predicted')
                                st.plotly_chart(fig_scatter, use_container_width=True)
                            
                            with row1_col2:
                                st.subheader("📈 Cumulative Error (Tracking Signal)")
                                fitted['cum_error'] = fitted['error'].cumsum()
                                fig_cum = px.line(fitted, x='ds', y='cum_error', markers=True, title='Cumulative Error')
                                fig_cum.update_traces(line=dict(color='#E91E63', width=3))
                                # Add control band: +/- 2*std
                                std = fitted['error'].std()
                                if not np.isnan(std):
                                    fig_cum.add_hrect(y0=-2*std, y1=2*std, fillcolor='rgba(76,175,80,0.12)', line_width=0)
                                fig_cum.update_layout(template=dark_template, height=380)
                                st.plotly_chart(fig_cum, use_container_width=True)
                                
                                st.subheader("🌡️ Error Heatmap (Year x Month)")
                                # Pivot
                                pivot = fitted.pivot_table(index='year', columns='month', values='error', aggfunc='mean')
                                if pivot.shape[0] > 0 and pivot.shape[1] > 0:
                                    fig_heat = px.imshow(pivot, color_continuous_scale='RdBu', aspect='auto', labels=dict(color='Error'))
                                    fig_heat.update_layout(template=dark_template, height=380)
                                    st.plotly_chart(fig_heat, use_container_width=True)
                                else:
                                    st.info("Not enough data to create a heatmap.")
                        
                        # ===============================
                        # FORECAST SUMMARY TABLE
                        # ===============================
                        st.markdown("### Résumé des Prévisions")
                        
                        summary_data = []
                        current_value = data.iloc[-1]
                        
                        for model_name, forecast_data in forecasts.items():
                            forecast_final = forecast_data['forecast'].iloc[-1] if hasattr(forecast_data['forecast'], 'iloc') else forecast_data['forecast'][-1]
                            growth_rate = ((forecast_final - current_value) / current_value) * 100
                            
                            summary_data.append({
                                'Modèle': model_name,
                                'Valeur Actuelle': f"{current_value / scale_factor:.2f} {scale_label}",
                                f'Prévision ({forecast_periods} trimestres)': f"{forecast_final / scale_factor:.2f} {scale_label}",
                                'Croissance Prévue': f"{growth_rate:+.1f}%",
                                'MAE (si disponible)': f"{forecast_data['mae']:.0f}" if forecast_data['mae'] else "N/A"
                            })
                        
                        summary_df = pd.DataFrame(summary_data)
                        st.dataframe(summary_df, use_container_width=True, hide_index=True)
                        
                        # ===============================
                        # MODEL COMPARISON ANALYSIS
                        # ===============================
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write("    ")
                            st.markdown("#### Analyse de Croissance")
                            for model_name, forecast_data in forecasts.items():
                                st.write("    ")
                                forecast_values = forecast_data['forecast']
                                current_val = current_value
                                final_val = forecast_values.iloc[-1] if hasattr(forecast_values, 'iloc') else forecast_values[-1]
                                
                                # Calculate CAGR
                                periods_per_year = 4  # Quarterly data
                                years = forecast_periods / periods_per_year
                                cagr = ((final_val / current_val) ** (1/years) - 1) * 100 if years > 0 and current_val > 0 else 0
                                
                                st.metric(
                                    f"CAGR {model_name}",
                                    f"{cagr:.1f}%",
                                    delta=f"{((final_val - current_val) / current_val * 100):+.1f}% total"
                                )
                        
                        with col2:
                            st.write("    ")
                            st.markdown("#### Variance des Prévisions")
                            st.write("    ")
                            if len(forecasts) > 1:
                                final_forecasts = []
                                for forecast_data in forecasts.values():
                                    final_val = forecast_data['forecast'].iloc[-1] if hasattr(forecast_data['forecast'], 'iloc') else forecast_data['forecast'][-1]
                                    final_forecasts.append(final_val)
                                
                                forecast_std = np.std(final_forecasts)
                                forecast_mean = np.mean(final_forecasts)
                                coefficient_variation = (forecast_std / forecast_mean) * 100 if forecast_mean > 0 else 0
                                
                                st.metric("Écart-Type", f"{forecast_std / scale_factor:.2f} {scale_label}")
                                st.metric("Coefficient de Variation", f"{coefficient_variation:.1f}%")
                                
                                if coefficient_variation < 5:
                                    st.success("✅ Prévisions convergentes (faible variance)")
                                elif coefficient_variation < 15:
                                    st.warning("⚠️ Variance modérée entre les modèles")
                                else:
                                    st.error("❌ Forte divergence entre les modèles")
                        
                        # Model comparison chart
                        if len(forecasts) > 1:
                            st.markdown("### Comparaison des Modèles")
                            
                            fig_comparison = go.Figure()
                            
                            for model_name, forecast_data in forecasts.items():
                                fig_comparison.add_trace(go.Scatter(
                                    x=future_dates,
                                    y=forecast_data['forecast'] / scale_factor,
                                    mode='lines+markers',
                                    name=model_name,
                                    line=dict(width=3)
                                ))
                            
                            fig_comparison.update_layout(
                                template=dark_template,
                                title='Comparaison des Prévisions par Modèle',
                                xaxis_title='Date',
                                yaxis_title=f'Valeur Prévue ({scale_label})',
                                height=400
                            )
                            
                            st.plotly_chart(fig_comparison, use_container_width=True)
                        
                        # Export functionality
                        st.markdown("### 💾 Export Forecast Results")
                        col1, col2 = st.columns(2)
                        
                        # Prepare detailed forecast data for export
                        detailed_data = []
                        for i, date in enumerate(future_dates):
                            quarter = (date.month - 1) // 3 + 1
                            row = {'Trimestre': f"{date.year}-T{quarter}", 'Date': date.strftime('%Y-%m-%d')}
                            for model_name, forecast_data in forecasts.items():
                                value = forecast_data['forecast'].iloc[i] if hasattr(forecast_data['forecast'], 'iloc') else forecast_data['forecast'][i]
                                row[f'{model_name}'] = f"{value / scale_factor:.2f}"
                            detailed_data.append(row)
                        
                        detailed_df = pd.DataFrame(detailed_data)
                        
                        with col1:
                            st.write("    ")
                            csv_data = detailed_df.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                "📥 Download Forecasts (CSV)",
                                csv_data,
                                file_name=f"forecasts_{selected_metric}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv"
                            )
                        
                        with col2:
                            st.write("    ")
                            # Excel export with multiple sheets
                            xlsx_buffer = BytesIO()
                            with pd.ExcelWriter(xlsx_buffer, engine='xlsxwriter') as writer:
                                detailed_df.to_excel(writer, index=False, sheet_name='Forecasts')
                                summary_df.to_excel(writer, index=False, sheet_name='Summary')
                            
                            st.download_button(
                                "📥 Download Analysis (Excel)",
                                xlsx_buffer.getvalue(),
                                file_name=f"analysis_{selected_metric}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                            )
            
            except Exception as e:
                st.error(f"⚠️ Error generating forecast: {str(e)}")
                st.info("💡 Tip: Make sure you have enough historical data for the selected training period.")

# Display available metrics if data loaded but no forecast generated
elif st.session_state.data_loaded:
    st.markdown("<div class='control-card'>", unsafe_allow_html=True)
    st.success(f"✅ {st.session_state.selected_domain} data loaded successfully!")
    st.info("👆 **Select metrics and models above, then click 'Generate Advanced Forecast'**")
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Available models info
    st.markdown("### 🤖 Available Models")
    
    model_info = {
        "✅ Prophet": "Facebook's time series forecasting model with trend and seasonality detection",
        "✅ SARIMAX": "Seasonal ARIMA with external regressors" if HAS_STATSMODELS else "❌ SARIMAX: Requires statsmodels",
        "✅ ARIMA": "AutoRegressive Integrated Moving Average" if HAS_STATSMODELS else "❌ ARIMA: Requires statsmodels",
        "✅ Exponential Smoothing": "Classical time series forecasting" if HAS_STATSMODELS else "❌ Exponential Smoothing: Requires statsmodels",
        "✅ Linear Regression": "Simple trend-based forecasting" if HAS_SKLEARN else "❌ Linear Regression: Requires sklearn",
        "✅ XGBoost": "Gradient boosting with engineered time series features" if HAS_XGBOOST else "❌ XGBoost: Requires xgboost"
    }
    
    for model, description in model_info.items():
        st.write(f"• {model}: {description}")
    
    # Available metrics
    st.markdown("### 📈 Available Metrics")
    metrics = st.session_state.available_metrics
    if metrics:
        cols = st.columns(3)
        for i, metric in enumerate(metrics):
            with cols[i % 3]:
                st.write(f"• {metric}")

else:
    st.markdown("<div class='control-card'>", unsafe_allow_html=True)
    st.info("👆 **Click on a domain button above to load data and start advanced forecasting**")
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)