import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.ensemble import IsolationForest
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
import warnings
import io
import base64
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
import tempfile
warnings.filterwarnings('ignore')

# ===============================
# PAGE CONFIGURATION
# ===============================
st.set_page_config(
    page_title="Mobile Analysis - Algeria Telecom",
    page_icon="📱",
    layout="wide"
)

# ===============================
# HIDE STREAMLIT TOP BAR & PADDING + NAVBAR
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
# ENHANCED DARK THEME CSS
# ===============================
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0c0c0c, #1a1a2e, #16213e);
        color: #ffffff;
    }
    h1, h2, h3, h4 {
        color: #ffffff;
        border-bottom: 2px solid #2e7d32;
        padding-bottom: 10px;
    }
    .analysis-card {
        background: rgba(30, 30, 46, 0.9);
        padding: 25px; border-radius: 15px;
        border: 1px solid #2e7d32; margin-bottom: 25px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
    }
    .metric-card {
        background: linear-gradient(135deg, rgba(46, 125, 50, 0.2), rgba(76, 175, 80, 0.1));
        padding: 20px; border-radius: 12px;
        border: 1px solid #2e7d32; margin-bottom: 20px;
        text-align: center;
    }
    .metric-value {
        font-size: 2rem; font-weight: bold; color: #4caf50;
    }
    .metric-label {
        font-size: 0.9rem; color: #c8e6c9; margin-top: 5px;
    }
    .stButton button {
        background: linear-gradient(45deg, #2e7d32, #4caf50);
        color: white; font-weight: bold; border: none;
        padding: 12px 30px; border-radius: 8px;
        transition: all 0.3s ease;
        margin: 10px;    
    }
    
    /* Enhanced tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(30, 30, 46, 0.5);
        border-radius: 10px;
        padding: 5px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background: rgba(46, 125, 50, 0.2);
        border-radius: 8px;
        color: #c8e6c9;
        font-weight: 600;
        margin: 5px;
        padding: 10px;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(45deg, #2e7d32, #4caf50) !important;
        color: white !important;
    }
    
    /* Enhanced report button styling */
    .report-button {
        background: linear-gradient(45deg, #1565C0, #42A5F5) !important;
        color: white !important; border: 2px solid #1976D2 !important;
        padding: 15px 25px !important; border-radius: 10px !important;
        font-weight: bold !important; font-size: 16px !important;
        transition: all 0.3s ease !important;
        width: 100% !important;
    }
    .report-button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(21, 101, 192, 0.4) !important;
    }
</style>
""", unsafe_allow_html=True)

# ===============================
# DATA LOADING & PROCESSING FUNCTIONS
# ===============================
def get_dark_template():
    return {
        'layout': {
            'paper_bgcolor': 'rgba(0,0,0,0)',
            'plot_bgcolor': 'black',
            'font': {'color': 'white'},
            'xaxis': {'gridcolor': 'rgba(255,255,255,0.1)', 'linecolor': 'rgba(255,255,255,0.2)'},
            'yaxis': {'gridcolor': 'rgba(255,255,255,0.1)', 'linecolor': 'rgba(255,255,255,0.2)'},
            'colorway': ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98FB98', '#F4A460']
        }
    }

@st.cache_data
def load_data():
    """Load mobile data from CSV"""
    try:
        df = pd.read_csv("data/mobile_data.csv")
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

def process_mobile_data(df):
    """Process mobile data into time series format"""
    if df.empty:
        return pd.DataFrame()
    
    # Transpose data
    df_transposed = df.set_index(df.columns[0]).T.reset_index()
    df_transposed.columns = ['Period'] + df.iloc[:, 0].tolist()
    
    # Convert to long format
    melted_df = pd.melt(
        df_transposed,
        id_vars=['Period'],
        value_vars=df.iloc[:, 0].tolist(),
        var_name='Metric',
        value_name='Value'
    )
    
    # Extract year and semester
    melted_df['Year'] = melted_df['Period'].str.extract(r'(\d{4})').astype(int)
    melted_df['Semester'] = melted_df['Period'].str.extract(r'_S(\d)').astype(int)
    
    # Map semester to starting month
    semester_month_map = {1: 1, 2: 4, 3: 7, 4: 10}  # Jan, Apr, Jul, Oct
    melted_df['Month'] = melted_df['Semester'].map(semester_month_map)
    
    # Build proper datetime
    melted_df['Date'] = pd.to_datetime(
        dict(year=melted_df['Year'], month=melted_df['Month'], day=1),
        errors='coerce'
    )
    
    # Clean values
    melted_df['Value'] = (
        melted_df['Value'].astype(str)
        .str.replace(' ', '')
        .replace('', '0')
    )
    melted_df['Value'] = pd.to_numeric(melted_df['Value'], errors='coerce').fillna(0)
    
    return melted_df.dropna(subset=['Date'])

def analyze_metric_deep_dive(df, metric_name):
    """Comprehensive analysis of a single metric"""
    metric_data = df[df['Metric'] == metric_name]
    if metric_data.empty:
        return None
        
    data = metric_data['Value']
    dates = metric_data['Date']
    
    current_value = data.iloc[-1] if len(data) > 0 else 0
    peak_value = data.max()
    lowest_value = data.min()
    total_growth = ((current_value - data.iloc[0]) / data.iloc[0]) * 100 if data.iloc[0] > 0 else 0
    years = (dates.iloc[-1] - dates.iloc[0]).days / 365.25 if len(dates) > 1 else 1
    cagr = ((current_value / data.iloc[0]) ** (1/years) - 1) * 100 if years > 0 and data.iloc[0] > 0 else 0
    semester_growth = data.pct_change() * 100
    avg_semester_growth = semester_growth.mean()
    volatility = semester_growth.std()
    
    return {
        'data': data, 'dates': dates, 'current_value': current_value,
        'peak_value': peak_value, 'lowest_value': lowest_value,
        'total_growth': total_growth, 'cagr': cagr,
        'avg_semester_growth': avg_semester_growth, 'volatility': volatility,
        'semester_growth': semester_growth
    }

def detect_anomalies(df, metric_name):
    """Advanced anomaly detection"""
    metric_data = df[df['Metric'] == metric_name]['Value']
    if len(metric_data) < 4:
        return None
    
    z_scores = np.abs(stats.zscore(metric_data))
    statistical_outliers = z_scores > 2.5
    
    iso_forest = IsolationForest(contamination=0.1, random_state=42)
    isolation_outliers = iso_forest.fit_predict(metric_data.values.reshape(-1, 1)) == -1
    
    Q1 = metric_data.quantile(0.25)
    Q3 = metric_data.quantile(0.75)
    IQR = Q3 - Q1
    iqr_outliers = (metric_data < (Q1 - 1.5 * IQR)) | (metric_data > (Q3 + 1.5 * IQR))
    
    combined_anomalies = (statistical_outliers.astype(int) + 
                         isolation_outliers.astype(int) + 
                         iqr_outliers.astype(int)) >= 2
    
    return {
        'statistical_outliers': statistical_outliers,
        'isolation_outliers': isolation_outliers,
        'iqr_outliers': iqr_outliers,
        'combined_anomalies': combined_anomalies,
        'z_scores': z_scores
    }

def perform_stationarity_test(data):
    """Perform ADF test for stationarity"""
    try:
        adf_stat, adf_pvalue, _, _, adf_critical, _ = adfuller(data)
        return {
            'adf_statistic': adf_stat,
            'adf_pvalue': adf_pvalue,
            'adf_critical_values': adf_critical,
            'is_stationary': adf_pvalue < 0.05
        }
    except Exception:
        return None

# ===============================
# PDF REPORT GENERATION
# ===============================
def generate_pdf_report(processed_df, analysis_results):
    """Generate comprehensive PDF report"""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []
    
    # Title
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#002147'),
        alignment=1,
        spaceAfter=30
    )
    story.append(Paragraph("Rapport d'Analyse - Services Mobile", title_style))
    story.append(Spacer(1, 20))
    
    # Executive Summary
    story.append(Paragraph("Résumé Exécutif", styles['Heading2']))
    
    summary_data = [
        ['Métrique', 'Valeur Actuelle', 'Croissance Totale', 'CAGR'],
    ]
    
    for metric, results in analysis_results.items():
        if results:
            summary_data.append([
                metric.replace('_', ' '),
                f"{results['current_value']:,.0f}",
                f"{results['total_growth']:+.1f}%",
                f"{results['cagr']:+.1f}%"
            ])
    
    table = Table(summary_data)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#002147')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    
    story.append(table)
    story.append(Spacer(1, 20))
    
    # Key Insights
    story.append(Paragraph("Insights Clés", styles['Heading2']))
    insights = [
        "• Évolution stable du parc global d'abonnés mobiles",
        "• Transition progressive des technologies 2G/3G vers la 4G",
        "• Analyse comparative des opérateurs (Mobilis, Djezzi, Ooredoo)",
        "• Répartition équilibrée entre services prépayés et postpayés",
        "• Anomalies identifiées et analysées pour optimisation"
    ]
    
    for insight in insights:
        story.append(Paragraph(insight, styles['Normal']))
    
    # Build PDF
    doc.build(story)
    buffer.seek(0)
    return buffer

# ===============================
# MAIN PAGE CONTENT
# ===============================

# Back button
if st.button("← Back to Domain Selection", key="back_btn"):
    st.switch_page("pages/Analyze_Previous.py")

st.title("📱 Mobile Services Analysis")

# Load and process data
df = load_data()
if df.empty:
    st.error("Unable to load Mobile data. Please check if 'data/mobile_data.csv' exists.")
    st.stop()

processed_df = process_mobile_data(df)
if processed_df.empty:
    st.error("Unable to process Mobile data.")
    st.stop()

# ===============================
# ORGANIZED TABBED INTERFACE
# ===============================

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Vue d'Ensemble", 
    "🔧 Analyse Comparative", 
    "🚨 Détection d'Anomalies", 
    "🔍 Analyse Métrique Spécifique",
    "📄 Rapports & Export"
])

# ===============================
# TAB 1: OVERVIEW
# ===============================
with tab1:
    st.subheader("📋 Dataset Overview")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.write("    ")
        st.write("**Data Preview:**")
        preview_df = df.head()
        st.dataframe(preview_df, use_container_width=True)
    
    with col2:
        st.write("    ")
        st.write("**Summary Statistics:**")
        total_records = len(processed_df)
        date_range = f"{processed_df['Date'].min().strftime('%Y-%m')} to {processed_df['Date'].max().strftime('%Y-%m')}"
        unique_metrics = processed_df['Metric'].nunique()
        
        st.metric("Total Records", f"{total_records:,}")
        st.metric("Date Range", date_range)
        st.metric("Unique Metrics", unique_metrics)
        st.metric("Missing Values", df.isnull().sum().sum())

    # Key Metrics Overview
    st.subheader("📊 Key Metrics Overview")
    
    latest_data = processed_df[processed_df['Date'] == processed_df['Date'].max()]
    
    # Calculate total subscribers by operator
    operators = ['Mobilis', 'Djezzi', 'Ooredoo']
    operator_totals = {}
    
    for operator in operators:
        gsm_metric = f'GSM_{operator}'
        data_metric = f'3G4G_{operator}'
        
        gsm_val = latest_data[latest_data['Metric'] == gsm_metric]['Value'].iloc[0] if not latest_data[latest_data['Metric'] == gsm_metric].empty else 0
        data_val = latest_data[latest_data['Metric'] == data_metric]['Value'].iloc[0] if not latest_data[latest_data['Metric'] == data_metric].empty else 0
        
        operator_totals[operator] = gsm_val + data_val
    
    # Display key metrics
    parc_global = latest_data[latest_data['Metric'] == 'Parc_global']['Value'].iloc[0] if not latest_data[latest_data['Metric'] == 'Parc_global'].empty else 0
    total_prepaid = latest_data[latest_data['Metric'] == 'Total_Prepaye']['Value'].iloc[0] if not latest_data[latest_data['Metric'] == 'Total_Prepaye'].empty else 0
    total_postpaid = latest_data[latest_data['Metric'] == 'Total_Postpaye']['Value'].iloc[0] if not latest_data[latest_data['Metric'] == 'Total_Postpaye'].empty else 0

    col1, col2, col3, col4 = st.columns(4)
    
    metrics_to_show = [
        (parc_global, 'Total Mobile Subscribers'),
        (max(operator_totals.values()) if operator_totals else 0, f'Leading Operator ({max(operator_totals, key=operator_totals.get) if operator_totals else "N/A"})'),
        (total_prepaid, 'Prepaid Subscribers'),
        (total_postpaid, 'Postpaid Subscribers')
    ]
    
    for i, (value, label) in enumerate(metrics_to_show):
        formatted_value = f"{value/1e6:.1f}M" if value > 1e6 else f"{value:,.0f}"
        
        with [col1, col2, col3, col4][i]:
            st.write("    ")
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{formatted_value}</div>
                <div class="metric-label">{label}</div>
            </div>
            """, unsafe_allow_html=True)

    # Evolution of Total Mobile Subscribers
    st.subheader("📈 Evolution of Total Mobile Subscribers")
    
    mobile_subscribers = processed_df[processed_df['Metric'] == 'Parc_global'].copy()
    if not mobile_subscribers.empty:
        if len(mobile_subscribers) > 1:
            start_val = mobile_subscribers['Value'].iloc[0]
            end_val = mobile_subscribers['Value'].iloc[-1]
            growth = ((end_val - start_val) / start_val * 100) if start_val != 0 else 0
            
            fig = px.line(mobile_subscribers, x='Date', y='Value', 
                         title=f'Total Mobile Subscribers Evolution (Growth: {growth:+.1f}%)',
                         markers=True, line_shape='linear')
            fig.update_layout(template=get_dark_template(), height=400)
            fig.update_traces(line_color='#4ECDC4', marker_color='#4ECDC4')
            st.plotly_chart(fig, use_container_width=True)

# ===============================
# TAB 2: COMPARATIVE ANALYSIS
# ===============================
with tab2:
    st.subheader("🏢 Operator Market Share Analysis")
    
    # Get all operator metrics (GSM + 3G4G combined)
    operator_data = []
    for operator in operators:
        gsm_data = processed_df[processed_df['Metric'] == f'GSM_{operator}'].copy()
        data_3g4g = processed_df[processed_df['Metric'] == f'3G4G_{operator}'].copy()
        
        if not gsm_data.empty and not data_3g4g.empty:
            # Merge GSM and 3G4G data
            combined = gsm_data.merge(data_3g4g, on='Date', suffixes=('_GSM', '_3G4G'))
            combined['Total'] = combined['Value_GSM'] + combined['Value_3G4G']
            combined['Operator'] = operator
            operator_data.append(combined[['Date', 'Total', 'Operator']])

    if operator_data:
        all_operators = pd.concat(operator_data)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Market share evolution
            fig = px.line(all_operators, x='Date', y='Total', color='Operator',
                         title='Operator Market Share Evolution',
                         markers=True)
            
            # Custom colors for operators
            operator_colors = {
                'Mobilis': '#FF6B6B',    # Red
                'Djezzi': '#4ECDC4',     # Cyan
                'Ooredoo': '#FFEAA7'     # Yellow
            }
            
            for trace in fig.data:
                if trace.name in operator_colors:
                    trace.line.color = operator_colors[trace.name]
                    trace.marker.color = operator_colors[trace.name]
            
            fig.update_layout(template=get_dark_template(), height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Current market share pie chart
            latest_operators = all_operators[all_operators['Date'] == all_operators['Date'].max()]
            fig_pie = px.pie(latest_operators, values='Total', names='Operator',
                            title='Current Market Share by Operator',
                            color_discrete_map=operator_colors)
            fig_pie.update_layout(template=get_dark_template(), height=400)
            fig_pie.update_traces(textinfo='percent+label')
            st.plotly_chart(fig_pie, use_container_width=True)

    st.subheader("📶 Technology Evolution: GSM vs 3G/4G")
    
    # Aggregate GSM and 3G4G data across all operators
    gsm_total = processed_df[processed_df['Metric'].str.contains('GSM_')].groupby('Date')['Value'].sum().reset_index()
    gsm_total['Technology'] = 'GSM'

    data_3g4g_total = processed_df[processed_df['Metric'].str.contains('3G4G_')].groupby('Date')['Value'].sum().reset_index()
    data_3g4g_total['Technology'] = '3G/4G'

    if not gsm_total.empty and not data_3g4g_total.empty:
        tech_comparison = pd.concat([gsm_total, data_3g4g_total])
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Technology evolution line chart
            fig = px.line(tech_comparison, x='Date', y='Value', color='Technology',
                         title='Technology Evolution: GSM vs 3G/4G Transition',
                         markers=True)
            
            # Custom colors
            for trace in fig.data:
                if 'GSM' in trace.name:
                    trace.line.color = '#FF6B6B'  # Red for legacy
                    trace.marker.color = '#FF6B6B'
                elif '3G/4G' in trace.name:
                    trace.line.color = '#96CEB4'  # Green for modern
                    trace.marker.color = '#96CEB4'
            
            fig.update_layout(template=get_dark_template(), height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Stacked area chart showing transition
            pivot_tech = tech_comparison.pivot_table(index='Date', columns='Technology', values='Value', fill_value=0)
            
            fig_area = px.area(tech_comparison, x='Date', y='Value', color='Technology',
                              title='Technology Market Share Over Time')
            fig_area.update_layout(template=get_dark_template(), height=400)
            st.plotly_chart(fig_area, use_container_width=True)

    st.subheader("💳 Prepaid vs Postpaid Analysis")
    
    prepaid_data = processed_df[processed_df['Metric'] == 'Total_Prepaye'].copy()
    postpaid_data = processed_df[processed_df['Metric'] == 'Total_Postpaye'].copy()

    if not prepaid_data.empty and not postpaid_data.empty:
        # Combine prepaid and postpaid data
        payment_comparison = pd.concat([
            prepaid_data.assign(Payment_Type='Prepaid'),
            postpaid_data.assign(Payment_Type='Postpaid')
        ])
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Payment type evolution
            fig = px.line(payment_comparison, x='Date', y='Value', color='Payment_Type',
                         title='Prepaid vs Postpaid Subscribers Evolution',
                         markers=True)
            
            # Custom colors
            for trace in fig.data:
                if 'Prepaid' in trace.name:
                    trace.line.color = '#45B7D1'  # Blue
                    trace.marker.color = '#45B7D1'
                elif 'Postpaid' in trace.name:
                    trace.line.color = '#DDA0DD'  # Purple
                    trace.marker.color = '#DDA0DD'
            
            fig.update_layout(template=get_dark_template(), height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Market share percentage over time
            payment_comparison['Total_Payment'] = payment_comparison.groupby('Date')['Value'].transform('sum')
            payment_comparison['Percentage'] = (payment_comparison['Value'] / payment_comparison['Total_Payment'] * 100)
            
            fig_area = px.area(payment_comparison, x='Date', y='Percentage', color='Payment_Type',
                              title='Prepaid vs Postpaid Market Share (%)')
            fig_area.update_layout(template=get_dark_template(), height=400)
            st.plotly_chart(fig_area, use_container_width=True)

    st.subheader("📋 Detailed Operator Analysis")
    
    # Create tabs for each operator
    tab1_op, tab2_op, tab3_op = st.tabs(["📱 Mobilis", "📱 Djezzi", "📱 Ooredoo"])

    for i, (tab_op, operator) in enumerate(zip([tab1_op, tab2_op, tab3_op], operators)):
        with tab_op:
            gsm_data = processed_df[processed_df['Metric'] == f'GSM_{operator}'].copy()
            data_3g4g = processed_df[processed_df['Metric'] == f'3G4G_{operator}'].copy()
            
            if not gsm_data.empty and not data_3g4g.empty:
                col1, col2 = st.columns(2)
                
                with col1:
                    # Technology breakdown for this operator
                    operator_tech = pd.concat([
                        gsm_data.assign(Technology='GSM'),
                        data_3g4g.assign(Technology='3G/4G')
                    ])
                    
                    fig = px.line(operator_tech, x='Date', y='Value', color='Technology',
                                 title=f'{operator} Technology Evolution',
                                 markers=True)
                    fig.update_layout(template=get_dark_template(), height=350)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Current split
                    latest_gsm = gsm_data['Value'].iloc[-1] if not gsm_data.empty else 0
                    latest_3g4g = data_3g4g['Value'].iloc[-1] if not data_3g4g.empty else 0
                    
                    split_data = pd.DataFrame({
                        'Technology': ['GSM', '3G/4G'],
                        'Subscribers': [latest_gsm, latest_3g4g]
                    })
                    
                    fig_pie = px.pie(split_data, values='Subscribers', names='Technology',
                                    title=f'{operator} Current Technology Split')
                    fig_pie.update_layout(template=get_dark_template(), height=350)
                    st.plotly_chart(fig_pie, use_container_width=True)

# ===============================
# TAB 3: ANOMALY DETECTION
# ===============================
with tab3:
    st.subheader("🚨 Global Anomaly Detection")
    
    st.info("This section analyzes anomalies in the most critical mobile metrics: Total Mobile Subscribers and Technology-specific data.")
    
    anomalies_found = []
    
    for metric in ['Parc_global', 'Total_Prepaye', 'Total_Postpaye']:
        st.subheader(f"Analysis for {metric.replace('_', ' ')}")
        
        metric_data = processed_df[processed_df['Metric'] == metric].copy().sort_values('Date')
        if len(metric_data) > 1:
            metric_data['Change'] = metric_data['Value'].pct_change() * 100
            
            changes = metric_data['Change'].dropna()
            if len(changes) > 0:
                std_dev = changes.std()
                mean_change = changes.mean()
                threshold = abs(mean_change) + 2 * std_dev
                
                anomaly_points = metric_data[abs(metric_data['Change']) > threshold]
                
                if not anomaly_points.empty:
                    for _, row in anomaly_points.iterrows():
                        anomalies_found.append({
                            'Metric': metric,
                            'Date': row['Date'],
                            'Change': row['Change'],
                            'Value': row['Value']
                        })
                
                # Visualization
                fig = px.line(metric_data, x='Date', y='Value',
                             title=f"{metric.replace('_', ' ')} with Anomalies Highlighted",
                             markers=True)
                fig.update_layout(template=get_dark_template(), height=400)
                fig.update_traces(line_color='#4ECDC4', marker_color='#4ECDC4')
                
                # Add anomaly markers
                if not anomaly_points.empty:
                    fig.add_scatter(
                        x=anomaly_points['Date'],
                        y=anomaly_points['Value'],
                        mode='markers+text',
                        name='Anomaly',
                        marker=dict(color='red', size=12, symbol='x'),
                        text=[f"{c:.1f}%" for c in anomaly_points['Change']],
                        textposition="top center"
                    )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Anomaly summary
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Anomalies Found", len(anomaly_points))
                with col2:
                    avg_change = anomaly_points['Change'].mean() if len(anomaly_points) > 0 else 0
                    st.metric("Avg Change", f"{avg_change:+.1f}%")
                with col3:
                    max_change = anomaly_points['Change'].abs().max() if len(anomaly_points) > 0 else 0
                    st.metric("Max Change", f"{max_change:.1f}%")

    # Summary of all anomalies
    st.subheader("📊 Anomaly Summary")
    
    if anomalies_found:
        st.warning(f"⚠️ Found {len(anomalies_found)} potential anomalies across all metrics:")
        for anomaly in anomalies_found[:10]:  # Show first 10
            st.write(f"• **{anomaly['Metric'].replace('_', ' ')}** on {anomaly['Date'].strftime('%Y-%m')}: {anomaly['Change']:.1f}% change")
    else:
        st.success("✅ No significant anomalies detected in the core metrics.")

# ===============================
# TAB 4: SPECIFIC METRIC ANALYSIS
# ===============================
with tab4:
    st.subheader("🔍 Deep Dive Analysis for Specific Metric")
    
    # Metric selector
    available_metrics = [
        'Parc_global',
        'Total_Prepaye', 
        'Total_Postpaye',
        'GSM_Mobilis',
        'GSM_Djezzi',
        'GSM_Ooredoo',
        '3G4G_Mobilis',
        '3G4G_Djezzi',
        '3G4G_Ooredoo'
    ]
    st.write("    ")
    selected_metric = st.selectbox(
        "Select a metric for detailed analysis:",
        options=available_metrics,
        format_func=lambda x: x.replace('_', ' ').title()
    )
    
    # Analysis type selector
    analysis_types = st.multiselect(
        "Choose analysis types:",
        options=[
            "Statistiques Descriptives",
            "Analyse de Tendance", 
            "Détection d'Anomalies",
            "Décomposition Saisonnière",
            "Tests de Stationnarité",
            "Analyse de Corrélation"
        ],
        default=["Statistiques Descriptives", "Analyse de Tendance"]
    )
    
    if selected_metric and analysis_types:
        metric_analysis = analyze_metric_deep_dive(processed_df, selected_metric)
        
        if metric_analysis:
            # Scale determination
            scale_factor = 1e6 if 'abonnés' in selected_metric.lower() or any(op in selected_metric for op in ['Mobilis', 'Djezzi', 'Ooredoo', 'Parc', 'Total']) else 1
            scale_label = 'M' if scale_factor == 1e6 else ''
            
            # Overview metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Current Value",
                    f"{metric_analysis['current_value']/scale_factor:.2f}{scale_label}",
                    delta=f"{metric_analysis['total_growth']:+.1f}% total"
                )
            
            with col2:
                st.metric("Peak Value", f"{metric_analysis['peak_value']/scale_factor:.2f}{scale_label}")
            
            with col3:
                st.metric("CAGR", f"{metric_analysis['cagr']:+.1f}%/year")
            
            with col4:
                st.metric("Volatility", f"{metric_analysis['volatility']:.1f}%")

            # Detailed Analysis based on selection
            if "Statistiques Descriptives" in analysis_types:
                st.subheader("📈 Descriptive Statistics")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Distribution plot
                    data = metric_analysis['data']
                    
                    fig_hist = go.Figure()
                    fig_hist.add_trace(go.Histogram(
                        x=data/scale_factor,
                        nbinsx=15,
                        name='Distribution',
                        marker_color='#002147',
                        opacity=0.7
                    ))
                    
                    fig_hist.update_layout(
                        title='Distribution des Valeurs',
                        xaxis_title=f'Valeur ({scale_label})',
                        yaxis_title='Fréquence',
                        template=get_dark_template(),
                        height=400
                    )
                    
                    st.plotly_chart(fig_hist, use_container_width=True)
                
                with col2:
                    # Box plot
                    fig_box = go.Figure()
                    fig_box.add_trace(go.Box(
                        y=data/scale_factor,
                        name=selected_metric,
                        marker_color='#00A859'
                    ))
                    
                    fig_box.update_layout(
                        title='Analyse des Quartiles',
                        yaxis_title=f'Valeur ({scale_label})',
                        template=get_dark_template(),
                        height=400
                    )
                    
                    st.plotly_chart(fig_box, use_container_width=True)
                
                # Summary statistics table
                stats_data = {
                    'Statistique': ['Moyenne', 'Médiane', 'Écart-type', 'Min', 'Max', 'Q1', 'Q3', 'Asymétrie', 'Aplatissement'],
                    'Valeur': [
                        f"{data.mean()/scale_factor:.2f}{scale_label}",
                        f"{data.median()/scale_factor:.2f}{scale_label}", 
                        f"{data.std()/scale_factor:.2f}{scale_label}",
                        f"{data.min()/scale_factor:.2f}{scale_label}",
                        f"{data.max()/scale_factor:.2f}{scale_label}",
                        f"{data.quantile(0.25)/scale_factor:.2f}{scale_label}",
                        f"{data.quantile(0.75)/scale_factor:.2f}{scale_label}",
                        f"{stats.skew(data):.3f}",
                        f"{stats.kurtosis(data):.3f}"
                    ]
                }
                
                st.dataframe(pd.DataFrame(stats_data), hide_index=True, use_container_width=True)

            if "Analyse de Tendance" in analysis_types:
                st.subheader("📊 Analyse de Tendance")
                
                data = metric_analysis['data']
                dates = metric_analysis['dates']
                
                # Trend analysis plot
                fig_trend = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=(
                        'Évolution Temporelle avec Tendance',
                        'Taux de Croissance Semestrielle',
                        'Moyennes Mobiles',
                        'Analyse des Résidus'
                    )
                )
                
                # Historical data with trend
                fig_trend.add_trace(
                    go.Scatter(x=dates, y=data/scale_factor, mode='lines+markers', 
                              name='Données', line=dict(color='#002147')),
                    row=1, col=1
                )
                
                # Add trend line
                z = np.polyfit(range(len(data)), data, 1)
                p = np.poly1d(z)
                fig_trend.add_trace(
                    go.Scatter(x=dates, y=p(range(len(data)))/scale_factor, 
                              mode='lines', name='Tendance', line=dict(color='#FF6B35', dash='dash')),
                    row=1, col=1
                )
                
                # Growth rates
                growth_rates = metric_analysis['semester_growth'][1:]
                colors_growth = ['green' if x >= 0 else 'red' for x in growth_rates]
                fig_trend.add_trace(
                    go.Bar(x=dates[1:], y=growth_rates, marker_color=colors_growth, 
                           name='Croissance Semestrielle', opacity=0.7),
                    row=1, col=2
                )
                
                # Moving averages
                ma_2 = data.rolling(window=2).mean()
                ma_4 = data.rolling(window=4).mean() if len(data) >= 4 else data.rolling(window=2).mean()
                
                fig_trend.add_trace(
                    go.Scatter(x=dates, y=data/scale_factor, mode='lines', 
                              name='Original', line=dict(color='#002147', width=1), opacity=0.5),
                    row=2, col=1
                )
                fig_trend.add_trace(
                    go.Scatter(x=dates, y=ma_2/scale_factor, mode='lines', 
                              name='MA-2S', line=dict(color='#00A859', width=2)),
                    row=2, col=1
                )
                if len(data) >= 4:
                    fig_trend.add_trace(
                        go.Scatter(x=dates, y=ma_4/scale_factor, mode='lines', 
                                  name='MA-4S', line=dict(color='#FF6B35', width=2)),
                        row=2, col=1
                    )
                
                # Residuals analysis
                residuals = data - p(range(len(data)))
                fig_trend.add_trace(
                    go.Scatter(x=dates, y=residuals/scale_factor, mode='lines+markers', 
                              name='Résidus', line=dict(color='#8E44AD')),
                    row=2, col=2
                )
                
                fig_trend.update_layout(height=600, showlegend=True, template=get_dark_template())
                st.plotly_chart(fig_trend, use_container_width=True)

            if "Détection d'Anomalies" in analysis_types:
                st.subheader("🚨 Détection d'Anomalies")
                
                anomaly_results = detect_anomalies(processed_df, selected_metric)
                
                if anomaly_results:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Anomalies Z-score", anomaly_results['statistical_outliers'].sum())
                    with col2:
                        st.metric("Isolation Forest", anomaly_results['isolation_outliers'].sum())
                    with col3:
                        st.metric("Anomalies IQR", anomaly_results['iqr_outliers'].sum())
                    
                    # Anomaly visualization
                    data = metric_analysis['data']
                    dates = metric_analysis['dates']
                    fig_anomaly = go.Figure()
                    
                    # Normal data
                    fig_anomaly.add_trace(go.Scatter(
                        x=dates, 
                        y=data/scale_factor,
                        mode='lines+markers',
                        name='Données Normales',
                        line=dict(color='#002147'),
                        marker=dict(size=4)
                    ))
                    
                    # Highlight anomalies
                    if anomaly_results['combined_anomalies'].any():
                        anomaly_indices = anomaly_results['combined_anomalies']
                        anomaly_dates = dates[anomaly_indices]
                        anomaly_values = data[anomaly_indices]
                        
                        fig_anomaly.add_trace(go.Scatter(
                            x=anomaly_dates,
                            y=anomaly_values/scale_factor,
                            mode='markers',
                            name='Anomalies Détectées',
                            marker=dict(color='red', size=10, symbol='x')
                        ))
                    
                    fig_anomaly.update_layout(
                        title='Détection d\'Anomalies dans les Données',
                        xaxis_title='Date',
                        yaxis_title=f'{selected_metric} ({scale_label})',
                        template=get_dark_template(),
                        height=400
                    )
                    
                    st.plotly_chart(fig_anomaly, use_container_width=True)

            if "Décomposition Saisonnière" in analysis_types:
                st.subheader("📅 Décomposition Saisonnière")
                
                data = metric_analysis['data']
                dates = metric_analysis['dates']
                
                if len(data) >= 8:
                    try:
                        # Create time series with proper frequency (semester-based)
                        ts = pd.Series(data.values, index=dates)
                        ts = ts.asfreq('6MS').interpolate()  # 6-month start frequency
                        
                        decomposition = seasonal_decompose(ts, model='additive', period=2)  # 2 semesters per year
                        
                        fig_decomp = make_subplots(
                            rows=4, cols=1,
                            subplot_titles=('Original', 'Tendance', 'Saisonnalité', 'Résidus'),
                            vertical_spacing=0.08
                        )
                        
                        fig_decomp.add_trace(
                            go.Scatter(x=dates, y=decomposition.observed/scale_factor, 
                                      name='Original', line=dict(color='#002147')),
                            row=1, col=1
                        )
                        
                        fig_decomp.add_trace(
                            go.Scatter(x=dates, y=decomposition.trend/scale_factor, 
                                      name='Tendance', line=dict(color='#00A859')),
                            row=2, col=1
                        )
                        
                        fig_decomp.add_trace(
                            go.Scatter(x=dates, y=decomposition.seasonal/scale_factor, 
                                      name='Saisonnalité', line=dict(color='#FF6B35')),
                            row=3, col=1
                        )
                        
                        fig_decomp.add_trace(
                            go.Scatter(x=dates, y=decomposition.resid/scale_factor, 
                                      name='Résidus', line=dict(color='#8E44AD')),
                            row=4, col=1
                        )
                        
                        fig_decomp.update_layout(
                            height=800,
                            showlegend=False,
                            template=get_dark_template()
                        )
                        
                        st.plotly_chart(fig_decomp, use_container_width=True)
                        
                        # Seasonal strength
                        seasonal_strength = np.var(decomposition.seasonal) / np.var(decomposition.resid + decomposition.seasonal)
                        st.info(f"Force Saisonnière: {seasonal_strength:.3f}")
                        
                    except Exception as e:
                        st.error(f"Erreur dans la décomposition saisonnière: {str(e)}")
                else:
                    st.warning("Pas assez de données pour la décomposition saisonnière (minimum 8 périodes)")

            if "Tests de Stationnarité" in analysis_types:
                st.subheader("🔬 Tests de Stationnarité")
                
                data = metric_analysis['data']
                stationarity_result = perform_stationarity_test(data)
                
                if stationarity_result:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**Test de Dickey-Fuller Augmenté**")
                        if stationarity_result['is_stationary']:
                            st.success(f"✅ Série stationnaire (p-value: {stationarity_result['adf_pvalue']:.4f})")
                        else:
                            st.warning(f"⚠️ Série non-stationnaire (p-value: {stationarity_result['adf_pvalue']:.4f})")
                        
                        st.write(f"Statistique ADF: {stationarity_result['adf_statistic']:.4f}")
                    
                    with col2:
                        st.markdown("**Valeurs Critiques ADF**")
                        for key, value in stationarity_result['adf_critical_values'].items():
                            st.write(f"{key}: {value:.4f}")

            if "Analyse de Corrélation" in analysis_types:
                st.subheader("🔗 Analyse de Corrélation")
                
                # Correlation matrix for all numeric metrics
                numeric_metrics = ['Parc_global', 'Total_Prepaye', 'Total_Postpaye', 
                                 'GSM_Mobilis', 'GSM_Djezzi', 'GSM_Ooredoo',
                                 '3G4G_Mobilis', '3G4G_Djezzi', '3G4G_Ooredoo']
                
                correlation_data = processed_df.pivot_table(
                    index='Date', 
                    columns='Metric', 
                    values='Value'
                )[numeric_metrics].corr()
                
                # Plotly heatmap
                fig_corr = go.Figure(data=go.Heatmap(
                    z=correlation_data.values,
                    x=[m.replace('_', ' ') for m in correlation_data.columns],
                    y=[m.replace('_', ' ') for m in correlation_data.index],
                    colorscale='RdBu',
                    zmid=0,
                    text=correlation_data.values,
                    texttemplate="%{text:.2f}",
                    textfont={"size": 10}
                ))
                
                fig_corr.update_layout(
                    title='Matrice de Corrélation',
                    template=get_dark_template(),
                    height=500
                )
                
                st.plotly_chart(fig_corr, use_container_width=True)
                
                # Strongest correlations with selected metric
                st.markdown("#### Corrélations les Plus Fortes")
                
                if selected_metric in correlation_data.columns:
                    selected_corr = correlation_data[selected_metric].abs().sort_values(ascending=False)
                    selected_corr = selected_corr[selected_corr.index != selected_metric]
                    
                    for metric, corr_value in selected_corr.head(3).items():
                        corr_direction = "positive" if correlation_data[selected_metric][metric] > 0 else "négative"
                        st.write(f"• **{metric.replace('_', ' ')}**: {corr_value:.3f} (corrélation {corr_direction})")

# ===============================
# TAB 5: REPORTS & EXPORT
# ===============================
with tab5:
    st.subheader("📄 Générer Rapport d'Analyse")
    
    # Store analysis results for report generation
    analysis_results = {}
    
    # Generate analysis for all key metrics
    key_metrics = [
        'Parc_global',
        'Total_Prepaye', 
        'Total_Postpaye',
        'GSM_Mobilis',
        '3G4G_Mobilis'
    ]
    
    for metric in key_metrics:
        metric_analysis = analyze_metric_deep_dive(processed_df, metric)
        if metric_analysis:
            analysis_results[metric] = metric_analysis
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("    ")
        if st.button("📊 Générer Rapport d'Analyse", key="generate_report", help="Générer un rapport PDF complet"):
            if analysis_results:
                with st.spinner("Génération du rapport en cours..."):
                    try:
                        pdf_buffer = generate_pdf_report(processed_df, analysis_results)
                        st.success("✅ Rapport généré avec succès!")
                        
                        # Provide download button
                        st.download_button(
                            label="📥 Télécharger le Rapport PDF",
                            data=pdf_buffer.getvalue(),
                            file_name=f"rapport_analyse_mobile_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                            mime="application/pdf"
                        )
                    except Exception as e:
                        st.error(f"Erreur lors de la génération du rapport: {str(e)}")
            else:
                st.warning("Aucun résultat d'analyse disponible pour générer un rapport.")
    
    with col2:
        st.write("    ")
        # Export raw data
        if st.button("📥 Exporter Données Brutes", key="export_data"):
            csv_data = processed_df.to_csv(index=False)
            st.download_button(
                label="Télécharger CSV",
                data=csv_data,
                file_name=f"donnees_mobile_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    # Summary insights from analysis
    st.subheader("💡 Key Insights from Analysis")
    
    insights = []
    st.write("    ")
    if not processed_df.empty:
        # Market leader insight
        if operator_totals:
            market_leader = max(operator_totals, key=operator_totals.get)
            leader_share = (operator_totals[market_leader] / sum(operator_totals.values()) * 100) if sum(operator_totals.values()) > 0 else 0
            insights.append(f"👑 {market_leader} leads the market with {leader_share:.1f}% share")
        
        # Technology transition insight
        if not gsm_total.empty and not data_3g4g_total.empty:
            latest_gsm = gsm_total['Value'].iloc[-1]
            latest_3g4g = data_3g4g_total['Value'].iloc[-1]
            total_tech = latest_gsm + latest_3g4g
            if total_tech > 0:
                gsm_percentage = (latest_gsm / total_tech * 100)
                if gsm_percentage < 50:
                    insights.append("📶 3G/4G has become the dominant technology, overtaking GSM")
                else:
                    insights.append(f"📶 GSM still represents {gsm_percentage:.1f}% of mobile subscriptions")
        
        # Prepaid vs Postpaid insight
        if not prepaid_data.empty and not postpaid_data.empty:
            latest_prepaid = prepaid_data['Value'].iloc[-1]
            latest_postpaid = postpaid_data['Value'].iloc[-1]
            total_payment = latest_prepaid + latest_postpaid
            if total_payment > 0:
                prepaid_percentage = (latest_prepaid / total_payment * 100)
                insights.append(f"💳 Prepaid dominates with {prepaid_percentage:.1f}% of subscribers")
        
        # Overall growth insight
        if not mobile_subscribers.empty and len(mobile_subscribers) > 1:
            overall_growth = ((mobile_subscribers['Value'].iloc[-1] - mobile_subscribers['Value'].iloc[0]) / mobile_subscribers['Value'].iloc[0] * 100)
            if overall_growth > 0:
                insights.append(f"📈 Mobile market grew by {overall_growth:.1f}% over the analysis period")
            else:
                insights.append(f"📉 Mobile market contracted by {abs(overall_growth):.1f}% over the analysis period")

    # Display insights
    if insights:
        for insight in insights:
            st.markdown(f"- {insight}")
    else:
        st.info("Analysis complete. Use the tabs above to explore detailed metrics and trends.")
    
    # Export options summary
    st.subheader("📤 Export Options")
    st.write("    ")
    st.markdown("""
    - **PDF Report**: Comprehensive analysis report with visualizations and insights
    - **CSV Export**: Raw data in spreadsheet format for further analysis
    - **Interactive Charts**: All charts can be downloaded as PNG images by hovering and clicking the camera icon
    """)

st.markdown("---")
st.info("💡 All charts are interactive: hover for details, zoom, pan, and download as PNG for reports.")

st.markdown("</div>", unsafe_allow_html=True)