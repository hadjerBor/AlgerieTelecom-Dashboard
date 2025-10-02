import streamlit as st

# ===============================
# PAGE CONFIGURATION
# ===============================
st.set_page_config(
    page_title="Analyze Previous Data",
    page_icon="📊",
    layout="wide"
)
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
            margin-bottom: 20px; /* extra spacing under titles */
        }
        .control-card {
            background: rgba(30, 30, 46, 0.9);
            padding: 25px;
            border-radius: 15px;
            border: 1px solid #2e7d32;
            margin: 30px 0; /* spacing between sections */
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
        }
        .metric-card {
            background: rgba(30, 30, 46, 0.8);
            padding: 25px;
            border-radius: 12px;
            border: 1px solid #2e7d32;
            margin: 25px 0; /* extra space */
        }
        .domain-card {
            background: rgba(46, 125, 50, 0.1);
            backdrop-filter: blur(15px);
            border-radius: 20px;
            padding: 2rem;
            border: 1px solid rgba(76, 175, 80, 0.3);
            transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
            position: relative;
            overflow: hidden;
            height: 370px; /* slightly taller for breathing room */
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            text-align: center;
            cursor: pointer;
            margin-bottom: 2rem; /* spacing between cards */
        }
        .domain-card:hover {
            transform: translateY(-10px);
            box-shadow: 0 20px 40px rgba(46, 125, 50, 0.3);
            background: rgba(46, 125, 50, 0.2);
            border-color: rgba(76, 175, 80, 0.6);
        }
        .domain-icon {
            font-size: 4rem;
            margin-bottom: 1.8rem;
            filter: drop-shadow(0 4px 8px rgba(0,0,0,0.5));
        }
        .domain-title {
            font-size: 1.9rem;
            font-weight: bold;
            margin-bottom: 1.2rem;
            color: #4caf50;
        }
        .domain-desc {
            font-size: 1rem;
            color: #c8e6c9;
            line-height: 1.7;
            flex-grow: 1;
            margin-bottom: 1rem;
        }
        .stButton button {
            background: linear-gradient(45deg, #2e7d32, #4caf50);
            color: white;
            font-weight: bold;
            border: none;
            padding: 15px 30px;
            border-radius: 10px;
            transition: all 0.3s ease;
            width: 100%;
            font-size: 1.1rem;
            margin-top: 15px;
        }
        .stButton button:hover {
            background: linear-gradient(45deg, #1b5e20, #2e7d32);
            transform: translateY(-2px);
        }
        .user-guide {
            background: linear-gradient(135deg, rgba(46, 125, 50, 0.2), rgba(76, 175, 80, 0.1));
            padding: 2.5rem;
            border-radius: 20px;
            border: 1px solid rgba(76, 175, 80, 0.3);
            margin: 40px 0; /* extra space around guide */
            box-shadow: 0 8px 32px rgba(46, 125, 50, 0.2);
            line-height: 1.8;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# ===============================
# TITLE & USER GUIDE
# ===============================
st.title("📊 Analyze Historical Data - Algeria Telecom")

st.markdown("""
<div class="user-guide">
<h3>ℹ️ User Guide</h3>
            <p></p>
<p>Welcome to the <b>Historical Data Analysis Dashboard</b>. This platform provides comprehensive analysis of Algeria's telecommunications sector across three key domains:</p>

<ul>
<li><b>📱 Mobile Services:</b> GSM/3G/4G subscriber trends, operator comparisons, prepaid vs postpaid analysis</li>
<li><b>🌐 Internet Services:</b> ADSL, Fiber, and LTE subscriber evolution, traffic patterns, and revenue metrics</li>
<li><b>📞 Fixed Line Services:</b> Voice traffic analysis, on-net/off-net patterns, and subscriber metrics</li>
</ul>

<p><b>Instructions:</b> Select a domain below to access detailed analytics with interactive visualizations, trend analysis, market share evolution, and anomaly detection.</p>
</div>
""", unsafe_allow_html=True)

# ===============================
# DOMAIN SELECTION CARDS
# ===============================
st.markdown("## 🌐 Select Analysis Domain")

col1, col2, col3 = st.columns(3)

domains = {
    "Internet": {
        "icon": "🌐",
        "title": "Internet Analysis",
        "description": "Comprehensive analysis of Internet subscribers across ADSL, Fiber (FTTH), and LTE technologies. Track traffic consumption, revenue trends, and technology adoption patterns.",
        "metrics": ["Global Subscribers", "ADSL Evolution", "Fiber Growth", "LTE Adoption", "Traffic Analysis", "ARPU Trends"],
        "page": "pages/analyze_internet.py"
    },
    "Mobile": {
        "icon": "📱", 
        "title": "Mobile Analysis",
        "description": "Deep dive into mobile services including GSM vs 3G/4G transitions, operator market share analysis, and prepaid vs postpaid subscriber patterns.",
        "metrics": ["GSM Migration", "3G/4G Growth", "Operator Comparison", "Prepaid vs Postpaid", "Market Share", "Technology Evolution"],
        "page": "pages/analyze_mobile.py"
    },
    "Fixed": {
        "icon": "📞",
        "title": "Fixed Line Analysis", 
        "description": "Analysis of fixed-line voice services including traffic patterns, subscriber evolution, on-net vs off-net usage analysis, and minutes per subscriber trends.",
        "metrics": ["Subscriber Base", "Traffic Patterns", "On-Net vs Off-Net", "Usage Minutes", "Service Performance", "Market Trends"],
        "page": "pages/analyze_fixe.py"
    }
}

columns = [col1, col2, col3]
for i, (domain_key, domain_info) in enumerate(domains.items()):
    with columns[i]:
        card_html = f"""
        <p></p>
        <p></p>
        <div class="domain-card" onclick="document.getElementById('{domain_key.lower()}_btn').click()">
            <div class="domain-icon">{domain_info['icon']}</div>
            <div class="domain-title">{domain_info['title']}</div>
            <div class="domain-desc">{domain_info['description']}</div>
            <div style="margin-top: 1.2rem;">
                <strong>Key Metrics:</strong><br>
                {'<br>'.join([f"• {metric}" for metric in domain_info['metrics'][:3]])}
                <br>• ... and more
            </div>
        </div>
        """
        st.markdown(card_html, unsafe_allow_html=True)
        
        if st.button(f"Analyze {domain_key}", key=f"{domain_key.lower()}_btn", help=f"Click to analyze {domain_key} data"):
            st.switch_page(domain_info['page'])

# ===============================
# ADDITIONAL INFORMATION SECTION
# ===============================
st.markdown("---")
st.markdown("## 📈 What You'll Find in Each Analysis")

info_col1, info_col2, info_col3 = st.columns(3)

with info_col1:
    st.markdown("""
    <div class="metric-card">
    <h4>🔍 Dataset Overview</h4>
                <p></p>
    <p>• Data preview and summary statistics<br>
    • Missing values analysis<br>
    • Time range and data quality insights</p>
    </div>
    """, unsafe_allow_html=True)

with info_col2:
    st.markdown("""
    <div class="metric-card">
    <h4>📊 Interactive Visualizations</h4>
                <p></p>
    <p>• Time-series analysis with trend lines<br>
    • Technology/service comparisons<br>
    • Market share evolution charts</p>
    </div>
    """, unsafe_allow_html=True)

with info_col3:
    st.markdown("""
    <div class="metric-card">
    <h4>🎯 Advanced Analytics</h4>
                <p></p>
    <p>• Anomaly detection and alerts<br>
    • Growth rate calculations<br>
    • Moving averages and forecasting</p>
    </div>
    """, unsafe_allow_html=True)

# ===============================
# TECHNICAL SPECIFICATIONS
# ===============================
with st.expander("🔧 Technical Specifications"):
    st.markdown("""
    ### Data Sources & Structure
    
    **Internet Data:**
    - Time Period: 2018 T2 - 2025 T1 (Quarterly data)
    - Technologies: ADSL, LTE Fixed, Fiber (FTTH), Mobile 3G/4G
    - Metrics: Subscribers, Traffic consumption, ARPU
    
    **Mobile Data:**
    - Time Period: 2019 S1 - 2025 S1 (Semi-annual data)  
    - Operators: Mobilis, Djezzy, Ooredoo
    - Technologies: GSM, 3G/4G
    - Payment Types: Prepaid, Postpaid
    
    **Fixed Line Data:**
    - Time Period: 2019 S1 - 2025 S1 (Semi-annual data)
    - Traffic Types: On-Net, Off-Net, Incoming, Outgoing
    - Metrics: Minutes per subscriber, Total traffic
    
    ### Visualization Features
    - Interactive Plotly charts with zoom and pan
    - Dark theme optimized for data visibility
    - Color-coded technology/service categories
    - Exportable charts and data tables
    """)

st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 2.5rem 0; color: rgba(200,230,201,0.8); line-height: 1.8;">
    ✨ <b>Done by Bordjiba Hadjer & Latreche Dhikra Maram</b><br>
    📅 <b>Internship Project for Algeria Telecom</b>
</div>
""", unsafe_allow_html=True)
