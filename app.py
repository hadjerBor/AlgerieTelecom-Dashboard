import streamlit as st

# =========================
# Page Config
# =========================
st.set_page_config(
    page_title="Algeria Telecom Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =========================
# Custom CSS for enhanced green theme styling
# =========================
page_bg = """
<style>
/* Full-screen gradient background with enhanced green theme */
.stApp {
    background: linear-gradient(135deg, #0d1117, #1a1a2e, #16213e);
    background-attachment: fixed;
    color: #ffffff;
    font-family: 'Segoe UI', 'Roboto', sans-serif;
    margin: 0;
    padding: 0;
}

/* Remove default Streamlit padding */
.main .block-container {
    padding-top: 1rem;
    padding-bottom: 2rem;
    max-width: 1400px;
}

/* Main container with glass morphism effect */
.main-container {
    background: rgba(0, 0, 0, 0.4);
    backdrop-filter: blur(10px);
    border-radius: 20px;
    margin: 1rem auto;
    padding: 2rem;
    box-shadow: 0 8px 32px rgba(46, 125, 50, 0.3);
    border: 1px solid rgba(46, 125, 50, 0.2);
}

/* Header section - fixed positioning */
.header-section {
    text-align: center;
    margin-bottom: 3rem;
    position: sticky;
    top: 0;
    background: rgba(0, 0, 0, 0.9);
    backdrop-filter: blur(15px);
    padding: 2rem 1rem;
    border-radius: 20px;
    z-index: 100;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
}

/* Logo styling with glow effect */
.logo-container {
    margin-bottom: 1.5rem;
    display: flex;
    justify-content: center;
}

.logo-placeholder {
    width: 120px; 
    height: 120px; 
    background: linear-gradient(135deg, #2e7d32, #4caf50); 
    border-radius: 50%; 
    display: flex; 
    align-items: center; 
    justify-content: center; 
    box-shadow: 0 8px 30px rgba(46, 125, 50, 0.6);
    border: 3px solid rgba(76, 175, 80, 0.4);
    animation: pulse 2s infinite;
}
/* Make logo image bigger inside the logo container */
.logo-container img {
    width: 180px !important;
    height: 180px !important;
    object-fit: contain;
}

@keyframes pulse {
    0% { box-shadow: 0 8px 30px rgba(46, 125, 50, 0.6); }
    50% { box-shadow: 0 8px 40px rgba(46, 125, 50, 0.8); }
    100% { box-shadow: 0 8px 30px rgba(46, 125, 50, 0.6); }
}

/* Enhanced title with gradient text */
.main-title {
    font-size: 3.5rem !important;
    font-weight: 800;
    background: linear-gradient(45deg, #4caf50, #81c784, #a5d6a7);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    text-shadow: 2px 2px 8px rgba(0,0,0,0.8);
    margin-bottom: 1rem;
    line-height: 1.2;
}

/* Subtitle styling */
.subtitle {
    font-size: 1.3rem;
    color: #c8e6c9;
    font-weight: 300;
    margin-bottom: 2rem;
    line-height: 1.8;
    text-shadow: 1px 1px 3px rgba(0,0,0,0.8);
}

/* Enhanced Start Button */
.main-cta-container {
    display: flex;
    justify-content: flex-end;
    margin: 2rem 0;
    padding-right: 2rem;
}

.stButton > button {
    padding: 35px 80px !important;
    font-size: 2.2rem !important;
    font-weight: 800 !important;
    background: linear-gradient(45deg, #1b5e20, #2e7d32, #388e3c) !important;
    color: #ffffff !important;
    border-radius: 50px !important;
    border: none !important;
    box-shadow: 0px 12px 35px rgba(27, 94, 32, 0.6) !important;
    transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1) !important;
    cursor: pointer !important;
}

.stButton > button:hover {
    transform: translateY(-8px) scale(1.08) !important;
    box-shadow: 0px 20px 45px rgba(27, 94, 32, 0.8) !important;
    background: linear-gradient(45deg, #0d3d12, #1b5e20, #2e7d32) !important;
}

/* Features section - fixed grid layout (always 4 columns) */
.features-container {
    margin-top: 3rem;
    display: grid;
    grid-template-columns: repeat(4, 1fr);  /* Always 4 side by side */
    gap: 1.5rem;
    max-width: 1400px;
    margin-left: auto;
    margin-right: auto;
    align-items: stretch;
}

.feature-card {
    background: rgba(46, 125, 50, 0.1);
    backdrop-filter: blur(15px);
    border-radius: 20px;
    padding: 1.5rem;
    border: 1px solid rgba(76, 175, 80, 0.3);
    transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
    position: relative;
    overflow: hidden;
    min-height: 300px;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    text-align: center;
}

.feature-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 4px;
    background: linear-gradient(90deg, #4caf50, #81c784, #4caf50);
    animation: shimmer 3s infinite;
}

@keyframes shimmer {
    0% { transform: translateX(-100%); }
    100% { transform: translateX(100%); }
}

.feature-card:hover {
    transform: translateY(-10px);
    box-shadow: 0 20px 40px rgba(46, 125, 50, 0.3);
    background: rgba(46, 125, 50, 0.2);
    border-color: rgba(76, 175, 80, 0.6);
}

.feature-icon {
    font-size: 3rem;
    margin-bottom: 1rem;
    filter: drop-shadow(0 4px 8px rgba(0,0,0,0.5));
}

.feature-title {
    font-size: 1.2rem;
    font-weight: bold;
    margin-bottom: 1rem;
    color: #4caf50;
}

.feature-desc {
    font-size: 0.95rem;
    color: #c8e6c9;
    line-height: 1.5;
    flex-grow: 1;
}

/* Hide streamlit elements */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Custom scrollbar */
::-webkit-scrollbar {
    width: 8px;
}

::-webkit-scrollbar-track {
    background: rgba(0, 0, 0, 0.3);
}

::-webkit-scrollbar-thumb {
    background: rgba(76, 175, 80, 0.6);
    border-radius: 4px;
}

::-webkit-scrollbar-thumb:hover {
    background: rgba(76, 175, 80, 0.8);
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# =========================
# Page Content
# =========================
# st.markdown('<div class="main-container">', unsafe_allow_html=True)

# # Header section - fixed at top
# st.markdown('<div class="header-section">', unsafe_allow_html=True)

# Logo section
st.markdown('<div class="logo-container">', unsafe_allow_html=True)
try:
    st.image("assets/logo.png", width=120)
except:
    # Create a placeholder logo with company initials
    st.markdown("""
    <div class="logo-placeholder">
        <span style="color: white; font-size: 2.5rem; font-weight: bold;">AT</span>
    </div>
    """, unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# Main title and subtitle
st.markdown('<h1 class="main-title">Algeria Telecom Dashboard</h1>', unsafe_allow_html=True)

st.markdown("""
<p class="subtitle">
Your comprehensive platform for analyzing and visualizing telecommunications data across 
<strong>Internet</strong>, <strong>Fixed-line</strong>, and <strong>Mobile</strong> services.  
Get deep insights, track performance trends, and make data-driven decisions with our advanced analytics tools.
</p>
            <p class="subtitle">
Done By <strong>Bordjiba Hadjer & Latreche Dhikra Maram</strong> | Built for Strategic Decision Making<br>
for <strong>Algeria Telecom</strong> internship project
</p>


""", unsafe_allow_html=True)


# Enhanced Start Button
if st.button(" Start Analysis", key="start_button"):
    st.switch_page("pages/select_role.py")

st.markdown('</div>', unsafe_allow_html=True)

# =========================
# Features section - cards side by side
# =========================
features = [
    {
        "icon": "📊",
        "title": "Interactive Analytics",
        "description": "Explore telecommunications data with dynamic, interactive charts and real-time filtering capabilities. Dive deep into trends with advanced visualizations."
    },
    {
        "icon": "📱",
        "title": "Mobile Insights",
        "description": "Comprehensive analysis of mobile services including GSM, 3G/4G adoption trends, operator comparisons, and prepaid vs postpaid analytics."
    },
    {
        "icon": "🌐",
        "title": "Internet Metrics",
        "description": "Monitor Internet subscriber growth across ADSL, Fiber, and LTE technologies. Track traffic consumption and revenue patterns over time."
    },
    {
        "icon": "📞",
        "title": "Fixed Line Data",
        "description": "Analyze fixed-line voice services including traffic patterns, subscriber evolution, on-net vs off-net usage, and service performance metrics."
    }
]

# Build all cards into one HTML block
cards_html = '<div class="features-container">'
for feature in features:
    cards_html += f'''
<div class="feature-card">
    <div class="feature-icon">{feature['icon']}</div>
    <div class="feature-title">{feature['title']}</div>
    <div class="feature-desc">{feature['description']}</div>
</div>'''
cards_html += '</div>'

# Render once
st.markdown(cards_html, unsafe_allow_html=True)



# =========================
# Footer
# =========================
st.markdown("""
<div style="text-align: center; padding: 2rem 0; color: rgba(200,230,201,0.8); font-size: 0.95rem;">
    📈 <strong>Done by Bordjiba Hadjer & Latreche Dhikra Maram</strong> | Built for Strategic Decision Making<br>
    <span style="color: rgba(76,175,80,0.8);">🔍 Unlock insights • 📊 Drive growth • 🎯 Make informed decisions</span>
</div>
""", unsafe_allow_html=True)
