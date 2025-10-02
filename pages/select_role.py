import streamlit as st

# =========================
# Page Config
# =========================
st.set_page_config(
    page_title="Telecom Dashboard - Select Role",
    layout="wide"
)

# =========================
# Custom CSS
# =========================
role_page_css = """
<style>
/* Page background */
.stApp {
    background: linear-gradient(135deg, #0c0c0c, #1a1a2e, #16213e);
    font-family: 'Poppins', sans-serif;
    color: white;
}

/* Logo */
.logo {
    display: block;
    margin: 0 auto 20px auto;
    width: 90px;
}

/* Page title */
h1 {
    text-align: center;
    font-size: 2.5em !important;
    color: #ffffff;
    margin-top: 20px;
    margin-bottom: 20px;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
}

/* Guide text */
p.guide {
    text-align: center;
    font-size: 1.1em;
    color: #e0e0e0;
    margin-bottom: 40px;
    max-width: 800px;
    margin-left: auto;
    margin-right: auto;
    line-height: 1.6;
}

/* Role cards */
.role-card {
    background: linear-gradient(135deg, #2e7d32, #4caf50);
    color: #ffffff;
    height: 280px;
    border-radius: 15px;
    text-align: center;
    padding: 25px 20px;
    box-shadow: 0px 8px 25px rgba(0,0,0,0.3);
    transition: transform 0.3s ease, box-shadow 0.3s ease;
    border: 1px solid rgba(255,255,255,0.1);
    display: flex;
    flex-direction: column;
    justify-content: space-between;
}

.role-card:hover {
    transform: translateY(-5px);
    box-shadow: 0px 12px 30px rgba(46, 125, 50, 0.4);
    background: linear-gradient(135deg, #1b5e20, #2e7d32);
}

/* Card title */
.role-card h3 {
    font-size: 1.5em;
    margin-bottom: 15px;
    color: #ffffff;
}

/* Card description */
.role-card p {
    font-size: 0.95em;
    line-height: 1.5;
    margin-bottom: 20px;
    color: #f0f0f0;
    flex-grow: 1;
}

/* Card button */
.role-btn {
    display: inline-block;
    padding: 10px 25px;
    font-size: 1em;
    font-weight: bold;
    color: #2e7d32 !important;
    background-color: #ffffff;
    border-radius: 25px;
    text-decoration: none;
    transition: all 0.3s ease;
    border: none;
    box-shadow: 0px 4px 10px rgba(0,0,0,0.2);
}

.role-btn:hover {
    background-color: #f0f0f0;
    transform: scale(1.05);
    color: #1b5e20 !important;
}

/* Center the columns container */
.columns-container {
    display: flex;
    justify-content: center;
    gap: 30px;
    margin-top: 30px;
}
</style>
"""
st.markdown(role_page_css, unsafe_allow_html=True)

# =========================
# Page Layout
# =========================

# Logo centered at top
st.image("assets/logo.png", width=90)

# Title
st.markdown("<h1>Select Your Role</h1>", unsafe_allow_html=True)

# Guide section
st.markdown("""
<p class="guide">
Choose your role below. Each role provides a different way to interact with the Telecom Dashboard:  
<b>Enter New Values</b> to keep the system updated,  
<b>Analyze Previous Trends</b> to view historical data insights,  
or <b>Forecast Future Values</b> to predict the coming years.
</p>
""", unsafe_allow_html=True)

# =========================
# Role Cards - USING STREAMLIT COLUMNS
# =========================

# Create three equal columns
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="role-card">
        <div>
            <h3>📥 Enter New Value</h3>
            <p>Update the system by entering data for past semesters, ensuring our dashboard stays accurate and up-to-date.</p>
        </div>
        <a href="/Enter_New_Value" target="_self" class="role-btn">Get Started</a>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="role-card">
        <div>
            <h3>📊 Analyze Previous Years</h3>
            <p>Explore historical data trends for Internet, Fixed-line, and Mobile services with interactive charts and insights.</p>
        </div>
        <a href="/Analyze_Previous" target="_self" class="role-btn">Explore Data</a>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="role-card">
        <div>
            <h3>🔮 Forecast Coming Years</h3>
            <p>Use predictive models to estimate Telecom metrics for future years and make informed strategic decisions.</p>
        </div>
        <a href="/Forecast_Coming" target="_self" class="role-btn">View Forecasts</a>
    </div>
    """, unsafe_allow_html=True)

# =========================
# Additional Information
# =========================
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #b0b0b0; margin-top: 40px;'>
        <p>💡 <b>Tip:</b> Each role offers specialized tools tailored for different analysis needs. Choose the one that matches your current objective.</p>
    </div>
    """,
    unsafe_allow_html=True
)