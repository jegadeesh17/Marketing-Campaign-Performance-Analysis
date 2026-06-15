import streamlit as st
import pandas as pd
import joblib
import numpy as np

st.set_page_config(page_title="Marketing Engine", page_icon="🎯", layout="wide")

st.markdown("""
    <style>
    /* The Ocean (Background) */
    .stApp {
        background: linear-gradient(135deg, #e0e7ff 0%, #c7d2fe 100%);
    }
    
    /* The Floating Island (Main Container) */
    .block-container {
        background-color: rgba(255, 255, 255, 0.95);
        padding: 3rem !important;
        border-radius: 20px;
        box-shadow: 0 15px 35px rgba(0,0,0,0.1);
        margin-top: 3rem !important;
        margin-bottom: 3rem !important;
        max-width: 1200px !important;
    }

    /* Clean modern button */
    .stButton > button {
        width: 100%;
        border-radius: 10px;
        height: 55px;
        font-size: 18px;
        font-weight: bold;
        background: linear-gradient(90deg, #6366f1 0%, #8b5cf6 100%);
        color: white;
        border: none;
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.4);
    }
    .stButton > button:hover {
        opacity: 0.9;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# Load trained pipelines
@st.cache_resource
def load_models():
    reg_model = joblib.load('models/revenue_regressor.joblib')
    cls_model = joblib.load('models/profit_classifier.joblib')
    return reg_model, cls_model

reg_model, cls_model = load_models()

st.title("🎯 Multi-Brand Marketing Performance Engine")
st.markdown("Predict campaign revenue metrics and operational profitability before deploying capital.")

# Input layout split into 3 columns to save vertical space
col1, col2, col3 = st.columns(3)

with col1:
    brand = st.selectbox("Corporate Brand", ["nykaa", "purplle", "tira"])
    campaign_type = st.selectbox("Campaign Strategy", ["Social Media", "Paid Ads", "Influencer", "Email", "SEO"])
    target_audience = st.selectbox("Target Core Audience", ["College Students", "Tier 2 City Customers", "Youth", "Working Women"])
    customer_segment = st.selectbox("Target Customer Tier", ["College Students", "Premium Shoppers", "Working Women", "Tier 2 City Customers"])
    language = st.selectbox("Content Language Context", ["English", "Hindi", "Tamil", "Bengali"])

with col2:
    month = st.slider("Execution Month", 1, 12, 5)
    impressions = st.number_input("Expected Impressions", min_value=0, value=50000)
    clicks = st.number_input("Expected Click volume", min_value=0, value=4000)
    engagement_score = st.slider("Target Engagement Score", 0.0, 30.0, 15.0)

with col3:
    leads = st.number_input("Projected Inbound Leads", min_value=0, value=1500)
    conversions = st.number_input("Target Conversions", min_value=0, value=500)
    acquisition_cost = st.number_input("Cost Per Acquisition (₹)", min_value=0.0, value=250.0)
    
    # Multi-select channels
    selected_channels = st.multiselect("Delivery Channels", ["YouTube", "Instagram", "Google", "WhatsApp", "Email", "Facebook"])

# Transform UI inputs back to structural DataFrame row format
ctr = clicks / impressions if impressions > 0 else 0
conversion_rate = conversions / clicks if clicks > 0 else 0
cpl = acquisition_cost / leads if leads > 0 else 0
month_sin = np.sin(2 * np.pi * float(month) / 12)
month_cos = np.cos(2 * np.pi * float(month) / 12)

input_data = {
    'campaign_type': campaign_type,
    'target_audience': target_audience,
    'language': language,
    'customer_segment': customer_segment,
    'brand': brand,
    'impressions': float(impressions),
    'clicks': float(clicks),
    'leads': float(leads),
    'conversions': float(conversions),
    'engagement_score': float(engagement_score),
    'month_sin': month_sin,
    'month_cos': month_cos,
    'ctr': ctr,
    'conversion_rate': conversion_rate,
    'cpl': cpl
}

# Add multi-label encoded channels to match expectations
channels_list = ["YouTube", "Instagram", "Google", "WhatsApp", "Email", "Facebook"]
for ch in channels_list:
    input_data[f'channel_{ch.lower()}'] = 1 if ch in selected_channels else 0

# Create separate analytical rows for both tasks
df_reg_input = pd.DataFrame([input_data])
df_reg_input['acquisition_cost'] = float(acquisition_cost)

if st.button("🚀 Calculate Campaign Forecasts"):
    # 1. Run Revenue Model
    predicted_revenue = reg_model.predict(df_reg_input)[0]
    
    # 2. Run Classification Model
    # Classification model now requires acquisition_cost and revenue
    df_cls_input = pd.DataFrame([input_data])
    df_cls_input['acquisition_cost'] = float(acquisition_cost)
    df_cls_input['revenue'] = float(predicted_revenue)
    
    predicted_profitability = cls_model.predict(df_cls_input)[0]
    
    st.subheader("Result Projections Analysis:")
    
    c1, c2 = st.columns(2)
    c1.metric(label="Forecasted Revenue Return", value=f"₹{predicted_revenue:,.2f}")
    
    if predicted_profitability == 1:
        c2.success("📈 Forecasted Status: PROFITABLE CAMPAIGN")
    else:
        c2.error("📉 Forecasted Status: NET OPERATIONAL LOSS")