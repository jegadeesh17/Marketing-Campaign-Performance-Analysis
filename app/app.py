import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load trained pipelines
@st.cache_resource
def load_models():
    reg_model = joblib.load('models/revenue_regressor.joblib')
    cls_model = joblib.load('models/profit_classifier.joblib')
    return reg_model, cls_model

reg_model, cls_model = load_models()

st.title("🎯 Multi-Brand Marketing Performance Engine")
st.markdown("Predict campaign revenue metrics and operational profitability before deploying capital.")

# Input layout split into columns
col1, col2 = st.columns(2)

with col1:
    brand = st.selectbox("Select Corporate Brand", ["nykaa", "purplle", "tira"])
    campaign_type = st.selectbox("Campaign Strategy", ["Social Media", "Paid Ads", "Influencer", "Email", "SEO"])
    target_audience = st.selectbox("Target Core Audience", ["College Students", "Tier 2 City Customers", "Youth", "Working Women"])
    customer_segment = st.selectbox("Target Customer Tier", ["College Students", "Premium Shoppers", "Working Women", "Tier 2 City Customers"])
    language = st.selectbox("Content Language Context", ["English", "Hindi", "Tamil", "Bengali"])
    duration = st.slider("Campaign Active Window (Days)", 1, 30, 14)
    month = st.slider("Target Execution Month", 1, 12, 5)

with col2:
    impressions = st.number_input("Expected Impressions", min_value=0, value=50000)
    clicks = st.number_input("Expected Click volume", min_value=0, value=4000)
    leads = st.number_input("Projected Inbound Leads", min_value=0, value=1500)
    conversions = st.number_input("Target Conversions achieved", min_value=0, value=500)
    engagement_score = st.slider("Target Engagement Score Matrix", 0.0, 30.0, 15.0)
    acquisition_cost = st.number_input("Cost Per Acquisition (₹)", min_value=0.0, value=250.0)

# Multi-select channels
selected_channels = st.multiselect("Select Delivery Channels", ["YouTube", "Instagram", "Google", "WhatsApp", "Email", "Facebook"])

# Transform UI inputs back to structural DataFrame row format
input_data = {
    'campaign_type': campaign_type,
    'target_audience': target_audience,
    'language': language,
    'customer_segment': customer_segment,
    'brand': brand,
    'duration': float(duration),
    'impressions': float(impressions),
    'clicks': float(clicks),
    'leads': float(leads),
    'conversions': float(conversions),
    'engagement_score': float(engagement_score),
    'month': float(month)
}

# Add multi-label encoded channels to match expectations
channels_list = ["YouTube", "Instagram", "Google", "WhatsApp", "Email", "Facebook"]
for ch in channels_list:
    input_data[f'channel_{ch.lower()}'] = 1 if ch in selected_channels else 0

# Create separate analytical rows for both tasks to prevent leakage tracking
df_reg_input = pd.DataFrame([input_data])
df_reg_input['acquisition_cost'] = float(acquisition_cost) # Only regressor requires cost metric

df_cls_input = pd.DataFrame([input_data])

if st.button("🚀 Calculate Campaign Forecasts"):
    # 1. Run Revenue Model
    predicted_revenue = reg_model.predict(df_reg_input)[0]
    
    # 2. Run Classification Model
    predicted_profitability = cls_model.predict(df_cls_input)[0]
    
    st.subheader("Result Projections Analysis:")
    
    c1, c2 = st.columns(2)
    c1.metric(label="Forecasted Revenue Return", value=f"₹{predicted_revenue:,.2f}")
    
    if predicted_profitability == 1:
        c2.success("📈 Forecasted Status: PROFITABLE CAMPAIGN")
    else:
        c2.error("📉 Forecasted Status: NET OPERATIONAL LOSS")