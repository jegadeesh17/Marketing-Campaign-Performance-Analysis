CREATE TABLE raw_campaign_data (
    campaign_id VARCHAR(50),
    campaign_type VARCHAR(50),
    target_audience VARCHAR(100),
    duration FLOAT,
    channel_used VARCHAR(255),
    impressions FLOAT,
    clicks FLOAT,
    leads FLOAT,
    conversions FLOAT,
    revenue FLOAT,
    acquisition_cost FLOAT,
    roi FLOAT,
    language VARCHAR(50),
    engagement_score FLOAT,
    customer_segment VARCHAR(100),
    date_str VARCHAR(50),
    brand VARCHAR(50) -- Engineered column to distinguish sources
);
