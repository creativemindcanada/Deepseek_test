import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import requests

# Title of the app
st.title("Customer Insights Dashboard")

# Current Dashboard (Primary)
st.header("Primary Dashboard: Upload CSV or Use Random Data")

# Function to generate random customer data
def generate_random_data():
    np.random.seed(42)
    data = {
        "customer_id": range(1, 101),
        "satisfaction_score": np.random.randint(1, 6, 100),
        "feedback": np.random.choice([
            "Great product!", "Good quality.", "Delivery was late.", "Poor packaging.", "Excellent service!"
        ], 100),
        "purchase_amount": np.random.uniform(10, 500, 100).round(2),
        "region": np.random.choice(["North", "South", "East", "West"], 100)
    }
    return pd.DataFrame(data)

# Upload customer data
uploaded_file = st.file_uploader("Upload your customer data (CSV file)", type=["csv"])

# Button to generate random data
if st.button("Use Randomly Generated Customer Data"):
    data = generate_random_data()
    st.session_state['data'] = data
elif uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.session_state['data'] = data
else:
    data = None

# Display current dashboard
if 'data' in st.session_state and st.session_state['data'] is not None:
    data = st.session_state['data']

    # Display raw data
    st.subheader("Raw Data")
    st.write(data)

    # Basic Analysis
    st.subheader("Basic Analysis")
    st.write(f"Total Customers: {len(data)}")
    st.write(f"Average Satisfaction Score: {data['satisfaction_score'].mean():.2f}")

    # Visualizations
    st.subheader("Customer Satisfaction Distribution")
    fig = px.histogram(data, x="satisfaction_score", nbins=10)
    st.plotly_chart(fig)

    # Insights and Recommendations
    st.subheader("Insights & Recommendations")
    if data['satisfaction_score'].mean() >= 4:
        st.success("Things are going well! Keep up the good work.")
        st.write("Recommendations: Consider launching a loyalty program to retain happy customers.")
    else:
        st.error("There are issues to address.")
        st.write("Recommendations: Investigate customer feedback and improve product/service quality.")

    # Sentiment Analysis (Optional)
    st.subheader("Sentiment Analysis")
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    import nltk
    nltk.download('vader_lexicon')

    def analyze_sentiment(text):
        sia = SentimentIntensityAnalyzer()
        return sia.polarity_scores(text)['compound']

    data['sentiment'] = data['feedback'].apply(analyze_sentiment)
    st.write(data[['customer_id', 'feedback', 'sentiment']])

    # Add filters
    st.subheader("Filter Data")
    min_score = st.slider("Minimum Satisfaction Score", 1, 5, 3)
    region_filter = st.selectbox("Select Region", ["All", "North", "South", "East", "West"])

    # Apply filters
    filtered_data = data[data['satisfaction_score'] >= min_score]
    if region_filter != "All":
        filtered_data = filtered_data[filtered_data['region'] == region_filter]

    st.write(filtered_data)

    # Add visualizations
    st.subheader("Customer Satisfaction by Region")
    fig = px.bar(filtered_data, x="region", y="satisfaction_score", color="region", barmode="group")
    st.plotly_chart(fig)

    # Add download button
    st.subheader("Download Analyzed Data")
    st.download_button(
        label="Download CSV",
        data=filtered_data.to_csv().encode('utf-8'),
        file_name="analyzed_data.csv",
        mime="text/csv"
    )
else:
    st.info("Please upload a CSV file or click the button to use randomly generated data.")

# Alternative Inputs (Below the Primary Dashboard)
st.header("Alternative Input Methods")

# Option 1: Company LinkedIn URL
# Option 1: Company LinkedIn URL
st.subheader("Analyze Company LinkedIn Profile")
linkedin_url = st.text_input("Enter Company LinkedIn URL")
if linkedin_url:
    st.write(f"Fetching insights for LinkedIn URL: {linkedin_url}")
    
    # Mock LinkedIn Data
    mock_linkedin_data = {
        "Company Name": "NVIDIA",
        "Followers": "2,500,000",
        "Employees": "20,000",
        "Industry": "Semiconductors",
        "Recent Posts": [
            "Excited to announce our new AI-powered GPU!",
            "Join us at the NVIDIA GTC conference next month!",
            "We're hiring! Check out our open positions."
        ]
    }
    
    # Display Mock Data
    st.subheader("Mock LinkedIn Insights")
    st.write(f"**Company Name:** {mock_linkedin_data['Company Name']}")
    st.write(f"**Followers:** {mock_linkedin_data['Followers']}")
    st.write(f"**Employees:** {mock_linkedin_data['Employees']}")
    st.write(f"**Industry:** {mock_linkedin_data['Industry']}")
    
    st.subheader("Recent Posts")
    for post in mock_linkedin_data['Recent Posts']:
        st.write(f"- {post}")
    
    # Add Disclaimer
    st.warning("""
    **Note:** LinkedIn API integration is not available in this demo. 
    The insights shown are for demonstration purposes only.
    """)

# Option 2: Company Website URL
st.subheader("Analyze Company Website")
website_url = st.text_input("Enter Company Website URL")
if website_url:
    st.write(f"Fetching insights for website: {website_url}")
    # Add website analysis logic here (e.g., Google PageSpeed Insights)
    st.warning("Website analysis is not implemented in this demo.")

# Option 3: Sharable Link (Google Sheets, Airtable)
st.subheader("Analyze Data from Sharable Link")
sharable_link = st.text_input("Enter Sharable Link (Google Sheets, Airtable, etc.)")
if sharable_link:
    st.write(f"Fetching data from sharable link: {sharable_link}")
    # Add Google Sheets or Airtable integration logic here
    st.warning("Sharable link integration is not implemented in this demo.")

# Option 4: Social Media Handles
st.subheader("Analyze Social Media Engagement")
social_media_handle = st.text_input("Enter Social Media Handle (e.g., @company)")
if social_media_handle:
    st.write(f"Fetching insights for social media handle: {social_media_handle}")
    # Add social media API integration logic here
    st.warning("Social media API integration is not implemented in this demo.")

# Option 5: Latest Trends
st.subheader("Latest Market Trends")
if st.button("Fetch Latest Trends"):
    st.write("Fetching the latest market trends...")
    # Add Google Trends or OpenAI GPT integration logic here
    st.warning("Trend analysis is not implemented in this demo.")
