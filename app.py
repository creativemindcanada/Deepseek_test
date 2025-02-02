import streamlit as st
import pandas as pd
import plotly.express as px
import requests
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

# Download the sentiment analysis model
nltk.download('vader_lexicon')

# Title of the app
st.title("Customer Insights Dashboard")

# Upload customer data
uploaded_file = st.file_uploader("Upload your customer data (CSV file)", type=["csv"])
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

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

    # Sentiment Analysis
    st.subheader("Sentiment Analysis")
    def analyze_sentiment(text):
        sia = SentimentIntensityAnalyzer()
        return sia.polarity_scores(text)['compound']
    
    data['sentiment'] = data['feedback'].apply(analyze_sentiment)
    st.write(data[['customer_id', 'feedback', 'sentiment']])

# Live Market Data
st.subheader("Live Market Data")
def get_market_data():
    try:
        api_key = "758AUAMY7VA84YW3"  # Replace with your actual API key
        url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=MSFT&apikey={api_key}"
        response = requests.get(url)
        
        # Check if the request was successful
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Failed to fetch market data. Status code: {response.status_code}")
            return None
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return None

market_data = get_market_data()
if market_data:
    st.write(market_data)
else:
    st.warning("No market data available.")
