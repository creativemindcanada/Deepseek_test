import streamlit as st
import pandas as pd
import plotly.express as px

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
else:
    st.info("Please upload a CSV file to get started.")
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

# Download the sentiment analysis model
nltk.download('vader_lexicon')

# Function to analyze sentiment
def analyze_sentiment(text):
    sia = SentimentIntensityAnalyzer()
    return sia.polarity_scores(text)['compound']

# Add sentiment analysis to the app
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    data['sentiment'] = data['feedback'].apply(analyze_sentiment)

    # Display sentiment analysis results
    st.subheader("Sentiment Analysis")
    st.write(data[['customer_id', 'feedback', 'sentiment']])
    import requests

# Function to get live market data
def get_market_data():
    api_key = "758AUAMY7VA84YW3"  # Replace with your Alpha Vantage API key
    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=MSFT&apikey={api_key}"
    response = requests.get(url)
    return response.json()

# Add market data to the app
st.subheader("Live Market Data")
market_data = get_market_data()
st.write(market_data)
# Add a slider for filtering satisfaction scores
st.subheader("Filter Data")
min_score = st.slider("Minimum Satisfaction Score", 1, 5, 3)

# Filter the data
filtered_data = data[data['satisfaction_score'] >= min_score]

# Display filtered data
st.write(filtered_data)
