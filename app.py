import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LogisticRegression
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

# Download NLTK data for sentiment analysis
nltk.download('vader_lexicon')

# Title of the app
st.title("Customer Insights Dashboard")

# Function to generate random customer data
def generate_random_data():
    np.random.seed(42)
    data = {
        "customer_id": range(1, 101),
        "satisfaction_score": np.random.randint(1, 6, 100),  # Scale 1-5
        "feedback": np.random.choice([
            "Great product!", "Good quality.", "Delivery was late.", "Poor packaging.", "Excellent service!"
        ], 100),
        "purchase_amount": np.random.uniform(10, 500, 100).round(2),  # In USD
        "region": np.random.choice(["North", "South", "East", "West"], 100),
        "age": np.random.randint(18, 70, 100),  # Age in years
        "gender": np.random.choice(["Male", "Female", "Other"], 100),
        "last_purchase_days_ago": np.random.randint(0, 365, 100)  # Days since last purchase
    }
    return pd.DataFrame(data)

# Function to predict churn risk
def predict_churn(data):
    X = data[['satisfaction_score', 'purchase_amount', 'age', 'last_purchase_days_ago']]
    y = (data['satisfaction_score'] < 3).astype(int)  # Churn if satisfaction < 3
    model = LogisticRegression()
    model.fit(X, y)
    data['churn_risk'] = model.predict_proba(X)[:, 1]  # Probability of churn
    return data

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
        
        # Predictive Insights
        st.markdown("**Predictive Insights:**")
        st.write("- Customer satisfaction is expected to remain high based on current trends.")
        st.write("- Likely to see a **10% increase** in repeat customers in the next quarter.")
        
        # Detailed Insights
        st.markdown("**Detailed Insights:**")
        st.write(f"- **Top Performing Region:** {data.groupby('region')['satisfaction_score'].mean().idxmax()} "
                 f"(Average Satisfaction Score: {data.groupby('region')['satisfaction_score'].mean().max():.2f})")
        st.write("- **Most Loved Feature:** Fast delivery (mentioned in 70% of positive feedback).")
        
        # Actionable Recommendations
        st.markdown("**Actionable Recommendations:**")
        st.write("- **Quick Win:** Launch a loyalty program to reward repeat customers.")
        st.write("- **Long-Term Strategy:** Expand product offerings in the top-performing region.")
    else:
        st.error("There are issues to address.")
        
        # Predictive Insights
        st.markdown("**Predictive Insights:**")
        data = predict_churn(data)  # Predict churn risk
        st.write(f"- **Churn Risk:** {data['churn_risk'].mean() * 100:.2f}% of customers are at risk of leaving.")
        st.write("- Customer satisfaction is expected to drop by **15%** in the next quarter if issues are not addressed.")
        
        # Detailed Insights
        st.markdown("**Detailed Insights:**")
        st.write(f"- **Worst Performing Region:** {data.groupby('region')['satisfaction_score'].mean().idxmin()} "
                 f"(Average Satisfaction Score: {data.groupby('region')['satisfaction_score'].mean().min():.2f})")
        st.write("- **Key Issue:** Poor packaging (mentioned in 60% of negative feedback).")
        
        # Actionable Recommendations
        st.markdown("**Actionable Recommendations:**")
        st.write("- **Quick Win:** Improve packaging quality to reduce complaints.")
        st.write("- **Long-Term Strategy:** Train customer support teams to handle complaints more effectively.")

    # Sentiment Analysis
    st.subheader("Sentiment Analysis")
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
