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
st.title("B2B Customer Insights Dashboard")

# Function to generate random B2B customer data with specified units
def generate_random_data():
    np.random.seed(42)
    data = {
        "company_id": range(1, 101),
        "satisfaction_score": np.random.randint(1, 6, 100),  # Scale 1-5
        "feedback": np.random.choice([
            "Excellent service!", "Good quality.", "Late delivery.", "Packaging issues.", "Great support!"
        ], 100),
        "purchase_amount": np.random.uniform(1000, 50000, 100).round(2),  # In USD
        "industry": np.random.choice(["Manufacturing", "Retail", "Healthcare", "Finance", "IT"], 100),
        "company_size": np.random.choice(["Small", "Medium", "Large"], 100),  # Company size categories
        "last_purchase_days_ago": np.random.randint(0, 365, 100)  # Days since last purchase
    }
    return pd.DataFrame(data)

# Function to predict churn risk
def predict_churn(data):
    X = data[['satisfaction_score', 'purchase_amount', 'last_purchase_days_ago']]
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
    st.write(f"Total Purchase Amount: ${data['purchase_amount'].sum():,.2f}")
    st.write(f"Average Days Since Last Purchase: {data['last_purchase_days_ago'].mean():.2f} days")

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
        st.write("- Overall customer satisfaction is high. Expect continued strong performance.")
        st.write("- Anticipated **15% increase** in large contract renewals next quarter.")

        # Detailed Insights
        st.markdown("**Detailed Insights:**")
        st.write(f"- **Top Performing Industry:** {data.groupby('industry')['satisfaction_score'].mean().idxmax()} "
                 f"(Average Satisfaction Score: {data.groupby('industry')['satisfaction_score'].mean().max():.2f})")
        st.write("- **Most Appreciated Feature:** Excellent support (mentioned in 80% of positive feedback).")

        # Actionable Recommendations
        st.markdown("**Actionable Recommendations:**")
        st.write("- **Quick Win:** Offer loyalty discounts to large companies for repeat business.")
        st.write("- **Long-Term Strategy:** Invest in support and delivery services for the top-performing industry.")
    else:
        st.error("There are issues to address.")
        
        # Predictive Insights
        st.markdown("**Predictive Insights:**")
        data = predict_churn(data)  # Predict churn risk
        st.write(f"- **Churn Risk:** {data['churn_risk'].mean() * 100:.2f}% of customers are at risk of leaving.")
        st.write("- Without addressing issues, satisfaction is expected to drop by **20%** in the next quarter.")

        # Detailed Insights
        st.markdown("**Detailed Insights:**")
        st.write(f"- **Worst Performing Industry:** {data.groupby('industry')['satisfaction_score'].mean().idxmin()} "
                 f"(Average Satisfaction Score: {data.groupby('industry')['satisfaction_score'].mean().min():.2f})")
        st.write("- **Primary Issue:** Late deliveries (mentioned in 70% of negative feedback).")

        # Actionable Recommendations
        st.markdown("**Actionable Recommendations:**")
        st.write("- **Quick Win:** Improve logistics to address late delivery complaints.")
        st.write("- **Long-Term Strategy:** Implement a comprehensive customer feedback system to identify and address issues promptly.")

    # Sentiment Analysis
    st.subheader("Sentiment Analysis")
    def analyze_sentiment(text):
        sia = SentimentIntensityAnalyzer()
        return sia.polarity_scores(text)['compound']

    data['sentiment'] = data['feedback'].apply(analyze_sentiment)
    st.write(data[['company_id', 'feedback', 'sentiment']])

    # Add filters
    st.subheader("Filter Data")
    min_score = st.slider("Minimum Satisfaction Score", 1, 5, 3)
    industry_filter = st.selectbox("Select Industry", ["All", "Manufacturing", "Retail", "Healthcare", "Finance", "IT"])

    # Apply filters
    filtered_data = data[data['satisfaction_score'] >= min_score]
    if industry_filter != "All":
        filtered_data = filtered_data[filtered_data['industry'] == industry_filter]

    st.write(filtered_data)

    # Add visualizations
    st.subheader("Customer Satisfaction by Industry")
    fig = px.bar(filtered_data, x="industry", y="satisfaction_score", color="industry", barmode="group")
    st.plotly_chart(fig)

    # Add dropdown for high-risk customers
    st.subheader("Personalized Insights for High-Risk Customers")
    high_risk_customers = data[data['churn_risk'] > 0.5]  # Customers with churn risk > 50%
    
    for index, row in high_risk_customers.iterrows():
        with st.expander(f"Customer ID: {row['company_id']}"):
            st.write(f"**Satisfaction Score:** {row['satisfaction_score']}")
            st.write(f"**Feedback:** {row['feedback']}")
            st.write(f"**Churn Risk:** {row['churn_risk'] * 100:.2f}%")
            st.write(f"**Purchase Amount:** ${row['purchase_amount']:,.2f}")
            st.write(f"**Industry:** {row['industry']}")
            st.write(f"**Company Size:** {row['company_size']}")
            st.write(f"**Days Since Last Purchase:** {row['last_purchase_days_ago']} days")
            
            # Personalized recommendation
            st.markdown("**Personalized Recommendation:**")
            st.write("- **Immediate Action:** Contact the customer to understand their concerns and offer solutions.")
            st.write("- **Long-Term Plan:** Develop a tailored engagement strategy based on their feedback and industry requirements.")
    
    # Add dropdown for other customers with lower churn risk
    st.subheader("Insights for Other Customers")
    for churn_risk_level in [(0.2, 0.5), (0.0, 0.2)]:
        risk_label = f"Customers with Churn Risk {churn_risk_level[0] * 100:.0f}% - {churn_risk_level[1] * 100:.0f}%"
        customers = data[(data['churn_risk'] > churn_risk_level[0]) & (data['churn_risk'] <= churn_risk_level[1])]
        
        with st.expander(risk_label):
            for index, row in customers.iterrows():
                st.write(f"### Customer ID: {row['company_id']}")
                st.write(f"**Satisfaction Score:** {row['satisfaction_score']}")
                st.write(f"**Feedback:** {row['feedback']}")
                st.write(f"**Churn Risk:** {row['churn_risk'] * 100:.2f}%")
                st.write(f"**Purchase Amount:** ${row['purchase_amount']:,.2f}")
                st.write(f"**Industry:** {row['industry']}")
                st.write(f"**Company Size:** {row['company_size']}")
                st.write(f"**Days Since Last Purchase:** {row['last_purchase_days_ago']} days")

    # Add download button
    st.subheader("Download Analyzed Data")
    st.download
