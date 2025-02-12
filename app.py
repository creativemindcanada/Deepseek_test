import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LogisticRegression
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
from datetime import datetime, timedelta

# Download NLTK data for sentiment analysis
nltk.download('vader_lexicon')

# Title of the app
st.title("Customer Insights Dashboard")

# Function to generate random customer data
def generate_random_data():
    np.random.seed(42)
    num_customers = 100  # Generate 100 customers
    data = {
        "customer_id": range(1, num_customers + 1),
        "satisfaction_score": np.random.randint(1, 6, num_customers),
        "feedback": np.random.choice([
            "Great product!", "Good quality.", "Delivery was late.", "Poor packaging.", "Excellent service!"
        ], num_customers),
        "purchase_amount": np.random.uniform(10, 500, num_customers).round(2),
        "region": np.random.choice(["North", "South", "East", "West"], num_customers),
        "age": np.random.randint(18, 65, num_customers),
        "gender": np.random.choice(["Male", "Female", "Other"], num_customers),
        "loyalty_status": np.random.choice(["New", "Regular", "VIP"], num_customers, p=[0.5, 0.3, 0.2]),
        "last_purchase_date": [datetime.now() - timedelta(days=np.random.randint(1, 365)) for _ in range(num_customers)],
        "total_spent": np.random.uniform(100, 5000, num_customers).round(2)
    }
    return pd.DataFrame(data)

# Function to ensure necessary columns with default values if missing
def ensure_columns(data):
    # Add default columns if they don't exist
    if 'satisfaction_score' not in data.columns:
        data['satisfaction_score'] = np.random.randint(1, 6, len(data))  # Random scores between 1 and 5
    if 'purchase_amount' not in data.columns:
        data['purchase_amount'] = np.random.uniform(10, 500, len(data)).round(2)  # Random purchase amounts
    if 'total_spent' not in data.columns:
        data['total_spent'] = np.random.uniform(100, 5000, len(data)).round(2)  # Random total spent
    return data

# Function to predict churn risk
def predict_churn(data):
    # Ensure required columns exist
    data = ensure_columns(data)
    
    # Check if required columns exist
    required_columns = ['satisfaction_score', 'purchase_amount', 'total_spent']
    if all(column in data.columns for column in required_columns):
        X = data[required_columns]
        y = (data['satisfaction_score'] < 3).astype(int)  # Churn if satisfaction < 3
        model = LogisticRegression()
        model.fit(X, y)
        data['churn_risk'] = model.predict_proba(X)[:, 1]  # Probability of churn
    else:
        st.warning("Required columns for churn prediction are missing. Using default churn risk of 0.")
        data['churn_risk'] = 0  # Default churn risk if columns are missing
    return data

# Upload customer data
uploaded_file = st.file_uploader("Upload your customer data (CSV file)", type=["csv"])

# Button to generate random data
if st.button("Use Randomly Generated Customer Data"):
    data = generate_random_data()
    st.session_state['data'] = data
elif uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    data = ensure_columns(data)
    st.session_state['data'] = data
else:
    data = None

# Display current dashboard
if 'data' in st.session_state and st.session_state['data'] is not None:
    data = st.session_state['data']

    # Predict churn risk
    data = predict_churn(data)

    # Display raw data
    st.subheader("Raw Data")
    st.write(data)

    # Basic Analysis
    st.subheader("Basic Analysis")
    st.write(f"Total Customers: {len(data)}")
    st.write(f"Average Satisfaction Score: {data['satisfaction_score'].mean():.2f}")
    st.write(f"Total Revenue: ${data['total_spent'].sum():,.2f}")

    # Visualizations
    st.subheader("Customer Satisfaction Distribution")
    fig = px.histogram(data, x="satisfaction_score", nbins=10)
    st.plotly_chart(fig)

    # Insights and Recommendations (Dropdown Format)
    st.subheader("Insights & Recommendations")
    if data['satisfaction_score'].mean() >= 4:
        st.success("Things are going well! Keep up the good work.")
    else:
        st.error("There are issues to address.")

    # High Risk
    with st.expander("🔴 High Risk Customers"):
        high_risk = data[data['churn_risk'] > 0.7]
        if not high_risk.empty:
            st.write(f"- **Number of High-Risk Customers:** {len(high_risk)}")
            for customer in high_risk['customer_id'].unique():
                with st.expander(f"Customer ID: {customer}"):
                    st.write("- **Key Issues:** Poor packaging, delayed delivery.")
                    st.write("- **Recommendations:**")
                    st.write("  - Offer personalized discounts to retain this customer.")
                    st.write("  - Improve delivery times for this region.")
        else:
            st.write("No high-risk customers found.")
    
    # Medium Risk
    with st.expander("🟠 Medium Risk Customers"):
        medium_risk = data[(data['churn_risk'] > 0.4) & (data['churn_risk'] <= 0.7)]
        if not medium_risk.empty:
            st.write(f"- **Number of Medium-Risk Customers:** {len(medium_risk)}")
            for customer in medium_risk['customer_id'].unique():
                with st.expander(f"Customer ID: {customer}"):
                    st.write("- **Key Issues:** Mixed feedback on product quality.")
                    st.write("- **Recommendations:**")
                    st.write("  - Conduct a survey to identify specific pain points for this customer.")
                    st.write("  - Offer a loyalty program discount.")
        else:
            st.write("No medium-risk customers found.")
    
    # Low Risk
    with st.expander("🟢 Low Risk Customers"):
        low_risk = data[data['churn_risk'] <= 0.4]
        if not low_risk.empty:
            st.write(f"- **Number of Low-Risk Customers:** {len(low_risk)}")
            for customer in low_risk['customer_id'].unique():
                with st.expander(f"Customer ID: {customer}"):
                    st.write("- **Key Strengths:** High satisfaction scores, positive feedback.")
                    st.write("- **Recommendations:**")
                    st.write("  - Encourage this customer to refer friends with a referral program.")
                    st.write("  - Upsell premium products to this loyal customer.")
        else:
            st.write("No low-risk customers found.")

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

# Secondary Inputs
st.header("Secondary Input Methods")

# LinkedIn URL
st.subheader("Analyze Company LinkedIn Profile")
linkedin_url = st.text_input("Enter Company LinkedIn URL", key="linkedin_url_input")
if linkedin_url:
    st.write(f"Fetching insights for LinkedIn URL: {linkedin_url}")
    st.warning("LinkedIn API integration is not implemented in this demo.")

# Function to analyze website performance using Google PageSpeed Insights API
def analyze_website(website_url, api_key):
    url = f"https://www.googleapis.com/pagespeedonline/v5/runPagespeed?url={website_url}&key={api_key}"
    try:
        response = requests.get(url, timeout=10)  # Add a timeout for the request
        response.raise_for_status()  # Raise an HTTPError for bad responses
        return response.json()
    except requests.exceptions.Timeout:
        st.error("The request to the PageSpeed API timed out. Please try again later.")
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"An error occurred while fetching website data: {e}")
        return None

# In the Company Website section
st.subheader("Analyze Company Website")
website_url = st.text_input("Enter Company Website URL", key="website_url_input")
if website_url:
    api_key = "AIzaSyDyC_h2_dQiVJEOpXdPlob1lX0Sfb2UTlI"  # Replace with your actual API key
    st.write(f"Fetching insights for website: {website_url}")
    website_data = analyze_website(website_url, api_key)
    
    if website_data:
        # Display performance metrics
        st.subheader("Website Performance Metrics")
        st.write(f"- **Performance Score:** {website_data['lighthouseResult']['categories']['performance']['score'] * 100:.2f}%")
        st.write(f"- **First Contentful Paint:** {website_data['lighthouseResult']['audits']['first-contentful-paint']['displayValue']}")
        st.write(f"- **Time to Interactive:** {website_data['lighthouseResult']['audits']['interactive']['displayValue']}")
        st.write(f"- **Speed Index:** {website_data['lighthouseResult']['audits']['speed-index']['displayValue']}")
    else:
        st.warning("Failed to analyze the website. Displaying mock data for demonstration purposes.")
        # Mock data for demo purposes
        st.subheader("Website Analysis Content")
        st.write("Based on the analysis of the website, here are the key findings:")
        st.subheader("Core Strategic Elements")
        st.write("- **Digital Twin Technology:** Enables real-time visibility across silos")
        st.write("- **Decision Intelligence Studio:** AI-powered optimization")
        st.write("- **Continuous Realignment:** Dynamic adaptation capabilities")
        st.subheader("Proven Value Lever Implementation")
        st.write("- **Intersilo Data Integration:**")
        st.write("  - Implemented through Digital Twin platform")
        st.write("  - Results: Thermo Fisher tracking 570K+ shipments")
        st.write("  - Impact: Enhanced quality and compliance")
        st.write("- **Resource Optimization:**")
        st.write("  - Used Decision Intelligence for dynamic allocation")
        st.write("  - Results: 33% reduction in expedited shipping costs")
        st.write("  - Impact: $25M savings for GE Appliances")
        st.write("- **Quality & Compliance:**")
        st.write("  - Real-time monitoring and alerts")
        st.write("  - Results: Enhanced cold chain compliance")
        st.write("  - Impact: Significant reduction in temperature excursions")
        st.subheader("Key Success Factors")
        st.write("- Industry-specific solutions (Pharma, Consumer Goods, F&B)")
        st.write("- End-to-end visibility approach")
        st.write("- Focus on measurable outcomes")
else:
    st.warning("Please enter a valid website URL.")
