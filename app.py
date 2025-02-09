import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LogisticRegression
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
from datetime import datetime, timedelta
import io  # Import the io module
import time  # Import time module for sleep

# ---- CACHING FUNCTIONS ----
@st.cache_data
def load_data(uploaded_file, encoding='utf-8'):
    """Loads data from a CSV file with specified encoding."""
    try:
        df = pd.read_csv(uploaded_file, encoding=encoding)
        return df
    except UnicodeDecodeError as e:
        st.error(f"Error loading data with encoding '{encoding}': {e}.  Trying 'latin1'...")
        try:
            df = pd.read_csv(uploaded_file, encoding='latin1')
            return df
        except Exception as e2:
            st.error(f"Error loading data with encoding 'latin1': {e2}.  Please try a different encoding or ensure the file is properly encoded.")
            return None
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None


@st.cache_data
def generate_random_data(num_customers=100):
    """Generates random customer data."""
    np.random.seed(42)
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
        "last_purchase_date": [datetime.now() - timedelta(days=np.random.randint(1, 365)) for _ in
                                 range(num_customers)],
        "total_spent": np.random.uniform(100, 5000, num_customers).round(2)
    }
    return pd.DataFrame(data)


@st.cache_resource
def load_sentiment_analyzer():
    """Loads the VADER sentiment analyzer."""
    return SentimentIntensityAnalyzer()


# ---- PREPROCESSING FUNCTIONS ----
def ensure_columns(data):
    """Ensures that 'total_spent' column exists and has no missing values."""
    if 'total_spent' not in data.columns:
        data['total_spent'] = 0
    data['total_spent'] = data['total_spent'].fillna(0)  # Fill NaN values with 0
    return data


def preprocess_data(data):
    """Handles data type conversions and missing values."""
    # Convert 'last_purchase_date' to datetime if it exists
    if 'last_purchase_date' in data.columns:
        try:
            data['last_purchase_date'] = pd.to_datetime(data['last_purchase_date'], errors='coerce')
        except Exception as e:
            st.error(f"Error converting 'last_purchase_date' to datetime: {e}")

    # Handle missing values (more robust approach)
    for col in data.columns:
        if data[col].isnull().any():
            if pd.api.types.is_numeric_dtype(data[col]):
                data[col] = data[col].fillna(data[col].mean())  # Fill numeric with mean
            else:
                data[col] = data[col].fillna(
                    data[col].mode()[0])  # Fill others with mode (most frequent value)
    return data


# ---- CHURN PREDICTION ----
def predict_churn(data):
    """Predicts customer churn risk using Logistic Regression."""
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


# ---- SENTIMENT ANALYSIS ----
def analyze_sentiment(text, sia):
    """Analyzes the sentiment of a given text."""
    try:
        return sia.polarity_scores(text)['compound']
    except TypeError:
        return 0  # Handle potential errors with non-string inputs


# ---- GOOGLE PAGESPEED INSIGHTS API ----
@st.cache_data(ttl=3600)  # Cache the API response for 1 hour
def analyze_website(website_url, api_key, retry_count=3):
    """Analyzes website performance using Google PageSpeed Insights API with retry logic."""
    url = f"https://www.googleapis.com/pagespeedonline/v5/runPagespeed?url={website_url}&key={api_key}"
    for attempt in range(retry_count):
        try:
            response = requests.get(url, timeout=20)  # Increased timeout to 20 seconds
            response.raise_for_status()  # Raise an HTTPError for bad responses
            return response.json()
        except requests.exceptions.Timeout:
            st.warning(f"Timeout occurred (attempt {attempt + 1}/{retry_count}). Retrying in 5 seconds...")
            time.sleep(5)  # Wait before retrying
        except requests.exceptions.RequestException as e:
            st.error(f"An error occurred (attempt {attempt + 1}/{retry_count}) while fetching website data: {e}")
            break  # Don't retry for other request exceptions
        except Exception as e:
            st.error(f"An unexpected error occurred (attempt {attempt + 1}/{retry_count}): {e}")
            break  # Don't retry for unexpected errors

    st.error("Failed to analyze the website after multiple retries.")
    return None


# ---- MAIN DASHBOARD ----
def main():
    st.title("Customer Insights Dashboard")

    # ---- DATA INPUT ----
    uploaded_file = st.file_uploader("Upload your customer data (CSV file)", type=["csv"])
    if st.button("Use Randomly Generated Customer Data"):
        data = generate_random_data()
        data = ensure_columns(data)
        data = preprocess_data(data)  # Preprocess the generated data
        st.session_state['data'] = data
    elif uploaded_file is not None:
        data = load_data(uploaded_file)
        if data is not None:
            data = ensure_columns(data)
            data = preprocess_data(data)  # Preprocess the uploaded data
            st.session_state['data'] = data
    else:
        data = None

    # ---- DASHBOARD DISPLAY ----
    if 'data' in st.session_state and st.session_state['data'] is not None:
        data = st.session_state['data']

        # Predict churn risk
        data = predict_churn(data)

        # ---- BASIC ANALYSIS ----
        st.subheader("Basic Analysis")
        st.write(f"Total Customers: {len(data)}")
        st.write(f"Average Satisfaction Score: {data['satisfaction_score'].mean():.2f}")
        st.write(f"Total Revenue: ${data['total_spent'].sum():,.2f}")

        # ---- VISUALIZATIONS ----
        st.subheader("Customer Satisfaction Distribution")
        fig = px.histogram(data, x="satisfaction_score", nbins=10)
        st.plotly_chart(fig)

        # ---- INSIGHTS AND RECOMMENDATIONS ----
        st.subheader("Insights & Recommendations")
        if data['satisfaction_score'].mean() >= 4:
            st.success("Things are going well! Keep up the good work.")
        else:
            st.error("There are issues to address.")

        # ---- RISK SEGMENTATION ----
        with st.expander("🔴 High Risk Customers"):
            high_risk = data[data['churn_risk'] > 0.7]
            if not high_risk.empty:
                st.write(f"- **Number of High-Risk Customers:** {len(high_risk)}")
                st.write("- **Key Issues:** Poor packaging, delayed delivery.")
                st.write("- **Recommendations:**")
                st.write("  - Offer personalized discounts to retain high-risk customers.")
                st.write("  - Improve delivery times in regions with high churn risk.")
            else:
                st.write("No high-risk customers found.")

        with st.expander("🟠 Medium Risk Customers"):
            medium_risk = data[(data['churn_risk'] > 0.4) & (data['churn_risk'] <= 0.7)]
            if not medium_risk.empty:
                st.write(f"- **Number of Medium-Risk Customers:** {len(medium_risk)}")
                st.write("- **Key Issues:** Mixed feedback on product quality.")
                st.write("- **Recommendations:**")
                st.write("  - Conduct customer surveys to identify specific pain points.")
                st.write("  - Launch a customer loyalty program.")
            else:
                st.write("No medium-risk customers found.")

        with st.expander("🟢 Low Risk Customers"):
            low_risk = data[data['churn_risk'] <= 0.4]
            if not low_risk.empty:
                st.write(f"- **Number of Low-Risk Customers:** {len(low_risk)}")
                st.write("- **Key Strengths:** High satisfaction scores, positive feedback.")
                st.write("- **Recommendations:**")
                st.write("  - Encourage low-risk customers to refer friends with a referral program.")
                st.write("  - Upsell premium products to loyal customers.")
            else:
                st.write("No low-risk customers found.")

        # ---- SENTIMENT ANALYSIS ----
        st.subheader("Sentiment Analysis")
        sia = load_sentiment_analyzer()  # Load sentiment analyzer using caching
        data['sentiment'] = data['feedback'].apply(lambda text: analyze_sentiment(text, sia))
        st.write(data[['customer_id', 'feedback', 'sentiment']])

        # ---- DATA FILTERING ----
        st.subheader("Filter Data")
        min_score = st.slider("Minimum Satisfaction Score", 1, 5, 3)
        region_filter = st.selectbox("Select Region", ["All", "North", "South", "East", "West"])

        # Apply filters
        filtered_data = data[data['satisfaction_score'] >= min_score]
        if region_filter != "All":
            filtered_data = filtered_data[filtered_data['region'] == region_filter]

        st.write("Filtered Data")
        st.write(filtered_data)

        # ---- REGIONAL SATISFACTION VISUALIZATION ----
        st.subheader("Customer Satisfaction by Region")
        fig = px.bar(filtered_data, x="region", y="satisfaction_score", color="region", barmode="group")
        st.plotly_chart(fig)

        # ---- DATA DOWNLOAD ----
        st.subheader("Download Analyzed Data")

        # Use io.StringIO to create an in-memory text buffer
        csv_buffer = io.StringIO()
        filtered_data.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)

        st.download_button(
            label="Download CSV",
            data=csv_buffer.getvalue().encode("utf-8"),  # Explicitly encode as UTF-8
            file_name="analyzed_data.csv",
            mime="text/csv"
        )

        # ---- DISPLAY RAW DATA (Optional) ----
        st.subheader("Raw Data")
        st.write(data)

    else:
        st.info("Please upload a CSV file or click the button to use randomly generated data.")

    # ---- SECONDARY INPUTS ----
    st.header("Secondary Input Methods")

    # ---- LINKEDIN ANALYSIS (Placeholder) ----
    st.subheader("Analyze Company LinkedIn Profile")
    linkedin_url = st.text_input("Enter Company LinkedIn URL", key="linkedin_url_input")
    if linkedin_url:
        st.write(f"Fetching insights for LinkedIn URL: {linkedin_url}")
        st.warning("LinkedIn API integration is not implemented in this demo.")

    # ---- COMPANY WEBSITE ANALYSIS ----
    st.subheader("Analyze Company Website")
    website_url = st.text_input("Enter Company Website URL", key="website_url_input")
    if website_url:
        api_key = st.secrets.get("google_page_speed_api")  # Get API key from Streamlit secrets

        if not api_key:
            st.error("Please configure the Google PageSpeed API key in Streamlit secrets.")
        else:
            st.write(f"Fetching insights for website: {website_url}")
            website_data = analyze_website(website_url, api_key)

            if website_data:
                # Display performance metrics
                st.subheader("Website Performance Metrics")
                try:
                    st.write(
                        f"- **Performance Score:** {website_data['lighthouseResult']['categories']['performance']['score'] * 100:.2f}%")
                    st.write(
                        f"- **First Contentful Paint:** {website_data['lighthouseResult']['audits']['first-contentful-paint']['displayValue']}")
                    st.write(
                        f"- **Time to Interactive:** {website_data['lighthouseResult']['audits']['interactive']['displayValue']}")
                    st.write(
                        f"- **Speed Index:** {website_data['lighthouseResult']['audits']['speed-index']['displayValue']}")
                except KeyError as e:
                    st.error(f"KeyError: {e}.  The API response structure may have changed.")
                    st.write("Raw API Response (for debugging):")
                    st.write(website_data)  # Display the raw response
            else:
                st.warning("Failed to analyze the website. Displaying mock data for demonstration purposes.")
                # Mock data for demo purposes
                st.subheader("Website Performance Metrics (Mock Data)")
                st.write("- **Performance Score:** 85.00%")
                st.write("- **First Contentful Paint:** 1.5s")
                st.write("- **Time to Interactive:** 3.2s")
                st.write("- **Speed Index:** 2.8s")
    else:
        st.warning("Please enter a valid website URL.")


if __name__ == "__main__":
    main()
