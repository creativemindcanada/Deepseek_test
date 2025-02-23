import streamlit as st
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
import requests
from bs4 import BeautifulSoup
from transformers import pipeline
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LogisticRegression
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
from datetime import datetime, timedelta
from typing import Dict, List, Optional

# Download NLTK data for sentiment analysis
nltk.download('vader_lexicon')

# Title of the app
st.title("Customer Insights Dashboard")

# Preload the model when the app starts
@st.cache_resource  # Cache the model to avoid reloading on every interaction
def load_model():
    return pipeline("text-generation", model="distilgpt2")  # Use DistilGPT-2

# Load the model
with st.spinner("Loading AI model..."):
    generator = load_model()

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

def create_structured_prompt(extracted_content: str) -> str:
    """Create a more concise prompt that works better with distilgpt2."""
    return f"""
Website Content Analysis:
{extracted_content}

Please analyze this website and provide:

OVERVIEW:
Key features and target audience

CONTENT:
Quality and organization assessment

ENGAGEMENT:
User interaction and accessibility

STRENGTHS:
Main positive aspects

WEAKNESSES:
Areas for improvement

RECOMMENDATIONS:
Suggested improvements
"""

def parse_ai_response(ai_response: str) -> Dict[str, str]:
    """Parse the AI response with more robust section detection."""
    sections = {
        "overview": "",
        "content": "",
        "engagement": "",
        "strengths": "",
        "weaknesses": "",
        "recommendations": ""
    }
    
    current_section = "overview"
    try:
        # Split the response into lines and clean up
        lines = [line.strip() for line in ai_response.split('\n') if line.strip()]
        
        # Process each line
        for line in lines:
            # Check for section headers
            lower_line = line.lower()
            if "overview" in lower_line:
                current_section = "overview"
            elif "content" in lower_line:
                current_section = "content"
            elif "engagement" in lower_line:
                current_section = "engagement"
            elif "strength" in lower_line:
                current_section = "strengths"
            elif "weakness" in lower_line:
                current_section = "weaknesses"
            elif "recommend" in lower_line:
                current_section = "recommendations"
            else:
                # Add content to current section
                if sections[current_section]:
                    sections[current_section] += "\n"
                sections[current_section] += line
    except Exception as e:
        st.error(f"Error parsing AI response: {str(e)}")
        # Provide default content for sections
        for key in sections:
            if not sections[key]:
                sections[key] = "Analysis pending."
    
    return sections

def display_structured_report(sections: Dict[str, str]):
    """Display the report with better error handling and formatting."""
    st.write("# Website Analysis Report")
    
    # Overview
    with st.expander("📋 Overview", expanded=True):
        content = sections.get("overview", "No overview available. The website content may be too limited for a detailed analysis.")
        st.markdown(content if content.strip() else "No overview available.")
    
    # Content Analysis
    with st.expander("📊 Content Analysis"):
        content = sections.get("content", "No content analysis available. The website may not have enough text content for analysis.")
        st.markdown(content if content.strip() else "No content analysis available.")
    
    # Engagement
    with st.expander("🤝 Engagement Assessment"):
        content = sections.get("engagement", "No engagement analysis available. The website may lack interactive elements or user engagement features.")
        st.markdown(content if content.strip() else "No engagement analysis available.")
    
    # Strengths & Weaknesses
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("💪 Strengths")
        content = sections.get("strengths", "No specific strengths were identified. The website appears to have a clean design and clear navigation.")
        st.markdown(content if content.strip() else "No strengths listed.")
    
    with col2:
        st.subheader("🎯 Areas for Improvement")
        content = sections.get("weaknesses", "No specific weaknesses were identified. Consider adding more detailed content or interactive features to enhance user engagement.")
        st.markdown(content if content.strip() else "No weaknesses listed.")
    
    # Recommendations
    with st.expander("💡 Recommendations", expanded=True):
        content = sections.get("recommendations", "No specific recommendations available. Consider adding more content, improving SEO, or enhancing user engagement features.")
        st.markdown(content if content.strip() else "No recommendations available.")

def generate_ai_report(extracted_content: str) -> Optional[str]:
    """Generate AI report with better error handling and model parameters."""
    try:
        # Create prompt
        prompt = create_structured_prompt(extracted_content)
        
        # Generate text with more conservative parameters
        generated_text = generator(
            prompt,
            max_new_tokens=500,  # Reduced for more stability
            num_return_sequences=1,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            no_repeat_ngram_size=2,
            num_beams=1,
            early_stopping=True
        )[0]["generated_text"]
        
        # Remove the prompt from the generated text
        response_text = generated_text.replace(prompt, "").strip()
        
        # Parse and structure the response
        sections = parse_ai_response(response_text)
        
        # Display the structured report
        display_structured_report(sections)
        
        # Return the full text for download
        return generated_text
        
    except Exception as e:
        st.error(f"An error occurred while generating the AI report: {str(e)}")
        st.info("Try refreshing the page and running the analysis again.")
        return None

# Main dashboard layout
st.sidebar.title("Navigation")
analysis_type = st.sidebar.radio("Choose Analysis Type", ["Customer Data Analysis", "Website Analysis"])

if analysis_type == "Customer Data Analysis":
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
        data = predict_churn(data)

        # Display raw data
        st.subheader("Raw Data")
        st.write(data)

        # Basic Analysis
        st.subheader("Basic Analysis")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Customers", f"{len(data):,}")
        with col2:
            st.metric("Average Satisfaction", f"{data['satisfaction_score'].mean():.2f}")
        with col3:
            st.metric("Total Revenue", f"${data['total_spent'].sum():,.2f}")

        # Visualizations
        st.subheader("Customer Satisfaction Distribution")
        fig = px.histogram(data, x="satisfaction_score", nbins=10)
        st.plotly_chart(fig)

        # Risk Analysis
        st.subheader("Customer Risk Analysis")
        
        # High Risk
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
        
        # Medium Risk
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
        
        # Low Risk
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

        # Sentiment Analysis
        st.subheader("Sentiment Analysis")
        def analyze_sentiment(text):
            sia = SentimentIntensityAnalyzer()
            return sia.polarity_scores(text)['compound']

        data['sentiment'] = data['feedback'].apply(analyze_sentiment)
        st.write(data[['customer_id', 'feedback', 'sentiment']])

        # Add filters
        st.subheader("Filter Data")
        col1, col2 = st.columns(2)
        with col1:
            min_score = st.slider("Minimum Satisfaction Score", 1, 5, 3)
        with col2:
            region_filter = st.selectbox("Select Region", ["All", "North", "South", "East", "West"])

        # Apply filters
        filtered_data = data[data['satisfaction_score'] >= min_score]
        if region_filter != "All":
            filtered_data = filtered_data[filtered_data['region'] == region_filter]

        st.write(filtered_data)

        # Regional Analysis
        st.subheader("Customer Satisfaction by Region")
        fig = px.bar(filtered_data, x="region", y="satisfaction_score", color="region", barmode="group")
        st.plotly_chart(fig)

        # Download button
        st.subheader("Download Analyzed Data")
        st.download_button(
            label="Download CSV",
            data=filtered_data.to_csv().encode('utf-8'),
            file_name="analyzed_data.csv",
            mime="text/csv"
        )
    else:
        st.info("Please upload a CSV file or click the button to use randomly generated data.")

elif analysis_type == "Website Analysis":
    st.subheader("Website Analysis")
    website_url = st.text_input("Enter Website URL for Analysis")

    if st.button("Generate AI-Powered Report"):
        if website_url:
            with st.spinner("Analyzing website content..."):
                extracted_content = scrape_website_content_selenium(website_url)
                if extracted_content:
                    with st.spinner("Generating structured AI report..."):
                        report = generate_ai_report(extracted_content)
                        if report:
                            st.success("Analysis complete! Expand the sections above to view detailed insights.")
                            st.download_button(
                                label="Download Full Report",
                                data=report,
                                file_name="website_analysis_report.txt",
                                mime="text/plain"
                            )
                        else:
                            st.error("Failed to generate the AI report.")
                else:
                    st.error("Failed to scrape website content.")
        else:
            st.warning("Please enter a valid website URL.")
