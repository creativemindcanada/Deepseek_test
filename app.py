import streamlit as st
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
@st.cache_resource  
def load_model():
    return pipeline("text-generation", model="distilgpt2")

# Load the model
with st.spinner("Loading AI model..."):
    generator = load_model()

# Function to generate random customer data
def generate_random_data():
    np.random.seed(42)
    num_customers = 100  
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
    if 'satisfaction_score' not in data.columns:
        data['satisfaction_score'] = np.random.randint(1, 6, len(data))
    if 'purchase_amount' not in data.columns:
        data['purchase_amount'] = np.random.uniform(10, 500, len(data)).round(2)
    if 'total_spent' not in data.columns:
        data['total_spent'] = np.random.uniform(100, 5000, len(data)).round(2)
    return data

# Function to predict churn risk
def predict_churn(data):
    data = ensure_columns(data)
    
    required_columns = ['satisfaction_score', 'purchase_amount', 'total_spent']
    if all(column in data.columns for column in required_columns):
        X = data[required_columns]
        y = (data['satisfaction_score'] < 3).astype(int)
        model = LogisticRegression()
        model.fit(X, y)
        data['churn_risk'] = model.predict_proba(X)[:, 1]
    else:
        st.warning("Required columns for churn prediction are missing. Using default churn risk of 0.")
        data['churn_risk'] = 0  
    return data

def create_structured_prompt(extracted_content: str) -> str:
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
        lines = [line.strip() for line in ai_response.split('\n') if line.strip()]
        
        for line in lines:
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
                if sections[current_section]:
                    sections[current_section] += "\n"
                sections[current_section] += line
    except Exception as e:
        st.error(f"Error parsing AI response: {str(e)}")
        
        for key in sections:
            if not sections[key]:
                sections[key] = "Analysis pending."
    
    return sections

def display_structured_report(sections: Dict[str, str]):
    st.write("# Website Analysis Report")
    
    with st.expander("📋 Overview", expanded=True):
        content = sections.get("overview", "Analysis pending.")
        st.markdown(content if content.strip() else "No overview available.")
    
    with st.expander("📊 Content Analysis"):
        content = sections.get("content", "Analysis pending.")
        st.markdown(content if content.strip() else "No content analysis available.")
    
    with st.expander("🤝 Engagement Assessment"):
        content = sections.get("engagement", "Analysis pending.")
        st.markdown(content if content.strip() else "No engagement analysis available.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("💪 Strengths")
        content = sections.get("strengths", "Analysis pending.")
        st.markdown(content if content.strip() else "No strengths listed.")
    
    with col2:
        st.subheader("🎯 Areas for Improvement")
        content = sections.get("weaknesses", "Analysis pending.")
        st.markdown(content if content.strip() else "No weaknesses listed.")
    
    with st.expander("💡 Recommendations", expanded=True):
        content = sections.get("recommendations", "Analysis pending.")
        st.markdown(content if content.strip() else "No recommendations available.")

def generate_ai_report(extracted_content: str) -> Optional[str]:
    try:
        prompt = create_structured_prompt(extracted_content)
        
        generated_text = generator(
            prompt,
            max_new_tokens=500,
            num_return_sequences=1,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            no_repeat_ngram_size=2,
            num_beams=1,
            early_stopping=True
        )[0]["generated_text"]
        
        response_text = generated_text.replace(prompt, "").strip()
        
        sections = parse_ai_response(response_text)
        
        display_structured_report(sections)
        
        return generated_text
        
    except Exception as e:
        st.error(f"An error occurred while generating the AI report: {str(e)}")
        st.info("Try refreshing the page and running the analysis again.")
        return None

# Main dashboard layout
st.sidebar.title("Navigation")
analysis_type = st.sidebar.radio("Choose Analysis Type", ["Customer Data Analysis", "Website Analysis"])

if analysis_type == "Customer Data Analysis":
    uploaded_file = st.file_uploader("Upload your customer data (CSV file)", type=["csv"])

    if st.button("Use Randomly Generated Customer Data"):
        data = generate_random_data()
        st.session_state['data'] = data
    elif uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        data = ensure_columns(data)
        st.session_state['data'] = data
    else:
        data = None

    if 'data' in st.session_state and st.session_state['data'] is not None:
        data = st.session_state['data']
        
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
       # Additional visualizations and analyses can go here...

else: 
   # Handle website analysis or other types of analysis...
   pass 
