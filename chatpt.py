import re
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
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional

# Download NLTK data for sentiment analysis
nltk.download('vader_lexicon')

# Title of the app
st.title("Customer Insights Dashboard")

# Dark mode toggle
dark_mode = st.sidebar.checkbox("🌙 Dark Mode")
if dark_mode:
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #1E1E1E;
            color: #FFFFFF;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

# Preload the model when the app starts using a free transformer model
@st.cache_resource
def load_model():
    return pipeline("text-generation", model="distilgpt2")  # Free model from Hugging Face

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

# Ensure required columns exist
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

# Create a structured prompt for AI analysis
def create_structured_prompt(extracted_content: str) -> str:
    return f"""
Analyze this business website to identify:
1) CORE BUSINESS MODEL:
   - What specific problems does this company solve?
   - What solutions/products do they offer?
   - What pricing model is implied?

2) GO-TO-MARKET STRATEGY:
   - Which channels are they using (social media, blogs)?
   - What customer evidence are they showcasing?
   - How do they position themselves?

3) CLIENT ANALYSIS:
   - List key clients mentioned
   - For 3 main clients: Problem → Solution → Outcome
   - Any missing client industries/types?

4) FUTURE RISKS:
   - What current limitations appear in testimonials?
   - What market needs are they not addressing?
   - What technical/support gaps exist?

WEBSITE CONTENT:
{extracted_content}
"""

def parse_ai_response(ai_response: str) -> Dict[str, str]:
    sections = {
        "business_model": "",
        "gtm_strategy": "",
        "client_analysis": "",
        "future_risks": ""
    }
    current_section = None
    for line in ai_response.split('\n'):
        line_lower = line.lower()
        if "core business model" in line_lower:
            current_section = "business_model"
        elif "go-to-market" in line_lower:
            current_section = "gtm_strategy"
        elif "client analysis" in line_lower:
            current_section = "client_analysis"
        elif "future risks" in line_lower:
            current_section = "future_risks"
        elif current_section and line.strip():
            sections[current_section] += line + "\n"
    return sections

def display_structured_report(sections: Dict[str, str]):
    st.write("# Website Analysis Report")
    with st.expander("📋 Overview", expanded=True):
        content = sections.get("business_model", "Analysis pending.")
        st.markdown(content if content.strip() else "No overview available.")
    with st.expander("🚀 GTM Strategy"):
        content = sections.get("gtm_strategy", "Analysis pending.")
        st.markdown(content if content.strip() else "No GTM analysis available.")
    with st.expander("📈 Client Insights"):
        content = sections.get("client_analysis", "Analysis pending.")
        st.markdown(content if content.strip() else "No client analysis available.")
    with st.expander("⚠️ Risk Assessment"):
        content = sections.get("future_risks", "Analysis pending.")
        st.markdown(content if content.strip() else "No risk assessment available.")

# Generate AI report using the free transformer model
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

# Clear session data
def clear_data():
    st.session_state.pop('data', None)
    st.toast("Data cleared successfully!", icon="✅")

# Simplified website scraping function using requests and BeautifulSoup (no Selenium)
def scrape_website_content(website_url: str) -> Optional[str]:
    """
    Enhanced scraping that extracts targeted business content and falls back to full text
    if the extracted data is too sparse.
    """
    try:
        if not website_url.startswith(('http://', 'https://')):
            website_url = 'https://' + website_url
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        response = requests.get(website_url, headers=headers, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        
        # Remove unwanted elements
        for element in soup(['script', 'style', 'meta', 'link', 'noscript']):
            element.decompose()

        content = {
            "testimonials": [],
            "use_cases": [],
            "blog_posts": [],
            "social_links": {},
            "client_list": [],
            "main_content": []
        }

        # Extract testimonials
        for section in soup.find_all(['div', 'section'], class_=re.compile(r'testimonial|review|case-study', re.I)):
            content["testimonials"].extend([
                p.get_text(separator=" ", strip=True)
                for p in section.find_all('p') if p.get_text(strip=True)
            ])

        # Extract use cases
        use_case_section = soup.find(['div', 'section'], string=re.compile(r'case studies|use cases', re.I))
        if use_case_section:
            content["use_cases"] = [
                case.get_text(separator=" ", strip=True)
                for case in use_case_section.find_all(['div', 'li'])
            ]

        # Extract blog posts
        blog_section = soup.find(['div', 'section'], string=re.compile(r'blog|articles', re.I))
        if blog_section:
            content["blog_posts"] = [
                post.get_text(separator=" ", strip=True)
                for post in blog_section.find_all(['article', 'div'])
            ]

        # Extract social media links
        social_links = soup.find_all('a', href=re.compile(
            r'linkedin\.com|twitter\.com|facebook\.com|instagram\.com', re.I))
        for link in social_links:
            platform = re.search(r'(linkedin|twitter|facebook|instagram)', link['href'], re.I)
            if platform:
                content["social_links"][platform.group(1).lower()] = link['href']

        # Extract client list
        client_section = soup.find(['div', 'section'], string=re.compile(r'clients|partners', re.I))
        if client_section:
            content["client_list"] = [
                img['alt']
                for img in client_section.find_all('img')
                if 'client' in img.get('alt', '').lower()
            ]

        # Extract main content: first try extracting from <p> tags
        main_text = ' '.join([p.get_text(separator=" ", strip=True) 
                              for p in soup.find_all('p') if p.get_text(strip=True)])
        content["main_content"] = main_text

        # Format the extracted content
        formatted_content = f"""
BUSINESS ANALYSIS DATA:
1. Customer Evidence:
   - Testimonials: {content['testimonials'][:5] if content['testimonials'] else 'N/A'}
   - Use Cases: {content['use_cases'][:5] if content['use_cases'] else 'N/A'}
   - Clients: {content['client_list'][:10] if content['client_list'] else 'N/A'}

2. Digital Presence:
   - Social Media: {content['social_links'] if content['social_links'] else 'N/A'}
   - Blog Posts: {content['blog_posts'][:3] if content['blog_posts'] else 'N/A'}

3. Main Content:
   {content["main_content"][:1000] if content["main_content"] else 'N/A'} 
   <!-- Limiting to first 1000 characters for brevity -->
"""

        # Fallback: If all targeted content is 'N/A' or too short, use a fallback extraction of all visible text.
        extracted_length = len(formatted_content.strip())
        if extracted_length < 200:  # threshold can be adjusted
            fallback_text = soup.get_text(separator=" ", strip=True)
            formatted_content = f"FULL PAGE TEXT:\n{fallback_text[:2000]}"  # limiting to first 2000 characters

        return formatted_content

    except Exception as e:
        st.error(f"Scraping error: {str(e)}")
        return None


# Main dashboard layout
st.sidebar.title("Navigation")
analysis_type = st.sidebar.radio("Choose Analysis Type", ["Customer Data Analysis", "Website Analysis"])

if st.sidebar.button("🧹 Clear Data"):
    clear_data()

with st.sidebar.expander("ℹ️ Help"):
    st.write("""
    - **Customer Data Analysis**: Upload a CSV file or use randomly generated data to analyze customer insights.
    - **Website Analysis**: Enter a website URL to generate an AI-powered analysis report.
    - **Dark Mode**: Toggle dark mode for better visibility in low-light environments.
    - **Clear Data**: Reset the app to its initial state.
    """)

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
        data = predict_churn(data)
        st.subheader("Raw Data")
        st.write(data)
        st.subheader("Basic Analysis")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Customers", f"{len(data):,}")
        with col2:
            st.metric("Average Satisfaction", f"{data['satisfaction_score'].mean():.2f}")
        with col3:
            st.metric("Total Revenue", f"${data['total_spent'].sum():,.2f}")
        st.subheader("Customer Satisfaction Distribution")
        fig = px.histogram(data, x="satisfaction_score", nbins=10)
        st.plotly_chart(fig)
        st.subheader("Customer Risk Analysis")
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
        st.subheader("Sentiment Analysis")
        def analyze_sentiment(text):
            sia = SentimentIntensityAnalyzer()
            return sia.polarity_scores(text)['compound']
        data['sentiment'] = data['feedback'].apply(analyze_sentiment)
        st.write(data[['customer_id', 'feedback', 'sentiment']])
        st.subheader("Filter Data")
        col1, col2 = st.columns(2)
        with col1:
            min_score = st.slider("Minimum Satisfaction Score", 1, 5, 3)
        with col2:
            region_filter = st.selectbox("Select Region", ["All", "North", "South", "East", "West"])
        filtered_data = data[data['satisfaction_score'] >= min_score]
        if region_filter != "All":
            filtered_data = filtered_data[filtered_data['region'] == region_filter]
        st.write(filtered_data)
        st.subheader("Customer Satisfaction by Region")
        fig = px.bar(filtered_data, x="region", y="satisfaction_score", color="region", barmode="group")
        st.plotly_chart(fig)
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
    st.subheader("Strategic Business Analysis")
    website_url = st.text_input("Enter Company Website URL")
    if st.button("Generate Business Report"):
        if website_url:
            with st.spinner("Analyzing business model..."):
                extracted_content = scrape_website_content(website_url)
                if extracted_content:
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
