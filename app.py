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

# Function to create structured prompt for AI analysis
def create_structured_prompt(extracted_content: str) -> str:
    return f"""
Analyze the following website content and provide a structured report:

{extracted_content}

Please provide a detailed analysis in the following format:

1. Website Overview:
   - Key features identified
   - Main focus areas
   - Target audience

2. Content Analysis:
   - Products and Services assessment
   - Quality of information
   - Content organization
   - Unique selling propositions

3. Engagement Elements:
   - Call-to-actions
   - User interaction features
   - Social proof elements
   - Contact accessibility

4. SWOT Analysis:
   Strengths:
   - List key strengths identified
   
   Weaknesses:
   - List areas needing improvement
   
   Opportunities:
   - List potential growth areas
   
   Threats:
   - List competitive challenges

5. Recommendations:
   - Short-term improvements (0-3 months)
   - Medium-term strategy (3-6 months)
   - Long-term goals (6-12 months)

Please provide specific, actionable insights for each section.
"""

# Function to parse and structure AI response
def parse_ai_response(ai_response: str) -> Dict[str, str]:
    sections = {
        "overview": "",
        "content_analysis": "",
        "engagement": "",
        "swot": "",
        "recommendations": ""
    }
    
    current_section = "overview"
    lines = ai_response.split('\n')
    
    for line in lines:
        if "Website Overview" in line:
            current_section = "overview"
        elif "Content Analysis" in line:
            current_section = "content_analysis"
        elif "Engagement Elements" in line:
            current_section = "engagement"
        elif "SWOT Analysis" in line:
            current_section = "swot"
        elif "Recommendations" in line:
            current_section = "recommendations"
        else:
            sections[current_section] += line + "\n"
    
    return sections

# Function to display structured report
def display_structured_report(sections: Dict[str, str]):
    st.write("# Website Analysis Report")
    
    with st.expander("📋 Website Overview", expanded=True):
        st.markdown(sections["overview"])
    
    with st.expander("📊 Content Analysis"):
        st.markdown(sections["content_analysis"])
    
    with st.expander("🤝 Engagement Elements"):
        st.markdown(sections["engagement"])
    
    with st.expander("📈 SWOT Analysis"):
        st.markdown(sections["swot"])
    
    with st.expander("💡 Recommendations"):
        st.markdown(sections["recommendations"])

# Function to analyze website URL and extract content
def scrape_website_content(website_url):
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        response = requests.get(website_url, headers=headers, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        content = {
            "products_services": soup.find_all(text=lambda text: "product" in text.lower() or "service" in text.lower()),
            "success_stories": soup.find_all(text=lambda text: "success" in text.lower() or "testimonial" in text.lower()),
            "blog_resources": soup.find_all(text=lambda text: "blog" in text.lower() or "resources" in text.lower()),
            "about_us": soup.find_all(text=lambda text: "about" in text.lower() and "us" in text.lower()),
            "contact_us": soup.find_all(text=lambda text: "contact" in text.lower() and "us" in text.lower()),
        }

        extracted_content = ""
        for section, items in content.items():
            extracted_content += f"**{section.replace('_', ' ').title()}:**\n"
            for item in items[:5]:
                extracted_content += f"- {item.strip()}\n"
            extracted_content += "\n"

        return extracted_content

    except requests.exceptions.RequestException as e:
        st.error(f"An error occurred while fetching the website: {e}")
        return None
    except Exception as e:
        st.error(f"An error occurred during scraping: {e}")
        return None

# Function to generate AI-powered report
def generate_ai_report(extracted_content: str) -> Optional[str]:
    try:
        prompt = create_structured_prompt(extracted_content)
        
        generated_text = generator(
            prompt,
            max_new_tokens=800,
            num_return_sequences=1,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.2
        )[0]["generated_text"]
        
        sections = parse_ai_response(generated_text)
        display_structured_report(sections)
        
        return generated_text
        
    except Exception as e:
        st.error(f"An error occurred while generating the AI report: {e}")
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

else:  # Website Analysis
    st.subheader("Website Analysis")
    website_url = st.text_input("Enter Website URL for Analysis")
    
    if st.button("Generate AI-Powered Report"):
        if website_url:
            with st.spinner("Analyzing website content..."):
                extracted_content = scrape_website_content(website_url)
                if extracted_content:
                    with st.spinner("Generating structured AI report..."):
                        report = generate_ai_report(extracted_content)
                        if report:
                            st.success("Analysis complete! Expand the sections above to view detailed insights.")
                            
                            # Add download button for the report
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
