import streamlit as st
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
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
st.title("Customer Insights & Competitor Analysis Dashboard")

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
    
   def scrape_website_content_selenium(website_url: str) -> Optional[str]:
    """
    Scrape and extract content from a website using Selenium with improved error handling and content processing.
    """
    try:
        # Validate URL format
        if not website_url.startswith(('http://', 'https://')):
            website_url = 'https://' + website_url

        # Set up Selenium options
        chrome_options = Options()
        chrome_options.add_argument("--headless")  # Run in headless mode
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")

        # Initialize the WebDriver
        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
        
        # Navigate to the website
        driver.get(website_url)
        
        # Wait for the page to load completely
        driver.implicitly_wait(10)  # Wait up to 10 seconds for elements to load

        # Get the page source
        page_source = driver.page_source

        # Close the browser
        driver.quit()

        # Parse the content using BeautifulSoup
        soup = BeautifulSoup(page_source, "html.parser")

        # Remove unwanted elements
        for element in soup(['script', 'style', 'meta', 'link', 'noscript']):
            element.decompose()

        # Initialize content dictionary
        content = {
            "title": "",
            "meta_description": "",
            "main_content": [],
            "navigation": [],
            "products_services": [],
            "about": [],
            "contact": []
        }

        # Extract title
        if soup.title:
            content["title"] = soup.title.string.strip() if soup.title.string else ""

        # Extract meta description
        meta_desc = soup.find('meta', attrs={'name': 'description'})
        if meta_desc:
            content["meta_description"] = meta_desc.get('content', '').strip()

        # Extract main content
        main_content = soup.find(['main', 'article', 'div'], class_=['content', 'main', 'main-content'])
        if main_content:
            content["main_content"] = [p.text.strip() for p in main_content.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6']) if p.text.strip()]

        # Extract navigation
        nav = soup.find(['nav', 'menu'])
        if nav:
            content["navigation"] = [link.text.strip() for link in nav.find_all('a') if link.text.strip()]

        # Extract products/services
        products_section = soup.find(['div', 'section'], string=lambda text: text and any(word in text.lower() for word in ['product', 'service']))
        if products_section:
            content["products_services"] = [item.text.strip() for item in products_section.find_all(['p', 'li', 'h3']) if item.text.strip()]

        # Extract about information
        about_section = soup.find(['div', 'section'], string=lambda text: text and 'about' in text.lower())
        if about_section:
            content["about"] = [item.text.strip() for item in about_section.find_all(['p', 'li']) if item.text.strip()]

        # Extract contact information
        contact_section = soup.find(['div', 'section'], string=lambda text: text and 'contact' in text.lower())
        if contact_section:
            content["contact"] = [item.text.strip() for item in contact_section.find_all(['p', 'li']) if item.text.strip()]

        # Format the extracted content
        formatted_content = f"""
Website: {website_url}

Title: {content['title']}

Description: {content['meta_description']}

Navigation Menu:
{chr(10).join([f"- {item}" for item in content['navigation'][:5]])}

Main Content:
{chr(10).join([f"- {item}" for item in content['main_content'][:10]])}

Products/Services:
{chr(10).join([f"- {item}" for item in content['products_services'][:5]])}

About Information:
{chr(10).join([f"- {item}" for item in content['about'][:5]])}

Contact Information:
{chr(10).join([f"- {item}" for item in content['contact'][:5]])}
"""
        
        # Check if we got meaningful content
        if not any([content['main_content'], content['products_services'], content['about'], content['contact']]):
            st.warning("Limited content could be extracted from this website. The analysis may be incomplete.")
            
        return formatted_content

    except Exception as e:
        st.error(f"An error occurred while scraping the website: {str(e)}")
        return None

def display_structured_report(sections: Dict[str, str]):
    """Display the report with better error handling and formatting."""
    st.write("# Competitor Analysis Report")
    
    # Core Strategic Elements
    with st.expander("💡 Core Strategic Elements", expanded=True):
        content = sections.get("core_strategic_elements", "Analysis pending.")
        st.markdown(content if content.strip() else "No core strategic elements available.")
    
    # Proven Value Levers
    with st.expander("📊 Proven Value Levers"):
        content = sections.get("proven_value_levers", "Analysis pending.")
        st.markdown(content if content.strip() else "No proven value levers available.")
    
    # Key Success Factors
    with st.expander("🏆 Key Success Factors"):
        content = sections.get("key_success_factors", "Analysis pending.")
        st.markdown(content if content.strip() else "No key success factors available.")

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
analysis_type = st.sidebar.radio("Choose Analysis Type", ["Customer Data Analysis", "Competitor Website Analysis"])

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

elif analysis_type == "Competitor Website Analysis":
    st.subheader("Competitor Website Analysis")
    website_url = st.text_input("Enter Competitor Website URL for Analysis")

    if st.button("Generate AI-Powered Competitor Report"):
        if website_url:
            with st.spinner("Analyzing competitor website content..."):
                extracted_content = scrape_website_content_selenium(website_url)
                if extracted_content:
                    with st.spinner("Generating structured AI report..."):
                        report = generate_ai_report(extracted_content)
                        if report:
                            st.success("Competitor analysis complete! Expand the sections below to view detailed insights.")
                            
                            # Display a sample report for logistics industry
                            st.write("### Sample Competitor Analysis Report (Logistics Industry)")
                            st.write("""
**Core Strategic Elements:**

- **Digital Twin Technology:** Enables real-time visibility across silos.
- **Decision Intelligence Studio:** AI-powered optimization for logistics operations.
- **Continuous Realignment:** Dynamic adaptation capabilities to changing market conditions.

**Proven Value Lever Implementation:**





1. **Intersilo Data Integration:**
   - Implemented through Digital Twin platform.
   - **Results:** Thermo Fisher tracking 570K+ shipments.
   - **Impact:** Enhanced quality and compliance.

2. **Resource Optimization:**
   - Used Decision Intelligence for dynamic allocation.
   - **Results:** 33% reduction in expedited shipping costs.
   - **Impact:** $25M savings for GE Appliances.

3. **Quality & Compliance:**
   - Real-time monitoring and alerts.
   - **Results:** Enhanced cold chain compliance.
   - **Impact:** Significant reduction in temperature excursions.

**Key Success Factors:**

- Industry-specific solutions (Pharma, Consumer Goods, F&B).
- End-to-end visibility approach.
- Focus on measurable outcomes.
                            """)
                            
                            # Download the full report
                            st.download_button(
                                label="Download Full Competitor Report",
                                data=report,
                                file_name="competitor_analysis_report.txt",
                                mime="text/plain"
                            )
                        else:
                            st.error("Failed to generate the AI report.")
                else:
                    st.error("Failed to scrape competitor website content.")
        else:
            st.warning("Please enter a valid competitor website URL.")
