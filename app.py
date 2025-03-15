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
    """Business-focused prompt"""
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
    """Parse business-focused response"""
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
    """Business-focused display"""
    st.write("# Strategic Business Analysis")
    
    with st.expander("💼 Business Model", expanded=True):
        st.markdown(sections.get("business_model", "Analysis pending..."))
        
    with st.expander("🚀 GTM Strategy"):
        st.markdown(sections.get("gtm_strategy", "Analysis pending..."))
        
    with st.expander("📈 Client Insights"):
        st.markdown(sections.get("client_analysis", "Analysis pending..."))
        
    with st.expander("⚠️ Risk Assessment"):
        st.markdown(sections.get("future_risks", "Analysis pending..."))
        
    # Add summary metrics
    st.subheader("Key Metrics")
    cols = st.columns(4)
    with cols[0]:
        st.metric("Clients Listed", len(re.findall(r"\bClient:", sections["client_analysis"])))
    with cols[1]:
        st.metric("Social Channels", len(sections["gtm_strategy"].split("Channel:"))-1)
    with cols[2]:
        st.metric("Key Solutions", len(re.findall(r"\bSolution:", sections["business_model"])))
    with cols[3]:
        st.metric("Identified Risks", len(re.findall(r"\-\s", sections["future_risks"])))

# Function to generate actionable recommendations
def generate_actionable_recommendations(sections: Dict[str, str]) -> List[str]:
    """
    Generate actionable recommendations based on the analysis.
    """
    recommendations = []

    # Recommendations based on weaknesses
    if sections.get("weaknesses"):
        if "slow" in sections["weaknesses"].lower():
            recommendations.append("🚀 **Improve website speed** by optimizing images and using a CDN.")
        if "navigation" in sections["weaknesses"].lower():
            recommendations.append("🧭 **Simplify navigation** by reducing menu items and adding a search bar.")
        if "mobile" in sections["weaknesses"].lower():
            recommendations.append("📱 **Optimize for mobile devices** by using responsive design.")

    # Recommendations based on content
    if sections.get("content"):
        if "outdated" in sections["content"].lower():
            recommendations.append("📅 **Update content regularly** to keep it relevant and engaging.")
        if "length" in sections["content"].lower():
            recommendations.append("✂️ **Break up long content** into smaller sections with headings.")

    # Default recommendations if no specific weaknesses are found
    if not recommendations:
        recommendations.append("🌟 **Focus on user experience** by gathering feedback and making iterative improvements.")

    return recommendations

# Function to display visualizations
def display_website_visualizations(sections: Dict[str, str]):
    """
    Display visualizations for website analysis.
    """
    st.subheader("📊 Visualizations")

    # Word cloud for main content
    if sections.get("main_content"):
        from wordcloud import WordCloud
        import matplotlib.pyplot as plt

        st.write("### Word Cloud for Main Content")
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(" ".join(sections["main_content"]))
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        st.pyplot(plt)

    # Bar chart for content quality
    if sections.get("content"):
        st.write("### Content Quality Score")
        quality_score = calculate_content_quality_score(sections["content"])
        st.metric("Quality Score", quality_score)

        # Display quality score as a bar chart
        fig = px.bar(x=["Content Quality"], y=[quality_score], labels={"x": "Metric", "y": "Score"}, text=[quality_score])
        fig.update_traces(textposition='auto')
        st.plotly_chart(fig)

# Update the display_structured_report function to include visualizations and recommendations
def display_structured_report(sections: Dict[str, str]):
    """Display the report with visualizations and actionable recommendations."""
    st.write("# Website Analysis Report")
    
    # Overview
    with st.expander("📋 Overview", expanded=True):
        content = sections.get("overview", "Analysis pending.")
        st.markdown(content if content.strip() else "No overview available.")
    
    # Content Analysis
    with st.expander("📊 Content Analysis"):
        content = sections.get("content", "Analysis pending.")
        st.markdown(content if content.strip() else "No content analysis available.")
    
    # Engagement
    with st.expander("🤝 Engagement Assessment"):
        content = sections.get("engagement", "Analysis pending.")
        st.markdown(content if content.strip() else "No engagement analysis available.")
    
    # Strengths & Weaknesses
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("💪 Strengths")
        content = sections.get("strengths", "Analysis pending.")
        st.markdown(content if content.strip() else "No strengths listed.")
    
    with col2:
        st.subheader("🎯 Areas for Improvement")
        content = sections.get("weaknesses", "Analysis pending.")
        st.markdown(content if content.strip() else "No weaknesses listed.")
    
    # Recommendations
    with st.expander("💡 Recommendations", expanded=True):
        content = sections.get("recommendations", "Analysis pending.")
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
        
# Function to clear session state
def clear_data():
    st.session_state.pop('data', None)
    st.toast("Data cleared successfully!", icon="✅")

# Main dashboard layout
st.sidebar.title("Navigation")
analysis_type = st.sidebar.radio("Choose Analysis Type", ["Customer Data Analysis", "Website Analysis"])


# Clear data button
if st.sidebar.button("🧹 Clear Data"):
    clear_data()

# Help section in the sidebar
with st.sidebar.expander("ℹ️ Help"):
    st.write("""
    - **Customer Data Analysis**: Upload a CSV file or use randomly generated data to analyze customer insights.
    - **Website Analysis**: Enter a website URL to generate an AI-powered analysis report.
    - **Dark Mode**: Toggle dark mode for better visibility in low-light environments.
    - **Clear Data**: Reset the app to its initial state.
    """)

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
def scrape_website_content_selenium(website_url: str) -> Optional[str]:
    """Enhanced scraping focusing on business insights extraction"""
    try:
        if not website_url.startswith(('http://', 'https://')):
            website_url = 'https://' + website_url

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        }

        response = requests.get(website_url, headers=headers, timeout=15)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        # Remove unwanted elements
        for element in soup(['script', 'style', 'meta', 'link', 'noscript']):
            element.decompose()

        # Enhanced content extraction
        content = {
            "success_stories": [],
            "testimonials": [],
            "use_cases": [],
            "blog_posts": [],
            "social_links": {},
            "client_list": [],
            "main_content": []
        }

        # Extract success stories/testimonials
        for section in soup.find_all(['div', 'section'], class_=re.compile(r'testimonial|review|case-study', re.I)):
            content["testimonials"].extend([p.text.strip() for p in section.find_all('p') if p.text.strip()])

        # Extract client use cases
        use_case_section = soup.find(['div', 'section'], string=re.compile(r'case studies|use cases', re.I))
        if use_case_section:
            content["use_cases"] = [case.text.strip() for case in use_case_section.find_all(['div', 'li'])]

        # Extract blog posts
        blog_section = soup.find(['div', 'section'], string=re.compile(r'blog|articles', re.I))
        if blog_section:
            content["blog_posts"] = [post.text.strip() for post in blog_section.find_all(['article', 'div'])]

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
            content["client_list"] = [img['alt'] for img in client_section.find_all('img') if 'client' in img.get('alt', '').lower()]

        # Format for AI analysis
        formatted_content = f"""
BUSINESS ANALYSIS DATA:
1. Customer Evidence:
   - Testimonials: {content['testimonials'][:5]}
   - Use Cases: {content['use_cases'][:5]}
   - Clients: {content['client_list'][:10]}

2. Digital Presence:
   - Social Media: {content['social_links']}
   - Blog Posts: {content['blog_posts'][:3]}

3. Main Content:
   {'. '.join([p for p in soup.find_all('p')[:3] if p.text.strip()])}
"""
        return formatted_content

    except Exception as e:
        st.error(f"Scraping error: {str(e)}")
        return None
# Function to optimize AI report generation with caching
@st.cache_data(ttl=3600)  # Cache results for 1 hour
def generate_ai_report_optimized(extracted_content: str) -> Optional[str]:
    """
    Optimized AI report generation with caching to improve performance.
    """
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

# Function to add a loading spinner with progress bar
def show_loading_spinner(message: str):
    """
    Display a loading spinner with a progress bar for long-running tasks.
    """
    with st.spinner(message):
        progress_bar = st.progress(0)
        for i in range(100):
            time.sleep(0.02)  # Simulate a long-running task
            progress_bar.progress(i + 1)
        st.success("Done!")
        # Website Analysis
if analysis_type == "Customer Data Analysis":  
    # Customer data logic here  
    st.sidebar.write("Analyzing customer data...")  

elif analysis_type == "Website Analysis":
    st.subheader("Strategic Business Analysis")
    website_url = st.text_input("Enter Company Website URL")
    
    if st.button("Generate Business Report"):
        if website_url:
            with st.spinner("Analyzing business model..."):
                extracted_content = scrape_website_content_selenium(website_url)
                if extracted_content:
                    report = generate_ai_report(extracted_content)
                    # Fixed indentation from here
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
