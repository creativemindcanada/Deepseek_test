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

    # Function to calculate content quality score
def calculate_content_quality_score(content: str) -> float:
    """
    Calculate a content quality score based on readability, length, and keyword density.
    """
    from textstat import flesch_reading_ease, syllable_count, lexicon_count
    import re

    # Calculate readability score
    readability_score = flesch_reading_ease(content)

    # Calculate word count
    word_count = len(re.findall(r'\b\w+\b', content))

    # Calculate keyword density (basic example: count of common words)
    common_words = ["product", "service", "quality", "customer", "experience"]
    keyword_density = sum(content.lower().count(word) for word in common_words) / word_count if word_count > 0 else 0

    # Combine scores (weights can be adjusted)
    quality_score = (readability_score * 0.4) + (word_count * 0.3) + (keyword_density * 0.3)
    return round(quality_score, 2)

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
    """
    Scrape and extract content from a website with improved error handling and content processing.
    """
    try:
        # Validate URL format
        if not website_url.startswith(('http://', 'https://')):
            website_url = 'https://' + website_url

        # Set up headers to mimic a browser request
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
        }

        # Make the request with timeout
        response = requests.get(
            website_url,
            headers=headers,
            timeout=15,
            verify=True  # Enable SSL verification
        )
        response.raise_for_status()

        # Check if response is HTML
        content_type = response.headers.get('content-type', '').lower()
        if 'text/html' not in content_type:
            st.error(f"Invalid content type: {content_type}. Please provide a valid website URL.")
            return None

        # Parse the content
        soup = BeautifulSoup(response.text, "html.parser")

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

    except requests.exceptions.MissingSchema:
        st.error("Invalid URL. Please include 'http://' or 'https://' in the URL.")
        return None
    except requests.exceptions.ConnectionError:
        st.error("Could not connect to the website. Please check the URL and try again.")
        return None
    except requests.exceptions.Timeout:
        st.error("The request timed out. Please try again later.")
        return None
    except requests.exceptions.TooManyRedirects:
        st.error("Too many redirects. Please check the URL.")
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"An error occurred while fetching the website: {str(e)}")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
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

elif analysis_type == "Website Analysis":  # Ensure this is properly aligned
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
