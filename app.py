import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import time  # Import the time module

# Title of the app
st.title("Customer Insights Dashboard")

# Function to generate random customer data
def generate_random_data():
    np.random.seed(42)
    data = {
        "customer_id": range(1, 101),
        "satisfaction_score": np.random.randint(1, 6, 100),
        "feedback": np.random.choice([
            "Great product!", "Good quality.", "Delivery was late.", "Poor packaging.", "Excellent service!"
        ], 100),
        "purchase_amount": np.random.uniform(10, 500, 100).round(2),
        "region": np.random.choice(["North", "South", "East", "West"], 100)
    }
    return pd.DataFrame(data)

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

    # Visualizations
    st.subheader("Customer Satisfaction Distribution")
    fig = px.histogram(data, x="satisfaction_score", nbins=10)
    st.plotly_chart(fig)

    # Insights and Recommendations
    st.subheader("Insights & Recommendations")
    if data['satisfaction_score'].mean() >= 4:
        st.success("Things are going well! Keep up the good work.")
        st.write("Recommendations: Consider launching a loyalty program to retain happy customers.")
    else:
        st.error("There are issues to address.")
        st.write("Recommendations: Investigate customer feedback and improve product/service quality.")

    # Sentiment Analysis (Optional)
    st.subheader("Sentiment Analysis")
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    import nltk
    nltk.download('vader_lexicon')

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

# Alternative Inputs (Below the Primary Dashboard)
st.header("Alternative Input Methods")

# Option 1: Company LinkedIn URL
st.subheader("Analyze Company LinkedIn Profile")
linkedin_url = st.text_input("Enter Company LinkedIn URL", key="linkedin_url_input")  # Unique key

if linkedin_url:
    with st.spinner("Fetching LinkedIn insights..."):
        time.sleep(2)  # Simulate a delay
        company_name = linkedin_url.split("/")[-1].replace("-", " ").title()
        mock_linkedin_data = {
            "Company Name": company_name,
            "Followers": f"{random.randint(1000, 1000000):,}",
            "Employees": f"{random.randint(100, 50000):,}",
            "Industry": random.choice(["Technology", "Healthcare", "Finance", "Retail"]),
            "Location": random.choice(["San Francisco, CA", "New York, NY", "London, UK", "Bangalore, India"]),
            "Recent Posts": [
                f"Excited to announce our new {random.choice(['AI-powered', 'blockchain-based', 'cloud-native'])} product!",
                f"Join us at the {company_name} {random.choice(['conference', 'summit', 'hackathon'])} next month!",
                f"We're hiring! Check out our open positions in {random.choice(['engineering', 'marketing', 'sales'])}."
            ]
        }
    
    # Display Mock Data
    st.subheader("Mock LinkedIn Insights")
    st.write(f"**Company Name:** {mock_linkedin_data['Company Name']}")
    st.write(f"**Followers:** {mock_linkedin_data['Followers']}")
    st.write(f"**Employees:** {mock_linkedin_data['Employees']}")
    st.write(f"**Industry:** {mock_linkedin_data['Industry']}")
    st.write(f"**Location:** {mock_linkedin_data['Location']}")
    
    st.subheader("Recent Posts")
    for post in mock_linkedin_data['Recent Posts']:
        st.write(f"- {post}")
    
    # Feedback Button
    st.write("Was this helpful?")
    if st.button("👍 Yes", key="feedback_yes"):  # Unique key
        st.success("Thanks for your feedback!")
    if st.button("👎 No", key="feedback_no"):  # Unique key
        st.error("We'll improve this feature soon!")
    
    # Try Another Company
    if st.button("Try Another Company", key="try_another_company"):  # Unique key
        st.session_state['linkedin_url'] = ""  # Clear the input
        st.experimental_rerun()  # Refresh the app
    
    # Learn More Section
    st.subheader("Why Can't We Access Real LinkedIn Data?")
    st.write("""
    LinkedIn's API is restricted and requires special permissions to access. 
    For this demo, we're using mock data to simulate LinkedIn insights. 
    If you need real LinkedIn data, consider applying for LinkedIn's API access or using alternative tools like Clearbit or Crunchbase.
    """)
