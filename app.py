import streamlit as st
import requests
from bs4 import BeautifulSoup
from transformers import pipeline  # Ensure transformers and torch/tensorflow are installed

# Title of the app
st.title("AI-Powered Website Analysis Tool")

# Preload the model when the app starts
@st.cache_resource  # Cache the model to avoid reloading on every interaction
def load_model():
    return pipeline("text-generation", model="distilgpt2")  # Use DistilGPT-2

# Load the model
with st.spinner("Loading AI model..."):
    generator = load_model()

# Function to analyze website URL and extract content
def scrape_website_content(website_url):
    try:
        # Fetch the website content
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        response = requests.get(website_url, headers=headers, timeout=10)
        response.raise_for_status()  # Raise an error for bad responses (4xx, 5xx)

        # Parse the website content using BeautifulSoup
        soup = BeautifulSoup(response.text, "html.parser")

        # Extract key sections
        content = {
            "products_services": soup.find_all(text=lambda text: "product" in text.lower() or "service" in text.lower()),
            "success_stories": soup.find_all(text=lambda text: "success" in text.lower() or "testimonial" in text.lower()),
            "blog_resources": soup.find_all(text=lambda text: "blog" in text.lower() or "resources" in text.lower()),
            "about_us": soup.find_all(text=lambda text: "about" in text.lower() and "us" in text.lower()),
            "contact_us": soup.find_all(text=lambda text: "contact" in text.lower() and "us" in text.lower()),
        }

        # Convert extracted content to a readable format
        extracted_content = ""
        for section, items in content.items():
            extracted_content += f"**{section.replace('_', ' ').title()}:**\n"
            for item in items[:5]:  # Display up to 5 examples per section
                extracted_content += f"- {item.strip()}\n"
            extracted_content += "\n"

        return extracted_content

    except requests.exceptions.RequestException as e:
        st.error(f"An error occurred while fetching the website: {e}")
        return None
    except Exception as e:
        st.error(f"An error occurred during scraping: {e}")
        return None

# Function to generate AI-powered report using Hugging Face Transformers
def generate_ai_report(extracted_content):
    try:
        # Define the prompt for the AI model
        prompt = f"""
        Analyze the following website content and provide a detailed report with actionable strategies for improvement:

        {extracted_content}

        The report should include:
        1. **Strengths:** Highlight the strengths of the website based on the content.
        2. **Weaknesses:** Identify any missing or underdeveloped sections.
        3. **Opportunities:** Suggest opportunities for growth or improvement.
        4. **Strategies:** Provide actionable strategies to enhance the website's effectiveness.

        Format the report in a structured way with clear headings and bullet points.
        """

        # Truncate the prompt if it exceeds the model's maximum context length
        max_context_length = 512  # DistilGPT-2 has a smaller context window
        if len(prompt) > max_context_length:
            prompt = prompt[:max_context_length]

        # Generate the report using the AI model
        report = generator(prompt, max_new_tokens=300, num_return_sequences=1)[0]["generated_text"]
        return report

    except Exception as e:
        st.error(f"An error occurred while generating the AI report: {e}")
        return None

# Function to display the report in a structured format
def display_structured_report(report):
    st.markdown("### AI-Powered Analysis Report")
    st.markdown("---")  # Add a horizontal line for separation

    # Use HTML/CSS for better alignment and structure
    st.markdown(
        f"""
        <div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px;">
            <h4 style="color: #2e86c1;">Strengths</h4>
            <ul style="list-style-type: disc; margin-left: 20px;">
                <li>{report.split("Strengths:")[1].split("Weaknesses:")[0].strip()}</li>
            </ul>

            <h4 style="color: #e74c3c;">Weaknesses</h4>
            <ul style="list-style-type: disc; margin-left: 20px;">
                <li>{report.split("Weaknesses:")[1].split("Opportunities:")[0].strip()}</li>
            </ul>

            <h4 style="color: #27ae60;">Opportunities</h4>
            <ul style="list-style-type: disc; margin-left: 20px;">
                <li>{report.split("Opportunities:")[1].split("Strategies:")[0].strip()}</li>
            </ul>

            <h4 style="color: #8e44ad;">Strategies</h4>
            <ul style="list-style-type: disc; margin-left: 20px;">
                <li>{report.split("Strategies:")[1].strip()}</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Input for website URL
st.subheader("Enter Website URL for Analysis")
website_url = st.text_input("Website URL", key="website_url_input")

# Analyze button
if st.button("Generate AI-Powered Report"):
    if website_url:
        with st.spinner("Analyzing website content..."):
            # Step 1: Scrape website content
            extracted_content = scrape_website_content(website_url)
            if extracted_content:
                # Step 2: Generate AI-powered report
                with st.spinner("Generating AI-powered report..."):
                    report = generate_ai_report(extracted_content)
                    if report:
                        st.success("AI-powered report generated successfully!")
                        display_structured_report(report)  # Display the structured report
                    else:
                        st.error("Failed to generate the AI report.")
            else:
                st.error("Failed to scrape website content.")
    else:
        st.warning("Please enter a valid website URL.")
