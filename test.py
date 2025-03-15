# Add these imports at the top
import re
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

def universal_scrape(url: str) -> str:
    """Robust scraper for any website"""
    try:
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        driver = webdriver.Chrome(
            service=Service(ChromeDriverManager().install()),
            options=chrome_options
        )
        driver.get(url)
        
        # Wait for core content
        WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.TAG_NAME, "main")) or
            EC.presence_of_element_located((By.ID, "content"))
        )
        
        # Extract vital sections
        soup = BeautifulSoup(driver.page_source, "html.parser")
        
        # Remove clutter
        for tag in soup(["script", "style", "nav", "footer"]):
            tag.decompose()
            
        # Universal content extraction
        content = {
            "main_heading": soup.find("h1").text if soup.find("h1") else "",
            "key_points": [p.text for p in soup.select("ul:not(nav ul) li")][:10],
            "testimonials": [div.text for div in soup.select(".testimonial, .review")],
            "clients": [img["alt"] for img in soup.select("img[alt*='client'], img[alt*='logo']") if "alt" in img.attrs],
            "pricing": soup.select_one(".pricing, .plans").text if soup.select(".pricing, .plans") else "",
            "cta": soup.select_one(".cta, .signup").text if soup.select(".cta, .signup") else ""
        }
        
        driver.quit()
        return f"""
        WEBSITE CONTENT ANALYSIS:
        Main Heading: {content['main_heading']}
        Key Features: {content['key_points']}
        Client Logos: {content['clients'][:5]}
        Pricing Info: {content['pricing'][:500]}
        Testimonials: {content['testimonials'][:3]}
        Call-to-Action: {content['cta']}
        """
        
    except Exception as e:
        st.error(f"Scraping error: {str(e)}")
        return ""

def universal_prompt(content: str) -> str:
    return f"""Analyze any business website and structure response with:
    
1. BUSINESS MODEL - Core offerings and revenue streams
2. TARGET AUDIENCE - Ideal customer demographics
3. COMPETITIVE EDGE - Unique value proposition
4. GROWTH POTENTIAL - Scalability indicators
5. KEY RISKS - Visible weaknesses

Website Content:
{content}

Format with these exact section headers:
### BUSINESS MODEL
### TARGET AUDIENCE 
### COMPETITIVE EDGE
### GROWTH POTENTIAL
### KEY RISKS"""

def parse_universal_response(response: str) -> dict:
    sections = {
        "Business Model": [],
        "Target Audience": [],
        "Competitive Edge": [],
        "Growth Potential": [],
        "Key Risks": []
    }
    current_section = None
    
    for line in response.split('\n'):
        line = line.strip()
        if line.startswith("### "):
            current_section = line[4:].strip()
        elif current_section and line:
            sections[current_section].append(line)
            
    return sections

# In your Website Analysis section:
elif analysis_type == "Website Analysis":
    url = st.text_input("Enter Website URL", "https://www.example.com")
    
    if st.button("Analyze Website"):
        raw_content = universal_scrape(url)
        if raw_content:
            prompt = universal_prompt(raw_content)
            response = generator(prompt, max_new_tokens=600)[0]["generated_text"]
            analysis = parse_universal_response(response)
            
            st.write("# Universal Business Analysis")
            for section, content in analysis.items():
                with st.expander(f"**{section}**"):
                    st.write("\n".join(content) if content else "No analysis available")
