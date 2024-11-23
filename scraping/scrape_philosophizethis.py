import requests
from bs4 import BeautifulSoup
import os

# Define the base URL and the page to scrape
base_url = "https://www.philosophizethis.org"
start_url = f"{base_url}/transcripts"

# Create a directory to save transcripts
output_dir = "transcripts"
os.makedirs(output_dir, exist_ok=True)

# Step 1: Fetch the page and parse
response = requests.get(start_url)
soup = BeautifulSoup(response.text, 'html.parser')

# Step 2: Find all links ending in 'transcript'
links = soup.find_all('a', href=True)
transcript_links = [link['href'] for link in links if link['href'].endswith('transcript')]

# Step 3: Visit each transcript link
for link in transcript_links:
    # Handle relative links
    full_url = link if link.startswith("http") else base_url + link
    transcript_response = requests.get(full_url)
    transcript_soup = BeautifulSoup(transcript_response.text, 'html.parser')
    
    # Find the content in the specific div
    content_divs = transcript_soup.find_all('div', class_='blog-item-content e-content')
    content = "\n\n".join(div.get_text(strip=True) for div in content_divs)
    
    # Save to .txt file
    if content:
        file_name = os.path.join(output_dir, f"{link.split('/')[-1]}.txt")
        with open(file_name, 'w', encoding='utf-8') as file:
            file.write(content)
        print(f"Saved: {file_name}")
    else:
        print(f"No content found for: {full_url}")
