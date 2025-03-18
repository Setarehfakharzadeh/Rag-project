#!/usr/bin/env python3
"""
Simple script to fetch content from the Model Context Protocol documentation.
"""
import requests
import json
from bs4 import BeautifulSoup

# URLs to fetch
URLS = [
    "https://modelcontextprotocol.io/quickstart/server",
    "https://modelcontextprotocol.io/quickstart/client",
    "https://modelcontextprotocol.io/specs/latest"
]

def fetch_content(url):
    """Fetch content from a URL"""
    print(f"Fetching {url}...")
    try:
        resp = requests.get(url)
        resp.raise_for_status()
        
        print(f"Response status: {resp.status_code}, content length: {len(resp.text)}")
        
        soup = BeautifulSoup(resp.text, 'html.parser')
        
        # Remove nav, headers, footers
        for elem in soup.select('nav, header, footer, script, style'):
            elem.extract()
        
        # Get main content
        main = soup.select_one('main, .content, .markdown-body, article') or soup.body
        if not main:
            print("Could not find main content element!")
            return None
            
        print(f"Found main content element: {main.name}")
        
        # Get title
        title_elem = soup.select_one('h1, h2')
        title = title_elem.get_text().strip() if title_elem else "Untitled"
        print(f"Found title: {title}")
        
        # Get content
        content = main.get_text().strip()
        print(f"Extracted content length: {len(content)}")
        
        return {
            "source": url,
            "title": title,
            "content": content
        }
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return None

def main():
    """Main function"""
    chunks = []
    
    # Fetch content from URLs
    for url in URLS:
        content = fetch_content(url)
        if content:
            print(f"Got content from {url}: {len(content['content'])} chars")
            # Split into smaller chunks of 1000 chars
            text = content['content']
            for i in range(0, len(text), 1000):
                chunk = text[i:i+1000]
                chunks.append({
                    "source": content['source'],
                    "content": f"{content['title']}\n\n{chunk}"
                })
    
    # Save to file
    if chunks:
        print(f"Saving {len(chunks)} chunks to mcp_chunks.json")
        with open('mcp_chunks.json', 'w') as f:
            json.dump(chunks, f, indent=2)
        print("Done!")
    else:
        print("No content found!")

if __name__ == "__main__":
    main() 