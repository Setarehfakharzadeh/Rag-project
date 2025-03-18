"""
Script to crawl the Model Context Protocol documentation website
and prepare data for RAG (Retrieval-Augmented Generation).
"""

import requests
from bs4 import BeautifulSoup
import re
import json
import os
import numpy as np
from sentence_transformers import SentenceTransformer

# Base URL and starting point
BASE_URL = "https://modelcontextprotocol.io"
START_URLS = [
    "https://modelcontextprotocol.io/quickstart/server",
    "https://modelcontextprotocol.io/quickstart/client",
    "https://modelcontextprotocol.io/specs/latest",
    "https://modelcontextprotocol.io/resources"
]

# Output files
CHUNKS_FILE = "mcp_chunks.json"
EMBEDDINGS_FILE = "mcp_embeddings.npy"

# Keep track of visited URLs
visited = set()
content_by_url = {}

def clean_text(text):
    """Clean and normalize text"""
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def extract_content(url):
    """Extract main content from a page"""
    try:
        print(f"Fetching: {url}")
        response = requests.get(url)
        if response.status_code != 200:
            print(f"  Failed with status code {response.status_code}")
            return None
            
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove navigation, header, footer elements
        for element in soup.select('nav, header, footer, script, style'):
            element.extract()
            
        # Get the main content - adjust selector based on the website structure
        main_content = soup.select_one('main') or soup.select_one('.content') or soup.select_one('.markdown-body') or soup.body
        
        if main_content:
            # Extract title if available
            title = soup.select_one('h1, h2')
            title_text = title.get_text() if title else ""
            
            # Extract main text content
            content_text = clean_text(main_content.get_text())
            
            # Format with title
            if title_text:
                return f"{title_text}\n\n{content_text}"
            return content_text
        return None
    except Exception as e:
        print(f"Error extracting content from {url}: {e}")
        return None

def get_links(url):
    """Extract links from a page"""
    links = set()
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        for link in soup.find_all('a', href=True):
            href = link['href']
            
            # Handle relative URLs
            if href.startswith('/'):
                next_url = BASE_URL + href
            elif href.startswith('http'):
                next_url = href
            else:
                continue
                
            # Skip non-documentation pages
            if BASE_URL not in next_url:
                continue
                
            links.add(next_url)
    except Exception as e:
        print(f"Error fetching links from {url}: {e}")
    
    return links

def crawl(start_url, max_pages=100):
    """Crawl the website breadth-first"""
    queue = [start_url]
    page_count = 0
    
    while queue and page_count < max_pages:
        url = queue.pop(0)
        
        if url in visited:
            continue
            
        visited.add(url)
        page_count += 1
        
        # Extract content
        content = extract_content(url)
        if content:
            content_by_url[url] = content
            print(f"  Added content from: {url}")
        
        # Get links for BFS
        if page_count < max_pages:
            links = get_links(url)
            for link in links:
                if link not in visited and link not in queue:
                    queue.append(link)

def create_chunks(texts, chunk_size=1000, chunk_overlap=200):
    """Split texts into chunks for RAG"""
    all_chunks = []
    
    for url, text in texts.items():
        # Simple chunking by paragraphs first
        paragraphs = text.split('\n\n')
        
        current_chunk = ""
        for para in paragraphs:
            # If adding this paragraph would exceed chunk size, save current chunk and start new one
            if len(current_chunk) + len(para) > chunk_size and current_chunk:
                all_chunks.append({
                    "content": current_chunk.strip(),
                    "source": url
                })
                
                # Start new chunk with overlap
                words = current_chunk.split()
                overlap_text = " ".join(words[-chunk_overlap:]) if len(words) > chunk_overlap else current_chunk
                current_chunk = overlap_text + "\n\n" + para
            else:
                if current_chunk:
                    current_chunk += "\n\n" + para
                else:
                    current_chunk = para
        
        # Add the last chunk if it's not empty
        if current_chunk.strip():
            all_chunks.append({
                "content": current_chunk.strip(),
                "source": url
            })
    
    return all_chunks

def generate_embeddings(chunks):
    """Generate embeddings for chunks"""
    model = SentenceTransformer('all-MiniLM-L6-v2')
    texts = [chunk["content"] for chunk in chunks]
    embeddings = model.encode(texts)
    return embeddings

def main():
    # Crawl each start URL
    for url in START_URLS:
        crawl(url)
    
    print(f"\nCrawled {len(content_by_url)} pages")
    
    # Create chunks
    chunks = create_chunks(content_by_url)
    print(f"Created {len(chunks)} chunks")
    
    # Save chunks
    with open(CHUNKS_FILE, "w") as f:
        json.dump(chunks, f, indent=2)
    print(f"Saved chunks to {CHUNKS_FILE}")
    
    # Generate and save embeddings
    if chunks:
        embeddings = generate_embeddings(chunks)
        np.save(EMBEDDINGS_FILE, embeddings)
        print(f"Saved embeddings to {EMBEDDINGS_FILE}")

if __name__ == "__main__":
    main() 