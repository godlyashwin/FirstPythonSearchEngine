import threading
import requests
from bs4 import BeautifulSoup
import wikipediaapi
import re
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.tag import pos_tag
import time
# Ensure NLTK resources are downloaded
import nltk
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger_eng')

# Global progress variable
progress = {"progress": 0, "status": "Processing", "results": None}
_progress_lock = threading.Lock()

# Function to simplify query (unchanged)
def simplify_query(query):
    global progress
    with _progress_lock:
        progress = {"progress": 1, "status": "Simplifying Query", "results": None}
    print("Progress: " + progress["status"])
    tokens = word_tokenize(query.lower())
    stop_words = set(stopwords.words('english'))
    custom_stopwords = {'was', 'when', 'where', 'who', 'what', 'how', 'made', 'created'}
    stop_words.update(custom_stopwords)
    tagged_tokens = pos_tag(tokens)
    key_terms = [word for word, pos in tagged_tokens 
                 if (pos in ['NNP', 'NN']) and word not in stop_words]
    if not key_terms:
        key_terms = [word for word in tokens if word not in stop_words]
    simplified = ' '.join(key_terms) if key_terms else query
    return simplified

# Function to clean text (unchanged)
def clean_text(text):
    global progress
    with _progress_lock:
        progress = {"progress": 13, "status": "Cleaning paragraphs", "results": None}
    print("Progress: " + progress["status"])
    pattern = r'\[\d+\]'
    cleaned = re.sub(pattern, '', text).strip()
    return cleaned if cleaned else None

# Function to clean Google URL (unchanged)
from urllib.parse import urlparse, parse_qs
def clean_google_url(url):
    global progress
    with _progress_lock:
        progress = {"progress": 2, "status": "Cleaning Google URL: " + url, "results": None}
    print("Progress: " + progress["status"])
    if url.startswith('/url?q='):
        parsed = urlparse(url)
        return parse_qs(parsed.query).get('q', [url])[0]
    return url

# Function to scrape paragraphs (modified for reliability)
def scrape_paragraphs(url, use_selenium=False):
    global progress
    try:
        if use_selenium:
            options = webdriver.ChromeOptions()
            options.add_argument('--headless')
            options.add_argument('--no-sandbox')  # Improve stability in some environments
            options.add_argument('--disable-dev-shm-usage')
            with webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options) as driver:
                with _progress_lock:
                    progress = {"progress": progress["progress"], "status": f"Scraping {url}", "results": None}
                print("Progress: " + progress["status"])
                driver.set_page_load_timeout(10)  # Timeout after 10 seconds
                driver.get(url)
                soup = BeautifulSoup(driver.page_source, 'html.parser')
        else:
            response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = soup.find_all('p')
        paragraph_texts = [p.get_text().strip() for p in paragraphs]
        draft_paragraph_texts = [None if len(p.strip()) < 10 else p for p in paragraph_texts]
        print("LenWords for each paragraph: " + str([len(p.strip()) for p in draft_paragraph_texts]))
        final_paragraph_texts = [re.sub(r'\[\d+\]', '', p) for p in draft_paragraph_texts if p]
        return final_paragraph_texts
    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return []

# Function to fetch Wikipedia content (unchanged)
def fetch_wikipedia_content(query):
    global progress
    with _progress_lock:
        progress = {"progress": 10, "status": "Fetching Wikipedia Content", "results": None}
    print("Progress: " + progress["status"])
    wiki = wikipediaapi.Wikipedia(user_agent='WikiSearchEngineBot/1.0 (gashwin503@gmail.com)', language='en')
    page = wiki.page(query)
    #test = wikipediaapi.Wikipedia(user_agent='SimpleTestBot/1.0 (gashwin503@gmail.com)').page
    if not page.exists():
        print(f"Wikipedia page for '{query}' not found.")
        return []
    paragraphs = page.text.split('\n')
    paragraph_texts = [clean_text(p) for p in paragraphs if clean_text(p)]
    return paragraph_texts, page.fullurl

# Function to fetch Google Custom Search API results (unchanged)
def fetch_google_api_results(query, api_key, cse_id):
    global progress
    try:
        with _progress_lock:
            progress = {"progress": 20, "status": "Fetching Google API", "results": None}
        print("Progress: " + progress["status"])
        url = f"https://www.googleapis.com/customsearch/v1?key={api_key}&cx={cse_id}&q={requests.utils.quote(query)}"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        results = response.json().get('items', [])
        return [(clean_text(item['snippet']), item['link']) for item in results if clean_text(item['snippet'])]
    except requests.exceptions.HTTPError as e:
        print(f"HTTP error fetching Google API results: {e}")
        print(f"Response: {response.text}")
        return []
    except Exception as e:
        print(f"Error fetching Google API results: {e}")
        return []

# Main function to process user input and retrieve content (modified for debugging)
def search_dynamic_content(query, docs, sources, APIKEY, SEARCHENGINEID):
    global progress
    with _progress_lock:
        progress = {"progress": 0, "status": "Searching", "results": None}
    print(f"[DEBUG] Starting search_dynamic_content for query: {query}")
    try:
        # Simplify the query
        query = simplify_query(query)
        print(f"[DEBUG] Simplified Query: {query}")

        # 1. Fetch Wikipedia content
        print(f"[DEBUG] Fetching Wikipedia content for '{query}'...")
        wiki_paragraphs, wiki_source = fetch_wikipedia_content(query)
        for i, wiki in enumerate(wiki_paragraphs):
            print(f"[DEBUG] Wiki {i}: {wiki[:50]}...")
        if wiki_paragraphs:
            docs.extend(wiki_paragraphs)
            sources.extend([str(wiki_source)] * len(wiki_paragraphs))

        # 2. Scrape Google search results
        print(f"[DEBUG] Fetching Google search results for '{query}'...")
        google_snippets_sources = fetch_google_api_results(query, APIKEY, SEARCHENGINEID)
        start_time = time.time()
        timeout = 30
        for i, source in enumerate(google_snippets_sources):
            if time.time() - start_time > timeout:
                print(f"Skipping {url} due to timeout")
                continue
            print(f"[DEBUG] Source {i+1}: {source[1]}")
            url = source[1]
            with _progress_lock:
                if len(url) > 50:
                    displayURL = url[:50] + "..." # Crops out the url
                else:
                    displayURL = url
                progress = {"progress": 20 + (i+1)*(50/len(google_snippets_sources)), "status": f"Fetching Content from {displayURL}", "results": None}
            print("Progress: " + progress["status"])
            google_paragraphs = scrape_paragraphs(url, use_selenium=True)
            if google_paragraphs:
                docs.extend(google_paragraphs)
                sources.extend([url] * len(google_paragraphs))
            if (i+1) >= 10:  # Early termination after 1 result
                print("Ended program!")
                break
        with _progress_lock:
            progress = {"progress": 100, "status": "Completed", "results": None}
        print("[DEBUG] search_dynamic_content completed")
        return docs, sources
    except Exception as e:
        print(f"[DEBUG] Error in search_dynamic_content: {e}")
        with _progress_lock:
            progress = {"progress": progress["progress"], "status": f"Error: {e}", "results": None}
        raise

# Function to get progress
def getProgress():
    global progress
    with _progress_lock:
        return progress