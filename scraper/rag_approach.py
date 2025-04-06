import argparse
import requests
from bs4 import BeautifulSoup, NavigableString, Comment
from urllib.parse import urlparse, urljoin
from collections import deque
import time
import logging
import re
import html
import os
from dotenv import load_dotenv

# Langchain & Embeddings specific imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings # Use Langchain wrapper
from langchain.schema.document import Document # To represent chunks

# --- Environment Variable Loading ---
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY") # Still load it for potential later use
# Default embedding model, good for general purpose & runs locally
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
embedding_model_name = os.getenv("EMBEDDING_MODEL_NAME", DEFAULT_EMBEDDING_MODEL)
# Default path to save/load the vector store
DEFAULT_VECTOR_STORE_PATH = "./faiss_vector_store"
vector_store_path = os.getenv("VECTOR_STORE_PATH", DEFAULT_VECTOR_STORE_PATH)

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Crawler/Processor Configuration (Keep these defined) ---
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}
MAIN_CONTENT_SELECTORS = [
    "main", "article", "div.article-body", "div.content__body", "div.DocsPage__body",
    "div#main", "div#main-content", "div.main-content", "div#content", "div.content",
    "div[role='main']",
]
NOISE_TAGS = [
    "nav", "header", "footer", "aside", "script", "style", "noscript",
    "button", "form", "meta", "link", "svg", "path", ".sidebar", ".navigation",
    ".footer", ".header", ".toc", ".breadcrumb", "#sidebar", "#navigation",
    "#footer", "#header", "#toc", "#breadcrumb", "*[aria-hidden='true']",
    "div[class*='NavBar__']", "div[class*='Sidebar__']", "div[class*='Footer__']",
    "div[class*='Breadcrumb__']", "div.article-votes", "section.article-relatives",
    "div.article-subscribe",
]
TEXT_BEARING_TAGS = ['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li', 'td', 'th', 'pre', 'code', 'span', 'div', 'label', 'figcaption']

# --- Helper Functions (Keep these defined) ---
def is_valid_url(url):
    parsed = urlparse(url)
    return bool(parsed.scheme) and bool(parsed.netloc)

def fetch_page(url, session, timeout=10):
    try:
        response = session.get(url, headers=HEADERS, timeout=timeout, allow_redirects=True)
        response.raise_for_status()
        content_type = response.headers.get('content-type', '').lower()
        if 'text/html' not in content_type:
            logging.warning(f"Skipping non-HTML content at {url} (Content-Type: {content_type})")
            return None, response.url
        if len(response.content) < 100:
             logging.warning(f"Skipping potentially empty page at {url} (Length: {len(response.content)})")
             return None, response.url
        response.encoding = response.apparent_encoding or 'utf-8'
        return response.text, response.url
    except requests.exceptions.RequestException as e:
        logging.error(f"Request error fetching {url}: {e}")
    except Exception as e:
        logging.error(f"Unexpected error fetching {url}: {e}")
    return None, url

def crawl_website(start_url, max_pages):
    # (Keep the crawl_website function implementation as defined in the previous answer)
    # ... (implementation omitted for brevity, assume it's here) ...
    if not is_valid_url(start_url):
        logging.error(f"Invalid start URL provided: {start_url}")
        return None
    parsed_start_url = urlparse(start_url)
    base_domain = parsed_start_url.netloc
    urls_to_visit = deque([start_url])
    visited_urls = set()
    crawled_data = {}
    with requests.Session() as session:
        while urls_to_visit and len(crawled_data) < max_pages:
            current_url = urls_to_visit.popleft()
            parsed_current = urlparse(current_url)
            normalized_url = parsed_current._replace(fragment="").geturl()
            if normalized_url in visited_urls: continue
            visited_urls.add(normalized_url)
            logging.info(f"Attempting to crawl: {current_url} ({len(crawled_data) + 1}/{max_pages})")
            html_content, final_url = fetch_page(current_url, session)
            parsed_final = urlparse(final_url)
            normalized_final_url = parsed_final._replace(fragment="").geturl()
            visited_urls.add(normalized_final_url)
            if html_content:
                final_domain = urlparse(final_url).netloc
                if final_domain != base_domain:
                    logging.warning(f"Skipping external link after redirect: {final_url}")
                    continue
                if final_url not in crawled_data and len(crawled_data) < max_pages:
                    crawled_data[final_url] = html_content
                soup = BeautifulSoup(html_content, 'html.parser')
                for link in soup.find_all('a', href=True):
                    href = link['href']
                    next_url = urljoin(final_url, href)
                    parsed_next_url = urlparse(next_url)
                    normalized_next_url = parsed_next_url._replace(fragment="").geturl()
                    if parsed_next_url.scheme in ['http', 'https'] and \
                       parsed_next_url.netloc == base_domain and \
                       normalized_next_url not in visited_urls and \
                       normalized_next_url not in urls_to_visit:
                        urls_to_visit.append(normalized_next_url)
            time.sleep(0.1)
    if not crawled_data:
         logging.warning(f"Could not retrieve any content starting from {start_url}.")
         return None
    logging.info(f"Crawling finished. Fetched content for {len(crawled_data)} unique pages.")
    return crawled_data


def format_list_item(tag, index):
    # (Keep implementation)
    parent_type = tag.parent.name if tag.parent else 'ul'
    prefix = f"{index + 1}. " if parent_type == 'ol' else "- "
    text = get_cleaned_text(tag)
    return prefix + text if text else ""

def format_table(tag):
    # (Keep implementation)
    rows = []
    for row in tag.find_all('tr'):
        cells = [get_cleaned_text(cell) for cell in row.find_all(['td', 'th'])]
        if any(cells): rows.append(" | ".join(cells))
    return "\n".join(rows)

def get_cleaned_text(element):
     # (Keep implementation)
    if not element: return ""
    text = element.get_text(separator=' ', strip=True)
    text = html.unescape(text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def process_element_recursive(element, extracted_blocks, processed_elements):
    # (Keep implementation as defined in the previous answer)
    # ... (implementation omitted for brevity, assume it's here) ...
    if element in processed_elements or not element.name or not element.parent: return
    element_text = ""
    should_process_children = True
    if element.name == 'ul' or element.name == 'ol':
        items = element.find_all('li', recursive=False)
        list_text = [fmt for i, item in enumerate(items) if (fmt := format_list_item(item, i))]
        element_text = "\n".join(list_text)
        should_process_children = False
    elif element.name == 'table':
        element_text = format_table(element)
        should_process_children = False
    elif element.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
        level = int(element.name[1]); cleaned_text = get_cleaned_text(element)
        if cleaned_text: element_text = "#" * level + " " + cleaned_text
        should_process_children = False
    elif element.name in ['p', 'pre', 'blockquote']:
        element_text = get_cleaned_text(element); should_process_children = False
    elif element.name in TEXT_BEARING_TAGS:
        if should_process_children:
            for child in element.find_all(True, recursive=False):
                process_element_recursive(child, extracted_blocks, processed_elements)
        direct_text = ''.join(element.find_all(string=True, recursive=False)).strip()
        block_children = set(['p', 'h1', 'h2', 'h3','h4','h5','h6','ul','ol','table','pre','blockquote'])
        if direct_text and not element.find(block_children):
             element_text = get_cleaned_text(element)
    if element_text: extracted_blocks.append(element_text.strip())
    if element_text or not should_process_children:
        processed_elements.add(element); processed_elements.update(element.find_all(True))
    elif should_process_children:
         for child in element.find_all(True, recursive=False):
             process_element_recursive(child, extracted_blocks, processed_elements)


def extract_meaningful_content(url, html_content):
    # (Keep implementation as defined in the previous answer)
    # ... (implementation omitted for brevity, assume it's here) ...
    if not html_content: return None
    try:
        soup = BeautifulSoup(html_content, 'html.parser')
        page_title = get_cleaned_text(soup.title) if soup.title else "No Title"
        for noise_selector in NOISE_TAGS:
            for element in soup.select(noise_selector): element.decompose()
        for comment in soup.find_all(string=lambda text: isinstance(text, Comment)): comment.extract()
        main_content_area = None
        for selector in MAIN_CONTENT_SELECTORS:
            if main_content_area := soup.select_one(selector):
                logging.debug(f"Found main content via: '{selector}' for {url}"); break
        if not main_content_area:
            logging.warning(f"No specific main content area for {url}. Using <body>.");
            if not (main_content_area := soup.body):
                 logging.error(f"No <body> tag in {url}. Skipping."); return None
        extracted_blocks = []; processed_elements = set()
        for element in main_content_area.find_all(True, recursive=False):
             process_element_recursive(element, extracted_blocks, processed_elements)
        full_text = "\n\n".join(block for block in extracted_blocks if block)
        full_text = re.sub(r'\n{3,}', '\n\n', full_text).strip()
        if not full_text or len(full_text) < 30:
             logging.warning(f"Extracted minimal text from {url}. Skipping."); return None
        logging.info(f"Successfully extracted ~{len(full_text)} chars from {url} (Title: {page_title})")
        return {"url": url, "title": page_title, "text": full_text}
    except Exception as e:
        logging.error(f"Error processing content for {url}: {e}", exc_info=False)
        return None

def process_crawled_data(website_data):
     # (Keep implementation)
    processed_docs = []
    if not website_data: return processed_docs
    logging.info(f"Processing content for {len(website_data)} pages...")
    for url, html_content in website_data.items():
        if extracted_data := extract_meaningful_content(url, html_content):
            processed_docs.append(extracted_data)
    logging.info(f"Extracted content from {len(processed_docs)} pages.")
    return processed_docs

# --- NEW: Combined Crawl and Process Function ---
def crawl_and_process(start_url: str, max_pages: int = 50) -> list[dict]:
    """
    Crawls a website starting from start_url, extracts meaningful content,
    and returns a list of processed document dictionaries.

    Args:
        start_url: The URL to begin crawling from.
        max_pages: The maximum number of pages to crawl.

    Returns:
        A list of dictionaries, where each dict has 'url', 'title', 'text'.
        Returns an empty list if crawling or processing fails significantly.
    """
    logging.info(f"--- Starting Crawl & Process for {start_url} (Max Pages: {max_pages}) ---")
    website_data = crawl_website(start_url, max_pages)
    if not website_data:
        logging.error("Crawling failed or yielded no data.")
        return []

    processed_documents = process_crawled_data(website_data)
    if not processed_documents:
        logging.error("Processing failed to extract content from any crawled page.")
        return []

    logging.info(f"--- Crawl & Process Completed: {len(processed_documents)} documents processed ---")
    return processed_documents

# --- NEW: Text Chunking Function ---
def chunk_documents(processed_docs: list[dict], chunk_size: int = 1000, chunk_overlap: int = 150) -> list[Document]:
    """
    Splits the text content of processed documents into smaller chunks.

    Args:
        processed_docs: List of dictionaries from crawl_and_process.
        chunk_size: The target size for each text chunk (in characters).
        chunk_overlap: The number of characters to overlap between consecutive chunks.

    Returns:
        A list of LangChain Document objects, each representing a chunk with metadata.
    """
    logging.info(f"--- Starting Document Chunking (Size: {chunk_size}, Overlap: {chunk_overlap}) ---")
    if not processed_docs:
        logging.warning("No processed documents provided for chunking.")
        return []

    # Using RecursiveCharacterTextSplitter is generally recommended
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=True, # Useful for potential context referencing later
    )

    all_chunks = []
    for doc in processed_docs:
        if not doc.get('text'): # Skip docs that somehow lost text
             logging.warning(f"Skipping document with missing text: {doc.get('url', 'Unknown URL')}")
             continue

        # Create LangChain Documents with metadata preserved
        # We split one document's text at a time to keep metadata association clear
        chunks = text_splitter.create_documents(
            texts=[doc['text']], # Pass text as a list
            metadatas=[{'url': doc['url'], 'title': doc['title']}] # Pass metadata as a list matching texts
        )
        all_chunks.extend(chunks)

    logging.info(f"--- Chunking Completed: {len(processed_docs)} documents split into {len(all_chunks)} chunks ---")
    return all_chunks

# --- NEW: Embedding Model Function ---
def get_embedding_model(model_name: str):
    """
    Initializes and returns a SentenceTransformer embedding model wrapped by LangChain.

    Args:
        model_name: The name of the sentence-transformer model to use
                    (e.g., 'all-MiniLM-L6-v2').

    Returns:
        A LangChain Embeddings object.
    """
    logging.info(f"--- Loading Embedding Model: {model_name} ---")
    try:
        # Use the Langchain wrapper for consistency
        # It will download the model on first use if not cached
        embeddings = SentenceTransformerEmbeddings(
            model_name=model_name,
            cache_folder=os.getenv("SENTENCE_TRANSFORMERS_HOME") # Optional: Specify cache dir via env var
            # Add other SentenceTransformer options if needed, e.g., device='cuda'
        )
        # You could add a small test encode here if needed: embeddings.embed_query("test")
        logging.info("--- Embedding Model Loaded Successfully ---")
        return embeddings
    except Exception as e:
        logging.error(f"Failed to load embedding model '{model_name}': {e}", exc_info=True)
        raise # Re-raise the exception to halt the process if embeddings fail

# --- NEW: Vector Store Creation Function ---
def create_faiss_vector_store(chunked_docs: list[Document], embeddings, save_path: str = None) -> FAISS | None :
    """
    Creates a FAISS vector store from document chunks and their embeddings.

    Args:
        chunked_docs: List of LangChain Document objects (the chunks).
        embeddings: The initialized LangChain Embeddings object.
        save_path: Optional path to save the created FAISS index locally.

    Returns:
        A LangChain FAISS vector store object, or None if creation fails.
    """
    logging.info(f"--- Starting FAISS Vector Store Creation ---")
    if not chunked_docs:
        logging.warning("No document chunks provided for vector store creation.")
        return None
    if not embeddings:
        logging.error("No embedding model provided for vector store creation.")
        return None

    try:
        # FAISS.from_documents handles embedding generation and indexing
        logging.info(f"Creating index from {len(chunked_docs)} chunks...")
        vector_store = FAISS.from_documents(documents=chunked_docs, embedding=embeddings)
        logging.info("--- FAISS Index Created in Memory ---")

        # Save the index locally if a path is provided
        if save_path:
            try:
                vector_store.save_local(save_path)
                logging.info(f"--- FAISS Index Saved Locally to: {save_path} ---")
            except Exception as e:
                logging.error(f"Failed to save FAISS index to {save_path}: {e}", exc_info=True)
                # Continue with the in-memory store even if saving fails

        return vector_store

    except Exception as e:
        logging.error(f"Failed to create FAISS vector store: {e}", exc_info=True)
        return None

# --- Main Execution ---
# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crawl website, process content, chunk, embed, and create FAISS index.")
    parser.add_argument("--url", required=True, help="The starting URL of the help website.")
    parser.add_argument("--max_pages", type=int, default=25, help="Maximum number of pages to crawl.")
    parser.add_argument("--force_reindex", action='store_true', help="Force crawling and re-indexing even if a local index exists.")
    parser.add_argument("--chunk_size", type=int, default=1000, help="Target size for text chunks.")
    parser.add_argument("--chunk_overlap", type=int, default=150, help="Overlap between text chunks.")

    args = parser.parse_args()

    # --- Decide whether to load existing index or re-index ---
    vector_store = None
    embedding_model = None # Initialize embedding_model

    # Check if index exists and we are not forcing re-index
    if os.path.exists(vector_store_path) and not args.force_reindex:
        logging.info(f"--- Found existing index at {vector_store_path}. Loading... ---")
        try:
            # Need embedding model to load the index
            embedding_model = get_embedding_model(embedding_model_name) # Load embeddings first
            vector_store = FAISS.load_local(
                vector_store_path,
                embedding_model,
                allow_dangerous_deserialization=True # Required for loading pickle files safely
            )
            logging.info("--- Existing FAISS Index Loaded Successfully ---")
        except Exception as e:
            logging.error(f"Failed to load existing index from {vector_store_path}: {e}", exc_info=True)
            logging.warning("Proceeding with re-indexing.")
            vector_store = None

    # If index wasn't loaded (doesn't exist or loading failed or forced re-index)
    if vector_store is None:
        logging.info(f"--- Starting Full Indexing Pipeline ---")

        # 1. Crawl and Process
        processed_documents = crawl_and_process(args.url, args.max_pages)

        if processed_documents:
            # 2. Chunk Documents
            chunked_documents = chunk_documents(processed_documents, args.chunk_size, args.chunk_overlap)

            if chunked_documents:
                # 3. Get Embedding Model (Load only if needed for indexing)
                if not embedding_model: # Load only if not loaded already
                    embedding_model = get_embedding_model(embedding_model_name)

                # 4. Create and Save FAISS Vector Store
                vector_store = create_faiss_vector_store(
                    chunked_docs=chunked_documents,
                    embeddings=embedding_model,
                    save_path=vector_store_path
                )
            else:
                logging.error("Chunking resulted in no documents. Cannot create vector store.")
        else:
            logging.error("Crawling and processing yielded no documents. Cannot create vector store.")

    # --- Ready for Q&A (Using Retriever) --- # MODIFIED SECTION
    if vector_store:
        print("\n--- Index Ready ---")
        print(f"Vector store contains {vector_store.index.ntotal} chunks.")
        print(f"Using embedding model: {embedding_model_name}") # Corrected variable name
        print(f"Index location: {vector_store_path if os.path.exists(vector_store_path) else 'In Memory'}")

        # --- Create and Test Retriever ---
        print("\n--- Creating and Testing Retriever ---")
        try:
            # Create a retriever object from the vector store
            # search_kwargs={"k": 4} tells the retriever to return the top 4 results
            retriever = vector_store.as_retriever(search_kwargs={"k": 4})
            logging.info("Retriever created successfully.")

            # --- Example Query ---
            # Use a query relevant to the content you crawled (e.g., from help.zluri.com)
            # Or use the user's example: query = "what is langgraph" (might not find anything relevant)
            # Example for help.zluri.com:
            query = "How do I integrate with Google Workspace?"
            # Example for help.slack.com:
            # query = "How do channels work?"

            print(f"\nInvoking retriever with query: '{query}'")
            # Use the retriever's invoke method to find relevant documents
            # This embeds the query and performs the similarity search in FAISS
            retrieved_docs = retriever.invoke(query)

            # --- Display Retrieved Documents ---
            print(f"\nRetrieved {len(retrieved_docs)} documents:")
            if not retrieved_docs:
                print("No relevant documents found for this query.")
            else:
                for i, doc in enumerate(retrieved_docs):
                    print(f"\n--- Document {i+1} ---")
                    print(f"Source URL: {doc.metadata.get('url', 'N/A')}")
                    print(f"Page Title: {doc.metadata.get('title', 'N/A')}")
                    # Clean up whitespace for preview
                    content_preview = ' '.join(doc.page_content.split())
                    print(f"Content Snippet: {content_preview[:300]}...") # Show first 300 chars

            # --- Next Steps Reminder ---
            logging.info("\n--- Next Steps (Q&A Interface) ---")
            logging.info("1. Build a loop to accept user questions.")
            logging.info("2. Use the 'retriever' created above to get relevant docs for each question.")
            logging.info("3. Use an LLM (like OpenAI GPT) with the retrieved docs as context to generate an answer (RAG).")
            logging.info("4. Print the final answer and source URLs.")

        except Exception as e:
            logging.error(f"Error during retriever creation or invocation: {e}", exc_info=True)

    else:
        logging.error("--- Failed to create or load a vector store. Cannot proceed. ---")