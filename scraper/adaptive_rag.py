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
from pprint import pprint

# --- Core Langchain & Document Processing ---
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.schema import Document
from langchain import hub
from langchain_core.output_parsers import StrOutputParser

# --- LLM ---
from langchain_groq import ChatGroq

# --- Tools ---
from langchain_community.tools.tavily_search import TavilySearchResults

# --- Graph & State ---
from typing import List, Literal, Sequence, Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages # Although not used in this state, good practice

# --- Prompts & Parsing ---
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from pydantic import BaseModel, Field

# #############################################################################
# Load Environment Variables & Basic Config
# #############################################################################
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
tavily_api_key = os.getenv("TAVILY_API_KEY")
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
embedding_model_name = os.getenv("EMBEDDING_MODEL_NAME", DEFAULT_EMBEDDING_MODEL)
DEFAULT_VECTOR_STORE_PATH = "./faiss_adaptive_rag_store" # Use a different default path
vector_store_path = os.getenv("VECTOR_STORE_PATH", DEFAULT_VECTOR_STORE_PATH)
GROQ_MODEL_NAME = "llama3-8b-8192" # Or "mixtral-8x7b-32768" etc.

# #############################################################################
# Logging Setup
# #############################################################################
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# Suppress excessive logs from libraries if needed
# logging.getLogger("httpx").setLevel(logging.WARNING)
# logging.getLogger("httpcore").setLevel(logging.WARNING)

# #############################################################################
# Crawler, Processor, Indexing Functions (Keep as previously corrected)
# #############################################################################
# Assume the functions are defined here:
# HEADERS, MAIN_CONTENT_SELECTORS, NOISE_TAGS, TEXT_BEARING_TAGS
# is_valid_url, fetch_page, crawl_website, format_list_item, format_table,
# get_cleaned_text, process_element_recursive, extract_meaningful_content,
# process_crawled_data, crawl_and_process, chunk_documents,
# get_embedding_model, create_faiss_vector_store
# ... (Full implementations omitted for brevity - paste your corrected versions here) ...
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
    # Add site-specific noise selectors if needed by inspecting the target URL's HTML
]
TEXT_BEARING_TAGS = ['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li', 'td', 'th', 'pre', 'code', 'span', 'div', 'label', 'figcaption']
def is_valid_url(url): parsed = urlparse(url); return bool(parsed.scheme) and bool(parsed.netloc)
def format_table(tag):
    """Formats a <table> tag into a simple text representation."""
    rows = []
    # Loop through each table row <tr>
    for r in tag.find_all('tr'):
        # Create a list of cleaned text from each cell <td> or <th> in the row
        cells = [get_cleaned_text(c) for c in r.find_all(['td', 'th'])]
        # Check if any cell in the row has content after cleaning
        if any(cells):
            # If relevant, join cells and append to rows
            rows.append(" | ".join(cells))
    # Return joined rows.
    return "\n".join(rows)
def fetch_page(url, session, timeout=10):
    try:
        response = session.get(url, headers=HEADERS, timeout=timeout, allow_redirects=True); response.raise_for_status()
        ct = response.headers.get('content-type', '').lower();
        if 'text/html' not in ct or len(response.content) < 100: return None, response.url
        response.encoding = response.apparent_encoding or 'utf-8'; return response.text, response.url
    except Exception as e: logging.debug(f"Fetch error {url}: {e}"); return None, url
def crawl_website(start_url, max_pages):
    if not is_valid_url(start_url): logging.error(f"Invalid start URL: {start_url}"); return None
    base_domain = urlparse(start_url).netloc; urls_to_visit = deque([start_url]); visited_urls = set(); crawled_data = {}
    with requests.Session() as session:
        while urls_to_visit and len(crawled_data) < max_pages:
            current_url = urls_to_visit.popleft(); norm_url = urlparse(current_url)._replace(fragment="").geturl()
            if norm_url in visited_urls: continue
            visited_urls.add(norm_url); logging.info(f"Crawling: {current_url} ({len(crawled_data)+1}/{max_pages})")
            html_content, final_url = fetch_page(current_url, session); norm_final = urlparse(final_url)._replace(fragment="").geturl()
            visited_urls.add(norm_final)
            if html_content and urlparse(final_url).netloc == base_domain:
                if final_url not in crawled_data and len(crawled_data) < max_pages: crawled_data[final_url] = html_content
                soup = BeautifulSoup(html_content, 'html.parser')
                for link in soup.find_all('a', href=True):
                    next_url = urljoin(final_url, link['href']); p_next = urlparse(next_url); norm_next = p_next._replace(fragment="").geturl()
                    if p_next.scheme in ['http', 'https'] and p_next.netloc == base_domain and norm_next not in visited_urls and norm_next not in urls_to_visit: urls_to_visit.append(norm_next)
            time.sleep(0.05) # Be gentle
    logging.info(f"Crawled {len(crawled_data)} pages."); return crawled_data
def format_list_item(tag, index):
    """Formats an <li> tag based on its parent (<ol> or <ul>)."""
    parent_type = tag.parent.name if tag.parent else 'ul'
    prefix = f"{index + 1}. " if parent_type == 'ol' else "- "

    # Assign text first
    text = get_cleaned_text(tag)

    # Now check the assigned value
    return prefix + text if text else "" # Return prefix + text only if text is not emptydef format_table(tag):
    """Formats a <table> tag into a simple text representation."""
    rows = []
    # Loop through each table row <tr>
    for r in tag.find_all('tr'):
        # Create a list of cleaned text from each cell <td> or <th> in the row
        # This line should be indented one level under the 'for'
        cells = [get_cleaned_text(c) for c in r.find_all(['td', 'th'])]

        # Check if any cell in the row has content after cleaning
        # This line should be at the SAME indentation level as 'cells = ...'
        if any(cells):
            # If relevant, join cells and append to rows
            # This line should be indented one level under the 'if'
            rows.append(" | ".join(cells))

    # Return joined rows. This should be aligned with the 'for' loop.
    return "\n".join(rows)
def get_cleaned_text(element):
    """Extracts and cleans text from a BeautifulSoup element. Always returns a string."""
    if not element:
        return "" # Return empty string if element is None
    try:
        # Get text, using a space separator, and strip leading/trailing whitespace
        text = element.get_text(separator=' ', strip=True)
        # Decode HTML entities like &
        text = html.unescape(text)
        # Normalize whitespace: replace multiple spaces/newlines with a single space
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    except Exception as e:
        # Log the error if needed, but return an empty string to prevent downstream errors
        logging.debug(f"Error in get_cleaned_text for element {str(element)[:50]}: {e}")
        return "" # Return empty string on error
def process_element_recursive(element, extracted_blocks, processed):
    if element in processed or not element.name or not element.parent: return
    txt = ""; process_children = True; name = element.name
    if name in ['ul', 'ol']: items=element.find_all('li', recursive=False); txt="\n".join(fmt for i, item in enumerate(items) if (fmt:=format_list_item(item,i))); process_children=False
    elif name == 'table': txt=format_table(element); process_children=False
    elif name.startswith('h') and name[1].isdigit():
        # Get the heading level (1-6)
        level = int(name[1])
        # Get the cleaned text content of the heading element
        cleaned = get_cleaned_text(element)
        # Only format if there's actual text content
        if cleaned:
            txt = "#" * level + " " + cleaned
        # Since get_cleaned_text processes descendants, don't process children separately
        process_children = False
    elif name in ['p', 'pre', 'blockquote']: txt=get_cleaned_text(element); process_children=False
    elif name in TEXT_BEARING_TAGS:
        if process_children:
            for child in element.find_all(True, recursive=False): process_element_recursive(child, extracted_blocks, processed)
        direct_text = ''.join(element.find_all(string=True, recursive=False)).strip()
        block_children = {'p','h1','h2','h3','h4','h5','h6','ul','ol','table','pre','blockquote'}
        if direct_text and not element.find(lambda tag: tag.name in block_children): txt=get_cleaned_text(element)
    if txt: extracted_blocks.append(txt.strip())
    if txt or not process_children: processed.add(element); processed.update(element.find_all(True))
    elif process_children:
        for child in element.find_all(True, recursive=False): process_element_recursive(child, extracted_blocks, processed)
def extract_meaningful_content(url, html_content):
    if not html_content: return None
    try:
        soup = BeautifulSoup(html_content, 'html.parser'); title = get_cleaned_text(soup.title) or "No Title"
        for sel in NOISE_TAGS:
            for el in soup.select(sel): el.decompose()
        for c in soup.find_all(string=lambda t: isinstance(t, Comment)): c.extract()
        main = None;
        for sel in MAIN_CONTENT_SELECTORS:
            if main := soup.select_one(sel): break
        if not main: main = soup.body
        if not main: return None
        blocks = []; processed = set();
        for el in main.find_all(True, recursive=False): process_element_recursive(el, blocks, processed)
        full_text = "\n\n".join(b for b in blocks if b); full_text = re.sub(r'\n{3,}', '\n\n', full_text).strip()
        if len(full_text) < 30: return None
        logging.debug(f"Extracted ~{len(full_text)} chars from {url}"); return {"url": url, "title": title, "text": full_text}
    except Exception as e: logging.error(f"Error processing {url}: {e}"); return None
# --- Start of First Attempt/Fragment ---
def process_crawled_data(website_data):
    processed_docs = [];
    if not website_data: return processed_docs
    for url, html_content in website_data.items():
        if extracted_data := extract_meaningful_content(url, html_content): processed_docs.append(extracted_data)
    logging.info(f"Extracted content from {len(processed_docs)} pages."); return processed_docs
def crawl_and_process(url, max_pages=50): data=crawl_website(url, max_pages); return process_crawled_data(data) if data else []
def chunk_documents(docs, chunk_size=1000, chunk_overlap=150):
    splitter=RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len, add_start_index=True)
    chunks=[]
    for doc in docs:
        if txt := doc.get('text'): chunks.extend(splitter.create_documents([txt], metadatas=[{'url':doc['url'], 'title':doc['title']}]))
    logging.info(f"Split into {len(chunks)} chunks."); return chunks
def get_embedding_model(model_name):
    logging.info(f"Loading embeddings: {model_name}"); emb = None
    try: emb = SentenceTransformerEmbeddings(model_name=model_name, cache_folder=os.getenv("SENTENCE_TRANSFORMERS_HOME"))
    except Exception as e: logging.error(f"Embedding load failed: {e}", exc_info=True); raise
    logging.info("Embeddings loaded."); return emb
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
    logging.info(f"--- Starting FAISS Vector Store Creation from {len(chunked_docs)} chunks ---")
    if not chunked_docs:
        logging.warning("No document chunks provided for vector store creation.")
        return None
    if not embeddings:
        logging.error("No embedding model provided for vector store creation.")
        return None

    vector_store = None # Initialize to None
    try:
        # FAISS.from_documents handles embedding generation and indexing
        logging.info(f"Embedding and indexing {len(chunked_docs)} chunks...")
        vector_store = FAISS.from_documents(documents=chunked_docs, embedding=embeddings)
        # If the above line succeeds, the in-memory store is created.

        # Save the index locally if a path is provided (still within the try block, as saving can also fail)
        if save_path and vector_store:
            try:
                vector_store.save_local(save_path)
                logging.info(f"--- FAISS Index Saved Locally to: {save_path} ---")
            except Exception as e_save:
                # Log the saving error, but don't necessarily fail the whole function
                # We still have the in-memory store which might be usable.
                logging.error(f"Failed to save FAISS index to {save_path}: {e_save}", exc_info=True)
                # Decide if you want to return None here or just proceed with the in-memory store.
                # Let's proceed for now.

    except Exception as e_create:
        # This catches errors during FAISS.from_documents()
        logging.error(f"Failed to create FAISS vector store: {e_create}", exc_info=True)
        return None # Return None if the core index creation fails

    # If the try block completed the FAISS.from_documents() call successfully,
    # vector_store should hold the index (even if saving failed).
    if vector_store:
        logging.info("--- FAISS Store Created Successfully (in memory) ---")
        return vector_store
    else:
        # This case should technically be covered by the except block returning None,
        # but added for clarity.
        logging.error("Vector store creation resulted in None, despite no explicit exception caught.")
        return None

# #############################################################################
# Adaptive RAG Components
# #############################################################################

# --- LLM Initialization ---
llm = None
if not groq_api_key:
    logging.error("GROQ_API_KEY missing. Adaptive RAG agent will not function.")
else:
    try:
        # Use a model known to be good at following instructions, e.g., Mixtral or Llama3
        llm = ChatGroq(temperature=0, model_name=GROQ_MODEL_NAME)
        llm.invoke("Respond with 'OK'") # Test call
        logging.info(f"Groq LLM ({GROQ_MODEL_NAME}) initialized successfully.")
    except Exception as e:
        logging.error(f"Failed to initialize Groq LLM: {e}. Agent will not function.")

# --- Tool Initialization ---
web_search_tool = None
if not tavily_api_key:
    logging.warning("TAVILY_API_KEY missing. Web search functionality will be disabled.")
else:
    web_search_tool = TavilySearchResults(k=3, tavily_api_key=tavily_api_key)
    logging.info("Tavily web search tool initialized.")

# --- Retriever (initialized later after vector store is ready) ---
retriever = None

# --- RAG Chain ---
rag_chain = None
if llm:
    try:
        prompt = hub.pull("rlm/rag-prompt")
        rag_chain = prompt | llm | StrOutputParser()
        logging.info("RAG generation chain initialized.")
    except Exception as e:
        logging.error(f"Failed to pull RAG prompt or build chain: {e}")

# --- Router ---
class RouteQuery(BaseModel):
    datasource: Literal["vectorstore", "web_search"] = Field(description="Route to 'vectorstore' for specific product documentation queries, or 'web_search' for general queries.")
question_router = None
if llm:
    structured_llm_router = llm.with_structured_output(RouteQuery)
    # System prompt dynamically set later
    route_prompt_template = """You are an expert at routing a user question to a vectorstore or web search.
The vectorstore contains documents related to {topic}.
Use the vectorstore for questions specifically about {topic_short}. Otherwise, use web-search.
Based on the question: '{question}', choose the best datasource."""

# --- Retrieval Grader ---
class GradeDocuments(BaseModel):
    binary_score: str = Field(description="Is the document relevant? 'yes' or 'no'.")
retrieval_grader = None
if llm:
    structured_llm_retrieval_grader = llm.with_structured_output(GradeDocuments)
    system_ret_grade = "You are a grader assessing relevance of a retrieved document to a user question. Grade 'yes' if it contains keywords or semantic meaning related to the question, otherwise 'no'. Goal is to filter errors."
    grade_prompt = ChatPromptTemplate.from_messages(
        [("system", system_ret_grade), ("human", "Document:\n{document}\n\nQuestion: {question}")]
    )
    retrieval_grader = grade_prompt | structured_llm_retrieval_grader

# --- Hallucination Grader ---
class GradeHallucinations(BaseModel):
    binary_score: str = Field(description="Is the answer grounded in the facts? 'yes' or 'no'.")
hallucination_grader = None
if llm:
    structured_llm_hallu_grader = llm.with_structured_output(GradeHallucinations)
    system_hallu = "You are a grader assessing whether an answer is grounded in/supported by a set of facts. Score 'yes' if grounded, 'no' otherwise."
    hallucination_prompt = ChatPromptTemplate.from_messages(
        [("system", system_hallu), ("human", "Facts:\n{documents}\n\nAnswer: {generation}")]
    )
    hallucination_grader = hallucination_prompt | structured_llm_hallu_grader

# --- Answer Grader ---
class GradeAnswer(BaseModel):
    binary_score: str = Field(description="Does the answer address the question? 'yes' or 'no'.")
answer_grader = None
if llm:
    structured_llm_ans_grader = llm.with_structured_output(GradeAnswer)
    system_ans = "You are a grader assessing whether an answer addresses/resolves a question. Score 'yes' if it does, 'no' otherwise."
    answer_prompt = ChatPromptTemplate.from_messages(
        [("system", system_ans), ("human", "Question:\n{question}\n\nAnswer: {generation}")]
    )
    answer_grader = answer_prompt | structured_llm_ans_grader

# --- Question Rewriter ---
question_rewriter = None
if llm:
    system_rew = "You are a question re-writer optimizing a question for vectorstore retrieval based on underlying semantic intent/meaning."
    re_write_prompt = ChatPromptTemplate.from_messages(
        [("system", system_rew), ("human", "Initial question: {question}\nFormulate an improved question.")]
    )
    question_rewriter = re_write_prompt | llm | StrOutputParser()

# #############################################################################
# Graph State & Nodes
# #############################################################################

class GraphState(TypedDict):
    """Represents the state of our graph."""
    question: str           # Original or rewritten question
    generation: str | None  # LLM generation
    documents: List[Document] # List of retrieved documents
    web_searched: bool      # Flag indicating if web search was used

def retrieve_node(state):
    """Retrieve documents from vector store."""
    print("--- NODE: RETRIEVE ---"); question = state["question"]
    if retriever is None: print("ERROR: Retriever not initialized."); return {"documents": [], "question": question}
    documents = retriever.invoke(question)
    print(f"Retrieved {len(documents)} docs.")
    return {"documents": documents, "question": question, "web_searched": False} # Reset web search flag

def grade_documents_node(state):
    """Determines whether the retrieved documents are relevant."""
    print("--- NODE: GRADE DOCUMENTS ---"); question = state["question"]; documents = state["documents"]
    if not documents: return {"documents": [], "question": question} # Skip if no docs
    if retrieval_grader is None: print("WARN: Retrieval grader not ready."); return {"documents": documents, "question": question}

    filtered_docs = []
    for d in documents:
        try:
            score = retrieval_grader.invoke({"question": question, "document": d.page_content})
            grade = score.binary_score.lower()
            if grade == "yes": print("+ Relevant doc found"); filtered_docs.append(d)
            else: print("- Irrelevant doc discarded")
        except Exception as e: print(f"Error grading doc: {e}"); # Optionally keep doc if grading fails?
    print(f"Filtered down to {len(filtered_docs)} relevant docs.")
    return {"documents": filtered_docs, "question": question}

def generate_node(state):
    """Generate answer using RAG."""
    print("--- NODE: GENERATE ---"); question = state["question"]; documents = state["documents"]
    if rag_chain is None: print("ERROR: RAG chain not ready."); return {"generation": None, **state} # Pass existing state
    if not documents: print("WARN: No documents to generate from."); generation = "I couldn't find relevant information in the documentation to answer that."
    else:
        try: generation = rag_chain.invoke({"context": documents, "question": question})
        except Exception as e: print(f"Error during RAG generation: {e}"); generation = "Error generating answer."
    print(f"Generated response snippet: {generation[:100]}...")
    return {"documents": documents, "question": question, "generation": generation}

def transform_query_node(state):
    """Transform the query to produce a better question."""
    print("--- NODE: TRANSFORM QUERY ---"); question = state["question"]
    if question_rewriter is None: print("WARN: Question rewriter not ready."); return state # Return unchanged state
    try: better_question = question_rewriter.invoke({"question": question})
    except Exception as e: print(f"Error rewriting question: {e}"); better_question = question # Fallback
    print(f"Rewritten question: {better_question}")
    # Keep original docs if any, but update question
    return {"documents": state.get("documents", []), "question": better_question}

def web_search_node(state):
    """Web search based question."""
    print("--- NODE: WEB SEARCH ---"); question = state["question"]
    if web_search_tool is None: print("ERROR: Web search tool not ready."); return {"documents": [], **state}
    try:
        docs = web_search_tool.invoke({"query": question})
        web_results = "\n\n".join([d["content"] for d in docs])
        web_doc = Document(page_content=web_results, metadata={"source": "web_search"})
        print(f"Web search snippet: {web_results[:100]}...")
        return {"documents": [web_doc], "question": question, "web_searched": True} # Flag that web search was used
    except Exception as e: print(f"Error during web search: {e}"); return {"documents": [], **state}

# #############################################################################
# Graph Edges (Conditional Logic)
# #############################################################################

def route_question_edge(state):
    """Route question to web search or vector store."""
    print("--- EDGE: ROUTE QUESTION ---"); question = state["question"]
    if question_router is None: print("WARN: Router not ready, defaulting to vectorstore."); return "vectorstore"
    try:
        source = question_router.invoke({"question": question})
        print(f"Routing decision: {source.datasource}")
        # Only allow web search if the tool is available
        if source.datasource == "web_search" and web_search_tool: return "web_search"
        else: return "vectorstore" # Default to vectorstore if routing fails or suggests web search without tool
    except Exception as e: print(f"Error routing question: {e}. Defaulting to vectorstore."); return "vectorstore"

def decide_to_generate_edge(state):
    """Determines whether to generate an answer or re-generate query."""
    print("--- EDGE: DECIDE TO GENERATE ---")
    if state.get("web_searched"): # If web search results came in, go straight to generate
         print("Decision: Generate from web search results.")
         return "generate"

    filtered_documents = state["documents"]
    if not filtered_documents: print("Decision: No relevant docs, transform query."); return "transform_query"
    else: print("Decision: Relevant docs found, generate answer."); return "generate"

def grade_generation_edge(state):
    """Determines whether the generation is grounded and answers the question."""
    print("--- EDGE: GRADE GENERATION ---"); question = state["question"]; documents = state["documents"]; generation = state["generation"]

    if hallucination_grader is None or answer_grader is None:
        print("WARN: Graders not ready, accepting generation.")
        return END # End the graph if graders aren't available

    if not generation: # Handle cases where generation failed
        print("Decision: Generation failed, transform query to retry.")
        return "transform_query"

    try:
        # Check for hallucinations
        hallu_score = hallucination_grader.invoke({"documents": documents, "generation": generation})
        if hallu_score.binary_score.lower() == "no":
            print("Decision: Generation hallucinates, try generating again.")
            # Potential loop risk: maybe transform query instead? For now, retry generation
            return "generate" # Or "transform_query"

        print("--- Generation Grounded ---")
        # Check if answer addresses the question
        ans_score = answer_grader.invoke({"question": question, "generation": generation})
        if ans_score.binary_score.lower() == "yes":
            print("Decision: Generation is useful.")
            return END # Useful answer, end the graph
        else:
            print("Decision: Generation doesn't address question, transform query.")
            return "transform_query" # Try rewriting the question

    except Exception as e:
        print(f"Error during generation grading: {e}. Accepting generation.")
        return END # End if grading fails

# #############################################################################
# Main Execution Logic
# #############################################################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Adaptive RAG Q&A Agent.")
    parser.add_argument("--url", required=True, help="The starting URL.")
    parser.add_argument("--max_pages", type=int, default=25, help="Max pages.")
    parser.add_argument("--force_reindex", action='store_true', help="Force re-indexing.")
    parser.add_argument("--chunk_size", type=int, default=1000, help="Chunk size.")
    parser.add_argument("--chunk_overlap", type=int, default=150, help="Chunk overlap.")
    args = parser.parse_args()

    # --- Check Core Dependencies ---
    if not llm:
        exit("Error: Groq LLM failed to initialize. Exiting.")
    if not tavily_api_key:
        logging.warning("Tavily API Key missing - web search disabled.")
    if not groq_api_key:
         exit("Error: Groq API Key missing. Exiting.")


    # --- Indexing ---
    vector_store = None
    embedding_model = None
    if os.path.exists(vector_store_path) and not args.force_reindex:
        try:
            logging.info(f"Loading index: {vector_store_path}")
            embedding_model = get_embedding_model(embedding_model_name)
            vector_store = FAISS.load_local(vector_store_path, embedding_model, allow_dangerous_deserialization=True)
            logging.info("Index loaded.")
        except Exception as e: logging.error(f"Failed load index: {e}", exc_info=True); vector_store = None

    if vector_store is None:
        logging.info("Running indexing pipeline...")
        processed_docs = crawl_and_process(args.url, args.max_pages)
        if processed_docs:
            chunked_docs = chunk_documents(processed_docs, args.chunk_size, args.chunk_overlap)
            if chunked_docs:
                if not embedding_model: embedding_model = get_embedding_model(embedding_model_name)
                vector_store = create_faiss_vector_store(chunked_docs, embedding_model, vector_store_path)
            else: logging.error("Chunking failed.")
        else: logging.error("Crawling/Processing failed.")

    # --- Agent Setup ---
    if not vector_store:
        exit("Error: Vector store not available. Exiting.")

    # Initialize retriever now that vector_store is ready
    retriever = vector_store.as_retriever(search_kwargs={"k": 3}) # Retrieve top 3
    logging.info("FAISS Retriever initialized.")

    # Dynamically set the router prompt topic based on the URL
    try:
        site_name = urlparse(args.url).netloc or "the provided documentation"
        topic_short = site_name.split('.')[1] if '.' in site_name else site_name # e.g., 'zluri' or 'slack'
        dynamic_route_system_prompt = route_prompt_template.format(topic=f"help documentation for {site_name}", topic_short=topic_short, question="{question}")
        route_prompt = ChatPromptTemplate.from_messages([("system", dynamic_route_system_prompt), ("human", "{question}")])
        question_router = route_prompt | structured_llm_router
        logging.info(f"Question router configured for topic: {site_name}")
    except Exception as e:
        logging.error(f"Failed to create dynamic router prompt: {e}. Routing might be impaired.")
        # Fallback or exit? For now, let it proceed, might default to vectorstore

    # --- Build Graph ---
    workflow = StateGraph(GraphState)

    workflow.add_node("web_search", web_search_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("grade_documents", grade_documents_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("transform_query", transform_query_node)

    workflow.add_conditional_edges(START, route_question_edge, {"web_search": "web_search", "vectorstore": "retrieve"})
    workflow.add_edge("web_search", "generate") # Directly generate after web search
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges("grade_documents", decide_to_generate_edge, {"transform_query": "transform_query", "generate": "generate"})
    # Loop back to retrieve after transforming query
    workflow.add_edge("transform_query", "retrieve")
    # Conditional edge after generation based on grading
    workflow.add_conditional_edges("generate", grade_generation_edge, {"transform_query": "transform_query", END: END}) # Removed "not supported": "generate" edge to avoid potential infinite generation loops. Go back to rewrite.

    # Compile the graph
    try:
        app = workflow.compile()
        logging.info("Adaptive RAG graph compiled successfully.")
    except Exception as e:
        exit(f"Error compiling graph: {e}")

    # --- Q&A Loop ---
    print("\n--- Adaptive RAG Agent Ready ---")
    print("Enter your question (type 'exit' or 'quit' to stop):")
    while True:
        user_input = input("> ").strip()
        if user_input.lower() in ["exit", "quit"]: break
        if not user_input: continue

        print("\n--- Running Agent ---")
        inputs = {"question": user_input}
        final_generation = "Agent finished without generating a final answer." # Default message
        try:
            for output in app.stream(inputs, {"recursion_limit": 15}): # Increased limit slightly
                for key, value in output.items():
                    print(f"\nOutput from node '{key}':")
                    # Optionally print state details for debugging
                    # pprint(value, indent=2, width=100)
                print("\n---\n")
            # Try to get the final generation from the last state
            if value and 'generation' in value and value['generation']:
                 final_generation = value['generation']
            elif value and 'question' in value: # Maybe it ended after rewrite?
                 final_generation = f"(Agent stopped. Last question state: '{value['question']}')"

        except Exception as e:
            print(f"\n--- Error during graph execution: {e} ---")
            final_generation = "An error occurred while processing your question."

        print("\n--- Agent Final Response ---")
        print(final_generation)
        print("-" * 40)