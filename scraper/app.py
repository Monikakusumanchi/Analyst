import streamlit as st
import argparse # Keep for potential future CLI, but not used by Streamlit directly
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
import io # For capturing print statements
import contextlib # For redirecting stdout

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
# Note: add_messages isn't used in this adaptive RAG state, but keep import if needed later
# from langgraph.graph.message import add_messages

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
# Logging Setup (Configure basicConfig, but output primarily via Streamlit)
# #############################################################################
# Basic config is still useful for library logs potentially going to console
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# #############################################################################
# Crawler, Processor, Indexing Functions (Paste your fully corrected versions here)
# #############################################################################
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

def get_cleaned_text(element):
    if not element: return ""
    try:
        text = element.get_text(separator=' ', strip=True); text = html.unescape(text)
        text = re.sub(r'\s+', ' ', text).strip(); return text
    except Exception as e: logging.debug(f"Error get_cleaned_text: {e}"); return ""

def format_list_item(tag, index):
    parent_type = tag.parent.name if tag.parent else 'ul'; prefix = f"{index + 1}. " if parent_type == 'ol' else "- "
    text = get_cleaned_text(tag); return prefix + text if text else ""

def process_element_recursive(element, extracted_blocks, processed):
    # PASTE YOUR CORRECTED process_element_recursive function here
    # Ensure it calls the corrected format_table and format_list_item
    if element in processed or not element.name or not element.parent: return
    txt = ""; process_children = True; name = element.name
    if name in ['ul', 'ol']: items=element.find_all('li', recursive=False); txt="\n".join(fmt for i, item in enumerate(items) if (fmt:=format_list_item(item,i))); process_children=False
    elif name == 'table': txt=format_table(element); process_children=False # Ensure format_table call is correct
    elif name.startswith('h') and name[1].isdigit():
        level = int(name[1]); cleaned = get_cleaned_text(element)
        if cleaned: txt = "#" * level + " " + cleaned
        process_children = False
    elif name in ['p', 'pre', 'blockquote']: txt=get_cleaned_text(element); process_children=False
    elif name in TEXT_BEARING_TAGS:
        if process_children:
            for child in element.find_all(True, recursive=False): process_element_recursive(child, extracted_blocks, processed)
        direct_text = ''.join(element.find_all(string=True, recursive=False)).strip()
        block_children = {'p','h1','h2','h3','h4','h5','h6','ul','ol','table','pre','blockquote'}
        # Simplified check to avoid potential errors in lambda
        has_block_child = any(element.find(block) for block in block_children)
        if direct_text and not has_block_child: txt=get_cleaned_text(element)
    if txt: extracted_blocks.append(txt.strip())
    if txt or not process_children: processed.add(element); processed.update(element.find_all(True))
    elif process_children:
        for child in element.find_all(True, recursive=False): process_element_recursive(child, extracted_blocks, processed)

def extract_meaningful_content(url, html_content):
    # PASTE YOUR CORRECTED extract_meaningful_content function here
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
        if not main: logging.warning(f"No main content or body found for {url}"); return None
        blocks = []; processed = set();
        for el in main.find_all(True, recursive=False): process_element_recursive(el, blocks, processed)
        full_text = "\n\n".join(b for b in blocks if b); full_text = re.sub(r'\n{3,}', '\n\n', full_text).strip()
        if len(full_text) < 30: logging.debug(f"Extracted text too short for {url}"); return None
        logging.debug(f"Extracted ~{len(full_text)} chars from {url}"); return {"url": url, "title": title, "text": full_text}
     except Exception as e: logging.error(f"Error processing {url}: {e}", exc_info=True); return None # Log full traceback

def fetch_page(url, session, timeout=10):
    # PASTE YOUR CORRECTED fetch_page function here
    try:
        response = session.get(url, headers=HEADERS, timeout=timeout, allow_redirects=True); response.raise_for_status()
        ct = response.headers.get('content-type', '').lower();
        # More robust check for html
        if 'html' not in ct: logging.debug(f"Skipping non-HTML {url}"); return None, response.url
        # Check length before decoding
        if len(response.content) < 100: logging.debug(f"Skipping potentially empty page {url}"); return None, response.url
        response.encoding = response.apparent_encoding or 'utf-8'; return response.text, response.url
    except Exception as e: logging.debug(f"Fetch error {url}: {e}"); return None, url

def crawl_website(start_url, max_pages):
    # PASTE YOUR CORRECTED crawl_website function here
    if not is_valid_url(start_url): logging.error(f"Invalid start URL: {start_url}"); return None
    base_domain = urlparse(start_url).netloc; urls_to_visit = deque([start_url]); visited_urls = set(); crawled_data = {}
    logging.info(f"Starting crawl from {start_url} (max: {max_pages})")
    with requests.Session() as session:
        page_count = 0
        while urls_to_visit and page_count < max_pages:
            current_url = urls_to_visit.popleft(); norm_url = urlparse(current_url)._replace(fragment="").geturl()
            if norm_url in visited_urls: continue
            # Add before fetch
            visited_urls.add(norm_url)

            # Fetch page - fetch_page handles logging now mostly
            html_content, final_url = fetch_page(current_url, session);
            norm_final = urlparse(final_url)._replace(fragment="").geturl()
            # Also add final url to visited
            visited_urls.add(norm_final)

            if html_content:
                # Check domain again after redirects
                if urlparse(final_url).netloc != base_domain:
                    logging.debug(f"Skipping external redirect: {final_url}")
                    continue

                # Only add if new and within limit
                if final_url not in crawled_data:
                    crawled_data[final_url] = html_content
                    page_count += 1 # Increment count only when adding new page data
                    logging.info(f"Crawled ({page_count}/{max_pages}): {final_url}") # Log final URL

                    soup = BeautifulSoup(html_content, 'html.parser')
                    for link in soup.find_all('a', href=True):
                        next_url = urljoin(final_url, link['href']); p_next = urlparse(next_url); norm_next = p_next._replace(fragment="").geturl()
                        if p_next.scheme in ['http', 'https'] and p_next.netloc == base_domain and norm_next not in visited_urls and norm_next not in urls_to_visit:
                            urls_to_visit.append(norm_next) # Append normalized URL to avoid duplicates in queue
            else:
                 logging.warning(f"Failed to fetch or skipped: {current_url}")

            time.sleep(0.05) # Be gentle

    logging.info(f"Crawled {len(crawled_data)} unique/valid pages."); return crawled_data


def process_crawled_data(website_data):
    # PASTE YOUR CORRECTED process_crawled_data function here
    docs = []
    if not website_data: logging.warning("No website data to process."); return docs
    logging.info(f"Processing content for {len(website_data)} pages...")
    processed_count = 0
    for url, html in website_data.items():
        if data := extract_meaningful_content(url, html):
            docs.append(data)
            processed_count +=1
    logging.info(f"Successfully processed content from {processed_count} pages."); return docs

def crawl_and_process(url, max_pages=50):
    # PASTE YOUR CORRECTED crawl_and_process function here
     logging.info(f"--- Starting Crawl & Process for {url} (Max Pages: {max_pages}) ---")
     website_data = crawl_website(url, max_pages)
     if not website_data: logging.error("Crawling failed or yielded no data."); return []
     processed_documents = process_crawled_data(website_data)
     if not processed_documents: logging.warning("Processing failed to extract content."); return []
     logging.info(f"--- Crawl & Process Completed: {len(processed_documents)} docs ---")
     return processed_documents

def chunk_documents(docs, chunk_size=1000, chunk_overlap=150):
    # PASTE YOUR CORRECTED chunk_documents function here
    logging.info(f"Chunking {len(docs)} documents (size={chunk_size}, overlap={chunk_overlap})")
    splitter=RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len, add_start_index=True)
    chunks=[]
    for i, doc in enumerate(docs):
        if txt := doc.get('text'):
            try:
                doc_chunks = splitter.create_documents([txt], metadatas=[{'url':doc.get('url', ''), 'title':doc.get('title', '')}])
                chunks.extend(doc_chunks)
            except Exception as e:
                logging.error(f"Error chunking doc {i} ({doc.get('url')}): {e}")
    logging.info(f"Split into {len(chunks)} chunks."); return chunks

def get_embedding_model(model_name):
    # PASTE YOUR CORRECTED get_embedding_model function here
    logging.info(f"Loading embeddings: {model_name}"); emb = None
    try:
        emb = SentenceTransformerEmbeddings(model_name=model_name, cache_folder=os.getenv("SENTENCE_TRANSFORMERS_HOME"))
    except Exception as e: logging.error(f"Embedding load failed: {e}", exc_info=True); raise
    logging.info("Embeddings loaded."); return emb

def create_faiss_vector_store(chunks, embeddings, path=None):
    # PASTE YOUR CORRECTED create_faiss_vector_store function here
    logging.info(f"Creating FAISS store ({len(chunks)} chunks)"); vs = None
    if not chunks: logging.warning("No chunks to create vector store from."); return None
    if not embeddings: logging.error("No embedding model for vector store."); return None
    try:
        vs = FAISS.from_documents(documents=chunks, embedding=embeddings)
        logging.info("FAISS store created in memory.")
        if path and vs:
            try: vs.save_local(path); logging.info(f"FAISS saved: {path}")
            except Exception as e_save: logging.error(f"FAISS save failed: {e_save}", exc_info=True)
    except Exception as e_create: logging.error(f"FAISS creation failed: {e_create}", exc_info=True); return None
    return vs


# #############################################################################
# Adaptive RAG Components (Initialize Globals or within Functions)
# #############################################################################

# --- LLM Initialization ---
llm = None
if not groq_api_key:
    st.error("GROQ_API_KEY missing from .env. Agent cannot function.", icon="🚨")
else:
    try:
        llm = ChatGroq(temperature=0, model_name=GROQ_MODEL_NAME, groq_api_key=groq_api_key)
        # Simple test
        # llm.invoke("OK")
        # logging.info(f"Groq LLM ({GROQ_MODEL_NAME}) initialized.")
    except Exception as e:
        st.error(f"Failed to initialize Groq LLM: {e}", icon="🚨")
        logging.error(f"Failed to initialize Groq LLM: {e}")

# --- Tool Initialization ---
web_search_tool = None
if not tavily_api_key:
    logging.warning("TAVILY_API_KEY missing. Web search disabled.")
else:
    try:
        web_search_tool = TavilySearchResults(k=3, tavily_api_key=tavily_api_key)
        logging.info("Tavily web search tool initialized.")
    except Exception as e:
        logging.warning(f"Failed to initialize Tavily: {e}")


# --- RAG Chain & Graders (Initialize if LLM is available) ---
rag_chain = None
question_router = None
retrieval_grader = None
hallucination_grader = None
answer_grader = None
question_rewriter = None

if llm:
    try: prompt = hub.pull("rlm/rag-prompt"); rag_chain = prompt | llm | StrOutputParser(); logging.info("RAG chain OK.")
    except Exception as e: logging.error(f"Failed init RAG chain: {e}")

    try:
        class RouteQuery(BaseModel): datasource: Literal["vectorstore", "web_search"] = Field(description="Route to 'vectorstore' for specific product documentation queries, or 'web_search' for general queries.")
        structured_llm_router = llm.with_structured_output(RouteQuery)
        route_prompt_template = """You are an expert at routing a user question to a vectorstore or web search. The vectorstore contains documents related to {topic}. Use the vectorstore for questions specifically about {topic_short}. Otherwise, use web-search. Based on the question: '{question}', choose the best datasource."""
        # Actual router initialized later when topic is known
    except Exception as e: logging.error(f"Failed init router components: {e}")

    try:
        class GradeDocuments(BaseModel): binary_score: str = Field(description="Is the document relevant? 'yes' or 'no'.")
        structured_llm_ret_grader = llm.with_structured_output(GradeDocuments)
        system_ret_grade = "You are a grader assessing relevance of a retrieved document to a user question. Grade 'yes' if it contains keywords or semantic meaning related to the question, otherwise 'no'. Filter errors."
        grade_prompt = ChatPromptTemplate.from_messages([("system", system_ret_grade), ("human", "Document:\n{document}\n\nQuestion: {question}")])
        retrieval_grader = grade_prompt | structured_llm_ret_grader; logging.info("Retrieval grader OK.")
    except Exception as e: logging.error(f"Failed init retrieval grader: {e}")

    try:
        class GradeHallucinations(BaseModel): binary_score: str = Field(description="Is the answer grounded in the facts? 'yes' or 'no'.")
        structured_llm_hallu_grader = llm.with_structured_output(GradeHallucinations)
        system_hallu = "Assess if an answer is grounded in/supported by facts. Score 'yes' if grounded, 'no' otherwise."
        hallucination_prompt = ChatPromptTemplate.from_messages([("system", system_hallu), ("human", "Facts:\n{documents}\n\nAnswer: {generation}")])
        hallucination_grader = hallucination_prompt | structured_llm_hallu_grader; logging.info("Hallucination grader OK.")
    except Exception as e: logging.error(f"Failed init hallucination grader: {e}")

    try:
        class GradeAnswer(BaseModel): binary_score: str = Field(description="Does the answer address the question? 'yes' or 'no'.")
        structured_llm_ans_grader = llm.with_structured_output(GradeAnswer)
        system_ans = "Assess if an answer addresses/resolves a question. Score 'yes' if it does, 'no' otherwise."
        answer_prompt = ChatPromptTemplate.from_messages([("system", system_ans), ("human", "Question:\n{question}\n\nAnswer: {generation}")])
        answer_grader = answer_prompt | structured_llm_ans_grader; logging.info("Answer grader OK.")
    except Exception as e: logging.error(f"Failed init answer grader: {e}")

    try:
        system_rew = "You are a question re-writer optimizing a question for vectorstore retrieval based on underlying semantic intent."
        re_write_prompt = ChatPromptTemplate.from_messages([("system", system_rew), ("human", "Initial question: {question}\nFormulate an improved question.")])
        question_rewriter = re_write_prompt | llm | StrOutputParser(); logging.info("Question rewriter OK.")
    except Exception as e: logging.error(f"Failed init question rewriter: {e}")


# #############################################################################
# Graph State & Nodes (Define functions)
# #############################################################################

class GraphState(TypedDict):
    """Represents the state of our graph."""
    question: str
    generation: str | None
    documents: List[Document]
    web_searched: bool
    log: List[str] # Add log to state

def retrieve_node(state):
    """Retrieve documents from vector store."""
    log_entry = "--- NODE: RETRIEVE ---"
    print(log_entry) # Keep console log for debugging if needed
    question = state["question"]
    current_log = state.get("log", [])

    if 'retriever' not in st.session_state or st.session_state.retriever is None:
        err_msg = "ERROR: Retriever not initialized."
        print(err_msg)
        return {"documents": [], "question": question, "log": current_log + [log_entry, err_msg]}

    try:
        documents = st.session_state.retriever.invoke(question)
        log_entry += f"\nRetrieved {len(documents)} docs."
        print(f"Retrieved {len(documents)} docs.")
        return {"documents": documents, "question": question, "web_searched": False, "log": current_log + [log_entry]}
    except Exception as e:
        err_msg = f"ERROR during retrieval: {e}"
        print(err_msg)
        return {"documents": [], "question": question, "log": current_log + [log_entry, err_msg]}


def grade_documents_node(state):
    """Determines whether the retrieved documents are relevant."""
    log_entry = "--- NODE: GRADE DOCUMENTS ---"
    print(log_entry)
    question = state["question"]; documents = state["documents"]
    current_log = state.get("log", [])
    log_updates = [log_entry]

    if not documents: log_updates.append("No documents to grade."); return {"documents": [], "question": question, "log": current_log + log_updates}
    if retrieval_grader is None: log_updates.append("WARN: Retrieval grader not ready."); return {"documents": documents, "question": question, "log": current_log + log_updates}

    filtered_docs = []
    for i, d in enumerate(documents):
        try:
            score = retrieval_grader.invoke({"question": question, "document": d.page_content})
            grade = score.binary_score.lower()
            if grade == "yes": msg = f"+ Doc {i+1}: Relevant"; print(msg); log_updates.append(msg); filtered_docs.append(d)
            else: msg = f"- Doc {i+1}: Irrelevant"; print(msg); log_updates.append(msg)
        except Exception as e: msg = f"Error grading doc {i+1}: {e}"; print(msg); log_updates.append(msg); # Optionally keep doc if grading fails?
    log_updates.append(f"Filtered down to {len(filtered_docs)} relevant docs.")
    return {"documents": filtered_docs, "question": question, "log": current_log + log_updates}

def generate_node(state):
    """Generate answer using RAG."""
    log_entry = "--- NODE: GENERATE ---"
    print(log_entry)
    question = state["question"]; documents = state["documents"]
    current_log = state.get("log", [])
    log_updates = [log_entry]
    generation = None # Default

    if rag_chain is None: err_msg="ERROR: RAG chain not ready."; print(err_msg); log_updates.append(err_msg)
    elif not documents: msg="WARN: No documents to generate from."; print(msg); log_updates.append(msg); generation = "I couldn't find relevant information in the documentation to answer that."
    else:
        try:
             # Format documents for the prompt context
             formatted_docs = "\n\n".join([doc.page_content for doc in documents])
             generation = rag_chain.invoke({"context": formatted_docs, "question": question}) # Pass formatted docs
             msg = f"Generated response snippet: {generation[:100]}..."
             print(msg); log_updates.append(msg)
        except Exception as e: msg = f"Error during RAG generation: {e}"; print(msg); log_updates.append(msg); generation = "Error generating answer."

    return {"documents": documents, "question": question, "generation": generation, "log": current_log + log_updates}


def transform_query_node(state):
    """Transform the query to produce a better question."""
    log_entry = "--- NODE: TRANSFORM QUERY ---"
    print(log_entry)
    question = state["question"]
    current_log = state.get("log", [])
    log_updates = [log_entry]
    better_question = question # Default

    if question_rewriter is None: msg="WARN: Question rewriter not ready."; print(msg); log_updates.append(msg)
    else:
        try: better_question = question_rewriter.invoke({"question": question})
        except Exception as e: msg=f"Error rewriting question: {e}"; print(msg); log_updates.append(msg); # Fallback to original
    msg = f"Rewritten question: {better_question}"; print(msg); log_updates.append(msg)
    return {"documents": state.get("documents", []), "question": better_question, "log": current_log + log_updates}

def web_search_node(state):
    """Web search based question."""
    log_entry = "--- NODE: WEB SEARCH ---"
    print(log_entry)
    question = state["question"]
    current_log = state.get("log", [])
    log_updates = [log_entry]
    web_docs = [] # Default

    if web_search_tool is None: msg="ERROR: Web search tool not ready."; print(msg); log_updates.append(msg)
    else:
        try:
            search_results = web_search_tool.invoke({"query": question})
            if search_results:
                 web_results_content = "\n\n".join([d.get("content", "") for d in search_results])
                 web_doc = Document(page_content=web_results_content, metadata={"source": "web_search"})
                 web_docs = [web_doc]
                 msg=f"Web search snippet: {web_results_content[:100]}..."; print(msg); log_updates.append(msg)
            else: msg="Web search returned no results."; print(msg); log_updates.append(msg)
        except Exception as e: msg=f"Error during web search: {e}"; print(msg); log_updates.append(msg)

    return {"documents": web_docs, "question": question, "web_searched": True, "log": current_log + log_updates} # Flag web search


# #############################################################################
# Graph Edges (Conditional Logic - Define functions)
# #############################################################################

def route_question_edge(state):
    log_entry = "--- EDGE: ROUTE QUESTION ---"
    print(log_entry)
    question = state["question"]
    current_log = state.get("log", [])
    decision = "vectorstore" # Default

    if 'question_router' not in st.session_state or st.session_state.question_router is None:
         msg="WARN: Router not ready, defaulting to vectorstore."; print(msg); log_updates = [log_entry, msg]
    else:
        try:
            source = st.session_state.question_router.invoke({"question": question})
            decision = source.datasource.lower()
            msg=f"Routing decision: {decision}"; print(msg); log_updates = [log_entry, msg]
            # Only allow web search if the tool is available
            if decision == "web_search" and not web_search_tool:
                 msg="WARN: Web search chosen but tool unavailable. Routing to vectorstore."; print(msg)
                 log_updates.append(msg); decision = "vectorstore"
        except Exception as e:
            msg=f"Error routing question: {e}. Defaulting to vectorstore."; print(msg); log_updates = [log_entry, msg]

    # Append log to state before returning decision
    state["log"] = current_log + log_updates
    return decision


def decide_to_generate_edge(state):
    log_entry = "--- EDGE: DECIDE TO GENERATE ---"
    print(log_entry)
    current_log = state.get("log", [])
    log_updates = [log_entry]
    decision = "generate" # Default

    if state.get("web_searched"):
         msg="Decision: Generate from web search results."; print(msg); log_updates.append(msg); decision = "generate"
    else:
        filtered_documents = state.get("documents", [])
        if not filtered_documents: msg="Decision: No relevant docs, transform query."; print(msg); log_updates.append(msg); decision = "transform_query"
        else: msg="Decision: Relevant docs found, generate answer."; print(msg); log_updates.append(msg); decision = "generate"

    state["log"] = current_log + log_updates
    return decision

def format_docs_for_grade(docs: List[Document]) -> str:
     """Helper to format docs for hallucination/answer grading prompts."""
     return "\n\n".join([f"--- Document {i+1} ---\n{doc.page_content}" for i, doc in enumerate(docs)])

def grade_generation_edge(state):
    log_entry = "--- EDGE: GRADE GENERATION ---"
    print(log_entry)
    question = state["question"]; documents = state["documents"]; generation = state["generation"]
    current_log = state.get("log", [])
    log_updates = [log_entry]
    decision = END # Default to useful if grading fails or isn't available

    if not generation: # Handle cases where generation failed in the node
        msg="Decision: Generation failed, transform query to retry."; print(msg); log_updates.append(msg); decision = "transform_query"
    elif hallucination_grader is None or answer_grader is None:
        msg="WARN: Graders not ready, accepting generation."; print(msg); log_updates.append(msg); decision = END
    else:
        try:
            formatted_docs = format_docs_for_grade(documents) # Format for prompt
            # Check for hallucinations
            hallu_score = hallucination_grader.invoke({"documents": formatted_docs, "generation": generation})
            if hallu_score.binary_score.lower() == "no":
                msg="Decision: Generation hallucinates, transforming query."; print(msg); log_updates.append(msg); decision = "transform_query" # Transform query might be safer than regenerating
            else:
                msg = "--- Generation Grounded ---"; print(msg); log_updates.append(msg)
                # Check if answer addresses the question
                ans_score = answer_grader.invoke({"question": question, "generation": generation})
                if ans_score.binary_score.lower() == "yes":
                    msg = "Decision: Generation is useful."; print(msg); log_updates.append(msg); decision = END
                else:
                    msg = "Decision: Generation doesn't address question, transform query."; print(msg); log_updates.append(msg); decision = "transform_query"
        except Exception as e:
            msg = f"Error during generation grading: {e}. Accepting generation."; print(msg); log_updates.append(msg); decision = END

    state["log"] = current_log + log_updates
    return decision


# #############################################################################
# Streamlit UI Application
# #############################################################################

st.set_page_config(page_title="Webpage Q&A Agent", layout="wide")
st.title("🌐 Adaptive RAG Webpage Q&A Agent")

# --- Session State Initialization ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "indexed_url" not in st.session_state:
    st.session_state.indexed_url = None
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "graph_app" not in st.session_state:
    st.session_state.graph_app = None
if "processing" not in st.session_state:
    st.session_state.processing = False # Flag for indexing
if "thinking" not in st.session_state:
    st.session_state.thinking = False # Flag for agent running
if "agent_logs" not in st.session_state:
    st.session_state.agent_logs = [] # Store logs from the last run
if "question_router" not in st.session_state:
    st.session_state.question_router = None # Store the initialized router


# --- Sidebar for Indexing Control ---
with st.sidebar:
    st.header("Website Indexing")
    st.markdown("Enter a URL to index its content for Q&A.")

    url_input = st.text_input("Website URL:", value=st.session_state.get("indexed_url", "https://help.slack.com/")) # Default example
    max_pages_input = st.number_input("Max Pages to Crawl:", min_value=1, max_value=200, value=25)
    force_reindex_input = st.checkbox("Force Re-index?", value=False)

    # Disable button while processing
    index_button = st.button("Index Website", disabled=st.session_state.processing)

    if index_button:
        if not url_input or not is_valid_url(url_input):
            st.error("Please enter a valid URL.")
        elif not groq_api_key or not llm:
             st.error("Groq API Key or LLM not configured. Cannot index.")
        else:
            st.session_state.processing = True
            st.session_state.messages = [] # Clear chat on new index
            st.session_state.indexed_url = None # Clear current index info
            st.session_state.vector_store = None
            st.session_state.retriever = None
            st.session_state.graph_app = None
            st.session_state.question_router = None
            st.rerun() # Rerun to show spinner and clear state visually

    # Indexing Logic (runs after rerun if processing is True)
    if st.session_state.processing:
        with st.spinner(f"Indexing {url_input}... Please wait."):
            vector_store = None
            embedding_model = None # Define embedding_model here

            # Check if index exists and we are *not* forcing reindex
            if os.path.exists(vector_store_path) and not force_reindex_input:
                try:
                    st.info(f"Loading existing index: {vector_store_path}")
                    embedding_model = get_embedding_model(embedding_model_name)
                    vector_store = FAISS.load_local(vector_store_path, embedding_model, allow_dangerous_deserialization=True)
                    st.success("Existing index loaded.")
                    st.session_state.indexed_url = url_input # Assume loaded index matches input URL if not forcing
                except Exception as e:
                    st.warning(f"Failed load index: {e}. Will re-index.")
                    vector_store = None # Ensure re-indexing happens

            # If index wasn't loaded (doesn't exist, loading failed, or forced re-index)
            if vector_store is None:
                st.info(f"Running indexing pipeline for {url_input}...")
                processed_docs = crawl_and_process(url_input, max_pages_input)
                if processed_docs:
                    chunked_docs = chunk_documents(processed_docs) # Use default chunk settings for simplicity
                    if chunked_docs:
                        if not embedding_model: embedding_model = get_embedding_model(embedding_model_name)
                        vector_store = create_faiss_vector_store(chunked_docs, embedding_model, vector_store_path)
                        if vector_store: st.success("Indexing successful!")
                        else: st.error("Failed to create vector store during indexing.")
                    else: st.error("Chunking failed during indexing.")
                else: st.error("Crawling/Processing failed during indexing.")

            # --- Post-Indexing Setup (if successful) ---
            if vector_store:
                st.session_state.vector_store = vector_store
                st.session_state.indexed_url = url_input
                st.session_state.retriever = vector_store.as_retriever(search_kwargs={"k": 3})

                # Dynamically configure router
                try:
                    site_name = urlparse(url_input).netloc or "the indexed documentation"
                    topic_short = site_name.split('.')[1] if '.' in site_name else site_name
                    dynamic_prompt_str = route_prompt_template.format(topic=f"help documentation for {site_name}", topic_short=topic_short, question="{question}")
                    route_prompt = ChatPromptTemplate.from_messages([("system", dynamic_prompt_str), ("human", "{question}")])
                    st.session_state.question_router = route_prompt | structured_llm_router # Assign to session state
                    logging.info(f"Question router configured for topic: {site_name}")
                except Exception as e:
                     st.error(f"Failed to configure question router: {e}")


                # --- Build and Compile Graph ---
                if llm and st.session_state.retriever and st.session_state.question_router: # Ensure router is ready
                    try:
                        workflow = StateGraph(GraphState)
                        # Define nodes
                        workflow.add_node("web_search", web_search_node)
                        workflow.add_node("retrieve", retrieve_node)
                        workflow.add_node("grade_documents", grade_documents_node)
                        workflow.add_node("generate", generate_node)
                        workflow.add_node("transform_query", transform_query_node)
                        # Define edges
                        workflow.add_conditional_edges(START, route_question_edge, {"web_search": "web_search", "vectorstore": "retrieve"})
                        workflow.add_edge("web_search", "generate")
                        workflow.add_edge("retrieve", "grade_documents")
                        workflow.add_conditional_edges("grade_documents", decide_to_generate_edge, {"transform_query": "transform_query", "generate": "generate"})
                        workflow.add_edge("transform_query", "retrieve")
                        workflow.add_conditional_edges("generate", grade_generation_edge, {"transform_query": "transform_query", END: END})

                        st.session_state.graph_app = workflow.compile()
                        st.success("Agent graph compiled.")
                    except Exception as e:
                        st.error(f"Failed to compile agent graph: {e}")
                else:
                     st.warning("LLM, Retriever, or Router not ready. Graph not compiled.")

            else:
                 st.session_state.indexed_url = None # Clear if indexing failed

            # Indexing finished
            st.session_state.processing = False
            st.rerun() # Rerun to update UI after processing finishes

    # Display current status
    if st.session_state.indexed_url and not st.session_state.processing:
        st.success(f"Indexed: {st.session_state.indexed_url}")
        if st.session_state.graph_app:
             st.info("Agent is ready for questions.")
        else:
             st.warning("Indexing done, but agent graph failed to compile.")
    elif not st.session_state.processing:
        st.info("No website indexed. Please enter a URL and click 'Index Website'.")


# --- Main Chat Area ---
st.header("Chat with Indexed Website")

if not st.session_state.indexed_url:
    st.warning("Please index a website using the sidebar first.")
elif not st.session_state.graph_app:
     st.error("Agent graph is not ready. Indexing might have failed or graph compilation failed.")
else:
    # Display chat messages
    for msg_data in st.session_state.messages:
        with st.chat_message(msg_data["role"]):
            st.markdown(msg_data["content"])
            # Optionally display logs associated with AI messages
            if msg_data["role"] == "assistant" and "logs" in msg_data:
                 with st.expander("Agent Thought Process"):
                      st.text("".join(msg_data["logs"])) # Join list of log strings

    # Chat input
    if prompt := st.chat_input("Ask a question about the website...", disabled=st.session_state.thinking):
        # Add user message to state and display
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Prepare for agent response
        st.session_state.thinking = True
        st.session_state.agent_logs = [] # Clear logs for the new run
        final_generation = "Agent encountered an issue." # Default
        agent_run_logs = [] # Store logs for this specific run

        # Display thinking indicator and run agent
        with st.chat_message("assistant"):
            with st.spinner("Agent is thinking..."):
                inputs = {"question": prompt, "log": []} # Initialize log in state
                captured_output = io.StringIO() # Capture print statements

                try:
                    # Redirect stdout to capture print statements from nodes/edges
                    with contextlib.redirect_stdout(captured_output):
                        # Use invoke for simplicity, get final state
                        # Stream is better for live updates, but harder to manage logs cleanly here
                        final_state = st.session_state.graph_app.invoke(inputs, {"recursion_limit": 15})

                    # Get logs captured via print statements
                    agent_run_logs = captured_output.getvalue().splitlines(keepends=True)

                    # Get logs potentially passed through state (if nodes updated it)
                    # state_logs = final_state.get("log", []) # If nodes appended to state['log']
                    # agent_run_logs.extend([f"STATE LOG: {l}\n" for l in state_logs]) # Combine if needed


                    if final_state and 'generation' in final_state and final_state['generation']:
                        final_generation = final_state['generation']
                    elif final_state and 'question' in final_state:
                         final_generation = f"(Agent stopped. Last question state: '{final_state['question']}')"
                    else:
                         final_generation = "Agent finished unexpectedly. Check logs."

                except Exception as e:
                    final_generation = f"An error occurred: {e}"
                    agent_run_logs.append(f"\n--- Error during graph execution: {e} ---")
                    logging.error(f"Error during graph execution: {e}", exc_info=True)

            # Display final answer
            st.markdown(final_generation)
            # Store logs with the message
            st.session_state.messages.append({
                "role": "assistant",
                "content": final_generation,
                "logs": agent_run_logs # Attach logs to this message
            })
            # Display logs in expander right after the message
            with st.expander("Agent Thought Process"):
                 st.text("".join(agent_run_logs)) # Display captured print logs

        # Agent finished thinking
        st.session_state.thinking = False
        # No rerun needed here as chat messages update dynamically
        # st.rerun()