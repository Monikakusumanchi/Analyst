import streamlit as st
# import argparse
import requests
from bs4 import BeautifulSoup two, the `generate` node should still produce the "not found" message, NavigableString, Comment
from urllib.parse import urlparse, url.

**Updated `app.py` Code:**

```python
import streamlit asjoin
from collections import deque
import time
import logging
import re
import html
import os st
# import argparse # Not needed for pure Streamlit app
import requests
from bs4 import BeautifulSoup, NavigableString, Comment
from urllib.parse import urlparse, urljoin
from dotenv import load_dotenv
from pprint import pprint
import io
import contextlib

# --- Core Langchain & Document Processing ---
from langchain
from collections import deque
import time
import logging
import re
import html
import os.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.
from dotenv import load_dotenv
from pprint import pprint
import io
import contextlib

# --- Corevectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.schema import Document
from langchain import hub
from langchain_core.output Langchain & Document Processing ---
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from_parsers import StrOutputParser

# --- LLM ---
from langchain_groq import ChatGroq

# --- Tools ---
# REMOVED: from langchain langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.schema import_community.tools.tavily_search import TavilySearchResults

# --- Graph & State ---
from typing import List, Literal, Sequence, Annotated Document
from langchain import hub
from langchain_core.output_parsers import
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END StrOutputParser

# --- LLM ---
from langchain_groq import ChatGroq

# --- Tools ---
# REMOVED: from langchain, START

# --- Prompts & Parsing ---
from langchain_core._community.tools.tavily_search import TavilySearchResults

#prompts import ChatPromptTemplate, PromptTemplate
from pydantic import BaseModel, Field

# --- Graph & State ---
from typing import List, Literal, Sequence, Annotated #############################################################################
# Load Environment Variables & Basic Config
#
from typing_extensions import TypedDict
from langgraph.graph import State #############################################################################
load_dotenv()
groq_apiGraph, END, START

# --- Prompts & Parsing ---
from langchain_core.prompts import_key = os.getenv("GROQ_API_KEY")
# ChatPromptTemplate, PromptTemplate
from pydantic import BaseModel, Field

# #############################################################################
# Load Environment Variables & Basic Config
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
embedding_modelgetenv("GROQ_API_KEY")
# REMOVED: tav_name = os.getenv("EMBEDDING_MODEL_NAME", DEFAULTily_api_key = os.getenv("TAVILY_API__EMBEDDING_MODEL)
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
tavily_api_key = os.getenv("TAVILY_API_KEY")
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
embedding_model_name = os.getenv("EMBEDDING_MODEL_NAME", DEFAULT_EMBEDDING_MODEL)
DEFAULT_VECTOR_STORE_PATH = "./faiss_adaptive_rag_store" # Use a different default path
vector_store_path = os.getenv("VECTOR_STORE_PATH", DEFAULT_VECTOR_STORE_PATH)
GROQ_MODEL_NAME = "llama3-8b-8192" # Or "mixtral-8x7b-32768" etc.
embedding_model_name = os. "./faiss_retry_rag_store" # New path
MAX_RETRIES = 2 # Max number of times to rewrite.getenv("VECTOR_STORE_PATH", DEFAULT_VECTOR_STORE_PATH)

# #############################################################################
# Logging Setup
# #############################################################################
logging.basicConfig(level=logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# #############################################################################
# Crawlerlogging.INFO, format='%(asctime)s - %(levelname)s -, Processor, Indexing Functions (Paste your fully corrected versions)
# ############################################################################# %(message)s')

# #############################################################################
# Crawler
# --- Paste your corrected functions here ---
# HEADERS, MAIN_CONTENT_SELECT, Processor, Indexing Functions (Paste your fully corrected versions)
# #############################################################################ORS, NOISE_TAGS, TEXT_BEARING_TAGS

# --- Paste your corrected functions here ---
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
# RAG Components Initialization
        vs = FAISS.from_documents(documents=chunks, embedding=embeddings); logging
# #############################################################################

# --- LLM ---
llm = None.info("FAISS store created in memory.")
        if path and vs:

if not groq_api_key: 
    st.error("GROQ_API_KEY missing.",            try: vs.save_local(path); logging.info(f"FAISS saved icon="🚨")
else:
    try: llm = ChatGroq(temperature=0: {path}")
            except Exception as e_save: logging.error, model_name=GROQ_MODEL_NAME, groq_api_key=groq_api_key)
    except Exception as e(f"FAISS save failed: {e_save}", exc_info: st.error(f"Failed init Groq LLM: {e=True)
    except Exception as e_create: logging.error(}", icon="🚨"); logging.error(f"Failed init Groq LLM: {e}")

f"FAISS creation failed: {e_create}", exc_info=# --- REMOVED: Tool Initialization ---

# --- Retriever (initialized later) ---
retrieverTrue); return None
    return vs

# ############################################################################# = None

# --- RAG Chain & Graders (Depends on LLM) ---
rag
# RAG Components Initialization
# #############################################################################

# ---_chain = None; retrieval_grader = None; hallucination_grader = None; answer_ LLM Initialization ---
llm = None
if not groq_api_key:
    grader = None
question_rewriter = None # RE-ADD Question Rewriter

if llm:
    # RAG Chain
    try: promptst.error("GROQ_API_KEY missing. Agent cannot function.", icon="🚨") = hub.pull("rlm/rag-prompt"); rag_chain =
else:
    try: llm = ChatGroq(temperature=0, model_name=GRO prompt | llm | StrOutputParser(); logging.info("RAG chain OK.")
    except ExceptionQ_MODEL_NAME, groq_api_key=groq_ as e: logging.error(f"Failed init RAG chain: {api_key)
    except Exception as e: st.error(fe}")
    # Retrieval Grader
    try:
        class GradeDocuments(BaseModel): binary"Failed to init Groq LLM: {e}", icon="🚨"); logging.error(f"Failed init Groq LLM: {e}")

# --- REMOVED: Tool_score: str = Field(description="Relevant? 'yes' or 'no'.")
        structured_llm_ret_grader = llm.with_structured_output( Initialization ---

# --- Retriever (initialized later) ---
retriever = None

# --- RAG ChainGradeDocuments)
        system_ret_grade = "Assess doc relevance. Grade 'yes' if keywords & Graders & Rewriter (Depends on LLM) ---
rag_chain = None; retrieval/meaning match question, else 'no'."
        grade_prompt = Chat_grader = None; hallucination_grader = None; answer_grader = None
PromptTemplate.from_messages([("system", system_ret_grade),question_rewriter = None # RE-ADD Rewriter
# REMOVED: question_router ("human", "Doc:\n{document}\n\nQ: {question}")])
        ret = None

if llm:
    # RAG Chain
    tryrieval_grader = grade_prompt | structured_llm_ret_: prompt = hub.pull("rlm/rag-prompt"); rag_grader; logging.info("Retrieval grader OK.")
    except Exception aschain = prompt | llm | StrOutputParser(); logging.info("RAG chain OK.")
    except e: logging.error(f"Failed init retrieval grader: {e}")
    # Exception as e: logging.error(f"Failed init RAG chain: Hallucination Grader
    try:
        class GradeHallucinations {e}")
    # Retrieval Grader
    try:
        class GradeDocuments(BaseModel): binary(BaseModel): binary_score: str = Field(description="Grounded?_score: str = Field(description="Is the document relevant? 'yes 'yes' or 'no'.")
        structured_llm_hallu_grader = ll' or 'no'.")
        structured_llm_ret_graderm.with_structured_output(GradeHallucinations)
        system = llm.with_structured_output(GradeDocuments)
        system_hallu = "Assess if answer is grounded in facts. Score 'yes'_ret_grade = "Assess doc relevance. Grade 'yes' if keywords if grounded, 'no' otherwise."
        hallucination_prompt = ChatPromptTemplate/meaning match question, else 'no'."
        grade_prompt = Chat.from_messages([("system", system_hallu), ("human",PromptTemplate.from_messages([("system", system_ret_grade), "Facts:\n{documents}\n\nAnswer: {generation}")]) ("human", "Doc:\n{document}\n\nQ: {question}")
        hallucination_grader = hallucination_prompt | structured_llm_hall])
        retrieval_grader = grade_prompt | structured_llm_u_grader; logging.info("Hallucination grader OK.")
    except Exception as e: logging.error(f"Failed init hallucination graderret_grader; logging.info("Retrieval grader OK.")
    except Exception as e: logging.error(f"Failed init retrieval grader: {: {e}")
    # Answer Grader
    try:
        e}")
    # Hallucination Grader
    try:
        class GradeAnswer(BaseModel): binary_score: str = Field(descriptionclass GradeHallucinations(BaseModel): binary_score: str = Field(description="Is answer grounded in facts? 'yes' or 'no'.")
        structured_="Addresses question? 'yes' or 'no'.")
        structured_llllm_hallu_grader = llm.with_structured_output(Gradem_ans_grader = llm.with_structured_output(GradeHallucinations)
        system_hallu = "Assess if an answer is grounded in/supportedAnswer)
        system_ans = "Assess if answer addresses question. Score 'yes' if it by facts. Score 'yes' if grounded, 'no' otherwise."
 does, 'no' otherwise."
        answer_prompt = ChatPromptTemplate        hallucination_prompt = ChatPromptTemplate.from_messages([(".from_messages([("system", system_ans), ("human", "Q:\n{question}\n\nAnswer: {generation}")])
        answersystem", system_hallu), ("human", "Facts:\n{documents_grader = answer_prompt | structured_llm_ans_grader;}\n\nAnswer: {generation}")])
        hallucination_grader = hallucination_prompt | structured_llm_hallu_grader; logging.info(" logging.info("Answer grader OK.")
    except Exception as e: loggingHallucination grader OK.")
    except Exception as e: logging.error.error(f"Failed init answer grader: {e}")
    # RE(f"Failed init hallucination grader: {e}")
    # Answer-ADD Question Rewriter
    try:
        system_rew = " Grader
    try:
        class GradeAnswer(BaseModel): binaryYou are a question re-writer optimizing a question for vectorstore retrieval based on underlying semantic intent."
_score: str = Field(description="Does answer address question? 'yes' or        re_write_prompt = ChatPromptTemplate.from_messages([(" 'no'.")
        structured_llm_ans_grader = llsystem", system_rew), ("human", "Initial question: {question}\m.with_structured_output(GradeAnswer)
        system_ans = "Assess if an answer addresses/resolves a question. Score 'yes' if it does,nFormulate an improved question.")])
        question_rewriter = re_write_ 'no' otherwise."
        answer_prompt = ChatPromptTemplate.fromprompt | llm | StrOutputParser(); logging.info("Question rewriter OK.")
    except_messages([("system", system_ans), ("human", "Q:\n{question}\n\n Exception as e: logging.error(f"Failed init question rewriter:Answer: {generation}")])
        answer_grader = answer_prompt | structured_llm_ans_ {e}")


# #############################################################################
# Graph State & Nodes
# #grader; logging.info("Answer grader OK.")
    except Exception as e: logging.error(f############################################################################

class GraphState(TypedDict):
    """"Failed init answer grader: {e}")
    # Question Rewriter
Represents the state of our graph."""
    original_question: str      try:
        system_rew = "You are a question re-writer optimizing a# Store the initial question
    question: str           # Current question (might question for vectorstore retrieval based on underlying semantic intent."
        re_write_prompt = be rewritten)
    generation: str | None
    documents: List[Document]
 ChatPromptTemplate.from_messages([("system", system_rew), ("human", "Initial question: {    log: List[str]
    retries_left: int       question}\nFormulate an improved question.")])
        question_rewriter = re_write_# Counter for retries

# Define node functions (retrieve_node, grade_documents_prompt | llm | StrOutputParser(); logging.info("Question rewriter OK.")
node, generate_node)
# Re-add transform_query_node
    except Exception as e: logging.error(f"Failed init question rewriter: {e}")# Add fail_node

def retrieve_node(state):
    log_entry =

# #############################################################################
# Graph State & Nodes
# #############################################################################

 "--- NODE: RETRIEVE ---"; print(log_entry)
class GraphState(TypedDict):
    """Represents the state of our    question = state["question"]; current_log = state.get("log graph."""
    question: str
    original_question: str # Store the", []); documents = []
    if 'retriever' not in st.session initial question
    generation: str | None
    documents: List[Document]
    log: List[str]
    rewrite_attempts: int #_state or st.session_state.retriever is None:
        err_msg = "ERROR: Counter for rewrite loops

def retrieve_node(state):
    log_entry = "--- Retriever not ready."; print(err_msg); log_updates = [log_entry, NODE: RETRIEVE (Vector Store) ---"; print(log_entry)
    question = state["question"] # Use current (potentially rewritten) question
    current_log err_msg]
    else:
        try:
            documents = st.session_state. = state.get("log", []); documents = []
    if 'retriever' not in st.sessionretriever.invoke(question); msg=f"Retrieved {len(documents)} docs for_state or st.session_state.retriever is None:
        err Q: '{question[:50]}...'"
            print(msg); log_updates =_msg = "ERROR: Retriever not ready."; print(err_msg); [log_entry, msg]
        except Exception as e: err_msg = f"ERROR log_updates = [log_entry, err_msg]
    else during retrieval: {e}"; print(err_msg); log_updates =:
        try:
            documents = st.session_state.retriever.invoke(question); [log_entry, err_msg]
    return {"documents": documents msg=f"Retrieved {len(documents)} docs for Q: '{question}'"
            print, "log": current_log + log_updates, **state} # Keep(msg); log_updates = [log_entry, msg]
         other state


def grade_documents_node(state):
    log_entry = "except Exception as e: err_msg = f"ERROR during retrieval: {e}"; print(--- NODE: GRADE DOCUMENTS ---"; print(log_entry)
    question = state["question"];err_msg); log_updates = [log_entry, err_msg documents = state["documents"]; current_log = state.get("log",]
    # Preserve other state fields like original_question, rewrite_attempts
    return {"documents []); log_updates = [log_entry]
    filtered_docs =": documents, **{k:v for k,v in state.items() if k != 'documents'}, "log": current_log + log_updates} []
    if not documents: log_updates.append("No docs to grade.")
    elif retrieval_


def grade_documents_node(state):
    log_entry =grader is None: log_updates.append("WARN: Grader not ready "--- NODE: GRADE DOCUMENTS ---"; print(log_entry)
    question = state["question"];, keeping docs."); filtered_docs = documents
    else:
        for i, d in enumerate(documents):
            try:
                score = retrieval_grader.invoke documents = state["documents"]; current_log = state.get("log",({"question": question, "document": d.page_content}); grade = score.binary_score. []); log_updates = [log_entry]
    filtered_docs =lower()
                if grade == "yes": msg = f"+ Relevant doc {i+1}"; print(msg); log_updates.append(msg); filtered []
    if not documents: log_updates.append("No docs to grade.")
    elif_docs.append(d)
                else: msg = f"- Irrelevant doc {i+1}"; retrieval_grader is None: log_updates.append("WARN: Grader not ready, keeping all docs print(msg); log_updates.append(msg)
            except Exception."); filtered_docs = documents
    else:
        for i, d in enumerate as e: msg = f"Error grading doc {i+1}: {(documents):
            try:
                score = retrieval_grader.invoke({"question": question, "e}"; print(msg); log_updates.append(msg); filtered_document": d.page_content}); grade = score.binary_score.docs.append(d) # Keep if grading fails
        log_updates.append(f"lower()
                if grade == "yes": msg = f"+ Doc {i+1}: Relevant"; printRelevant docs: {len(filtered_docs)}.")
    return {"documents": filtered_docs, "log(msg); log_updates.append(msg); filtered_docs.append": current_log + log_updates, **state}


def generate_node(state):
(d)
                else: msg = f"- Doc {i+1    log_entry = f"--- NODE: GENERATE ---"; print(log_entry)}: Irrelevant"; print(msg); log_updates.append(msg)
    question = state["question"]; documents = state["documents"]; current_
            except Exception as e: msg = f"Error grading doc {i+1}: {e}"; print(msg); log_updates.append(msg); filtered_docs.appendlog = state.get("log", []); log_updates = [log_(d) # Keep doc if grading fails?
        log_updates.append(f"entry]
    generation = None
    doc_source = "indexed documentation" #Relevant docs: {len(filtered_docs)}.")
    return {"documents": filtered_docs, ** Source is always vector store now

    if not documents:
        # This case should ideally{k:v for k,v in state.items() if k != be caught by the edge logic before calling generate
        # But as a fallback, handle 'documents'}, "log": current_log + log_updates}


def it here too.
        msg = "Logic Error: Generate called with no relevant generate_node(state):
    log_entry = f"--- NODE: GENERATE --- documents. Setting 'not found'."; print(msg); log_updates.append(msg)
        generation"; print(log_entry)
    question = state["question"]; documents = state["documents"]; current = f"Sorry, I couldn't find relevant information about '{state._log = state.get("log", []); log_updates = [logget('original_question', question)}' in the indexed documentation." # Use original Q_entry]
    generation = None

    # Check if documents list is empty * if available
    elif rag_chain is None:
        err_msg="ERROR: RAG chain not ready."; print(err_msg); log_updatesafter* grading
    if not documents:
        msg = "No relevant documents found in source.append(err_msg)
        generation = "Error: Generation component not ready after grading."; print(msg); log_updates.append(msg)
        #."
    else:
        log_updates.append(f"Generating Use original question in the "not found" message for clarity
        original_q answer from {len(documents)} relevant doc(s).")
        try:
              = state.get("original_question", question)
        generation = f"Sorry, I couldn't find specific information about '{original_q}' in the indexed documentation."
    elif rag_chainformatted_docs = "\n\n".join([doc.page_content for doc in documents is None:
        err_msg="ERROR: RAG chain not ready.";])
             generation = rag_chain.invoke({"context": formatted_docs, " print(err_msg); log_updates.append(err_msg)question": question})
             msg = f"Generated response snippet: {generation[:100]}
        generation = "Error: Generation component not ready."
    else:
        #..."; print(msg); log_updates.append(msg)
        except Documents exist, proceed with RAG
        log_updates.append(f Exception as e: msg = f"Error during RAG generation: {e"Generating answer based on {len(documents)} relevant document(s).")
        try:}"; print(msg); log_updates.append(msg); generation = "
             formatted_docs = "\n\n".join([doc.page_content for doc in documents])Error generating answer."

    if generation is None: generation = "Unexpected issue during generation."

             generation = rag_chain.invoke({"context": formatted_docs, "    return {"generation": generation, "log": current_log + log_updates, **statequestion": question}) # Use current question for generation
             msg = f"Generated response snippet: {generation[:100]}..."; print(msg); log}


# RE-ADD transform_query_node
def transform_query_node(state):
    _updates.append(msg)
        except Exception as e: msg = f"Error during Rlog_entry = "--- NODE: TRANSFORM QUERY ---"; print(log_entryAG generation: {e}"; print(msg); log_updates.append(msg); generation = "Error)
    # IMPORTANT: Always rewrite the *original* question to avoid drift generating answer."

    if generation is None: generation = "Unexpected issue during generation
    original_question = state["original_question"]
    current_."
    return {"generation": generation, **{k:v for k,v in state.itemslog = state.get("log", []); log_updates = [log_entry]
    retries = state["retries_left"] - 1 #() if k != 'generation'}, "log": current_log + log_updates}

def transform_query_node(state):
    """Transform the query to Decrement retries
    better_question = original_question # Default

    if question produce a better question."""
    log_entry = "--- NODE: TRANSFORM_rewriter is None: msg="WARN: Rewriter not ready."; print(msg); log_updates. QUERY ---"; print(log_entry)
    # Use original question for rewriteappend(msg)
    else:
        try: better_question = question_rewriter. to avoid drifting too far
    original_question = state["original_question"]
    current_invoke({"question": original_question})
        except Exception as e: msglog = state.get("log", []); log_updates = [log_=f"Error rewriting question: {e}"; print(msg); log_entry]
    rewritten_question = original_question # Default if rewrite fails
    rewrite_attempts =updates.append(msg); # Fallback
    msg = f"Rew state.get("rewrite_attempts", 0) + 1 # Increment counterritten question: {better_question}"; print(msg); log_updates.

    if question_rewriter is None: msg="WARN: Rewriter not ready."; print(msg); log_updates.append(msg)
    else:
        tryappend(msg)
    return {"question": better_question, "ret: rewritten_question = question_rewriter.invoke({"question": original_question})ries_left": retries, "log": current_log + log_updates, **state}



        except Exception as e: msg=f"Error rewriting question: {def fail_node(state):
    """Node to handle failure after retries."""
    log_e}"; print(msg); log_updates.append(msg); # Fallback to original
    msgentry = "--- NODE: FAIL (Max Retries Reached) ---"; print(log_ = f"Attempt {rewrite_attempts}: Rewritten question: {rewritten_question}"; print(msg);entry)
    current_log = state.get("log", [])
    original_question = state. log_updates.append(msg)
    # Update question, keep docsget("original_question", "your query")
    fail_message = f"Sorry, I tried (they were irrelevant), increment counter
    return {
        "documents": [], rewriting the question and searching again, but couldn't find relevant information for '{original_question}' in the documentation # Clear irrelevant docs before re-retrieving
        "question": rewritten."
    # Set generation to the failure message and clear documents
    return_question,
        "original_question": original_question, # Preserve {"generation": fail_message, "documents": [], "log": current_log + [ original
        "rewrite_attempts": rewrite_attempts,
        "loglog_entry]}

# REMOVED: web_search_node

# #############################################################################": current_log + log_updates
    }

# REMOVED:
# Graph Edges (Conditional Logic)
# #############################################################################

# REMOVED: route_question_edge

def decide_ web_search_node

# #############################################################################
# Graphgenerate_or_rewrite_edge(state):
    """Decide whether to generate, Edges (Conditional Logic)
# #############################################################################

# rewrite, or fail based on graded docs and retries."""
    log_entry REMOVED: route_question_edge

def decide_generate_or_rewrite_edge(state = "--- EDGE: DECIDE GENERATE/REWRITE ---"; print(log_):
    """Decide whether to generate, rewrite, or end if too many rewrites."""
    log_entry = "--- EDGE: DECIDE GENERentry)
    filtered_documents = state.get("documents", [])
    retATE or REWRITE ---"; print(log_entry)
    current_log = state.get("log", []); log_updates = [log_entry]
ries_left = state.get("retries_left", MAX_RETRIES)
    current_    decision = "generate" # Default

    if not state.get("documents"):log = state.get("log", []); log_updates = [log_ # Check if list is empty after grading
        rewrite_attempts = state.get("rewriteentry]
    decision = "generate" # Assume generate first

    if not filtered_documents:
        if retries_left > 0:
            msg = f_attempts", 0)
        MAX_REWRITES = 1"Decision: No relevant docs. Retries left: {retries_left # Allow only one rewrite attempt
        if rewrite_attempts < MAX_REWRITES:
            msg = f"Decision: No relevant docs (attempt {rewrite_attempts+}. Transform query."; print(msg); decision = "transform_query"
        else1}), transform query."; print(msg); decision = "transform_query"
        else:
:
            msg = f"Decision: No relevant docs & no retries left. Failing             msg = f"Decision: No relevant docs after {rewrite_attempts} rewrite."; print(msg); decision = "fail_node"
    else:
        msg="(s), generate 'not found' message."; print(msg)
             decision = "generateDecision: Relevant docs found. Generate answer."; print(msg); decision = "generate"" # Force generation with empty docs
    else:
        msg="Decision: Relevant

    log_updates.append(msg)
    state["log"] = current_log + log_updates
    return decision

def format_docs_for_grade(docs docs found, generate answer."; print(msg); decision = "generate"

: List[Document]) -> str:
     return "\n\n".    log_updates.append(msg)
    state["log"] = currentjoin([f"--- Doc {i+1} (Source: {doc.metadata_log + log_updates
    return decision

def format_docs_for_grade(docs.get('url', 'N/A')}) ---\n{doc.page: List[Document]) -> str:
     return "\n\n"._content}" for i, doc in enumerate(docs)])

# RE-ADD gradejoin([f"--- Doc {i+1} URL: {doc.metadata_generation_edge (modified to retry via transform_query)
def grade_generation.get('url', 'N/A')} ---\n{doc.page_content}"_edge(state):
    log_entry = "--- EDGE: GRADE for i, doc in enumerate(docs)])

def grade_generation_edge(state):
    """ GENERATION ---"; print(log_entry)
    question = state["question"]; documentsDetermines whether the generation is useful or if we should stop."""
    log_entry = " = state["documents"]; generation = state["generation"]
    retries_left = state.--- EDGE: GRADE GENERATION ---"; print(log_entry)
    question = state.get("retries_left", MAX_RETRIES)
    current_get("original_question", state["question"]) # Grade against original Q
    documents = state["documents"];log = state.get("log", []); log_updates = [log_entry]
    decision = END # Default

    if not generation: # generation = state["generation"]
    current_log = state.get("log", []); log_updates Handle if generation node itself failed
        msg="Decision: Generation failed in node."; print(msg); log_updates.append(msg)
        if ret = [log_entry]
    decision = END # Default to useful

ries_left > 0: decision = "transform_query" # Retry    if not generation or "error generating answer" in generation.lower() or "component by rewriting original Q
        else: decision = "fail_node"
 is not ready" in generation.lower():
        msg="Decision: Generation failed or erro    elif hallucination_grader is None or answer_grader is None:
red, ending."; print(msg); log_updates.append(msg); decision = END
    elif hallucination_grader is None or answer_grader is None:        msg="WARN: Graders not ready, accepting generation."; print(msg); log_updates.append(
        msg="WARN: Graders not ready, accepting generation as final.";msg); decision = END
    else:
        try:
            formatted_docs = format_docs_ print(msg); log_updates.append(msg); decision = END
for_grade(documents)
            hallu_score = hallucination_grader.invoke({"documents":    else:
        try:
            formatted_docs = format_docs formatted_docs, "generation": generation})
            if hallu_score_for_grade(documents)
            hallu_score = hallucination.binary_score.lower() == "no":
                msg="Decision: Generation hallucinates."; print(msg); log_updates.append(msg)
                if retries_left > 0: decision = "transform_query" # Retry by_grader.invoke({"documents": formatted_docs, "generation": generation}) rewriting original Q
                else: decision = "fail_node"
            else:
                msg
            if hallu_score.binary_score.lower() == " = "--- Generation Grounded ---"; print(msg); log_updates.appendno":
                # NOTE: If hallucinating, maybe provide a different message(msg)
                ans_score = answer_grader.invoke({"question": state? For now, just end.
                msg="Decision: Generation may hallucinate. Ending."; print(msg); log_updates.append(msg); decision = END
['original_question'], "generation": generation}) # Grade against ORIGINAL question
                if ans_score.            else:
                msg = "--- Generation Grounded ---"; print(msg); logbinary_score.lower() == "yes":
                    msg = "Decision: Generation is useful."; print(msg); log_updates.append(msg); decision_updates.append(msg)
                ans_score = answer_grader = END
                else:
                    msg = "Decision: Generation doesn't address question.";.invoke({"question": question, "generation": generation})
                if ans print(msg); log_updates.append(msg)
                    if ret_score.binary_score.lower() == "yes":
                    msgries_left > 0: decision = "transform_query" # Retry by rewriting original Q
                    else: decision = "fail_node"
 = "Decision: Generation is useful."; print(msg); log_updates.        except Exception as e:
            msg = f"Error during grading: {e}. Acceptingappend(msg); decision = END
                else:
                    # NOTE: If not useful, maybe provide a different message? For now, just end.
                    msg = "Decision: Generation may not fully address question. Ending."; print(msg); log_updates.append generation."; print(msg); log_updates.append(msg); decision = END(msg); decision = END
        except Exception as e:
            msg = f"Error during final

    state["log"] = current_log + log_updates
    return decision grading: {e}. Accepting generation."; print(msg); log_updates.append(msg); decision = END

# #############################################################################
# Streamlit UI Application
#

    state["log"] = current_log + log_updates
     #############################################################################

# --- UI ---
st.set_page_config(page_title="Doc Q&A Bot (Retry)", layout="wide")
streturn decision

# #############################################################################
# Streamlit UI Application.title("📚 Documentation Q&A Bot with Rewrite")

# --- Session State Init ---
# #############################################################################

# --- UI ---
st.set_page_config
if "messages" not in st.session_state: st.session_state.(page_title="Webpage Q&A Bot", layout="wide")
st.title("📚 Documentation Q&A Bot (Retrieve & Rewrite)")

# --- Sessionmessages = []
if "indexed_url" not in st.session_state: st.session_state State Init ---
if "messages" not in st.session_state: st.session_state..indexed_url = None
if "vector_store" not in stmessages = []
if "indexed_url" not in st.session_state: st.session_state.session_state: st.session_state.vector_store = None.indexed_url = None
if "vector_store" not in st
if "retriever" not in st.session_state: st..session_state: st.session_state.vector_store = Nonesession_state.retriever = None
if "graph_app" not in st.session_state: st.session_state.graph_app
if "retriever" not in st.session_state: st. = None
if "processing" not in st.session_state: stsession_state.retriever = None
if "graph_app" not in st.session_state: st.session_state.graph_app.session_state.processing = False
if "thinking" not in st.session_state = None
if "processing" not in st.session_state: st: st.session_state.thinking = False

# --- Sidebar ---
with st.sidebar:
    .session_state.processing = False
if "thinking" not in stst.header("Website Indexing")
    st.markdown("Index a.session_state: st.session_state.thinking = False

# URL. Bot answers based *only* on this content.")
    url_input = --- Sidebar ---
with st.sidebar:
    st.header("Website Index st.text_input("URL:", value=st.session_state.get("indexed_url", "ing")
    st.markdown("Index a URL. The bot answers basedhttps://docs.smith.langchain.com/"))
    max_pages_input = st.number_input("Max Pages:", min_value=1, max_value=100, value=2 only on indexed content, trying to rewrite questions if needed.")
    url_input0)
    force_reindex_input = st.checkbox("Force = st.text_input("URL:", value=st.session_state Re-index?", value=False)
    index_button = st.button("Index.get("indexed_url", "https://docs.smith.langchain.com/"))
 Website", disabled=st.session_state.processing or not llm)    max_pages_input = st.number_input("Max Pages:",
    if not llm: st.warning("LLM not init.", min_value=1, max_value=100, value= icon="⚠️")

    if index_button:
        if not url_input or not is_valid20)
    force_reindex_input = st.checkbox("_url(url_input): st.error("Invalid URL.")
        else:
            st.session_state.processing = True; st.session_state.Force Re-index?", value=False)
    index_button = st.button("Indexmessages = []
            st.session_state.indexed_url = None Website", disabled=st.session_state.processing or not llm); st.session_state.vector_store = None
            st.session_state.ret
    if not llm: st.warning("Groq LLM not initializedriever = None; st.session_state.graph_app = None
. Indexing disabled.", icon="⚠️")

    if index_button:
        if not url_input or not is_valid_url(url_input): st.            st.rerun()

    if st.session_state.processing:
error("Please enter a valid URL.")
        else:
            st.        with st.spinner(f"Indexing {url_input}..."):
            vectorsession_state.processing = True; st.session_state.messages =_store = None; embedding_model = None
            if os.path.exists(vector_store []
            st.session_state.indexed_url = None; st.session_state.vector_store = None
            st.session_state.ret_path) and not force_reindex_input:
                try:riever = None; st.session_state.graph_app = None

                    st.info(f"Loading index: {vector_store_            st.rerun()

    if st.session_state.processing:
path}"); embedding_model = get_embedding_model(embedding_model_        with st.spinner(f"Indexing {url_input}..."):
            vectorname)
                    vector_store = FAISS.load_local(vector_store_path,_store = None; embedding_model = None
            if os.path.exists(vector_store embedding_model, allow_dangerous_deserialization=True); st._path) and not force_reindex_input:
                try:success("Index loaded.")
                    st.session_state.indexed_url
                    st.info(f"Loading index: {vector_store_path}"); embedding_model = get_embedding_model(embedding_model_name)
                    vector_store = FAISS.load_local(vector_store_path, = url_input
                except Exception as e: st.warning(f"Load embedding_model, allow_dangerous_deserialization=True); st. failed: {e}. Re-indexing."); vector_store = None
            if vector_store is None:success("Index loaded.")
                    st.session_state.indexed_url
                st.info(f"Running indexing: {url_input}..."); processed_docs = crawl_and_process(url_input, max_pages_input)
                if processed_docs: chunked_docs = url_input
                except Exception as e: st.warning(f"Load = chunk_documents(processed_docs);
                    if chunked_docs index failed: {e}. Re-indexing."); vector_store = None
:
                        if not embedding_model: embedding_model = get_embedding_model(embedding_            if vector_store is None:
                st.info(f"model_name)
                        vector_store = create_faiss_vectorRunning indexing: {url_input}..."); processed_docs = crawl_and_store(chunked_docs, embedding_model, vector_store__process(url_input, max_pages_input)
                ifpath)
                        if vector_store: st.success("Indexing OK!")
                        else: st.error("VS creation failed.")
                    else: st.error("Chunking failed.")
 processed_docs: chunked_docs = chunk_documents(processed_docs                else: st.error("Crawl/Process failed.")
            if vector_store);
                    if chunked_docs:
                        if not embedding_model: embedding_model =:
                st.session_state.vector_store = vector_store; get_embedding_model(embedding_model_name)
                        vector_ st.session_state.indexed_url = url_input
                st.session_state.retstore = create_faiss_vector_store(chunked_docs,riever = vector_store.as_retriever(search_kwargs={"k embedding_model, vector_store_path)
                        if vector_store: st.success("Indexing successful!")
                        else: st.error("": 3}) # Keep k=3 or 4
                # --- Build Graph with Retry ---
                if llm and st.session_state.retriever:Vector store creation failed.")
                    else: st.error("Chunking failed
                    try:
                        workflow = StateGraph(GraphState)
                        # Define nodes.")
                else: st.error("Crawling/Processing failed.")

                        workflow.add_node("retrieve", retrieve_node)
                        workflow.add_node("            if vector_store:
                st.session_state.vector_store = vector_store;grade_documents", grade_documents_node)
                        workflow.add_ st.session_state.indexed_url = url_input
                st.session_state.retnode("generate", generate_node)
                        workflow.add_node("riever = vector_store.as_retriever(search_kwargs={"ktransform_query", transform_query_node) # Re-add
                        workflow.add_": 4}) # Retrieve maybe 4 initially
                # REMOVED: Routernode("fail_node", fail_node) # Add failure node
                        # REMOVED: configuration
                # --- Build Graph ---
                if llm and st.session_state.retriever: web_search node

                        # Define edges
                        workflow.add_edge
                    try:
                        workflow = StateGraph(GraphState)
                        # Define nodes
                        workflow(START, "retrieve")
                        workflow.add_edge("retrieve", "grade_documents").add_node("retrieve", retrieve_node)
                        workflow.add
                        workflow.add_conditional_edges(
                            "grade_documents_node("grade_documents", grade_documents_node)
                        workflow",
                            decide_generate_or_rewrite_edge, # Routes.add_node("generate", generate_node)
                        workflow.add_node("transform_query", transform_query_node) # RE- to generate, transform, or fail
                            {"generate": "generate", "transform_query": "transform_query", "fail_node": "fail_node"}ADD rewrite node
                        # REMOVED: web_search node

                        # Define
                        )
                        workflow.add_edge("transform_query", "retrieve") # Loop back after rewrite
                        workflow.add_conditional_edges(
                            "generate", edges - REVISED FLOW
                        workflow.add_edge(START, "retrieve")
                            grade_generation_edge, # Routes to END, transform, or fail
                            {END
                        workflow.add_edge("retrieve", "grade_documents")
                        workflow.add_conditional_: END, "transform_query": "transform_query", "fail_node":edges(
                            "grade_documents",
                            decide_generate_or_rewrite_edge, # "fail_node"}
                        )
                        workflow.add_edge(" Use edge that decides rewrite
                            {"transform_query": "transform_query", "generate": "generate"}
                        )
                        workflow.add_edge("transform_query", "retrieve") # Loop back to retrieve after rewrite
                        workflow.add_conditional_edges(
                            "generate",
fail_node", END) # Failure node always ends

                        st.session_state.graph_app                            grade_generation_edge, # Use final grading edge
                            {END: END, = workflow.compile()
                        st.success("Agent graph compiled.")
                    except Exception as "transform_query": "transform_query"} # Added back option to rewrite e: st.error(f"Graph compile error: {e}")
 if generation bad
                        )

                        st.session_state.graph_                else: st.warning("LLM/Retriever not ready.")
            else: st.session_state.indexed_url = None
            st.session_state.processing = False; st.rerun()

    if st.session_state.indexed_app = workflow.compile()
                        st.success("Agent graph compiled.")url and not st.session_state.processing:
        st.success(f"Ready
                    except Exception as e: st.error(f"Graph compile error: {e}")
                : {st.session_state.indexed_url}")
        if notelse: st.warning("LLM/Retriever not ready. Graph not compiled.")
            else: st st.session_state.graph_app: st.warning("Agent graph failed.").session_state.indexed_url = None
            st.session_
    elif not st.session_state.processing: st.info("state.processing = False; st.rerun()

    if st.session_state.indexed_Index a website.")

# --- Chat Area ---
st.divider()
sturl and not st.session_state.processing:
        st.success.header("Chat")

if not st.session_state.indexed_url or not st.(f"Ready: {st.session_state.indexed_url}")
        if not stsession_state.graph_app:
    st.info("Index a website using the sidebar.")
else.session_state.graph_app: st.warning("Agent graph failed to compile.")
    elif:
    # Display messages
    for msg_data in st.session_state.messages not st.session_state.processing: st.info("Index a website via sidebar.")

# --- Chat:
        with st.chat_message(msg_data["role"]):
            st.markdown Area ---
st.divider()
st.header("Chat")

if not st.session_state.indexed_url or not st.session_state(msg_data["content"])
            if msg_data["role"] == "assistant" and "logs" in msg_data and msg_data["logs"]:
                 with st.expander("Agent.graph_app:
    st.info("Index a website using the sidebar to begin.")
else: Workings"): st.text("".join(msg_data["logs"]))
    # Display messages
    for msg_data in st.session_state.messages:


    # Chat input
    if prompt := st.chat_input("Ask about        with st.chat_message(msg_data["role"]):
             the indexed docs...", disabled=st.session_state.thinking):
        stst.markdown(msg_data["content"])
            if msg_data.session_state.messages.append({"role": "user", "content": prompt["role"] == "assistant" and "logs" in msg_data and msg_data["})
        with st.chat_message("user"): st.markdown(prompt)
        logs"]:
                 with st.expander("Agent Workings"): st.text("".joinst.session_state.thinking = True
        final_generation = "(msg_data["logs"]))

    # Chat input
    if prompt := st.Agent error."
        agent_run_logs = []

        with stchat_input("Ask a question...", disabled=st.session_state.thinking):
        st.session_state.messages.append({"role": "user", "content.chat_message("assistant"):
            log_placeholder = st.empty": prompt})
        with st.chat_message("user"): st.markdown(prompt)
        ()
            answer_placeholder = st.empty()
            answer_placeholder.markdown("▌")

            # Initialize state for the graph run
            inputs = {st.session_state.thinking = True
        final_generation = "
                "original_question": prompt, # Store original question
                "question": promptAgent error."
        agent_run_logs = []

        with st,          # Current question starts as original
                "log": [],
                "retries_left": MAX_RETRIES
            }
            captured_output = io.StringIO()
.chat_message("assistant"):
            log_placeholder = st.empty()
            answer_placeholder = st.empty()
            answer_placeholder            full_log_text = ""

            try:
                with context.markdown("▌") # Typing cursor

            # Prepare initial state
            inputs = {lib.redirect_stdout(captured_output):
                    # Use stream

                "question": prompt,
                "original_question": prompt,                    for output in st.session_state.graph_app.stream( # Store original question
                "log": [],
                "rewrite_attempts": 0 # Initializeinputs, {"recursion_limit": 15}): # Limit loops
                        current rewrite counter
            }
            captured_output = io.StringIO()
_output_log = captured_output.getvalue()
                        captured_output.seek(            full_log_text = ""

            try:
                with contextlib.redirect_stdout(captured0); captured_output.truncate(0)
                        full_log_text += current_output):
                    # Stream output
                    for output in st.session_state.graph_app_output_log
                        with log_placeholder.expander("Agent Workings...", expanded=True): st.stream(inputs, {"recursion_limit": 10}): # Limit.text(full_log_text)
                        last_state_key recursion
                        current_output_log = captured_output.getvalue()
                        captured_output.seek( = list(output.keys())[0]; last_state_value =0); captured_output.truncate(0) # Reset buffer
                        full_log_text output[last_state_key]

                final_state = last_state_value += current_output_log
                        with log_placeholder.expander("Agent Workings...", expanded=
                if final_state and 'generation' in final_state and finalTrue): st.text(full_log_text)
                        last_state_key = list(output_state['generation']: final_generation = final_state['generation']
.keys())[0]; last_state_value = output[last_state_key]

                final_state = last_state_value
                if final_state and '                elif final_state and 'question' in final_state: final_generation = f"(Agent stopped. Last state: '{final_state['question']generation' in final_state and final_state['generation']: final_generation}')"
                else: final_generation = "Agent finished unexpectedly."

                state_logs = = final_state['generation']
                elif final_state and 'question final_state.get("log", [])
                agent_run_logs = full_log' in final_state: final_generation = f"(Agent stopped after modifying_text.splitlines(keepends=True) #+ [f" question to: '{final_state['question']}')"
                else: final_STATE_LOG: {l}\n" for l in state_logs]generation = "Agent finished unexpectedly."

                state_logs = final_state.

            except Exception as e:
                final_generation = f"Anget("log", [])
                agent_run_logs = full_log error occurred: {e}"; agent_run_logs = full_log__text.splitlines(keepends=True) #+ [f"STATE_LOG: {l}\n" for l in state_logs]

text.splitlines(keepends=True) + [f"\n--- Error: {e} ---"]
                logging.error(f"Graph execution error: {e}", exc_info=True)

            answer_placeholder.markdown(final_generation            except Exception as e:
                final_generation = f"An error occurred: {e}"; agent_)
            with log_placeholder.expander("Agent Workings", expandedrun_logs = full_log_text.splitlines(keepends==False): st.text("".join(agent_run_logs))True) + [f"\n--- Error: {e} ---"]


            st.session_state.messages.append({"role": "assistant", "content": final_generation, "logs": agent_run_logs                logging.error(f"Graph execution error: {e}", exc_})

        st.session_state.thinking = False
        # st.rerun()info=True)

            answer_placeholder.markdown(final_generation) # Update final answer
            with