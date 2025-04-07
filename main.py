import streamlit as st
# import argparse # Not needed for pure Streamlit app
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
import io
import contextlib

# --- Core Langchain & Document Processing ---
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.schema import Document
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage # Added AIMessage

# --- LLM ---
from langchain_groq import ChatGroq

# --- Tools ---
from langchain.tools.retriever import create_retriever_tool # Needed for the agentic part
from langchain_community.tools.tavily_search import TavilySearchResults # Keep import if needed elsewhere, but remove usage

# --- Graph & State ---
from typing import List, Literal, Sequence, Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages # For message history state

# --- Prompts & Parsing ---
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from pydantic import BaseModel, Field

# #############################################################################
# Load Environment Variables & Basic Config
# #############################################################################
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
# REMOVED: tavily_api_key = os.getenv("TAVILY_API_KEY")
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
embedding_model_name = os.getenv("EMBEDDING_MODEL_NAME", DEFAULT_EMBEDDING_MODEL)
DEFAULT_VECTOR_STORE_PATH = "./faiss_agentic_rag_store" # New path
vector_store_path = os.getenv("VECTOR_STORE_PATH", DEFAULT_VECTOR_STORE_PATH)
GROQ_MODEL_NAME = "llama3-8b-8192"
MAX_REWRITES = 1 # Limit rewrite attempts

# #############################################################################
# Logging Setup
# #############################################################################
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# #############################################################################
# Crawler, Processor, Indexing Functions (Paste your fully corrected versions)
# #############################################################################
# --- Paste your corrected functions here ---
# HEADERS, MAIN_CONTENT_SELECTORS, NOISE_TAGS, TEXT_BEARING_TAGS
# is_valid_url, fetch_page, crawl_website, format_list_item, format_table,
# get_cleaned_text, process_element_recursive, extract_meaningful_content,
# process_crawled_data, crawl_and_process, chunk_documents,
# get_embedding_model, create_faiss_vector_store
HEADERS = { 'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36' }
MAIN_CONTENT_SELECTORS = [ "main", "article", "div.article-body", "div.content__body", "div.DocsPage__body", "div#main", "div#main-content", "div.main-content", "div#content", "div.content", "div[role='main']", ]
NOISE_TAGS = [ "nav", "header", "footer", "aside", "script", "style", "noscript", "button", "form", "meta", "link", "svg", "path", ".sidebar", ".navigation", ".footer", ".header", ".toc", ".breadcrumb", "#sidebar", "#navigation", "#footer", "#header", "#toc", "#breadcrumb", "*[aria-hidden='true']", ]
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
    logging.info(f"Chunking {len(docs)} docs (size={chunk_size}, overlap={chunk_overlap})")
    splitter=RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len, add_start_index=True)
    chunks=[]
    for i, doc in enumerate(docs):
        if txt := doc.get('text'):
            try: chunks.extend(splitter.create_documents([txt], metadatas=[{'url':doc.get('url', ''), 'title':doc.get('title', '')}]))
            except Exception as e: logging.error(f"Error chunking doc {i} ({doc.get('url')}): {e}")
    logging.info(f"Split into {len(chunks)} chunks."); return chunks
def get_embedding_model(model_name):
    logging.info(f"Loading embeddings: {model_name}"); emb = None
    try: emb = SentenceTransformerEmbeddings(model_name=model_name, cache_folder=os.getenv("SENTENCE_TRANSFORMERS_HOME"))
    except Exception as e: logging.error(f"Embedding load failed: {e}", exc_info=True); raise
    logging.info("Embeddings loaded."); return emb
def create_faiss_vector_store(chunks, embeddings, path=None):
    logging.info(f"Creating FAISS store ({len(chunks)} chunks)"); vs = None
    if not chunks: logging.warning("No chunks for vector store."); return None
    if not embeddings: logging.error("No embedding model for vector store."); return None
    try:
        vs = FAISS.from_documents(documents=chunks, embedding=embeddings); logging.info("FAISS store created in memory.")
        if path and vs:
            try: vs.save_local(path); logging.info(f"FAISS saved: {path}")
            except Exception as e_save: logging.error(f"FAISS save failed: {e_save}", exc_info=True)
    except Exception as e_create: logging.error(f"FAISS creation failed: {e_create}", exc_info=True); return None
    return vs

# --- LLM ---
llm = None
if not groq_api_key: st.error("GROQ_API_KEY missing.", icon="🚨")
else:
    try: llm = ChatGroq(temperature=0, model_name=GROQ_MODEL_NAME, groq_api_key=groq_api_key)
    except Exception as e: st.error(f"Failed init Groq LLM: {e}", icon="🚨"); logging.error(f"Failed init Groq LLM: {e}")

# --- REMOVED: Web Search Tool ---

# --- Retriever (initialized later) ---
retriever = None

# --- RAG Chain & Graders & Rewriter ---
rag_chain = None; retrieval_grader = None; hallucination_grader = None; answer_grader = None; question_rewriter = None

if llm:
    # RAG Chain
    try: prompt = hub.pull("rlm/rag-prompt"); rag_chain = prompt | llm | StrOutputParser(); logging.info("RAG chain OK.")
    except Exception as e: logging.error(f"Failed init RAG chain: {e}")
    # Retrieval Grader
    try:
        class GradeDocuments(BaseModel): binary_score: str = Field(description="Is the document relevant? 'yes' or 'no'.")
        structured_llm_ret_grader = llm.with_structured_output(GradeDocuments)
        system_ret_grade = "Assess doc relevance. Grade 'yes' if keywords/meaning match question, else 'no'."
        grade_prompt = ChatPromptTemplate.from_messages([("system", system_ret_grade), ("human", "Doc:\n{document}\n\nQ: {question}")])
        retrieval_grader = grade_prompt | structured_llm_ret_grader; logging.info("Retrieval grader OK.")
    except Exception as e: logging.error(f"Failed init retrieval grader: {e}")
    # Hallucination Grader
    try:
        class GradeHallucinations(BaseModel): binary_score: str = Field(description="Is answer grounded in facts? 'yes' or 'no'.")
        structured_llm_hallu_grader = llm.with_structured_output(GradeHallucinations)
        system_hallu = "Assess if answer is grounded in facts. Score 'yes' if grounded, 'no' otherwise."
        hallucination_prompt = ChatPromptTemplate.from_messages([("system", system_hallu), ("human", "Facts:\n{documents}\n\nAnswer: {generation}")])
        hallucination_grader = hallucination_prompt | structured_llm_hallu_grader; logging.info("Hallucination grader OK.")
    except Exception as e: logging.error(f"Failed init hallucination grader: {e}")
    # Answer Grader
    try:
        class GradeAnswer(BaseModel): binary_score: str = Field(description="Does answer address question? 'yes' or 'no'.")
        structured_llm_ans_grader = llm.with_structured_output(GradeAnswer)
        system_ans = "Assess if answer addresses/resolves question. Score 'yes' if it does, 'no' otherwise."
        answer_prompt = ChatPromptTemplate.from_messages([("system", system_ans), ("human", "Q:\n{question}\n\nAnswer: {generation}")])
        answer_grader = answer_prompt | structured_llm_ans_grader; logging.info("Answer grader OK.")
    except Exception as e: logging.error(f"Failed init answer grader: {e}")
    # Question Rewriter
    try:
        system_rew = "You are a question re-writer optimizing a question for vectorstore retrieval based on underlying semantic intent."
        re_write_prompt = ChatPromptTemplate.from_messages([("system", system_rew), ("human", "Initial question: {question}\nFormulate an improved question.")])
        question_rewriter = re_write_prompt | llm | StrOutputParser(); logging.info("Question rewriter OK.")
    except Exception as e: logging.error(f"Failed init question rewriter: {e}")

# #############################################################################
# Graph State & Nodes
# #############################################################################

class GraphState(TypedDict):
    """Represents the state of our graph."""
    original_question: str      # Store the initial question
    question: str           # Current question (might be rewritten)
    generation: str | None
    documents: List[Document]
    log: List[str]
    rewrite_attempts: int       # Counter for retries

def retrieve_node(state):
    log_entry = "--- NODE: RETRIEVE ---"; print(log_entry)
    question = state["question"]; current_log = state.get("log", []); documents = []
    if 'retriever' not in st.session_state or st.session_state.retriever is None:
        err_msg = "ERROR: Retriever not ready."; print(err_msg); log_updates = [log_entry, err_msg]
    else:
        try:
            documents = st.session_state.retriever.invoke(question); msg=f"Retrieved {len(documents)} docs for Q: '{question[:50]}...'"
            print(msg); log_updates = [log_entry, msg]
        except Exception as e: err_msg = f"ERROR during retrieval: {e}"; print(err_msg); log_updates = [log_entry, err_msg]
    # Preserve other state fields
    return {"documents": documents, **{k:v for k,v in state.items() if k != 'documents'}, "log": current_log + log_updates}

def grade_documents_node(state):
    log_entry = "--- NODE: GRADE DOCUMENTS ---"; print(log_entry)
    question = state["question"]; documents = state["documents"]; current_log = state.get("log", []); log_updates = [log_entry]
    filtered_docs = []
    if not documents: log_updates.append("No docs to grade.")
    elif retrieval_grader is None: log_updates.append("WARN: Grader not ready, keeping docs."); filtered_docs = documents
    else:
        for i, d in enumerate(documents):
            try:
                score = retrieval_grader.invoke({"question": question, "document": d.page_content}); grade = score.binary_score.lower()
                if grade == "yes": msg = f"+ Doc {i+1}: Relevant"; print(msg); log_updates.append(msg); filtered_docs.append(d)
                else: msg = f"- Doc {i+1}: Irrelevant"; print(msg); log_updates.append(msg)
            except Exception as e: msg = f"Error grading doc {i+1}: {e}"; print(msg); log_updates.append(msg); filtered_docs.append(d) # Keep if grading fails?
        log_updates.append(f"Relevant docs: {len(filtered_docs)}.")
    # Preserve other state fields
    return {"documents": filtered_docs, **{k:v for k,v in state.items() if k != 'documents'}, "log": current_log + log_updates}

def generate_node(state):
    log_entry = f"--- NODE: GENERATE ---"; print(log_entry)
    question = state["question"]; documents = state["documents"]; current_log = state.get("log", []); log_updates = [log_entry]
    generation = None
    original_q = state.get("original_question", question)

    if not documents:
        msg = "No relevant documents found to generate answer."; print(msg); log_updates.append(msg)
        generation = f"Sorry, I couldn't find specific information about '{original_q}' in the indexed documentation."
    elif rag_chain is None:
        err_msg="ERROR: RAG chain not ready."; print(err_msg); log_updates.append(err_msg)
        generation = "Error: Generation component not ready."
    else:
        log_updates.append(f"Generating answer based on {len(documents)} relevant doc(s).")
        try:
             formatted_docs = "\n\n".join([doc.page_content for doc in documents])
             generation = rag_chain.invoke({"context": formatted_docs, "question": question})
             msg = f"Generated response snippet: {generation[:100]}..."; print(msg); log_updates.append(msg)
        except Exception as e: msg = f"Error during RAG generation: {e}"; print(msg); log_updates.append(msg); generation = "Error generating answer."

    if generation is None: generation = "Unexpected issue during generation."
    # Preserve other state fields
    return {"generation": generation, **{k:v for k,v in state.items() if k != 'generation'}, "log": current_log + log_updates}

def transform_query_node(state):
    """Transform the query based on the original question."""
    log_entry = "--- NODE: TRANSFORM QUERY ---"; print(log_entry)
    original_question = state["original_question"]
    current_log = state.get("log", []); log_updates = [log_entry]
    rewrite_attempts = state.get("rewrite_attempts", 0) + 1
    msg = f"Rewrite attempt {rewrite_attempts}/{MAX_REWRITES} based on: '{original_question}'"
    print(msg); log_updates.append(msg)
    rewritten_question = original_question # Default

    if question_rewriter is None: msg="WARN: Rewriter not ready."; print(msg); log_updates.append(msg)
    else:
        try: rewritten_question = question_rewriter.invoke({"question": original_question})
        except Exception as e: msg=f"Error rewriting question: {e}"; print(msg); log_updates.append(msg);
    msg = f"Rewritten question: {rewritten_question}"; print(msg); log_updates.append(msg)
    # Update question, keep original, increment counter, clear docs
    return {
        "documents": [], # Clear irrelevant docs before re-retrieving
        "question": rewritten_question,
        "original_question": original_question,
        "rewrite_attempts": rewrite_attempts,
        "log": current_log + log_updates,
        "generation": None # Clear previous generation
    }

def fail_node(state):
    """Node that sets the 'not found' message after exhausting retries."""
    log_entry = "--- NODE: FAIL (Graceful Exit) ---"; print(log_entry)
    current_log = state.get("log", [])
    original_question = state.get("original_question", "your query")
    fail_message = f"Sorry, I tried searching and rewriting the question, but couldn't find relevant information for '{original_question}' in the indexed documentation."
    # Set generation to the failure message and clear documents
    return {
        "generation": fail_message,
        "documents": [], # Ensure docs are empty
        "log": current_log + [log_entry],
        **{k:v for k,v in state.items() if k not in ['generation', 'documents', 'log']}
        }

# REMOVED: web_search_node

# #############################################################################
# Graph Edges (Conditional Logic)
# #############################################################################

# REMOVED: route_question_edge

def decide_generate_or_rewrite_edge(state):
    """Decide whether to generate, rewrite, or fail based on graded docs and rewrites."""
    log_entry = "--- EDGE: DECIDE GENERATE or REWRITE ---"; print(log_entry)
    current_log = state.get("log", []); log_updates = [log_entry]
    decision = "generate" # Assume generate first

    if not state.get("documents"): # Check if list is empty after grading
        rewrite_attempts = state.get("rewrite_attempts", 0)
        if rewrite_attempts < MAX_REWRITES:
            msg = f"Decision: No relevant docs (attempt {rewrite_attempts+1}), transform query."; print(msg); decision = "transform_query"
        else:
            msg = f"Decision: No relevant docs after {rewrite_attempts} rewrite(s), failing gracefully."; print(msg)
            decision = "fail_node" # Route to failure node
    else:
        msg="Decision: Relevant docs found, generate answer."; print(msg); decision = "generate"

    log_updates.append(msg)
    state["log"] = current_log + log_updates
    return decision

def format_docs_for_grade(docs: List[Document]) -> str:
     return "\n\n".join([f"--- Doc {i+1} URL: {doc.metadata.get('url', 'N/A')} ---\n{doc.page_content}" for i, doc in enumerate(docs)])

def grade_generation_edge(state):
    """Determines whether the generation is useful or if we should rewrite or fail."""
    log_entry = "--- EDGE: GRADE GENERATION ---"; print(log_entry)
    # Grade generation against the *original* question for usefulness check
    original_question = state["original_question"]
    documents = state["documents"]; generation = state["generation"]
    rewrite_attempts = state.get("rewrite_attempts", 0)
    current_log = state.get("log", []); log_updates = [log_entry]
    decision = END # Default to useful

    # Check for failed generation first
    if not generation or "error" in generation.lower() or "not ready" in generation.lower():
        msg="Decision: Generation failed/errored."; print(msg); log_updates.append(msg)
        if rewrite_attempts < MAX_REWRITES: decision = "transform_query"
        else: decision = "fail_node" # Give up if retries exhausted
    elif hallucination_grader is None or answer_grader is None:
        msg="WARN: Graders not ready, accepting generation as final."; print(msg); log_updates.append(msg); decision = END
    else:
        try:
            formatted_docs = format_docs_for_grade(documents)
            # 1. Check Hallucinations
            hallu_score = hallucination_grader.invoke({"documents": formatted_docs, "generation": generation})
            if hallu_score.binary_score.lower() == "no":
                msg="Decision: Generation may hallucinate."; print(msg); log_updates.append(msg)
                if rewrite_attempts < MAX_REWRITES: decision = "transform_query"
                else: decision = "fail_node"
            else:
                # 2. Check Answer Relevance (if not hallucinating)
                msg = "--- Generation Grounded ---"; print(msg); log_updates.append(msg)
                ans_score = answer_grader.invoke({"question": original_question, "generation": generation}) # Grade against original
                if ans_score.binary_score.lower() == "yes":
                    msg = "Decision: Generation is useful."; print(msg); log_updates.append(msg); decision = END
                else:
                    msg = "Decision: Generation doesn't address question well."; print(msg); log_updates.append(msg)
                    if rewrite_attempts < MAX_REWRITES: decision = "transform_query"
                    else: decision = "fail_node"
        except Exception as e:
            msg = f"Error during final grading: {e}. Accepting generation."; print(msg); log_updates.append(msg); decision = END

    state["log"] = current_log + log_updates
    return decision

# #############################################################################
# Streamlit UI Application
# #############################################################################

# --- UI ---
st.set_page_config(page_title="Doc Q&A Bot (Rewrite)", layout="wide")
st.title("📚 Documentation Q&A Bot with Rewrite")

# --- Session State Init ---
if "messages" not in st.session_state: st.session_state.messages = []
if "indexed_url" not in st.session_state: st.session_state.indexed_url = None
if "vector_store" not in st.session_state: st.session_state.vector_store = None
if "retriever" not in st.session_state: st.session_state.retriever = None
if "graph_app" not in st.session_state: st.session_state.graph_app = None
if "processing" not in st.session_state: st.session_state.processing = False
if "thinking" not in st.session_state: st.session_state.thinking = False

# --- Sidebar ---
with st.sidebar:
    st.header("Website Indexing")
    st.markdown("Index a URL. Bot answers based *only* on this content, rewriting questions if needed.")
    url_input = st.text_input("URL:", value=st.session_state.get("indexed_url", "https://docs.smith.langchain.com/"))
    max_pages_input = st.number_input("Max Pages:", min_value=1, max_value=100, value=20)
    force_reindex_input = st.checkbox("Force Re-index?", value=False)
    index_button = st.button("Index Website", disabled=st.session_state.processing or not llm)
    if not llm: st.warning("LLM not init.", icon="⚠️")

    if index_button:
        if not url_input or not is_valid_url(url_input): st.error("Invalid URL.")
        else:
            st.session_state.processing = True; st.session_state.messages = []
            st.session_state.indexed_url = None; st.session_state.vector_store = None
            st.session_state.retriever = None; st.session_state.graph_app = None
            st.rerun()

    if st.session_state.processing:
        with st.spinner(f"Indexing {url_input}..."):
            vector_store = None; embedding_model = None
            if os.path.exists(vector_store_path) and not force_reindex_input:
                try:
                    st.info(f"Loading index: {vector_store_path}"); embedding_model = get_embedding_model(embedding_model_name)
                    vector_store = FAISS.load_local(vector_store_path, embedding_model, allow_dangerous_deserialization=True); st.success("Index loaded.")
                    st.session_state.indexed_url = url_input
                except Exception as e: st.warning(f"Load failed: {e}. Re-indexing."); vector_store = None
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
            if vector_store:
                st.session_state.vector_store = vector_store; st.session_state.indexed_url = url_input
                st.session_state.retriever = vector_store.as_retriever(search_kwargs={"k": 4}) # Retrieve 4 initially
                # --- Build Graph with Retry ---
                if llm and st.session_state.retriever:
                    try:
                        workflow = StateGraph(GraphState)
                        # Add nodes
                        workflow.add_node("retrieve", retrieve_node)
                        workflow.add_node("grade_documents", grade_documents_node)
                        workflow.add_node("generate", generate_node)
                        workflow.add_node("transform_query", transform_query_node) # Re-add
                        workflow.add_node("fail_node", fail_node) # Add failure node
                        # REMOVED: web_search node

                        # Define edges
                        workflow.add_edge(START, "retrieve")
                        workflow.add_edge("retrieve", "grade_documents")
                        workflow.add_conditional_edges(
                            "grade_documents",
                            decide_generate_or_rewrite_edge, # Routes to generate, transform, or fail
                            {"generate": "generate", "transform_query": "transform_query", "fail_node": "fail_node"}
                        )
                        workflow.add_edge("transform_query", "retrieve") # Loop back after rewrite
                        workflow.add_conditional_edges(
                            "generate",
                            grade_generation_edge, # Routes to END, transform, or fail
                            {END: END, "transform_query": "transform_query", "fail_node": "fail_node"}
                        )
                        workflow.add_edge("fail_node", END) # Failure node always ends

                        st.session_state.graph_app = workflow.compile()
                        st.success("Agent graph compiled.")
                    except Exception as e: st.error(f"Graph compile error: {e}")
                else: st.warning("LLM/Retriever not ready.")
            else: st.session_state.indexed_url = None
            st.session_state.processing = False; st.rerun()

    if st.session_state.indexed_url and not st.session_state.processing:
        st.success(f"Ready: {st.session_state.indexed_url}")
        if not st.session_state.graph_app: st.warning("Agent graph failed.")
    elif not st.session_state.processing: st.info("Index a website.")

# --- Chat Area ---
st.divider()
st.header("Chat")

if not st.session_state.indexed_url or not st.session_state.graph_app:
    st.info("Index a website using the sidebar to begin.")
else:
    # Display messages
    for msg_data in st.session_state.messages:
        with st.chat_message(msg_data["role"]):
            st.markdown(msg_data["content"])
            if msg_data["role"] == "assistant" and "logs" in msg_data and msg_data["logs"]:
                 with st.expander("Agent Workings"): st.text("".join(msg_data["logs"]))

    # Chat input
    if prompt := st.chat_input("Ask about the indexed docs...", disabled=st.session_state.thinking):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)
        st.session_state.thinking = True
        final_generation = "(Agent error.)"
        agent_run_logs = []

        with st.chat_message("assistant"):
            log_placeholder = st.empty()
            answer_placeholder = st.empty()
            answer_placeholder.markdown("▌") # Typing cursor

            # Initialize state for the graph run
            inputs = {
                "original_question": prompt, # Store original question
                "question": prompt,          # Current question starts as original
                "log": [],
                "rewrite_attempts": 0         # Initialize rewrite counter
            }
            captured_output = io.StringIO()
            full_log_text = ""
            last_state_value = None # Define to store final state

            try:
                with contextlib.redirect_stdout(captured_output):
                    # Use stream
                    for output in st.session_state.graph_app.stream(inputs, {"recursion_limit": 15}): # Limit loops
                        current_output_log = captured_output.getvalue()
                        captured_output.seek(0); captured_output.truncate(0)
                        full_log_text += current_output_log
                        with log_placeholder.expander("Agent Workings...", expanded=True): st.text(full_log_text)
                        last_state_key = list(output.keys())[0]; last_state_value = output[last_state_key]

                final_state = last_state_value
                if final_state and 'generation' in final_state and final_state['generation']:
                    final_generation = final_state['generation']
                elif final_state and 'question' in final_state: # Ended possibly after rewrite/fail
                    # Attempt to get a more informative message if generation is missing
                    final_generation = final_state.get("generation", f"(Agent stopped. Last Q: '{final_state['question']}')")
                else: final_generation = "Agent finished unexpectedly (No generation found)."

                state_logs = final_state.get("log", []) if final_state else []
                agent_run_logs = full_log_text.splitlines(keepends=True) #+ [f"STATE_LOG: {l}\n" for l in state_logs]

            except Exception as e:
                final_generation = f"An error occurred: {e}"; agent_run_logs = full_log_text.splitlines(keepends=True) + [f"\n--- Error: {e} ---"]
                logging.error(f"Graph execution error: {e}", exc_info=True)

            answer_placeholder.markdown(final_generation)
            with log_placeholder.expander("Agent Workings", expanded=False): st.text("".join(agent_run_logs))

            st.session_state.messages.append({"role": "assistant", "content": final_generation, "logs": agent_run_logs})

        st.session_state.thinking = False
        # st.rerun()