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
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.schema.document import Document
from langchain.tools.retriever import create_retriever_tool
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage, AIMessage
from langchain_groq import ChatGroq
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field # For grading schema

# LangGraph specific imports
from typing import Annotated, Sequence, Literal
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages # Utility for AgentState
from langgraph.prebuilt import ToolNode, tools_condition # Prebuilt nodes/conditions

# --- Environment Variable Loading ---
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY") # Keep for potential future OpenAI use
groq_api_key = os.getenv("GROQ_API_KEY") # Load Groq key
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
embedding_model_name = os.getenv("EMBEDDING_MODEL_NAME", DEFAULT_EMBEDDING_MODEL)
DEFAULT_VECTOR_STORE_PATH = "./faiss_vector_store"
vector_store_path = os.getenv("VECTOR_STORE_PATH", DEFAULT_VECTOR_STORE_PATH)

# Basic check for Groq key
if not groq_api_key:
    logging.warning("GROQ_API_KEY not found in .env file. The agent requiring Groq LLM will likely fail.")
    # You might want to exit here if Groq is essential: exit("Error: Groq API Key missing.")

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Crawler/Processor Configuration & Helper Functions ---
# (Keep all functions from the previous step: HEADERS, SELECTORS, TAGS,
# is_valid_url, fetch_page, crawl_website, format_list_item, format_table,
# get_cleaned_text, process_element_recursive, extract_meaningful_content,
# process_crawled_data, crawl_and_process, chunk_documents,
# get_embedding_model, create_faiss_vector_store)
# ... (implementation omitted for brevity, assume they are all here) ...
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
def is_valid_url(url):
    parsed = urlparse(url)
    return bool(parsed.scheme) and bool(parsed.netloc)
def fetch_page(url, session, timeout=10):
    try:
        response = session.get(url, headers=HEADERS, timeout=timeout, allow_redirects=True)
        response.raise_for_status(); content_type = response.headers.get('content-type', '').lower()
        if 'text/html' not in content_type: return None, response.url
        if len(response.content) < 100: return None, response.url
        response.encoding = response.apparent_encoding or 'utf-8'; return response.text, response.url
    except Exception as e: logging.error(f"Error fetching {url}: {e}"); return None, url
def crawl_website(start_url, max_pages):
    if not is_valid_url(start_url): logging.error(f"Invalid start URL: {start_url}"); return None
    parsed_start_url = urlparse(start_url); base_domain = parsed_start_url.netloc
    urls_to_visit = deque([start_url]); visited_urls = set(); crawled_data = {}
    with requests.Session() as session:
        while urls_to_visit and len(crawled_data) < max_pages:
            current_url = urls_to_visit.popleft(); parsed_current = urlparse(current_url)
            normalized_url = parsed_current._replace(fragment="").geturl()
            if normalized_url in visited_urls: continue
            visited_urls.add(normalized_url); logging.info(f"Crawling: {current_url} ({len(crawled_data)+1}/{max_pages})")
            html_content, final_url = fetch_page(current_url, session)
            parsed_final = urlparse(final_url); normalized_final_url = parsed_final._replace(fragment="").geturl()
            visited_urls.add(normalized_final_url)
            if html_content:
                final_domain = urlparse(final_url).netloc
                if final_domain != base_domain: continue
                if final_url not in crawled_data and len(crawled_data) < max_pages: crawled_data[final_url] = html_content
                soup = BeautifulSoup(html_content, 'html.parser')
                for link in soup.find_all('a', href=True):
                    href = link['href']; next_url = urljoin(final_url, href); parsed_next_url = urlparse(next_url)
                    normalized_next_url = parsed_next_url._replace(fragment="").geturl()
                    if parsed_next_url.scheme in ['http', 'https'] and parsed_next_url.netloc == base_domain and \
                       normalized_next_url not in visited_urls and normalized_next_url not in urls_to_visit:
                        urls_to_visit.append(normalized_next_url)
            time.sleep(0.1)
    if not crawled_data: logging.warning(f"No content retrieved from {start_url}"); return None
    logging.info(f"Crawled {len(crawled_data)} pages."); return crawled_data
def format_list_item(tag, index):
    parent_type = tag.parent.name if tag.parent else 'ul'; prefix = f"{index + 1}. " if parent_type == 'ol' else "- "
    text = get_cleaned_text(tag); return prefix + text if text else ""
def format_table(tag):
    """Formats a <table> tag into a simple text representation."""
    rows = []
    for row in tag.find_all('tr'): # Find all table rows
        # Get cleaned text from all table cells (<td> or <th>) in the row
        cells = [get_cleaned_text(cell) for cell in row.find_all(['td', 'th'])]
        # Filter out rows that become empty after cleaning
        # Ensure this 'if' statement is indented exactly one level under the 'for' loop
        if any(cells):
            rows.append(" | ".join(cells)) # Join cells with " | "
    # Ensure this 'return' statement is aligned with the 'for' loop (one level out)
    return "\n".join(rows) # Join rows with newlines

def get_cleaned_text(element):
    if not element: return ""; text = element.get_text(separator=' ', strip=True); text = html.unescape(text)
    text = re.sub(r'\s+', ' ', text).strip(); return text
def process_element_recursive(element, extracted_blocks, processed_elements):
    if element in processed_elements or not element.name or not element.parent: return
    element_text = ""; should_process_children = True
    if element.name in ['ul', 'ol']:
        items = element.find_all('li', recursive=False)
        list_text = [fmt for i, item in enumerate(items) if (fmt := format_list_item(item, i))]
        element_text = "\n".join(list_text); should_process_children = False
    elif element.name == 'table': element_text = format_table(element); should_process_children = False
    elif element.name.startswith('h') and element.name[1].isdigit():
        level = int(element.name[1]); cleaned_text = get_cleaned_text(element)
        if cleaned_text: element_text = "#" * level + " " + cleaned_text; should_process_children = False
    elif element.name in ['p', 'pre', 'blockquote']: element_text = get_cleaned_text(element); should_process_children = False
    elif element.name in TEXT_BEARING_TAGS:
        if should_process_children:
            for child in element.find_all(True, recursive=False): process_element_recursive(child, extracted_blocks, processed_elements)
        direct_text = ''.join(element.find_all(string=True, recursive=False)).strip()
        block_children = {'p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'ul', 'ol', 'table', 'pre', 'blockquote'}
        if direct_text and not element.find(lambda tag: tag.name in block_children): element_text = get_cleaned_text(element)
    if element_text: extracted_blocks.append(element_text.strip())
    if element_text or not should_process_children: processed_elements.add(element); processed_elements.update(element.find_all(True))
    elif should_process_children:
        for child in element.find_all(True, recursive=False): process_element_recursive(child, extracted_blocks, processed_elements)
def extract_meaningful_content(url, html_content):
    if not html_content: return None
    try:
        soup = BeautifulSoup(html_content, 'html.parser'); page_title = get_cleaned_text(soup.title) if soup.title else "No Title"
        for noise_selector in NOISE_TAGS:
            for element in soup.select(noise_selector): element.decompose()
        for comment in soup.find_all(string=lambda text: isinstance(text, Comment)): comment.extract()
        main_content_area = None
        for selector in MAIN_CONTENT_SELECTORS:
            if main_content_area := soup.select_one(selector): break
        if not main_content_area: main_content_area = soup.body
        if not main_content_area: return None
        extracted_blocks = []; processed_elements = set()
        for element in main_content_area.find_all(True, recursive=False): process_element_recursive(element, extracted_blocks, processed_elements)
        full_text = "\n\n".join(block for block in extracted_blocks if block); full_text = re.sub(r'\n{3,}', '\n\n', full_text).strip()
        if not full_text or len(full_text) < 30: return None
        logging.info(f"Extracted ~{len(full_text)} chars from {url}"); return {"url": url, "title": page_title, "text": full_text}
    except Exception as e: logging.error(f"Error processing {url}: {e}"); return None
def process_crawled_data(website_data):
    processed_docs = [];
    if not website_data: return processed_docs
    for url, html_content in website_data.items():
        if extracted_data := extract_meaningful_content(url, html_content): processed_docs.append(extracted_data)
    logging.info(f"Extracted content from {len(processed_docs)} pages."); return processed_docs
def crawl_and_process(start_url: str, max_pages: int = 50) -> list[dict]:
    logging.info(f"--- Crawl & Process: {start_url} (Max: {max_pages}) ---"); website_data = crawl_website(start_url, max_pages)
    if not website_data: return []; processed_documents = process_crawled_data(website_data)
    if not processed_documents: return []; logging.info(f"--- Crawl & Process OK: {len(processed_documents)} docs ---"); return processed_documents
def chunk_documents(processed_docs: list[dict], chunk_size: int = 1000, chunk_overlap: int = 150) -> list[Document]:
    logging.info(f"--- Chunking {len(processed_docs)} docs ---"); text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len, add_start_index=True)
    all_chunks = []
    for doc in processed_docs:
        if not doc.get('text'): continue
        chunks = text_splitter.create_documents([doc['text']], metadatas=[{'url': doc['url'], 'title': doc['title']}])
        all_chunks.extend(chunks)
    logging.info(f"--- Chunking OK: {len(all_chunks)} chunks ---"); return all_chunks
def get_embedding_model(model_name: str):
    """
    Initializes and returns a SentenceTransformer embedding model wrapped by LangChain.

    Args:
        model_name: The name of the sentence-transformer model to use
                    (e.g., 'all-MiniLM-L6-v2').

    Returns:
        A LangChain Embeddings object.

    Raises:
        Exception: Re-raises any exception encountered during model loading.
    """
    logging.info(f"--- Loading Embedding Model: {model_name} ---")
    embeddings = None # Initialize to None
    try:
        # Use the Langchain wrapper for consistency
        # It will download the model on first use if not cached
        embeddings = SentenceTransformerEmbeddings(
            model_name=model_name,
            cache_folder=os.getenv("SENTENCE_TRANSFORMERS_HOME") # Optional: Specify cache dir via env var
            # Add other SentenceTransformer options if needed, e.g., device='cuda'
        )
        # If the above line succeeds without error, the embedding object is created.

    except Exception as e:
        # If any error occurs during SentenceTransformerEmbeddings initialization
        logging.error(f"Failed to load embedding model '{model_name}': {e}", exc_info=True)
        raise # Re-raise the exception to halt the process if embeddings fail

    # If the try block completed without raising an exception, embeddings should be valid.
    # We place the success message and return statement *after* the try...except block.
    logging.info("--- Embedding Model OK ---")
    return embeddings
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


# --- LangGraph Agent Components ---

# 1. Agent State
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

# 2. Tool (Define this globally after vector_store is potentially loaded/created)
retriever_tool = None

# 3. LLM (Using Groq)
# Ensure GROQ_API_KEY is loaded via load_dotenv() earlier
# Select a Groq model (check available models on GroqCloud)
# Common choices: "llama3-8b-8192", "mixtral-8x7b-32768", "gemma-7b-it"
groq_model_name = "llama3-8b-8192"
try:
    llm = ChatGroq(temperature=0, model_name=groq_model_name)
    # Test invocation (optional)
    # llm.invoke("Test prompt")
    logging.info(f"Initialized Groq LLM with model: {groq_model_name}")
except Exception as e:
    logging.error(f"Failed to initialize Groq LLM: {e}. Agent will likely fail.")
    llm = None # Set to None to handle gracefully later if needed

# 4. Node Functions
def agent_node(state):
    """Decides whether to use the retriever tool or end."""
    print("--- Calling Agent Node ---")
    if llm is None or retriever_tool is None: # Check if dependencies are ready
        print("LLM or Retriever Tool not initialized. Ending.")
        # Return a message indicating the issue or just end
        return {"messages": [AIMessage(content="Agent components not ready. Cannot proceed.")]}

    messages = state["messages"]
    # Bind the retriever tool to the LLM
    llm_with_tools = llm.bind_tools([retriever_tool])
    # Invoke the LLM with the current conversation history
    response = llm_with_tools.invoke(messages)
    print(f"Agent Response: {response.content[:100]}...") # Log snippet
    # The response will be an AIMessage. If it contains tool_calls, the graph routes to the ToolNode. Otherwise, it might end.
    return {"messages": [response]}

# ToolNode is prebuilt, handles calling the retriever_tool
retrieve_node = None # Will be initialized later if retriever_tool is created

def rewrite_node(state):
    """Rewrites the user query for better retrieval."""
    print("--- Rewriting Query ---")
    if llm is None: return {"messages": [AIMessage(content="LLM not ready. Cannot rewrite.")]}

    messages = state["messages"]
    # Typically, the original question is the first human message
    user_question = ""
    for msg in messages:
        if isinstance(msg, HumanMessage):
            user_question = msg.content
            break

    if not user_question:
        return {"messages": [AIMessage(content="Could not find original question to rewrite.")]}

    # Simple rewrite prompt
    rewrite_prompt = PromptTemplate(
        template="""Given the user question: '{question}'
        Improve this question to make it more suitable for retrieving information from a technical help documentation knowledge base.
        Focus on key terms and specific features mentioned. Output only the improved question.""",
        input_variables=["question"],
    )
    rewrite_chain = rewrite_prompt | llm | StrOutputParser()
    improved_question = rewrite_chain.invoke({"question": user_question})
    print(f"Rewritten question: {improved_question}")
    # Return as a HumanMessage to feed back into the agent
    return {"messages": [HumanMessage(content=improved_question)]}

# Schema for the document grader
class GradeDocuments(BaseModel):
    """Binary score for relevance check based on retrieved documents."""
    binary_score: str = Field(description="Score 'yes' or 'no'. 'yes' if the docs are relevant to the question, 'no' otherwise.")

def grade_documents_node(state) -> Literal["generate", "rewrite"]:
    """Determines if the retrieved documents are relevant to the question."""
    print("--- Grading Retrieved Documents ---")
    if llm is None: return "rewrite" # Default to rewrite if LLM isn't working

    messages = state["messages"]
    # Find the original question (first HumanMessage)
    question = ""
    for msg in messages:
        if isinstance(msg, HumanMessage):
            question = msg.content
            break

    # Find the ToolMessage containing retrieved docs
    retrieved_docs_content = ""
    tool_message_found = False
    for msg in reversed(messages): # Check recent messages first
        if isinstance(msg, ToolMessage):
            retrieved_docs_content = msg.content
            tool_message_found = True
            break

    if not tool_message_found:
        print("--- No ToolMessage found with documents. Assuming irrelevant. ---")
        return "rewrite" # Cannot grade without docs

    if not question:
        print("--- Cannot find original question for grading. Assuming irrelevant. ---")
        return "rewrite"

    print(f"Original Question for Grading: {question}")
    print(f"Retrieved Docs Snippet for Grading: {retrieved_docs_content[:500]}...")

    # Prompt for the grader
    prompt = PromptTemplate(
        template="""You are a grader assessing relevance of retrieved documentation to a user question.
        Retrieved Docs:
        {context}

        User Question: {question}
        If the documents contain keywords or semantic meaning directly addressing the user question, grade 'yes'. Otherwise, grade 'no'.
        Provide only the binary score 'yes' or 'no'.""",
        input_variables=["context", "question"],
    )

    # LLM with structured output forcing the GradeDocuments schema
    structured_llm_grader = llm.with_structured_output(GradeDocuments)
    chain = prompt | structured_llm_grader

    try:
        scored_result = chain.invoke({"question": question, "context": retrieved_docs_content})
        score = scored_result.binary_score.lower()
        print(f"--- Relevance Score: {score} ---")
        if score == "yes":
            return "generate"
        else:
            return "rewrite"
    except Exception as e:
        print(f"--- Error during grading: {e}. Defaulting to rewrite. ---")
        return "rewrite"


def generate_node(state):
    """Generates an answer using the retrieved documents."""
    print("--- Generating Answer ---")
    if llm is None: return {"messages": [AIMessage(content="LLM not ready. Cannot generate answer.")]}

    messages = state["messages"]
    # Find the original question
    question = ""
    for msg in messages:
        if isinstance(msg, HumanMessage):
            question = msg.content
            break

    # Find the ToolMessage containing retrieved docs
    retrieved_docs_content = ""
    docs = [] # Keep original Document objects if possible for metadata
    tool_message_found = False
    for msg in reversed(messages):
        if isinstance(msg, ToolMessage):
            # Assuming the content of ToolMessage is the list of Document objects or their string representations
            # Let's refine this - ToolNode output is typically a list of Documents if the tool returns them.
            # However, the state usually stores the *string* representation from tool output.
            # We might need to adjust how docs are passed or re-retrieve based on the query.
            # For now, let's assume the ToolMessage content is the *string* representation.
            retrieved_docs_content = msg.content # This might be a string joined by ToolNode, need to verify
            tool_message_found = True
            # Ideally, we'd have the actual Document objects here.
            # If ToolNode stringifies, we might lose easy access to metadata.
            # A workaround: the ToolNode could potentially store results elsewhere or we could re-run retriever.
            # Let's stick to the simple path assuming content is usable text:
            docs_for_prompt = retrieved_docs_content
            break

    if not tool_message_found:
        return {"messages": [AIMessage(content="Could not find retrieved documents to generate answer.")]}
    if not question:
        return {"messages": [AIMessage(content="Could not find original question for answer generation.")]}


    # RAG prompt
    prompt = hub.pull("rlm/rag-prompt")

    # Chain for RAG
    rag_chain = prompt | llm | StrOutputParser()

    # Run RAG chain
    print("--- Running RAG Chain ---")
    try:
        answer = rag_chain.invoke({"context": docs_for_prompt, "question": question})
        print(f"Generated Answer: {answer[:200]}...")
        # Add the final answer to the messages
        # We might want to add source info here if we had access to doc metadata
        return {"messages": [AIMessage(content=answer)]}
    except Exception as e:
        print(f"--- Error during RAG generation: {e} ---")
        return {"messages": [AIMessage(content="Sorry, I encountered an error while generating the answer.")]}


# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crawl, index, and run Q&A agent.")
    parser.add_argument("--url", required=True, help="The starting URL.")
    parser.add_argument("--max_pages", type=int, default=25, help="Max pages to crawl.")
    parser.add_argument("--force_reindex", action='store_true', help="Force re-indexing.")
    parser.add_argument("--chunk_size", type=int, default=1000, help="Chunk size.")
    parser.add_argument("--chunk_overlap", type=int, default=150, help="Chunk overlap.")
    args = parser.parse_args()

    # --- Indexing Phase ---
    vector_store = None
    embedding_model = None
    if os.path.exists(vector_store_path) and not args.force_reindex:
        logging.info(f"--- Loading existing index: {vector_store_path} ---")
        try:
            embedding_model = get_embedding_model(embedding_model_name)
            vector_store = FAISS.load_local(vector_store_path, embedding_model, allow_dangerous_deserialization=True)
            logging.info("--- Index Loaded OK ---")
        except Exception as e:
            logging.error(f"Failed load index: {e}", exc_info=True); vector_store = None
    if vector_store is None:
        logging.info(f"--- Running Full Indexing Pipeline ---")
        processed_docs = crawl_and_process(args.url, args.max_pages)
        if processed_docs:
            chunked_docs = chunk_documents(processed_docs, args.chunk_size, args.chunk_overlap)
            if chunked_docs:
                if not embedding_model: embedding_model = get_embedding_model(embedding_model_name)
                vector_store = create_faiss_vector_store(chunked_docs, embedding_model, vector_store_path)
            else: logging.error("Chunking failed.")
        else: logging.error("Crawling/Processing failed.")

    # --- Agent Setup & Q&A Phase ---
    if vector_store and llm: # Proceed only if index and LLM are ready
        print("\n--- Index & LLM Ready - Setting up Agent ---")
        # 1. Create Retriever
        retriever = vector_store.as_retriever(search_kwargs={"k": 3}) # Retrieve top 3 chunks

        # 2. Create Retriever Tool (using the actual retriever)
        # Give it a clear name and description for the agent LLM
        retriever_tool = create_retriever_tool(
            retriever,
            "search_help_documentation", # Tool name
            f"Searches and returns relevant excerpts from the {urlparse(args.url).netloc} help documentation." # Tool description
        )
        tools = [retriever_tool] # List of tools for the agent
        retrieve_node = ToolNode(tools) # Create the ToolNode using the tool list

        # 3. Define Graph
        workflow = StateGraph(AgentState)
        workflow.add_node("agent", agent_node)
        workflow.add_node("retrieve", retrieve_node) # Use the ToolNode instance
        workflow.add_node("rewrite", rewrite_node)
        workflow.add_node("generate", generate_node)

        # 4. Define Edges
        workflow.add_edge(START, "agent")
        workflow.add_conditional_edges(
            "agent",
            tools_condition, # Prebuilt condition checks if tool calls are in the last message
            {
                "tools": "retrieve", # If tool call -> retrieve node
                END: END             # If no tool call -> end (agent gives direct answer or error)
            },
        )
        workflow.add_conditional_edges(
            "retrieve",
             grade_documents_node, # Custom node checks relevance of retrieved docs
             {
                 "generate": "generate", # If relevant -> generate node
                 "rewrite": "rewrite"    # If not relevant -> rewrite node
             }
        )
        workflow.add_edge("generate", END) # After generation, end
        workflow.add_edge("rewrite", "agent")  # After rewriting, go back to agent node

        # 5. Compile Graph
        try:
            graph = workflow.compile()
            print("--- Agent Graph Compiled Successfully ---")
            print("--- Ready for Questions (type 'exit' or 'quit' to stop) ---")
        except Exception as e:
            logging.error(f"Failed to compile graph: {e}", exc_info=True)
            graph = None

        # 6. Interactive Q&A Loop
        if graph:
            while True:
                user_input = input("\n> ")
                if user_input.lower() in ["exit", "quit"]:
                    break
                if not user_input.strip():
                    continue

                # Format input for the graph
                inputs = {"messages": [HumanMessage(content=user_input)]}

                print("\n--- Running Agent ---")
                final_answer = "Could not get a final answer." # Default
                try:
                    # Use stream to see intermediate steps (optional, invoke is simpler for just final answer)
                    for output in graph.stream(inputs, {"recursion_limit": 10}): # Added recursion limit
                        # stream yields dictionaries with node names as keys
                        last_state = output
                        print(f"Output from node '{list(last_state.keys())[0]}':")
                        # print(f"  State: {last_state[list(last_state.keys())[0]]['messages'][-1]}") # Print last message
                        # print("-" * 10)

                    # Extract final response (usually the last AI message without tool calls)
                    if last_state:
                         final_messages = last_state[list(last_state.keys())[0]]['messages']
                         if final_messages:
                             last_msg = final_messages[-1]
                             if isinstance(last_msg, AIMessage) and not last_msg.tool_calls:
                                 final_answer = last_msg.content
                             # Handle cases where the last message might be from rewrite or error
                             elif isinstance(last_msg, AIMessage) and last_msg.tool_calls:
                                 final_answer = "Agent decided to use a tool, but processing might have stopped."
                             elif isinstance(last_msg, ToolMessage):
                                 final_answer = "Agent ended after using a tool. Check logs for details."
                             elif isinstance(last_msg, HumanMessage): # e.g. after rewrite
                                 final_answer = f"Agent rewrote question to: {last_msg.content}"


                except Exception as e:
                    print(f"\n--- Error during graph execution: {e} ---")
                    final_answer = "An error occurred while processing your question."

                print("\n--- Agent Final Response ---")
                print(final_answer)
                print("-" * 30)

    elif not llm:
        logging.error("--- Groq LLM failed to initialize. Cannot start Q&A agent. ---")
    else:
        logging.error("--- Vector store not ready. Cannot start Q&A agent. ---")