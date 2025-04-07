# Help Website Q&A Agent (Agentic RAG + Streamlit)

## 1. Overview

This project implements an AI-powered question-answering agent designed to process documentation from help websites (e.g., `help.slack.com`, `docs.smith.langchain.com`) and answer user queries.

The agent features an **Agentic Retrieval-Augmented Generation (RAG)** pipeline built using **LangGraph**. It accepts a target website URL, crawls its content, processes and indexes the text into a local FAISS vector store, and then answers user questions based **solely** on the information found within that indexed documentation.

Key characteristics of the agentic approach include:
*   Retrieving relevant document chunks.
*   Grading the relevance of retrieved documents using an LLM.
*   Rewriting the user's question (if initial retrieval yields no relevant documents) and retrying retrieval (with limits).
*   Generating a final answer based on relevant documents (if found).
*   Grading the generated answer for factual grounding (hallucination check) and relevance to the original question.
*   Gracefully responding that information couldn't be found if relevant documents aren't retrieved after retries, or if the generated answer fails final checks.

The user interacts with the agent through a **Streamlit** web interface, which allows for URL input, indexing control, and a chat-like Q&A experience, including viewing the agent's internal thought process.

## 2. Features

*   **Web Crawling:** Recursively crawls pages within the domain of the provided starting URL.
*   **Content Extraction:** Parses HTML, attempts to extract main textual content while filtering common noise elements (navbars, footers, sidebars) using CSS selectors. Handles basic lists and tables.
*   **Indexing:** Chunks extracted text, generates embeddings using Sentence Transformers, and stores them in a local FAISS vector store for efficient semantic search. Supports loading a previously saved index.
*   **Streamlit UI:** Provides a user-friendly web interface for:
    *   Entering the target website URL and max pages to crawl.
    *   Initiating the indexing process (with optional force re-index).
    *   Displaying indexing status and readiness.
    *   Chatting with the agent.
    *   Viewing the agent's internal execution steps ("Agent Workings").
*   **Agentic RAG Pipeline (LangGraph):**
    *   Retrieves relevant document chunks from the FAISS index.
    *   Grades document relevance using an LLM.
    *   Rewrites the original question using an LLM if initial documents are irrelevant (limited retries).
    *   Generates answers grounded in the retrieved relevant documents using an LLM and RAG prompt.
    *   Grades the final answer for hallucinations and relevance using an LLM.
    *   Provides a clear "Sorry, I couldn't find..." response if information is unavailable after retries or if generation fails checks.

## 3. Architecture

The application follows this high-level flow:

1.  **Initialization:** Load environment variables, initialize LLM (Groq) and necessary Langchain components (graders, rewriter, RAG chain).
2.  **Indexing (via Streamlit Sidebar):**
    *   User inputs URL, max pages, force re-index option.
    *   On "Index Website" click:
        *   Load existing FAISS index if available and not forcing re-index.
        *   Otherwise, run the **Indexing Pipeline**:
            *   `crawl_website` (requests, BeautifulSoup) -> HTML content
            *   `process_crawled_data` -> Extracted text per page
            *   `chunk_documents` (RecursiveCharacterTextSplitter) -> Document chunks
            *   `get_embedding_model` (SentenceTransformerEmbeddings) -> Embedding function
            *   `create_faiss_vector_store` (FAISS) -> Vector store created & saved locally
        *   Initialize the `retriever` from the vector store.
        *   Compile the **LangGraph Q&A Agent Workflow**.
        *   Store key components (`vector_store`, `retriever`, `graph_app`) in `st.session_state`.
3.  **Q&A (via Streamlit Chat):**
    *   User inputs a question.
    *   The compiled LangGraph `app` is invoked with the question and initial state.
    *   The graph executes the agentic workflow: Retrieve -> Grade Docs -> (Rewrite -> Retrieve loop OR Generate) -> Grade Generation -> END / Fail Node.
    *   Logs are captured using `contextlib.redirect_stdout`.
    *   The final answer or "not found" message is displayed.
    *   Logs are shown in an expander.

## 4. Dependencies

All dependencies are listed in `requirements.txt`. Key libraries include:

*   `streamlit`: Web application framework.
*   `langchain`, `langchain-community`, `langchain-groq`: Core LLM framework, components, Groq LLM integration.
*   `langgraph`: Building the agent workflow state machine.
*   `requests`, `beautifulsoup4`: Web crawling and HTML parsing.
*   `sentence-transformers`: Text embeddings model.
*   `faiss-cpu` (or `faiss-gpu`): Vector store implementation.
*   `python-dotenv`: Loading environment variables.
*   `pydantic`: Data validation for structured LLM outputs (graders).
*   `torch`, `transformers`, `accelerate`: Underlying ML libraries.

## 5. Setup Instructions

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/Monikakusumanchi/Analyst/
    cd Analyst
    ```
2.  **Create and Activate a Virtual Environment:**
    ```bash
    python -m venv venv
    # On macOS/Linux:
    source venv/bin/activate
    # On Windows:
    # venv\Scripts\activate
    ```
3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Create Environment File:** Create a `.env` file in the project root directory.
5.  **Add API Keys:** Add your Groq API key to the `.env` file.
    ```dotenv
    # .env
    GROQ_API_KEY="your_groq_api_key_here"

    # Optional: Define a specific embedding model or vector store path
    # EMBEDDING_MODEL_NAME="all-MiniLM-L6-v2"
    # VECTOR_STORE_PATH="./my_custom_faiss_store"
    ```
    **Important:** Add `.env` to your `.gitignore` file to avoid committing secrets.

## 6. Usage

1.  **Run the Streamlit Application:**
    ```bash
    streamlit run app.py
    ```
    *   *Note:* If you encounter `RuntimeError: no running event loop` related to PyTorch and the file watcher, try running with the watcher disabled:
        ```bash
        streamlit run app.py --server.fileWatcherType none
        ```
2.  **Index a Website:**
    *   Open the app in your browser (usually `http://localhost:8501`).
    *   Use the **sidebar** to enter the target website URL (e.g., `https://help.slack.com/`).
    *   Set the "Max Pages" to crawl (start low, e.g., 10-20, for initial tests).
    *   Optionally check "Force Re-index" to ignore any previously saved index for this path.
    *   Click the "Index Website" button.
    *   Wait for the indexing process to complete. Status messages will appear in the sidebar.
3.  **Chat with the Agent:**
    *   Once the sidebar shows "Ready: <indexed_url>" and "Agent graph compiled.", the chat interface in the main area is active.
    *   Type your question about the indexed website content in the input box at the bottom and press Enter.
    *   The agent will process the query. An "Agent is thinking..." spinner will show.
    *   The final answer (or a "Sorry, I couldn't find..." message) will be displayed.
    *   Click the "Agent Workings" expander below the answer to see the sequence of nodes and edges executed by the LangGraph agent.
4.  **Stop the Application:** Press `Ctrl+C` in the terminal where you ran the `streamlit run` command.

## 7. Design Decisions

*   **Streamlit UI:** Chosen over the requested terminal interface for enhanced usability, better visualization of the agent's process (logs), and faster development iteration.
*   **LangGraph for Agentic Flow:** Selected to manage the complex, stateful, and potentially cyclic workflow required for the adaptive RAG approach (retrieve -> grade -> decide -> rewrite/generate -> grade -> decide). LangGraph provides explicit control over state transitions and conditional logic.
*   **Agentic RAG (No Web Search):** Implemented document retrieval grading and question rewriting to handle cases where initial documents are irrelevant. Answer grading (hallucination, relevance) added for quality control. Web search was explicitly *removed* to adhere strictly to answering *only* from indexed documentation and providing a clear "not found" message when necessary.
*   **Retry Mechanism:** A limited number of question rewrites (`MAX_REWRITES`) are allowed if initial retrieval or final generation fails checks, preventing infinite loops and leading to a graceful failure message (`fail_node`).
*   **Local Vector Store (FAISS):** Used FAISS for efficient local semantic search, suitable for the assignment scope. Local persistence (`save_local`, `load_local`) avoids re-indexing the same URL repeatedly during development/testing.
*   **Local Embeddings (Sentence Transformers):** Employed `all-MiniLM-L6-v2` for good performance without needing external APIs or keys for the embedding step.
*   **LLM (Groq):** Utilized Groq (specifically Llama3-8b) for its high inference speed, which is crucial for the multiple LLM calls involved in the agentic workflow (grading, rewriting, generation). Requires a free API key.
*   **Content Processing:** Standard `requests` and `BeautifulSoup` are used for broad compatibility. Content extraction relies on common CSS selectors, a heuristic approach suitable for many help sites but potentially brittle. Basic list/table formatting preserves some structure.

## 8. Known Limitations

*   **JavaScript Rendering:** The crawler cannot execute JavaScript, so content loaded dynamically on websites will likely be missed.
*   **Content Extraction Robustness:** The heuristic CSS selector approach for finding main content and filtering noise might fail on websites with non-standard or complex layouts. Needs tuning per site type for optimal results.
*   **Source Citation:** The final answer presented to the user does not automatically include citations (URLs) pointing to the source documents used for generation, although this information is available within the `Document` objects during processing.
*   **Scalability:** The single-process crawler and local FAISS index will be slow and memory-intensive for very large websites.
*   **Streamlit State:** Chat history and indexed state are lost if the browser tab is closed or refreshed. The FAISS index persists locally on disk.
*   **Error Handling:** While basic error handling exists, specific failures (e.g., LLM rate limits during grading, parsing errors on specific pages) could be handled more granularly.

## 9. Future Improvements (Optional)

*   Implement source URL citation in the final generated answer.
*   Integrate more robust content extraction libraries (e.g., `Unstructured`) or use browser automation (Playwright) for JS-heavy sites.
*   Explore chunking strategies that retain more structural context.
*   Replace local FAISS with a scalable vector database (e.g., ChromaDB, Weaviate, Pinecone) for larger sites.
*   Containerize the application using Docker.
*   Add an optional API endpoint (e.g., using FastAPI).
*   Improve UI log streaming for a more real-time view of the agent's thoughts.
