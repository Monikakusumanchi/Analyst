from fastapi import FastAPI, File, UploadFile,Query
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import pandas as pd
import openai
import faiss
import numpy as np
from langchain_text_splitters import RecursiveJsonSplitter
from langchain_openai import OpenAIEmbeddings
import google.generativeai as genai
from agno.models.google import Gemini
from dotenv import load_dotenv
import json
import os
import pickle
from pydantic import BaseModel


load_dotenv()
GOOGLE_KEY = os.getenv("GOOGLE_API_KEY")
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
if not GOOGLE_KEY:
    raise ValueError("Error: GOOGLE_API_KEY is not set in environment variables.")
os.environ["GOOGLE_API_KEY"] = GOOGLE_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_KEY

# MongoDB connection
uri = "mongodb+srv://monika:wOcbxCsRVJIDsphl@crm.hd2v6c5.mongodb.net/?retryWrites=true&w=majority&appName=CRM"
client = MongoClient(uri, server_api=ServerApi('1'))
db = client["Demo"]  # Using a database
collection = db["embeddings_store"] 
# FastAPI app
app = FastAPI()


@app.post("/upload_json/")
async def upload_json(file: UploadFile = File(...)):
    try:
        # Extract table name from filename (before .json)
        table_name = os.path.splitext(file.filename)[0]

        # Read and parse JSON data
        contents = await file.read()
        data = json.loads(contents)

        # Ensure data is a list (for bulk insert)
        if isinstance(data, dict):
            data = [data]

        # Insert into MongoDB
        collection = db[table_name]  # Create/Use collection dynamically
        collection.insert_many(data)

        return {"message": f"Data inserted into table '{table_name}'", "total_records": len(data)}
    
    except Exception as e:
        return {"error": str(e)}

@app.get("/extract_collections")
def extract_collections():
    collections = db.list_collection_names()
    dataframes = {}

    for collection in collections:
        data = list(db[collection].find({}, {"_id": 0}))  # Exclude _id
        df = pd.DataFrame(data)
        df.to_csv(f"/tmp/{collection}.csv", index=False)  # Save as CSV
        dataframes[collection] = df.to_dict(orient="records")

    return {"message": "Data extracted successfully", "tables": collections}


@app.post("/split_json")
def split_json():
    collections = db.list_collection_names()
    json_splitter = RecursiveJsonSplitter(max_chunk_size=300)
    split_data = {}

    for collection in collections:
        data = list(db[collection].find({}, {"_id": 0}))  # Exclude _id field

        # ✅ Check if collection is empty
        if not data:
            continue

        # ✅ Convert list to dictionary format (if necessary)
        if isinstance(data, list):
            data = {"documents": data}  # Wrap list inside a dictionary
        
        try:
            json_chunks = json_splitter.split_json(data)
            split_data[collection] = json_chunks
        except Exception as e:
            return {"error": f"Error splitting collection {collection}: {str(e)}"}

    return {"message": "JSON split into chunks", "data": split_data}

embeddings_model = OpenAIEmbeddings(model="text-embedding-3-large")

@app.post("/generate_embeddings")
def generate_embeddings():
    collections = db.list_collection_names()
    json_splitter = RecursiveJsonSplitter(max_chunk_size=300)
    embeddings_store = db["embeddings_store"]  # Store embeddings here

    for collection in collections:
        data = list(db[collection].find({}, {"_id": 0}))  # Exclude _id field

        # ✅ Skip empty collections
        if not data:
            continue

        # ✅ Convert list to dictionary format for JSON splitting
        if isinstance(data, list):
            data = {"documents": data}

        try:
            # ✅ Split JSON into smaller chunks
            json_chunks = json_splitter.split_json(data)

            # ✅ Generate embeddings for each chunk
            for chunk in json_chunks:
                chunk_text = str(chunk)  # Convert chunk to string
                embedding_vector = embeddings_model.embed_query(chunk_text)

                # ✅ Store embeddings in MongoDB
                embeddings_store.insert_one({
                    "collection": collection,
                    "chunk": chunk,
                    "embedding": embedding_vector
                })

        except Exception as e:
            return {"error": f"Error processing collection {collection}: {str(e)}"}

    return {"message": "Embeddings created and stored in MongoDB!"}

FAISS_INDEX_PATH = "faiss_index.pkl"
with open(FAISS_INDEX_PATH, "rb") as f:
    index = pickle.load(f)

@app.post("/load_embeddings_to_faiss")
def load_embeddings_to_faiss():
    collection = db["embeddings_store"]
    embeddings_data = list(collection.find({}, {"_id": 0, "embedding": 1, "chunk": 1}))

    if not embeddings_data:
        return {"error": "No embeddings found in MongoDB"}

    # ✅ Convert stored embeddings to NumPy array
    vectors = np.array([data["embedding"] for data in embeddings_data]).astype('float32')

    # ✅ Initialize FAISS Index
    index = faiss.IndexFlatL2(vectors.shape[1])  # L2 distance (Euclidean)
    index.add(vectors)

    # ✅ Save FAISS index to a file
    with open(FAISS_INDEX_PATH, "wb") as f:
        pickle.dump(index, f)

    return {"message": "Embeddings loaded into FAISS and saved!"}

# ✅ Initialize Google Gemini API
genai.configure(api_key=GOOGLE_KEY)
model = genai.GenerativeModel("gemini-2.0-flash-exp")

# # ✅ Request Schema
# class QueryRequest(BaseModel):
#     query: str

@app.post("/query_agent/")
def query_agent(query: str = Query(..., description="User's natural language query")):
    try:
        # Step 1: Convert Query to Embedding
        query_embedding = embeddings_model.embed_query(query)
        query_embedding = np.array(query_embedding, dtype='float32').reshape(1, -1)

        # Step 2: Search FAISS Index for Top-K Similar Chunks
        k = 5  # Number of top matches to retrieve
        _, indices = index.search(query_embedding, k)

        # Step 3: Fetch Chunks from MongoDB
        relevant_chunks = []
        for idx in indices[0]:  # Iterate over retrieved indices
            if idx < 0:
                continue  # Skip invalid indices

            # Fetch corresponding chunk from MongoDB
            chunk_data = collection.find_one({"embedding": {"$exists": True}}, {"_id": 0, "chunk": 1})
            if chunk_data:
                relevant_chunks.append(chunk_data["chunk"])

        # Step 4: Construct Context for Gemini
        context = "\n".join([str(chunk) for chunk in relevant_chunks])

        # Step 5: Query Gemini for Response
        prompt = f"Using the following context, answer the question:\n\n{context}\n\nQuery: {query}"
        response = model.generate_content(prompt)

        return {
            "query": query,
            "response": response.text,
            "retrieved_chunks": relevant_chunks
        }

    except Exception as e:
        return {"error": str(e)}