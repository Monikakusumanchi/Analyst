# 
from fastapi import FastAPI, HTTPException
from pymongo import MongoClient
from datetime import datetime, timedelta
from langchain_groq import ChatGroq

from langchain.agents import initialize_agent
from langchain.tools import Tool
from langchain.agents import AgentType
import os
from dotenv import load_dotenv

app = FastAPI()

# MongoDB Connection
uri = "mongodb+srv://monika:wOcbxCsRVJIDsphl@crm.hd2v6c5.mongodb.net/?retryWrites=true&w=majority&appName=CRM"
client = MongoClient(uri)

# Set up Groq API Key
os.environ["GROQ_API_KEY"] = "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"  # Replace with your actual Groq API Key
# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("Error: GOOGLE_API_KEY is not set in environment variables.")
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
model = Gemini(id="gemini-2.0-flash-exp")
agent = Agent(
    model=model,
    markdown=True,
    response_model=InterviewAnalysis,
    structured_outputs=True,
)

# Function to Get All Field Names
def get_all_fields():
    db_collection_fields = {}

    for db_name in client.list_database_names():
        db = client[db_name]
        db_collection_fields[db_name] = {}

        for collection_name in db.list_collection_names():
            collection = db[collection_name]
            fields = set()

            for doc in collection.find({}, {"_id": 0}).limit(10):  # Exclude _id field
                fields.update(doc.keys())

            db_collection_fields[db_name][collection_name] = list(fields)

    return db_collection_fields

# Date Utilities
def start_of_day(dt: datetime) -> datetime:
    return dt.replace(hour=0, minute=0, second=0, microsecond=0)

def end_of_day(dt: datetime) -> datetime:
    return dt.replace(hour=23, minute=59, second=59, microsecond=999999)

# AGNO Agent Function Using Groq
def generate_mongo_query(natural_query: str):
    # Fetch all field details
    db_metadata = get_all_fields()

    # Prepare agent prompt
    prompt = f"""
    You are a MongoDB query generator using Groq AI.
    Given a natural language query, generate a valid MongoDB query using the correct database, collection, and fields.

    Available databases and their collections:
    {db_metadata}

    If the query is about counting calls made by a person in a week, assume the field 'call_time' contains timestamps.

    Example:
    Query: "How many calls did Priya Sharma make this week?"
    MongoDB Query:
    {{
        "caller_name": "Priya Sharma",
        "call_time": {{"$gte": "{start_of_day(datetime.utcnow() - timedelta(days=7)).isoformat()}", "$lte": "{end_of_day(datetime.utcnow()).isoformat()}"}}
    }}

    Now generate the correct MongoDB query for:
    "{natural_query}"
    """

    # Initialize LangChain with Groq
    llm = ChatGroq(model_name="llama3-8b-8192")  # You can also use "mixtral-8x7b-32768"

    # Define a tool that fetches the query
    query_tool = Tool(
        name="MongoQueryGenerator",
        func=lambda x: llm.predict(x),
        description="Generates MongoDB queries from natural language."
    )

    # Create an agent
    agent_executor = initialize_agent(
        tools=[query_tool],
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
    )

    # Generate the query
    generated_query = agent_executor.run(prompt)

    return generated_query

@app.get("/query_mongodb/")
def query_mongodb(natural_query: str):
    """
    API Endpoint to take a natural language query and return a MongoDB query.
    """
    try:
        query = generate_mongo_query(natural_query)
        return {"query": query}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run with: uvicorn filename:app --reload
