from fastapi import FastAPI, HTTPException
from pymongo import MongoClient
from pydantic import BaseModel
from datetime import datetime, timedelta
from typing import List, Optional

app = FastAPI()

# MongoDB Configuration (Replace with your actual credentials)
uri = "mongodb+srv://monika:wOcbxCsRVJIDsphl@crm.hd2v6c5.mongodb.net/?retryWrites=true&w=majority&appName=CRM"
client = MongoClient(uri)
db = client.Demo  

# --- Data Models ---

class CallLog(BaseModel):
    call_id: str
    agent: str
    customer: str
    call_time: datetime
    duration_minutes: int
    call_status: str

class Appointment(BaseModel):
    appointment_id: str
    agent: str
    customer: str
    appointment_time: datetime
    status: str
    notes: Optional[str] = None

class Email(BaseModel):
    email_id: str
    sender: str
    receiver: str
    subject: str
    timestamp: datetime
    status: str

class WhatsAppMessage(BaseModel):
    message_id: str
    agent: str
    customer: str
    timestamp: datetime
    message: str
    status: str

# --- Helper Functions (Date Range) ---

def start_of_day(dt: datetime) -> datetime:
    return dt.replace(hour=0, minute=0, second=0, microsecond=0)

def end_of_day(dt: datetime) -> datetime:
    return dt.replace(hour=23, minute=59, second=59, microsecond=999999)

# --- Call Logs Endpoints ---

@app.get("/calls/agent/{agent_name}/count")
def get_call_count_for_agent_this_week(agent_name: str):
    """How many calls did {agent_name} make this week?"""
    today = datetime.utcnow()
    print(today)
    start_of_week = today - timedelta(days=today.weekday())
    print(start_of_week)
    query = {
        "agent": agent_name,
        "call_time": {"$gte": start_of_week.isoformat()}
    }
    start_of_week = datetime.utcnow() - timedelta(days=datetime.utcnow().weekday())


    count = db.call_logs_sample.count_documents(query)
    return {"agent": agent_name, "call_count": count}

@app.get("/calls/failed")
def get_failed_calls_last_7_days() -> List[CallLog]:
    """List all failed calls in the last 7 days."""
    seven_days_ago = datetime.utcnow() - timedelta(days=7)
    query = {
        "call_time": {"$gte": seven_days_ago.isoformat()},
        "call_status": "failed"
    }
    calls = list(db.call_logs_sample.find(query))
    return [CallLog(**call) for call in calls]

@app.get("/calls/average_duration")
def get_average_call_duration():
    """What is the average call duration for completed calls?"""
    pipeline = [
        {"$match": {"call_status": "completed"}},
        {"$group": {"_id": None, "average_duration": {"$avg": "$duration_minutes"}}}
    ]
    result = list(db.call_logs_sample.aggregate(pipeline))
    if result:
        return {"average_duration": result[0]["average_duration"]}
    else:
        return {"average_duration": 0}

# --- Appointments Endpoints ---

@app.get("/appointments/today")
def get_confirmed_appointments_today() -> List[Appointment]:
    """List all confirmed appointments for today."""
    today_start = start_of_day(datetime.utcnow())
    today_end = end_of_day(datetime.utcnow())

    query = {
        "appointment_time": {"$gte": today_start.isoformat(), "$lte": today_end.isoformat()},
        "status": "confirmed"
    }
    appointments = list(db.appointments_sample.find(query))
    return [Appointment(**appt) for appt in appointments]

@app.get("/appointments/agent/{agent_name}/count")
def get_appointment_count_for_agent_this_week(agent_name: str):
    """How many appointments has {agent_name} had this week?"""
    today = datetime.utcnow()
    start_of_week = today - timedelta(days=today.weekday())
    query = {
        "agent": agent_name,
        "appointment_time": {"$gte": start_of_week.isoformat()}
    }
    count = db.appointments_sample.count_documents(query)
    return {"agent": agent_name, "appointment_count": count}

@app.get("/appointments/missed")
def get_missed_appointments_last_7_days() -> List[Appointment]:
    """Find missed appointments in the last 7 days."""
    seven_days_ago = datetime.utcnow() - timedelta(days=7)
    query = {
        "appointment_time": {"$gte": seven_days_ago.isoformat()},
        "status": "missed" # Assuming you have a 'missed' status
    }
    appointments = list(db.appointments_sample.find(query))
    return [Appointment(**appt) for appt in appointments]

# --- Email Endpoints ---

@app.get("/emails/sent_by/{sender_email}")
def get_emails_sent_by_this_week(sender_email: str) -> List[Email]:
    """List all emails sent by {sender_email} this week."""
    today = datetime.utcnow()
    start_of_week = today - timedelta(days=today.weekday())
    query = {
        "sender": sender_email,
        "timestamp": {"$gte": start_of_week.isoformat()}
    }
    emails = list(db.emails.find(query))
    return [Email(**email) for email in emails]

@app.get("/emails/onboarding_count")
def get_emails_with_onboarding_subject_last_month():
    """How many emails had 'onboarding' in the subject last month?"""
    today = datetime.utcnow()
    first_day_of_this_month = today.replace(day=1)
    first_day_of_last_month = (first_day_of_this_month - timedelta(days=1)).replace(day=1)
    last_day_of_last_month = first_day_of_this_month - timedelta(days=1)
    query = {
        "subject": {"$regex": "onboarding", "$options": "i"},
        "timestamp": {"$gte": first_day_of_last_month.isoformat(), "$lte": last_day_of_last_month.isoformat()}
    }
    count = db.emails_sample.count_documents(query)
    return {"email_count": count}

@app.get("/emails/not_delivered")
def get_emails_not_delivered() -> List[Email]:
    """Show emails that were not delivered successfully."""
    query = {"status": {"$ne": "delivered"}}  # Assuming 'delivered' is the successful status
    emails = list(db.emails_sample.find(query))
    return [Email(**email) for email in emails]

# --- WhatsApp Endpoints ---

@app.get("/whatsapp/agent/{agent_name}/today")
def get_whatsapp_messages_by_agent_today(agent_name: str) -> List[WhatsAppMessage]:
    """List all WhatsApp messages sent by {agent_name} today."""
    today_start = start_of_day(datetime.utcnow())
    today_end = end_of_day(datetime.utcnow())
    query = {
        "agent": agent_name,
        "timestamp": {"$gte": today_start.isoformat(), "$lte": today_end.isoformat()}
    }
    messages = list(db.whatsapp_sample.find(query))
    return [WhatsAppMessage(**msg) for msg in messages]

@app.get("/whatsapp/delivery_stats")
def get_whatsapp_delivery_stats_last_3_days():
    """How many messages were delivered vs failed in the last 3 days?"""
    three_days_ago = datetime.utcnow() - timedelta(days=3)
    pipeline = [
        {"$match": {"timestamp": {"$gte": three_days_ago.isoformat()}}},
        {"$group": {"_id": "$status", "count": {"$sum": 1}}}
    ]
    result = list(db.whatsapp_sample.aggregate(pipeline))
    delivery_stats = {}
    for item in result:
        delivery_stats[item["_id"]] = item["count"]
    return delivery_stats

@app.get("/whatsapp/latest_to/{customer_name}")
def get_latest_whatsapp_message_to_customer(customer_name: str) -> Optional[WhatsAppMessage]:
    """Find the latest message sent to {customer_name}."""
    message = db.whatsapp_sample.find({"customer": customer_name}).sort("timestamp", -1).limit(1)
    message = list(message)
    if message:
        return WhatsAppMessage(**message[0])
    else:
        return None

# --- Startup Event ---

@app.on_event("startup")
async def startup_event():
    print("Connecting to MongoDB...")
    try:
        client.admin.command('ping')
        print("Successfully connected to MongoDB!")
    except Exception as e:
        print(f"Could not connect to MongoDB: {e}")
        # Optionally, you might want to raise an exception to prevent the app from starting
        # raise

# --- Shutdown Event ---

@app.on_event("shutdown")
async def shutdown_event():
    print("Closing MongoDB connection...")
    client.close()