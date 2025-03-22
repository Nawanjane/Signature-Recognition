from fastapi import FastAPI, HTTPException, Depends, status, WebSocket
import asyncio
from datetime import datetime
from pydantic import BaseModel
from typing import List
import firebase_admin
from fastapi.middleware.cors import CORSMiddleware
from firebase_admin import credentials, auth
from pymongo import MongoClient
from bson.objectid import ObjectId
from passlib.context import CryptContext
from dotenv import load_dotenv
import os
import motor.motor_asyncio  # Add this import

# Load environment variables from .env file
load_dotenv()
firebase_credentials = os.getenv('FIREBASE_CREDENTIALS')
mongodb_uri = os.getenv('MONGODB_URI')


# Initialize Firebase
cred = credentials.Certificate(firebase_credentials)
firebase_admin.initialize_app(cred)

# Connect to MongoDB
client = motor.motor_asyncio.AsyncIOMotorClient(mongodb_uri)
db = client['attendance_db']
attendance_collection = db['attendance_records']
courses_collection = db['courses']
signature_collection = db['signature_validation']

# FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React app URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Pydantic models
class UserRegister(BaseModel):
    email: str
    user_id: str
    password: str
    firstName: str
    lastName: str
    phoneNumber: str

class UserLogin(BaseModel):
    email: str
    password: str

class AttendanceRecord(BaseModel):
    course_name: str
    date: str
    attendance: List[dict]

class Course(BaseModel):
    course_name: str
    date: str

class SignatureValidation(BaseModel):
    ID: str
    Name: str
    Morning: str
    Evening: str
    Morning_Prediction: str
    Morning_Confidence: float
    Evening_Prediction: str
    Evening_Confidence: float

# Helper functions
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

# Routes
@app.post("/register")
async def register(user: UserRegister):
    try:
        # Create user in Firebase
        user_record = auth.create_user(
            email=user.email,
            uid=user.user_id,
            password=user.password
        )
        
        # Hash the password
        hashed_password = get_password_hash(user.password)
        
        # Store user in MongoDB with additional fields
        user_data = {
            "user_id": user.user_id,
            "email": user.email,
            "hashed_password": hashed_password,
            "first_name": user.firstName,
            "last_name": user.lastName,
            "phone_number": user.phoneNumber
        }
        db.users.insert_one(user_data)
        
        return {"message": "User registered successfully"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/login")
async def login(user: UserLogin):
    try:
        # Verify user in Firebase
        user_record = auth.get_user_by_email(user.email)
        
        # Verify password and get user data
        user_data = db.users.find_one({"email": user.email})
        if not user_data:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found"
            )
            
        if not verify_password(user.password, user_data["hashed_password"]):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect password"
            )
        
        # Fetch all user related data
        all_user_data = await get_user_data(user.email)
        
        # Prepare response with data validation
        return {
            "message": "Login successful",
            "user_id": user_data.get("user_id", ""),
            "email": user_data.get("email", ""),
            "firstName": user_data.get("firstName", user_data.get("first_name", "")),
            "lastName": user_data.get("lastName", user_data.get("last_name", "")),
            "phoneNumber": user_data.get("phoneNumber", user_data.get("phone_number", "")),
            "attendance_records": all_user_data.get("attendance_records", []),
            "signature_records": all_user_data.get("signature_records", [])
        }
        
    except Exception as e:
        print(f"Login Error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )

@app.get("/attendance/{student_id}")
async def get_attendance(student_id: str):
    try:
        attendance_records = attendance_collection.find(
            {"attendance": {"$elemMatch": {"ID": student_id}}},
            {"_id": 0, "course_name": 1, "date": 1, "attendance.$": 1}
        )

        records = []
        for record in attendance_records:
            student_attendance = record["attendance"][0]
            if student_attendance["ID"] == student_id:
                morning = student_attendance.get("Morning", "Absent")
                evening = student_attendance.get("Evening", "Absent")
                
                records.append({
                    "course_name": record["course_name"],
                    "date": record["date"],
                    "morning_status": "Present" if morning == "Present" else "Absent",
                    "evening_status": "Present" if evening == "Present" else "Absent"
                })

        # Sort records by date
        records.sort(key=lambda x: x["date"])
        return {"attendance_records": records}
    except Exception as e:
        print(f"Attendance Error: {str(e)}")
        return {"attendance_records": []}

@app.get("/signature_validation/{user_id}")
async def get_signature_validation(user_id: str):
    try:
        signature_records = signature_collection.find(
            {"validation_results": {"$elemMatch": {"ID": user_id}}},
            {"_id": 0, "date": 1, "validation_results.$": 1}
        )

        validation_results = []
        for record in signature_records:
            result = record["validation_results"][0]
            if result["ID"] == user_id:
                morning_conf = float(result.get("Morning_Confidence", 0))
                evening_conf = float(result.get("Evening_Confidence", 0))
                
                confidence_data = {
                    "date": record["date"],
                    "Morning_Confidence": morning_conf,
                    "Evening_Confidence": evening_conf,
                    "Morning_Prediction": "Absent" if morning_conf == 0 else result.get("Morning_Prediction", "N/A"),
                    "Evening_Prediction": "Absent" if evening_conf == 0 else result.get("Evening_Prediction", "N/A"),
                    "Morning_Status": "Present" if morning_conf > 0 else "Absent",
                    "Evening_Status": "Present" if evening_conf > 0 else "Absent"
                }
                validation_results.append(confidence_data)

        validation_results.sort(key=lambda x: x["date"])
        return {"signature_records": validation_results}
    except Exception as e:
        print(f"Signature Error: {str(e)}")
        return {"signature_records": []}

@app.get("/user-data/{email}")
async def get_user_data(email: str):
    try:
        user_data = db.users.find_one({"email": email}, {"_id": 0})
        if not user_data:
            raise HTTPException(status_code=404, detail="User not found")

        user_id = user_data["user_id"]
        attendance_response = await get_attendance(user_id)
        signature_response = await get_signature_validation(user_id)

        # Calculate attendance percentage and check for warnings
        attendance_records = attendance_response["attendance_records"]
        total_sessions = len(attendance_records) * 2  # Morning and Evening
        present_sessions = sum(
            (1 if record["morning_status"] == "Present" else 0) +
            (1 if record["evening_status"] == "Present" else 0)
            for record in attendance_records
        )
        attendance_percentage = (present_sessions / total_sessions * 100) if total_sessions > 0 else 0

        # Calculate forged signature percentage
        signature_records = signature_response["signature_records"]
        total_signatures = len(signature_records) * 2  # Morning and Evening
        forged_signatures = sum(
            (1 if record["Morning_Prediction"] == "forged" else 0) +
            (1 if record["Evening_Prediction"] == "forged" else 0)
            for record in signature_records
        )
        forge_percentage = (forged_signatures / total_signatures * 100) if total_signatures > 0 else 0

        # Generate warnings
        warnings = []
        if attendance_percentage < 70:
            warnings.append({
                "type": "attendance",
                "message": f"Warning: Your attendance is {attendance_percentage:.1f}%, which is below the required 70%"
            })
        
        if forge_percentage > 50:
            warnings.append({
                "type": "signature",
                "message": f"Warning: {forge_percentage:.1f}% of your signatures have been detected as potentially forged"
            })

        response_data = {
            "user_data": user_data,
            "attendance_records": attendance_response["attendance_records"],
            "signature_records": signature_response["signature_records"],
            "attendance_percentage": attendance_percentage,
            "forge_percentage": forge_percentage,
            "warnings": warnings
        }

        return response_data

    except Exception as e:
        print(f"User Data Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/courses/{course_name}")
async def get_course(course_name: str):
    try:
        # Retrieve course details
        course = courses_collection.find_one({"course_name": course_name})
        if not course:
            raise HTTPException(status_code=404, detail="Course not found")
        
        return {"course": course}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Run the app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


# Add these after your existing routes
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    async def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            await connection.send_json(message)

manager = ConnectionManager()

@app.websocket("/ws/{user_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: str):
    await manager.connect(websocket)
    try:
        while True:
            # Fetch latest data every 5 seconds
            attendance_data = await get_attendance(user_id)
            signature_data = await get_signature_validation(user_id)
            
            await websocket.send_json({
                "attendance_records": attendance_data["attendance_records"],
                "signature_records": signature_data["signature_records"]
            })
            
            await asyncio.sleep(5)  # Update interval
    except Exception as e:
        print(f"WebSocket Error: {str(e)}")
    finally:
        await manager.disconnect(websocket)

#> npm start                                                                                      
#uvicorn main:app --reload                                                                      
