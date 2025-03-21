from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import numpy as np
import json
from datetime import datetime

from ..models.cognitive_assessment import CognitiveAssessment
from ..models.facial_analysis import FacialAnalysis
from ..models.speech_analysis import SpeechAnalysis
from ..utils.iris_vector_store import IRISVectorStore

app = FastAPI(title="Reflexion API", description="Smart Mirror Cognitive Health Monitoring System")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize our analysis models
cognitive_assessment = CognitiveAssessment()
facial_analysis = FacialAnalysis()
speech_analysis = SpeechAnalysis()
vector_store = IRISVectorStore()

class AssessmentResult(BaseModel):
    user_id: str
    timestamp: datetime
    risk_level: str
    cognitive_score: float
    speech_metrics: dict
    facial_metrics: dict
    recommendations: List[str]

@app.post("/analyze/interaction", response_model=AssessmentResult)
async def analyze_interaction(
    user_id: str,
    video: UploadFile = File(...),
    audio: UploadFile = File(...),
):
    try:
        # Process video for facial analysis
        facial_metrics = await facial_analysis.analyze_video(video)
        
        # Process audio for speech analysis
        speech_metrics = await speech_analysis.analyze_audio(audio)
        
        # Combine analyses for cognitive assessment
        assessment_result = cognitive_assessment.evaluate(
            facial_metrics=facial_metrics,
            speech_metrics=speech_metrics
        )
        
        # Store results in IRIS Vector Store
        vector_store.store_assessment(assessment_result)
        
        # Generate recommendations based on risk level
        recommendations = cognitive_assessment.generate_recommendations(
            risk_level=assessment_result["risk_level"]
        )
        
        return AssessmentResult(
            user_id=user_id,
            timestamp=datetime.now(),
            risk_level=assessment_result["risk_level"],
            cognitive_score=assessment_result["cognitive_score"],
            speech_metrics=speech_metrics,
            facial_metrics=facial_metrics,
            recommendations=recommendations
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/user/{user_id}/history")
async def get_user_history(user_id: str):
    try:
        history = vector_store.get_user_history(user_id)
        return {"history": history}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/user/{user_id}/trends")
async def get_cognitive_trends(user_id: str):
    try:
        trends = vector_store.analyze_trends(user_id)
        return {"trends": trends}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/alert/caregiver")
async def alert_caregiver(user_id: str, risk_level: str):
    if risk_level == "high":
        # Implement caregiver notification system
        # This could be email, SMS, or push notification
        pass
    return {"status": "alert_sent"} 