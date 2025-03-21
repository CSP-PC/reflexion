import numpy as np
from typing import Dict, List
from sklearn.preprocessing import StandardScaler
import json

class CognitiveAssessment:
    def __init__(self):
        self.scaler = StandardScaler()
        self.risk_thresholds = {
            "low": 0.7,
            "moderate": 0.4,
            "high": 0.0
        }
        
    def evaluate(self, facial_metrics: Dict, speech_metrics: Dict) -> Dict:
        """
        Evaluate cognitive health based on facial and speech metrics
        """
        # Extract key metrics
        response_time = speech_metrics.get("response_time", 0)
        speech_coherence = speech_metrics.get("coherence_score", 0)
        hesitation_count = speech_metrics.get("hesitation_count", 0)
        
        confusion_expressions = facial_metrics.get("confusion_count", 0)
        stress_indicators = facial_metrics.get("stress_level", 0)
        attention_score = facial_metrics.get("attention_score", 0)
        
        # Calculate cognitive score
        features = np.array([
            response_time,
            speech_coherence,
            hesitation_count,
            confusion_expressions,
            stress_indicators,
            attention_score
        ]).reshape(1, -1)
        
        # Normalize features
        normalized_features = self.scaler.fit_transform(features)
        
        # Calculate weighted cognitive score
        weights = [0.2, 0.2, 0.15, 0.15, 0.15, 0.15]
        cognitive_score = np.dot(normalized_features, weights)[0]
        
        # Determine risk level
        risk_level = self._determine_risk_level(cognitive_score)
        
        return {
            "cognitive_score": float(cognitive_score),
            "risk_level": risk_level,
            "metrics": {
                "response_time": response_time,
                "speech_coherence": speech_coherence,
                "hesitation_count": hesitation_count,
                "confusion_expressions": confusion_expressions,
                "stress_indicators": stress_indicators,
                "attention_score": attention_score
            }
        }
    
    def _determine_risk_level(self, cognitive_score: float) -> str:
        """
        Determine risk level based on cognitive score
        """
        if cognitive_score >= self.risk_thresholds["low"]:
            return "low"
        elif cognitive_score >= self.risk_thresholds["moderate"]:
            return "moderate"
        else:
            return "high"
    
    def generate_recommendations(self, risk_level: str) -> List[str]:
        """
        Generate personalized recommendations based on risk level
        """
        recommendations = {
            "low": [
                "Continue daily cognitive exercises",
                "Engage in social activities",
                "Try new brain training games",
                "Maintain regular physical exercise",
                "Practice mindfulness meditation"
            ],
            "moderate": [
                "Schedule a cognitive assessment with your healthcare provider",
                "Increase frequency of cognitive exercises",
                "Consider joining a cognitive support group",
                "Review and adjust daily routine for better cognitive support",
                "Monitor sleep patterns and stress levels"
            ],
            "high": [
                "Immediate consultation with healthcare provider recommended",
                "Daily monitoring of cognitive symptoms",
                "Caregiver support is advised",
                "Review current medications with doctor",
                "Consider professional cognitive rehabilitation therapy"
            ]
        }
        
        return recommendations.get(risk_level, []) 