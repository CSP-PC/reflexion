import irisnative
import numpy as np
from typing import Dict, List
import json
from datetime import datetime, timedelta

class IRISVectorStore:
    def __init__(self):
        # Initialize IRIS connection
        # Note: In production, these would be loaded from environment variables
        self.connection = irisnative.createConnection(
            "localhost",
            1972,
            "USER",
            "_SYSTEM",
            "SYS"
        )
        self.iris_native = irisnative.createIris(self.connection)
        
    def store_assessment(self, assessment_data: Dict) -> None:
        """
        Store assessment data in IRIS with vector embedding
        """
        try:
            # Create vector embedding from assessment metrics
            vector = self._create_vector_embedding(assessment_data)
            
            # Store data with timestamp
            timestamp = datetime.now().isoformat()
            
            # Convert assessment data to JSON string
            data_json = json.dumps(assessment_data)
            
            # Store in IRIS global
            global_name = "^CognitiveAssessment"
            self.iris_native.set(data_json, global_name, 
                               assessment_data["user_id"], timestamp)
            
            # Store vector embedding
            vector_json = json.dumps(vector.tolist())
            self.iris_native.set(vector_json, global_name + "Vector", 
                               assessment_data["user_id"], timestamp)
            
        except Exception as e:
            print(f"Error storing assessment: {str(e)}")
    
    def get_user_history(self, user_id: str) -> List[Dict]:
        """
        Retrieve user's assessment history
        """
        try:
            history = []
            global_name = "^CognitiveAssessment"
            
            # Iterate through user's assessments
            iterator = self.iris_native.iterator(global_name, user_id)
            for timestamp in iterator:
                data_json = self.iris_native.get(global_name, user_id, timestamp)
                assessment = json.loads(data_json)
                history.append(assessment)
            
            return sorted(history, 
                        key=lambda x: x.get('timestamp', ''), 
                        reverse=True)
            
        except Exception as e:
            print(f"Error retrieving history: {str(e)}")
            return []
    
    def analyze_trends(self, user_id: str) -> Dict:
        """
        Analyze cognitive health trends over time
        """
        try:
            history = self.get_user_history(user_id)
            
            if not history:
                return {
                    "trend": "insufficient_data",
                    "risk_level_changes": [],
                    "cognitive_score_trend": []
                }
            
            # Calculate trends
            scores = [entry.get('cognitive_score', 0) for entry in history]
            risk_levels = [entry.get('risk_level', 'unknown') for entry in history]
            
            # Calculate score trend
            score_trend = np.polyfit(range(len(scores)), scores, 1)[0]
            
            # Analyze risk level changes
            risk_changes = []
            for i in range(1, len(risk_levels)):
                if risk_levels[i] != risk_levels[i-1]:
                    risk_changes.append({
                        'from': risk_levels[i-1],
                        'to': risk_levels[i],
                        'timestamp': history[i].get('timestamp')
                    })
            
            # Determine overall trend
            if score_trend > 0.05:
                trend = "improving"
            elif score_trend < -0.05:
                trend = "declining"
            else:
                trend = "stable"
            
            return {
                "trend": trend,
                "risk_level_changes": risk_changes,
                "cognitive_score_trend": score_trend
            }
            
        except Exception as e:
            print(f"Error analyzing trends: {str(e)}")
            return {
                "trend": "error",
                "risk_level_changes": [],
                "cognitive_score_trend": 0
            }
    
    def find_similar_cases(self, assessment_data: Dict, limit: int = 5) -> List[Dict]:
        """
        Find similar cases using vector similarity search
        """
        try:
            query_vector = self._create_vector_embedding(assessment_data)
            
            similar_cases = []
            global_name = "^CognitiveAssessment"
            vector_global = global_name + "Vector"
            
            # Iterate through all assessments
            iterator = self.iris_native.iterator(vector_global)
            for user_id in iterator:
                user_iterator = self.iris_native.iterator(vector_global, user_id)
                for timestamp in user_iterator:
                    vector_json = self.iris_native.get(vector_global, 
                                                     user_id, timestamp)
                    case_vector = np.array(json.loads(vector_json))
                    
                    # Calculate cosine similarity
                    similarity = self._cosine_similarity(query_vector, case_vector)
                    
                    # Get case data
                    case_data = json.loads(self.iris_native.get(
                        global_name, user_id, timestamp))
                    
                    similar_cases.append({
                        "similarity": similarity,
                        "case": case_data
                    })
            
            # Sort by similarity and return top cases
            similar_cases.sort(key=lambda x: x["similarity"], reverse=True)
            return similar_cases[:limit]
            
        except Exception as e:
            print(f"Error finding similar cases: {str(e)}")
            return []
    
    def _create_vector_embedding(self, assessment_data: Dict) -> np.ndarray:
        """
        Create vector embedding from assessment metrics
        """
        metrics = assessment_data.get("metrics", {})
        
        # Create feature vector from metrics
        vector = np.array([
            metrics.get("response_time", 0),
            metrics.get("speech_coherence", 0),
            metrics.get("hesitation_count", 0),
            metrics.get("confusion_expressions", 0),
            metrics.get("stress_indicators", 0),
            metrics.get("attention_score", 0)
        ])
        
        return vector / np.linalg.norm(vector)  # Normalize vector
    
    def _cosine_similarity(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two vectors
        """
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)) 