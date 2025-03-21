import cv2
import numpy as np
import mediapipe as mp
from typing import Dict, List
import face_recognition

class FacialAnalysis:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Define facial landmarks for different expressions
        self.confusion_landmarks = [33, 133, 157, 158, 159, 160, 161, 246]  # Brow and forehead area
        self.attention_landmarks = [33, 133, 362, 263, 386]  # Eye area
        self.stress_landmarks = [61, 291, 0, 17, 61, 291]  # Mouth and jaw area
        
    async def analyze_video(self, video_file) -> Dict:
        """
        Analyze facial expressions from video stream
        """
        contents = await video_file.read()
        nparr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        metrics = {
            "confusion_count": 0,
            "stress_level": 0,
            "attention_score": 0,
            "micro_expressions": []
        }
        
        # Process frame
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(frame_rgb)
        
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            
            # Analyze confusion indicators
            confusion_score = self._analyze_confusion(face_landmarks)
            metrics["confusion_count"] = confusion_score
            
            # Analyze stress indicators
            stress_score = self._analyze_stress(face_landmarks)
            metrics["stress_level"] = stress_score
            
            # Analyze attention
            attention_score = self._analyze_attention(face_landmarks)
            metrics["attention_score"] = attention_score
            
            # Detect micro-expressions
            micro_expressions = self._detect_micro_expressions(frame_rgb)
            metrics["micro_expressions"] = micro_expressions
        
        return metrics
    
    def _analyze_confusion(self, landmarks) -> float:
        """
        Analyze facial features indicating confusion
        """
        # Calculate brow furrow intensity
        brow_points = [landmarks.landmark[i] for i in self.confusion_landmarks]
        brow_distances = self._calculate_distances(brow_points)
        
        # Normalize and scale confusion score
        confusion_score = np.mean(brow_distances)
        return min(1.0, max(0.0, confusion_score))
    
    def _analyze_stress(self, landmarks) -> float:
        """
        Analyze facial features indicating stress
        """
        # Calculate tension in jaw and mouth area
        stress_points = [landmarks.landmark[i] for i in self.stress_landmarks]
        stress_distances = self._calculate_distances(stress_points)
        
        # Normalize and scale stress score
        stress_score = np.mean(stress_distances)
        return min(1.0, max(0.0, stress_score))
    
    def _analyze_attention(self, landmarks) -> float:
        """
        Analyze facial features indicating attention level
        """
        # Calculate eye openness and gaze direction
        eye_points = [landmarks.landmark[i] for i in self.attention_landmarks]
        eye_distances = self._calculate_distances(eye_points)
        
        # Normalize and scale attention score
        attention_score = np.mean(eye_distances)
        return min(1.0, max(0.0, attention_score))
    
    def _detect_micro_expressions(self, frame) -> List[str]:
        """
        Detect and classify micro-expressions
        """
        face_locations = face_recognition.face_locations(frame)
        if not face_locations:
            return []
        
        face_encodings = face_recognition.face_encodings(frame, face_locations)
        
        # Analyze facial features for micro-expressions
        micro_expressions = []
        
        # Example micro-expression detection logic
        # This would be enhanced with a proper ML model trained on micro-expressions
        if len(face_encodings) > 0:
            features = face_encodings[0]
            
            # Simple threshold-based detection
            if features[0] > 0.5:  # Example threshold
                micro_expressions.append("surprise")
            if features[1] < -0.3:  # Example threshold
                micro_expressions.append("confusion")
            if features[2] > 0.4:  # Example threshold
                micro_expressions.append("stress")
        
        return micro_expressions
    
    def _calculate_distances(self, points) -> List[float]:
        """
        Calculate distances between facial landmarks
        """
        distances = []
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                distance = np.sqrt(
                    (points[i].x - points[j].x) ** 2 +
                    (points[i].y - points[j].y) ** 2 +
                    (points[i].z - points[j].z) ** 2
                )
                distances.append(distance)
        return distances 