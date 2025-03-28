import cv2
import numpy as np
import time
import logging
from pathlib import Path
import json
import asyncio
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QuickDemo:
    def __init__(self):
        self.setup_demo_data()
    
    def setup_demo_data(self):
        """Create mock data for demo"""
        self.mock_metrics = {
            "response_time": 0.8,
            "speech_coherence": 0.85,
            "hesitation_count": 2,
            "confusion_expressions": 1,
            "stress_indicators": 0.3,
            "attention_score": 0.9
        }
        
        self.risk_levels = {
            "low": "Continue daily cognitive exercises",
            "moderate": "Schedule cognitive assessment",
            "high": "Immediate medical consultation"
        }
    
    def generate_mock_assessment(self):
        """Generate mock assessment based on metrics"""
        cognitive_score = np.mean([
            self.mock_metrics["speech_coherence"],
            self.mock_metrics["attention_score"],
            1 - self.mock_metrics["stress_indicators"]
        ])
        
        if cognitive_score > 0.7:
            risk_level = "low"
        elif cognitive_score > 0.4:
            risk_level = "moderate"
        else:
            risk_level = "high"
        
        return {
            "timestamp": datetime.now().isoformat(),
            "cognitive_score": float(cognitive_score),
            "risk_level": risk_level,
            "metrics": self.mock_metrics,
            "recommendations": self.risk_levels[risk_level]
        }
    
    async def run_demo(self):
        """Run the quick demo"""
        try:
            # Initialize camera
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                raise Exception("Could not open camera")
            
            logger.info("Starting Reflexion Demo...")
            logger.info("Press 'q' to quit")
            
            # Create window for visualization
            cv2.namedWindow('Reflexion Demo', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Reflexion Demo', 800, 600)
            
            # Demo loop
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Add demo overlay
                overlay = frame.copy()
                cv2.rectangle(overlay, (50, 50), (750, 550), (0, 255, 0), 2)
                
                # Add mock metrics
                assessment = self.generate_mock_assessment()
                
                # Display metrics
                metrics_text = [
                    f"Risk Level: {assessment['risk_level']}",
                    f"Cognitive Score: {assessment['cognitive_score']:.2f}",
                    f"Speech Coherence: {assessment['metrics']['speech_coherence']:.2f}",
                    f"Attention Score: {assessment['metrics']['attention_score']:.2f}"
                ]
                
                for i, text in enumerate(metrics_text):
                    cv2.putText(
                        overlay,
                        text,
                        (70, 100 + i * 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2
                    )
                
                # Add recommendation
                cv2.putText(
                    overlay,
                    f"Recommendation: {assessment['recommendations']}",
                    (70, 200),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )
                
                # Blend overlay
                alpha = 0.7
                cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
                
                # Show frame
                cv2.imshow('Reflexion Demo', frame)
                
                # Break on 'q' press
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            # Cleanup
            cap.release()
            cv2.destroyAllWindows()
            
            logger.info("Demo completed successfully!")
            
        except Exception as e:
            logger.error(f"Demo failed: {str(e)}")
            raise

async def main():
    demo = QuickDemo()
    await demo.run_demo()

if __name__ == '__main__':
    asyncio.run(main())