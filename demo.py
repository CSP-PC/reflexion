import argparse
import yaml
import torch
import cv2
import sounddevice as sd
import soundfile as sf
import numpy as np
from pathlib import Path
import time
import logging
from models.facial_analysis import FacialAnalysis
from models.speech_analysis import SpeechAnalysis
from models.cognitive_assessment import CognitiveAssessment
from utils.iris_vector_store import IRISVectorStore
import asyncio
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ReflexionDemo:
    def __init__(self, config_path: str):
        """Initialize demo with configuration"""
        self.config = self._load_config(config_path)
        self.setup_components()
        
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def setup_components(self):
        """Initialize system components"""
        self.facial_analyzer = FacialAnalysis()
        self.speech_analyzer = SpeechAnalysis()
        self.cognitive_assessor = CognitiveAssessment()
        self.vector_store = IRISVectorStore()
        
        # Load pre-trained models if available
        self._load_models()
    
    def _load_models(self):
        """Load pre-trained models"""
        model_path = Path(self.config['model_save_path'])
        
        if (model_path / 'speech_model.pth').exists():
            logger.info("Loading pre-trained speech model...")
            # Load model weights here
        
        if (model_path / 'facial_model.pth').exists():
            logger.info("Loading pre-trained facial model...")
            # Load model weights here
    
    async def record_interaction(self, duration: int = 30):
        """Record video and audio for specified duration"""
        logger.info(f"Recording interaction for {duration} seconds...")
        
        # Initialize video capture
        cap = cv2.VideoCapture(0)
        frames = []
        
        # Initialize audio recording
        audio_data = []
        sample_rate = 16000
        
        # Record timestamp
        start_time = time.time()
        
        try:
            # Record video and audio simultaneously
            while time.time() - start_time < duration:
                # Capture video frame
                ret, frame = cap.read()
                if ret:
                    frames.append(frame)
                    cv2.imshow('Reflexion Demo', frame)
                
                # Capture audio
                audio_chunk = sd.rec(
                    int(0.1 * sample_rate),
                    samplerate=sample_rate,
                    channels=1
                )
                sd.wait()
                audio_data.extend(audio_chunk)
                
                # Break if 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
        
        # Save recordings
        video_path = "temp_video.avi"
        audio_path = "temp_audio.wav"
        
        # Save video
        if frames:
            out = cv2.VideoWriter(
                video_path,
                cv2.VideoWriter_fourcc(*'XVID'),
                30.0,
                (frames[0].shape[1], frames[0].shape[0])
            )
            for frame in frames:
                out.write(frame)
            out.release()
        
        # Save audio
        if audio_data:
            sf.write(audio_path, np.array(audio_data), sample_rate)
        
        return video_path, audio_path
    
    async def analyze_interaction(self, video_path: str, audio_path: str):
        """Analyze recorded interaction"""
        logger.info("Analyzing interaction...")
        
        try:
            # Analyze facial expressions
            with open(video_path, 'rb') as video_file:
                facial_metrics = await self.facial_analyzer.analyze_video(video_file)
            
            # Analyze speech
            with open(audio_path, 'rb') as audio_file:
                speech_metrics = await self.speech_analyzer.analyze_audio(audio_file)
            
            # Combine analyses for cognitive assessment
            assessment = self.cognitive_assessor.evaluate(
                facial_metrics=facial_metrics,
                speech_metrics=speech_metrics
            )
            
            # Store assessment in IRIS
            self.vector_store.store_assessment(assessment)
            
            return assessment
            
        except Exception as e:
            logger.error(f"Error during analysis: {str(e)}")
            raise
    
    def display_results(self, assessment: dict):
        """Display assessment results"""
        logger.info("\n=== Cognitive Health Assessment ===")
        logger.info(f"Risk Level: {assessment['risk_level']}")
        logger.info(f"Cognitive Score: {assessment['cognitive_score']:.2f}")
        
        logger.info("\nMetrics:")
        logger.info(json.dumps(assessment['metrics'], indent=2))
        
        logger.info("\nRecommendations:")
        recommendations = self.cognitive_assessor.generate_recommendations(
            assessment['risk_level']
        )
        for rec in recommendations:
            logger.info(f"- {rec}")
    
    async def run_demo(self):
        """Run the complete demo"""
        try:
            # Record interaction
            video_path, audio_path = await self.record_interaction()
            
            # Analyze interaction
            assessment = await self.analyze_interaction(video_path, audio_path)
            
            # Display results
            self.display_results(assessment)
            
            # Clean up temporary files
            Path(video_path).unlink(missing_ok=True)
            Path(audio_path).unlink(missing_ok=True)
            
        except Exception as e:
            logger.error(f"Demo failed: {str(e)}")
            raise

async def main():
    parser = argparse.ArgumentParser(description='Run Reflexion Demo')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to configuration file')
    args = parser.parse_args()
    
    demo = ReflexionDemo(args.config)
    await demo.run_demo()

if __name__ == '__main__':
    asyncio.run(main()) 