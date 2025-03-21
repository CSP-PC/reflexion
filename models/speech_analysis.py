import librosa
import numpy as np
from typing import Dict, List
import speech_recognition as sr
from transformers import pipeline
import torch

class SpeechAnalysis:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.sentiment_analyzer = pipeline("sentiment-analysis")
        self.coherence_threshold = 0.6
        
    async def analyze_audio(self, audio_file) -> Dict:
        """
        Analyze speech patterns from audio file
        """
        contents = await audio_file.read()
        
        metrics = {
            "response_time": 0.0,
            "coherence_score": 0.0,
            "hesitation_count": 0,
            "sentiment_score": 0.0,
            "speech_rate": 0.0,
            "language_complexity": 0.0
        }
        
        try:
            # Convert audio file to numpy array
            audio_data = np.frombuffer(contents, dtype=np.int16)
            
            # Calculate speech rate and pauses
            metrics.update(self._analyze_speech_patterns(audio_data))
            
            # Perform speech-to-text
            text = self._speech_to_text(audio_data)
            
            if text:
                # Analyze language patterns
                language_metrics = self._analyze_language_patterns(text)
                metrics.update(language_metrics)
                
                # Analyze sentiment
                sentiment = self._analyze_sentiment(text)
                metrics["sentiment_score"] = sentiment
            
        except Exception as e:
            print(f"Error in speech analysis: {str(e)}")
            
        return metrics
    
    def _speech_to_text(self, audio_data) -> str:
        """
        Convert speech to text using speech recognition
        """
        try:
            # Convert numpy array to audio source
            audio = sr.AudioData(audio_data.tobytes(), 
                               sample_rate=16000,
                               sample_width=2)
            
            # Perform speech recognition
            text = self.recognizer.recognize_google(audio)
            return text
        except Exception as e:
            print(f"Speech recognition error: {str(e)}")
            return ""
    
    def _analyze_speech_patterns(self, audio_data) -> Dict:
        """
        Analyze speech patterns including rate, pauses, and hesitations
        """
        # Convert to float32 for librosa
        audio_float = audio_data.astype(np.float32) / 32768.0
        
        # Calculate speech rate
        tempo, _ = librosa.beat.beat_track(y=audio_float, sr=16000)
        speech_rate = tempo / 60.0  # words per second
        
        # Detect pauses and hesitations
        onset_env = librosa.onset.onset_strength(y=audio_float, sr=16000)
        onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env)
        
        # Count hesitations (longer pauses)
        pause_threshold = 0.5  # seconds
        hesitation_count = len(np.where(np.diff(onset_frames) > 
                                      pause_threshold * 16000)[0])
        
        return {
            "speech_rate": float(speech_rate),
            "hesitation_count": hesitation_count
        }
    
    def _analyze_language_patterns(self, text: str) -> Dict:
        """
        Analyze language patterns for complexity and coherence
        """
        # Split text into words and sentences
        words = text.split()
        sentences = text.split('.')
        
        # Calculate average word length
        avg_word_length = np.mean([len(word) for word in words])
        
        # Calculate average sentence length
        avg_sentence_length = np.mean([len(sent.split()) 
                                     for sent in sentences if sent.strip()])
        
        # Simple coherence score based on sentence structure
        coherence_score = min(1.0, (avg_sentence_length / 20.0))
        
        # Language complexity score
        complexity_score = min(1.0, (avg_word_length / 8.0))
        
        return {
            "coherence_score": float(coherence_score),
            "language_complexity": float(complexity_score)
        }
    
    def _analyze_sentiment(self, text: str) -> float:
        """
        Analyze sentiment in speech
        """
        try:
            result = self.sentiment_analyzer(text)[0]
            # Convert sentiment to score between 0 and 1
            score = result['score']
            if result['label'] == 'NEGATIVE':
                score = 1 - score
            return float(score)
        except Exception as e:
            print(f"Sentiment analysis error: {str(e)}")
            return 0.5  # Neutral sentiment as fallback 