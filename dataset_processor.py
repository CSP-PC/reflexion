import pandas as pd
import numpy as np
import os
from typing import Dict, List, Tuple
import cv2
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DementiaDataset(Dataset):
    """Custom Dataset for loading dementia-related data"""
    def __init__(self, features: np.ndarray, labels: np.ndarray, transform=None):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        feature = self.features[idx]
        label = self.labels[idx]
        
        if self.transform:
            feature = self.transform(feature)
            
        return feature, label

class DatasetProcessor:
    def __init__(self, config: Dict):
        """
        Initialize dataset processor with configuration
        
        Args:
            config: Dictionary containing:
                - speech_data_path: Path to speech dataset
                - facial_data_path: Path to facial expression dataset
                - batch_size: Batch size for data loaders
                - test_size: Proportion of data for testing
                - random_state: Random seed
        """
        self.config = config
        self.scaler = StandardScaler()
        
    def process_speech_data(self) -> Tuple[DataLoader, DataLoader]:
        """
        Process speech dataset from Kaggle
        https://www.kaggle.com/datasets/tahouramorovati/dementia-detection-using-speech
        """
        try:
            logger.info("Processing speech dataset...")
            
            # Load speech data
            speech_df = pd.read_csv(self.config['speech_data_path'])
            
            # Extract features
            features = []
            labels = []
            
            for _, row in speech_df.iterrows():
                # Extract acoustic features
                audio_path = os.path.join(
                    os.path.dirname(self.config['speech_data_path']),
                    row['audio_filename']
                )
                
                if os.path.exists(audio_path):
                    # Load audio file
                    y, sr = librosa.load(audio_path, sr=None)
                    
                    # Extract features
                    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
                    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
                    zero_crossing_rate = librosa.feature.zero_crossing_rate(y)
                    
                    # Combine features
                    feature_vector = np.concatenate([
                        mfcc.mean(axis=1),
                        spectral_centroid.mean(axis=1),
                        [zero_crossing_rate.mean()]
                    ])
                    
                    features.append(feature_vector)
                    labels.append(row['dementia_label'])
            
            # Convert to numpy arrays
            X = np.array(features)
            y = np.array(labels)
            
            # Split data
            return self._create_data_loaders(X, y)
            
        except Exception as e:
            logger.error(f"Error processing speech data: {str(e)}")
            raise
    
    def process_facial_data(self) -> Tuple[DataLoader, DataLoader]:
        """
        Process facial expression dataset
        Using CK+ dataset structure
        """
        try:
            logger.info("Processing facial expression dataset...")
            
            features = []
            labels = []
            
            # Process CK+ dataset
            data_path = self.config['facial_data_path']
            for emotion_label in os.listdir(data_path):
                emotion_path = os.path.join(data_path, emotion_label)
                if os.path.isdir(emotion_path):
                    for image_file in os.listdir(emotion_path):
                        if image_file.endswith(('.jpg', '.png')):
                            image_path = os.path.join(emotion_path, image_file)
                            
                            # Read and preprocess image
                            image = cv2.imread(image_path)
                            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                            image = cv2.resize(image, (48, 48))
                            
                            # Extract facial landmarks using MediaPipe
                            features.append(self._extract_facial_features(image))
                            labels.append(self._emotion_to_label(emotion_label))
            
            # Convert to numpy arrays
            X = np.array(features)
            y = np.array(labels)
            
            # Split data
            return self._create_data_loaders(X, y)
            
        except Exception as e:
            logger.error(f"Error processing facial data: {str(e)}")
            raise
    
    def _create_data_loaders(
        self, 
        X: np.ndarray, 
        y: np.ndarray
    ) -> Tuple[DataLoader, DataLoader]:
        """Create train and test data loaders"""
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.config['test_size'],
            random_state=self.config['random_state']
        )
        
        # Scale features
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)
        
        # Create datasets
        train_dataset = DementiaDataset(X_train, y_train)
        test_dataset = DementiaDataset(X_test, y_test)
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False
        )
        
        return train_loader, test_loader
    
    def _extract_facial_features(self, image: np.ndarray) -> np.ndarray:
        """Extract facial features using MediaPipe"""
        import mediapipe as mp
        
        mp_face_mesh = mp.solutions.face_mesh
        
        with mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            min_detection_confidence=0.5
        ) as face_mesh:
            
            results = face_mesh.process(image)
            
            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0]
                
                # Extract key facial features
                features = []
                for landmark in landmarks.landmark:
                    features.extend([landmark.x, landmark.y, landmark.z])
                
                return np.array(features)
            
            return np.zeros(468 * 3)  # Default size for MediaPipe face mesh
    
    def _emotion_to_label(self, emotion: str) -> int:
        """Convert emotion string to numeric label"""
        emotion_map = {
            'neutral': 0,
            'confusion': 1,
            'stress': 2,
            'anxiety': 3,
            'cognitive_strain': 4
        }
        return emotion_map.get(emotion.lower(), 0) 