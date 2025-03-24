import unittest
import os
import torch
import numpy as np
from pathlib import Path
import shutil
import logging
import sys

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset_processor import DatasetProcessor
from models.model_trainer import ModelTrainer, FacialExpressionNet, SpeechNet

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestReflexionPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test data and configuration"""
        cls.test_dir = Path("test_data")
        cls.setup_test_data()
        
        cls.config = {
            'speech_data_path': str(cls.test_dir / "dementia_speech/test_speech.csv"),
            'facial_data_path': str(cls.test_dir / "facial_expressions"),
            'batch_size': 2,
            'test_size': 0.5,
            'random_state': 42,
            'learning_rate': 0.001,
            'num_epochs': 2,
            'hidden_size': 64,
            'device': 'cpu',
            'model_save_path': str(cls.test_dir / "models")
        }
    
    @classmethod
    def setup_test_data(cls):
        """Create test data directory structure and sample data"""
        # Create directories
        speech_dir = cls.test_dir / "dementia_speech"
        facial_dir = cls.test_dir / "facial_expressions/test_emotion"
        speech_dir.mkdir(parents=True, exist_ok=True)
        facial_dir.mkdir(parents=True, exist_ok=True)
        
        # Create sample speech data
        speech_data = "audio_filename,dementia_label\n"
        speech_data += "test_audio1.wav,0\n"
        speech_data += "test_audio2.wav,1\n"
        
        with open(speech_dir / "test_speech.csv", 'w') as f:
            f.write(speech_data)
        
        # Create sample audio files
        sample_audio = np.random.rand(16000)  # 1 second of random audio
        np.save(speech_dir / "test_audio1.wav", sample_audio)
        np.save(speech_dir / "test_audio2.wav", sample_audio)
        
        # Create sample facial expression images
        sample_image = np.random.randint(0, 255, (48, 48), dtype=np.uint8)
        np.save(facial_dir / "test_image1.npy", sample_image)
        np.save(facial_dir / "test_image2.npy", sample_image)
    
    def test_dataset_processor(self):
        """Test dataset processor functionality"""
        try:
            processor = DatasetProcessor(self.config)
            
            # Test speech data processing
            speech_train, speech_test = processor.process_speech_data()
            self.assertIsNotNone(speech_train)
            self.assertIsNotNone(speech_test)
            
            # Verify data loader properties
            self.assertEqual(speech_train.batch_size, self.config['batch_size'])
            
            # Test facial data processing
            facial_train, facial_test = processor.process_facial_data()
            self.assertIsNotNone(facial_train)
            self.assertIsNotNone(facial_test)
            
            logger.info("Dataset processor tests passed successfully")
            
        except Exception as e:
            self.fail(f"Dataset processor test failed: {str(e)}")
    
    def test_model_trainer(self):
        """Test model trainer functionality"""
        try:
            # Create dummy data
            batch_size = 2
            speech_features = 15
            facial_features = 468 * 3
            
            # Create dummy speech data
            speech_data = torch.randn(batch_size, speech_features)
            speech_labels = torch.randint(0, 2, (batch_size,))
            
            # Create dummy facial data
            facial_data = torch.randn(batch_size, facial_features)
            facial_labels = torch.randint(0, 5, (batch_size,))
            
            # Test speech model
            speech_model = SpeechNet(
                input_size=speech_features,
                hidden_size=self.config['hidden_size'],
                num_classes=2
            )
            speech_output = speech_model(speech_data)
            self.assertEqual(speech_output.shape, (batch_size, 2))
            
            # Test facial model
            facial_model = FacialExpressionNet(
                input_size=facial_features,
                hidden_size=self.config['hidden_size'],
                num_classes=5
            )
            facial_output = facial_model(facial_data)
            self.assertEqual(facial_output.shape, (batch_size, 5))
            
            logger.info("Model trainer tests passed successfully")
            
        except Exception as e:
            self.fail(f"Model trainer test failed: {str(e)}")
    
    def test_end_to_end_pipeline(self):
        """Test the entire pipeline from data processing to model training"""
        try:
            # Initialize components
            processor = DatasetProcessor(self.config)
            trainer = ModelTrainer(self.config)
            
            # Process data
            speech_train, speech_test = processor.process_speech_data()
            facial_train, facial_test = processor.process_facial_data()
            
            # Train models
            speech_model, speech_history = trainer.train_speech_model(
                speech_train, speech_test
            )
            facial_model, facial_history = trainer.train_facial_model(
                facial_train, facial_test
            )
            
            # Verify training history
            self.assertTrue(len(speech_history['train_loss']) > 0)
            self.assertTrue(len(facial_history['train_loss']) > 0)
            
            logger.info("End-to-end pipeline test passed successfully")
            
        except Exception as e:
            self.fail(f"End-to-end pipeline test failed: {str(e)}")
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test data"""
        shutil.rmtree(cls.test_dir)

if __name__ == '__main__':
    unittest.main() 