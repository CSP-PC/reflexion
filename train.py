import argparse
import yaml
import torch
import logging
from data.dataset_processor import DatasetProcessor
from models.model_trainer import ModelTrainer
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description='Train Reflexion models')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration file')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config['device'] = str(device)
    logger.info(f"Using device: {device}")
    
    # Initialize dataset processor
    dataset_processor = DatasetProcessor(config)
    
    # Process datasets
    logger.info("Processing speech dataset...")
    speech_train_loader, speech_test_loader = dataset_processor.process_speech_data()
    
    logger.info("Processing facial expression dataset...")
    facial_train_loader, facial_test_loader = dataset_processor.process_facial_data()
    
    # Initialize model trainer
    trainer = ModelTrainer(config)
    
    # Train speech model
    logger.info("Training speech analysis model...")
    speech_model, speech_history = trainer.train_speech_model(
        speech_train_loader,
        speech_test_loader
    )
    
    # Train facial expression model
    logger.info("Training facial expression model...")
    facial_model, facial_history = trainer.train_facial_model(
        facial_train_loader,
        facial_test_loader
    )
    
    # Save models
    os.makedirs(config['model_save_path'], exist_ok=True)
    
    torch.save(speech_model.state_dict(),
              os.path.join(config['model_save_path'], 'speech_model.pth'))
    torch.save(facial_model.state_dict(),
              os.path.join(config['model_save_path'], 'facial_model.pth'))
    
    logger.info("Training completed successfully!")

if __name__ == '__main__':
    main() 