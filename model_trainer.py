import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Tuple, List
import numpy as np
from sklearn.metrics import classification_report
import logging
import wandb
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FacialExpressionNet(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_classes: int):
        super(FacialExpressionNet, self).__init__()
        
        self.features = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size // 2, num_classes)
        )
        
    def forward(self, x):
        return self.features(x)

class SpeechNet(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_classes: int):
        super(SpeechNet, self).__init__()
        
        self.features = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size // 2, num_classes)
        )
        
    def forward(self, x):
        return self.features(x)

class ModelTrainer:
    def __init__(self, config: Dict):
        """
        Initialize model trainer with configuration
        
        Args:
            config: Dictionary containing:
                - learning_rate: Learning rate for optimization
                - num_epochs: Number of training epochs
                - hidden_size: Size of hidden layers
                - device: Training device (cuda/cpu)
                - wandb_project: W&B project name for logging
        """
        self.config = config
        self.device = torch.device(config['device'])
        
        # Initialize W&B for experiment tracking
        if config.get('wandb_project'):
            wandb.init(project=config['wandb_project'])
    
    def train_facial_model(
        self,
        train_loader: DataLoader,
        test_loader: DataLoader
    ) -> Tuple[FacialExpressionNet, Dict]:
        """
        Train facial expression analysis model
        """
        # Initialize model
        input_size = 468 * 3  # MediaPipe face mesh features
        model = FacialExpressionNet(
            input_size=input_size,
            hidden_size=self.config['hidden_size'],
            num_classes=5  # number of emotion classes
        ).to(self.device)
        
        return self._train_model(model, train_loader, test_loader, "facial")
    
    def train_speech_model(
        self,
        train_loader: DataLoader,
        test_loader: DataLoader
    ) -> Tuple[SpeechNet, Dict]:
        """
        Train speech analysis model
        """
        # Initialize model
        input_size = 15  # MFCC + spectral features
        model = SpeechNet(
            input_size=input_size,
            hidden_size=self.config['hidden_size'],
            num_classes=2  # binary classification for dementia
        ).to(self.device)
        
        return self._train_model(model, train_loader, test_loader, "speech")
    
    def _train_model(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        model_type: str
    ) -> Tuple[nn.Module, Dict]:
        """
        Generic training loop for both models
        """
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(
            model.parameters(),
            lr=self.config['learning_rate']
        )
        
        best_accuracy = 0.0
        best_model_state = None
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        for epoch in range(self.config['num_epochs']):
            # Training phase
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_features, batch_labels in tqdm(train_loader, 
                                                   desc=f"Epoch {epoch+1}"):
                batch_features = batch_features.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(batch_features)
                loss = criterion(outputs, batch_labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += batch_labels.size(0)
                train_correct += (predicted == batch_labels).sum().item()
            
            train_loss = train_loss / len(train_loader)
            train_accuracy = 100 * train_correct / train_total
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            all_predictions = []
            all_labels = []
            
            with torch.no_grad():
                for batch_features, batch_labels in test_loader:
                    batch_features = batch_features.to(self.device)
                    batch_labels = batch_labels.to(self.device)
                    
                    outputs = model(batch_features)
                    loss = criterion(outputs, batch_labels)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += batch_labels.size(0)
                    val_correct += (predicted == batch_labels).sum().item()
                    
                    all_predictions.extend(predicted.cpu().numpy())
                    all_labels.extend(batch_labels.cpu().numpy())
            
            val_loss = val_loss / len(test_loader)
            val_accuracy = 100 * val_correct / val_total
            
            # Log metrics
            if self.config.get('wandb_project'):
                wandb.log({
                    f'{model_type}_train_loss': train_loss,
                    f'{model_type}_train_accuracy': train_accuracy,
                    f'{model_type}_val_loss': val_loss,
                    f'{model_type}_val_accuracy': val_accuracy
                })
            
            # Save best model
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                best_model_state = model.state_dict()
            
            # Update history
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_accuracy)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_accuracy)
            
            logger.info(
                f"Epoch {epoch+1}/{self.config['num_epochs']} - "
                f"Train Loss: {train_loss:.4f} - "
                f"Train Acc: {train_accuracy:.2f}% - "
                f"Val Loss: {val_loss:.4f} - "
                f"Val Acc: {val_accuracy:.2f}%"
            )
        
        # Load best model
        model.load_state_dict(best_model_state)
        
        # Generate classification report
        report = classification_report(
            all_labels,
            all_predictions,
            output_dict=True
        )
        
        logger.info(f"\nBest validation accuracy: {best_accuracy:.2f}%")
        logger.info("\nClassification Report:")
        logger.info(report)
        
        return model, history 