# app/models/ml_models.py
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from typing import List
from app.core.config import get_settings

settings = get_settings()

class MLPPredictor(nn.Module):
    """Multi-layer perceptron for protein prediction tasks"""
    
    def __init__(self, input_dim: int, hidden_dim: int = None, output_dim: int = 1):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = settings.DEFAULT_HIDDEN_DIM
            
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(settings.DROPOUT_RATE_1),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(settings.DROPOUT_RATE_2),
            
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(settings.DROPOUT_RATE_3),
            
            nn.Linear(hidden_dim // 4, output_dim)
        )
    
    def forward(self, x):
        return self.network(x)

class ProteinDataset(Dataset):
    """PyTorch Dataset for protein embeddings and labels"""
    
    def __init__(self, embeddings: List[torch.Tensor], labels: List[float]):
        self.embeddings = embeddings
        self.labels = torch.tensor(labels, dtype=torch.float32)
    
    def __len__(self):
        return len(self.embeddings)
    
    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]