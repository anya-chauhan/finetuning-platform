# app/models/neural_networks.py
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from typing import List

class MLPPredictor(nn.Module):
    """Multi-layer perceptron for protein predictions"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 256, output_dim: int = 1):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 4, output_dim)
        )
    
    def forward(self, x):
        return self.network(x)

class ProteinDataset(Dataset):
    """Dataset for protein embeddings and labels"""
    
    def __init__(self, embeddings: List[torch.Tensor], labels: List[float]):
        self.embeddings = embeddings
        self.labels = torch.tensor(labels, dtype=torch.float32)
    
    def __len__(self):
        return len(self.embeddings)
    
    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]