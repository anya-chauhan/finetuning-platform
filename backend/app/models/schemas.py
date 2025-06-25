# app/models/schemas.py
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime

class TrainingRequest(BaseModel):
    """Request model for training"""
    job_name: str
    context: str
    task_type: str = "binary_classification"
    epochs: int = 30
    learning_rate: float = 0.001
    batch_size: int = 32
    selected_contexts: Optional[List[str]] = None

class PredictionRequest(BaseModel):
    """Request model for predictions"""
    model_id: str
    protein_ids: List[str]
    context: str

class ProteinPrediction(BaseModel):
    """Single protein prediction result"""
    protein_id: str
    prediction: Optional[float]
    probability: Optional[float] = None
    confidence: Optional[float] = None
    error: Optional[str] = None

class JobMetric(BaseModel):
    """Training metric for a single epoch"""
    epoch: int
    loss: float
    best_loss: float
    timestamp: str

class JobResponse(BaseModel):
    """Response model for job status"""
    job_id: str
    type: str
    status: str
    progress: int = 0
    job_name: Optional[str] = None
    context: Optional[str] = None
    config: Optional[Dict[str, Any]] = None
    timestamp: str
    metrics: List[JobMetric] = []
    model_id: Optional[str] = None
    final_loss: Optional[float] = None
    best_loss: Optional[float] = None
    error: Optional[str] = None

class DataUploadResponse(BaseModel):
    """Response after uploading training data"""
    data_id: str
    total_proteins: int
    positive_proteins: int
    negative_proteins: int

class ContextInfo(BaseModel):
    """Context information"""
    context_id: int
    context_name: str
    protein_count: int

class SimilarProtein(BaseModel):
    """Similar protein result"""
    protein_id: str
    similarity: float

class ProteinContextResponse(BaseModel):
    """Response for protein context query"""
    protein_id: str
    contexts: List[ContextInfo]
    total_contexts: int