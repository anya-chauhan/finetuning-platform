# app/core/storage.py
import uuid
from typing import Dict, Any, Optional, List
from datetime import datetime
import torch.nn as nn

class JobStorage:
    """In-memory storage for training jobs and data"""
    
    def __init__(self):
        self.jobs: Dict[str, Dict[str, Any]] = {}
    
    def create_data_job(self, protein_ids: List[str], labels: List[int]) -> str:
        """Create a data storage job"""
        data_id = str(uuid.uuid4())
        self.jobs[data_id] = {
            "type": "data",
            "protein_ids": protein_ids,
            "labels": labels,
            "timestamp": datetime.now().isoformat()
        }
        return data_id
    
    def create_training_job(self, job_name: str, context: str, 
                          data_id: str, config: Dict[str, Any]) -> str:
        """Create a training job"""
        job_id = str(uuid.uuid4())
        self.jobs[job_id] = {
            "type": "training",
            "status": "started",
            "progress": 0,
            "job_name": job_name,
            "context": context,
            "data_id": data_id,
            "config": config,
            "timestamp": datetime.now().isoformat(),
            "metrics": []
        }
        return job_id
    
    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get job by ID"""
        return self.jobs.get(job_id)
    
    def update_job(self, job_id: str, updates: Dict[str, Any]) -> None:
        """Update job with new data"""
        if job_id in self.jobs:
            self.jobs[job_id].update(updates)
    
    def add_metric(self, job_id: str, metric: Dict[str, Any]) -> None:
        """Add training metric to job"""
        if job_id in self.jobs and "metrics" in self.jobs[job_id]:
            self.jobs[job_id]["metrics"].append(metric)
    
    def get_training_jobs(self) -> Dict[str, Dict[str, Any]]:
        """Get all training jobs"""
        return {k: v for k, v in self.jobs.items() if v.get("type") == "training"}
    
    def job_exists(self, job_id: str, job_type: str = None) -> bool:
        """Check if job exists and optionally verify type"""
        job = self.jobs.get(job_id)
        if not job:
            return False
        if job_type and job.get("type") != job_type:
            return False
        return True

class ModelStorage:
    """In-memory storage for trained models"""
    
    def __init__(self):
        self.models: Dict[str, Dict[str, Any]] = {}
    
    def store_model(self, model_id: str, model: nn.Module, context_id: int) -> None:
        """Store a trained model"""
        self.models[model_id] = {
            'model': model,
            'context_id': context_id,
            'timestamp': datetime.now().isoformat()
        }
    
    def get_model(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get model by ID"""
        return self.models.get(model_id)
    
    def delete_model(self, model_id: str) -> bool:
        """Delete model by ID"""
        if model_id in self.models:
            del self.models[model_id]
            return True
        return False
    
    def model_exists(self, model_id: str) -> bool:
        """Check if model exists"""
        return model_id in self.models
    
    def list_models(self) -> Dict[str, Dict[str, Any]]:
        """List all stored models"""
        return {
            model_id: {
                'context_id': info['context_id'],
                'timestamp': info['timestamp']
            }
            for model_id, info in self.models.items()
        }