# app/services/prediction_service.py
import torch
import numpy as np
from typing import List, Dict
from app.models.schemas import PredictionRequest, ProteinPrediction
from app.services.data_service import DataService
from app.services.training_service import TrainingService

class PredictionService:
    """Service for making predictions with trained models"""
    
    def __init__(self, data_service: DataService):
        self.data_service = data_service
        
    def predict(self, request: PredictionRequest, trained_models: Dict) -> List[ProteinPrediction]:
        """Make predictions using a trained model"""
        if request.model_id not in trained_models:
            raise ValueError("Model not found")
        
        # Get model and its training context
        model_info = trained_models[request.model_id]
        model = model_info['model']
        trained_context_id = model_info['context_id']
        
        model.eval()
        
        # Parse context from request
        context_id = self.data_service.parse_context(request.context)
        
        if context_id != trained_context_id:
            print(f"Warning: Model was trained on context {trained_context_id}, but predicting on context {context_id}")
        
        # Get context information
        context_name = self.data_service.context_names_map.get(context_id)
        context_proteins = self.data_service.celltype_protein_dict.get(context_name, [])
        
        if context_id not in self.data_service.pinnacle_embeddings_dict:
            raise ValueError(f"Context {context_id} not found in embeddings")
        
        context_embeddings = self.data_service.pinnacle_embeddings_dict[context_id]
        
        predictions = []
        with torch.no_grad():
            for protein_id in request.protein_ids:
                if protein_id in context_proteins:
                    try:
                        # Find protein index in context
                        protein_idx = context_proteins.index(protein_id)
                        
                        # Get embedding for this protein
                        embedding = context_embeddings[protein_idx].unsqueeze(0)
                        
                        # Make prediction
                        pred = model(embedding).item()
                        
                        # Convert to probability for binary classification
                        probability = self._sigmoid(pred)
                        
                        predictions.append(ProteinPrediction(
                            protein_id=protein_id,
                            prediction=pred,
                            probability=probability,
                            confidence=abs(pred)
                        ))
                    except ValueError:
                        predictions.append(ProteinPrediction(
                            protein_id=protein_id,
                            prediction=None,
                            error=f"Protein not found in context {context_name}"
                        ))
                else:
                    predictions.append(ProteinPrediction(
                        protein_id=protein_id,
                        prediction=None,
                        error=f"Protein not found in context {context_name}"
                    ))
        
        return predictions
    
    def _sigmoid(self, x: float) -> float:
        """Sigmoid function with numerical stability"""
        if x < -20:
            return 0.0
        elif x > 20:
            return 1.0
        else:
            return 1 / (1 + np.exp(-x))