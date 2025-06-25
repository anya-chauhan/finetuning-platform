# app/services/training_service.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import asyncio
import uuid
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import json
import numpy as np  # Added missing import

from app.models.neural_networks import MLPPredictor, ProteinDataset
from app.models.schemas import TrainingRequest
from app.services.data_service import DataService

class TrainingService:
    """Service for handling model training with intelligent parameter suggestions"""
    
    def __init__(self, data_service: DataService):
        self.data_service = data_service
        self.jobs: Dict[str, dict] = {}
        self.trained_models: Dict[str, dict] = {}
        
    async def upload_training_data(self, positive_contents: bytes, negative_contents: bytes) -> Dict:
        """Process uploaded training data files"""
        try:
            # Parse JSON files
            pos_data = json.loads(positive_contents.decode('utf-8'))
            neg_data = json.loads(negative_contents.decode('utf-8'))
            
            # Extract protein lists from PINNACLE JSON structure
            positive_proteins = self._extract_proteins_from_json(pos_data)
            negative_proteins = self._extract_proteins_from_json(neg_data)
            
            # Remove duplicates
            positive_proteins = list(set(positive_proteins))
            negative_proteins = list(set(negative_proteins))
            
            print(f"Loaded {len(positive_proteins)} positive and {len(negative_proteins)} negative proteins")
            
            # Create training data
            all_protein_ids = positive_proteins + negative_proteins
            all_labels = [1] * len(positive_proteins) + [0] * len(negative_proteins)
            
            # Store training data
            data_id = str(uuid.uuid4())
            self.jobs[data_id] = {
                "type": "data",
                "protein_ids": all_protein_ids,
                "labels": all_labels,
                "timestamp": datetime.now().isoformat()
            }
            
            return {
                "data_id": data_id,
                "total_proteins": len(all_protein_ids),
                "positive_proteins": len(positive_proteins),
                "negative_proteins": len(negative_proteins)
            }
            
        except json.JSONDecodeError:
            raise ValueError("Files must be valid JSON format")
    
    def _extract_proteins_from_json(self, data: dict) -> List[str]:
        """Extract protein lists from PINNACLE JSON structure"""
        proteins = []
        
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, dict):  # EFO format
                    for context, protein_list in value.items():
                        if isinstance(protein_list, list):
                            proteins.extend(protein_list)
                elif isinstance(value, list):  # Direct format
                    proteins.extend(value)
        
        return proteins
    
    async def analyze_dataset_and_suggest_params(self, data_id: str) -> Dict:
        """Analyze uploaded data and suggest optimal training parameters"""
        
        if data_id not in self.jobs or self.jobs[data_id]["type"] != "data":
            raise ValueError("Training data not found")
        
        # Get the data
        data = self.jobs[data_id]
        protein_ids_data = data["protein_ids"]
        labels = data["labels"]
        
        # Calculate dataset statistics
        n_samples = len(labels)
        n_positive = sum(labels)
        n_negative = n_samples - n_positive
        imbalance_ratio = n_positive / n_negative if n_negative > 0 else 1
        
        # Suggest parameters based on dataset characteristics
        suggestions = {}
        
        # 1. Batch size based on dataset size
        if n_samples < 50:
            suggestions["batch_size"] = 8
            suggestions["batch_size_reason"] = "Small dataset - using smaller batches for more gradient updates"
        elif n_samples < 200:
            suggestions["batch_size"] = 16
            suggestions["batch_size_reason"] = "Medium dataset - balanced batch size"
        else:
            suggestions["batch_size"] = 32
            suggestions["batch_size_reason"] = "Large dataset - efficient training with larger batches"
        
        # 2. Epochs based on dataset size
        if n_samples < 50:
            suggestions["epochs"] = 100
            suggestions["epochs_reason"] = "Small dataset - more epochs but watch for overfitting"
            suggestions["early_stopping"] = True
        elif n_samples < 200:
            suggestions["epochs"] = 50
            suggestions["epochs_reason"] = "Medium dataset - standard training duration"
        else:
            suggestions["epochs"] = 30
            suggestions["epochs_reason"] = "Large dataset - fewer epochs needed"
        
        # 3. Learning rate based on dataset complexity
        if imbalance_ratio > 2 or imbalance_ratio < 0.5:
            suggestions["learning_rate"] = 0.0001
            suggestions["lr_reason"] = "Imbalanced dataset - slower learning for stability"
        else:
            suggestions["learning_rate"] = 0.001
            suggestions["lr_reason"] = "Balanced dataset - standard learning rate"
        
        # 4. Model architecture based on dataset size
        if n_samples < 30:
            suggestions["model_type"] = "linear"
            suggestions["model_reason"] = "Very small dataset - simple linear model to avoid overfitting"
        elif n_samples < 100:
            suggestions["model_type"] = "shallow_mlp"
            suggestions["architecture"] = [128, 32]
            suggestions["model_reason"] = "Small dataset - shallow network"
        else:
            suggestions["model_type"] = "mlp"
            suggestions["architecture"] = [128, 64, 32]
            suggestions["model_reason"] = "Sufficient data for deeper network"
        
        # 5. Regularization based on overfitting risk
        overfitting_risk = n_samples < 100
        if overfitting_risk:
            suggestions["dropout"] = 0.3
            suggestions["weight_decay"] = 0.01
            suggestions["regularization_reason"] = "Small dataset - adding regularization"
        else:
            suggestions["dropout"] = 0.1
            suggestions["weight_decay"] = 0.001
            suggestions["regularization_reason"] = "Large dataset - minimal regularization needed"
        
        # 6. Class imbalance handling
        if imbalance_ratio > 2:
            suggestions["use_class_weights"] = True
            suggestions["pos_weight"] = 1 / imbalance_ratio
            suggestions["imbalance_reason"] = f"Dataset has {n_positive} positive vs {n_negative} negative samples"
        elif imbalance_ratio < 0.5:
            suggestions["use_class_weights"] = True
            suggestions["pos_weight"] = 1 / imbalance_ratio
            suggestions["imbalance_reason"] = f"Dataset has {n_positive} positive vs {n_negative} negative samples"
        else:
            suggestions["use_class_weights"] = False
            suggestions["imbalance_reason"] = "Dataset is reasonably balanced"
        
        # 7. Validation strategy
        if n_samples < 50:
            suggestions["validation_strategy"] = "leave_one_out"
            suggestions["validation_reason"] = "Very small dataset - using leave-one-out CV"
        elif n_samples < 200:
            suggestions["validation_strategy"] = "5_fold_cv"
            suggestions["validation_reason"] = "Small dataset - using 5-fold cross-validation"
        else:
            suggestions["validation_strategy"] = "train_val_split"
            suggestions["validation_split"] = 0.2
            suggestions["validation_reason"] = "Large dataset - simple train/validation split"
        
        # Calculate confidence in suggestions
        confidence_factors = {
            "dataset_size": min(n_samples / 100, 1.0),  # More confidence with more data
            "balance": 1.0 - abs(0.5 - n_positive/n_samples) * 2,  # More confidence when balanced
        }
        suggestions["confidence"] = sum(confidence_factors.values()) / len(confidence_factors)
        
        # Summary recommendation
        suggestions["summary"] = f"""
        Based on your dataset with {n_samples} proteins ({n_positive} positive, {n_negative} negative):
        • Use a {'simple linear model' if n_samples < 30 else 'shallow neural network' if n_samples < 100 else 'standard MLP'}
        • Train for {suggestions['epochs']} epochs with early stopping
        • {'Address class imbalance with weighted loss' if suggestions['use_class_weights'] else 'No class balancing needed'}
        • {'High risk of overfitting - strong regularization applied' if overfitting_risk else 'Standard regularization'}
        """
        
        return {
            "dataset_stats": {
                "total_samples": n_samples,
                "positive_samples": n_positive,
                "negative_samples": n_negative,
                "imbalance_ratio": round(imbalance_ratio, 2)
            },
            "suggested_parameters": suggestions,
            "warnings": self._generate_warnings(n_samples, imbalance_ratio)
        }
    
    def _generate_warnings(self, n_samples: int, imbalance_ratio: float) -> List[Dict]:
        """Generate warnings based on dataset characteristics"""
        warnings = []
        
        if n_samples < 20:
            warnings.append({
                "level": "critical",
                "message": "Very small dataset - results may be unreliable. Consider gathering more data."
            })
        elif n_samples < 50:
            warnings.append({
                "level": "warning",
                "message": "Small dataset - use cross-validation to assess model reliability"
            })
        
        if imbalance_ratio > 5 or imbalance_ratio < 0.2:
            warnings.append({
                "level": "warning",
                "message": "Severe class imbalance detected - consider resampling techniques"
            })
        
        return warnings
    
    async def start_training(self, request: TrainingRequest, data_id: str) -> str:
        """Start a training job with optional suggested parameters"""
        if data_id not in self.jobs or self.jobs[data_id]["type"] != "data":
            raise ValueError("Training data not found")
        
        # Validate context
        context_id = self.data_service.parse_context(request.context)
        
        job_id = str(uuid.uuid4())
        
        # Get suggestions if not provided custom parameters
        suggestions = await self.analyze_dataset_and_suggest_params(data_id)
        
        # Initialize job status
        self.jobs[job_id] = {
            "type": "training",
            "status": "started",
            "progress": 0,
            "job_name": request.job_name,
            "context": request.context,
            "data_id": data_id,
            "config": request.dict(),
            "suggestions": suggestions["suggested_parameters"],
            "timestamp": datetime.now().isoformat(),
            "metrics": []
        }
        
        # Start training asynchronously
        asyncio.create_task(self._train_model(job_id, request, data_id, suggestions["suggested_parameters"]))
        
        return job_id
    
    async def _train_model(self, job_id: str, config: TrainingRequest, data_id: str, suggestions: Dict):
        """Background training function with optional suggested parameters"""
        try:
            print(f"Starting training for job {job_id}")
            
            # Get training data
            data = self.jobs[data_id]
            protein_ids_data = data["protein_ids"]
            labels = data["labels"]
            
            # ADD THIS - Handle multiple contexts
            # Get config dict to check for selected_contexts
            config_dict = config.dict()

            # Determine which contexts to use
            contexts_to_use = []
            if 'selected_contexts' in config_dict and config_dict['selected_contexts']:
                contexts_to_use = config_dict['selected_contexts']
            elif config.context:
                contexts_to_use = [config.context]
            else:
                raise ValueError("No contexts selected for training")

            print(f"Training with contexts: {contexts_to_use}")

            # Collect embeddings from ALL contexts
            all_embeddings = []
            all_labels = []
            all_valid_proteins = []  # Track which proteins were used

            for context in contexts_to_use:
                context_id = self.data_service.parse_context(context)
                embeddings_list, valid_indices = self.data_service.get_embeddings_for_proteins(
                    protein_ids_data, context_id
                )
                
                if len(embeddings_list) > 0:
                    # Add embeddings from this context
                    all_embeddings.extend(embeddings_list)
                    
                    # Get corresponding labels for valid proteins in this context
                    context_labels = [labels[i] for i in valid_indices]
                    all_labels.extend(context_labels)
        
                    # Track which proteins were valid (for gene importance later)
                    valid_proteins = [protein_ids_data[i] for i in valid_indices]
                    all_valid_proteins.extend(valid_proteins)
                    
                    print(f"Context {context}: {len(embeddings_list)} valid proteins")

            if len(all_embeddings) < 2:
                raise ValueError(f"Only {len(all_embeddings)} total embeddings found. Need at least 2.")

            print(f"Total training samples across all contexts: {len(all_embeddings)}")
            
            # Use suggested parameters if user hasn't overridden them
            batch_size = config.batch_size if hasattr(config, 'batch_size') and config.batch_size else suggestions.get("batch_size", 32)
            learning_rate = config.learning_rate if hasattr(config, 'learning_rate') and config.learning_rate else suggestions.get("learning_rate", 0.001)
            epochs = config.epochs if hasattr(config, 'epochs') and config.epochs else suggestions.get("epochs", 50)
            weight_decay = suggestions.get("weight_decay", 0.0)
            use_class_weights = suggestions.get("use_class_weights", False)
            pos_weight = suggestions.get("pos_weight", 1.0) if use_class_weights else None
            
            # Create dataset and dataloader
            dataset = ProteinDataset(all_embeddings, all_labels)
            dataloader = DataLoader(
                dataset, 
                batch_size=min(batch_size, len(dataset)), 
                shuffle=True
            )
            
            # Initialize model with suggested architecture if available
            if suggestions.get("model_type") == "linear":
                # Simple linear model
                model = nn.Sequential(
                    nn.Linear(self.data_service.embedding_dim, 1)
                )
            elif suggestions.get("architecture"):
                # Custom MLP with suggested architecture
                layers = []
                in_dim = self.data_service.embedding_dim
                for hidden_dim in suggestions["architecture"]:
                    layers.extend([
                        nn.Linear(in_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Dropout(suggestions.get("dropout", 0.1))
                    ])
                    in_dim = hidden_dim
                layers.append(nn.Linear(in_dim, 1))
                model = nn.Sequential(*layers)
            else:
                # Default model
                model = MLPPredictor(self.data_service.embedding_dim)
            
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
            
            # Set up loss function with optional class weights
            if config.task_type == "binary_classification":
                if pos_weight:
                    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))
                else:
                    criterion = nn.BCEWithLogitsLoss()
            else:
                criterion = nn.MSELoss()
            
            print(f"Training {config.task_type} model for {epochs} epochs...")
            print(f"Model input dim: {self.data_service.embedding_dim}, Dataset size: {len(dataset)}")
            print(f"Using suggested hyperparameters: batch_size={batch_size}, lr={learning_rate}, weight_decay={weight_decay}")
            
            # Training loop
            best_loss = float('inf')
            patience_counter = 0
            patience = 10 if suggestions.get("early_stopping") else epochs
            
            for epoch in range(epochs):
                model.train()
                epoch_loss = 0
                num_batches = 0
                
                for batch_embeddings, batch_labels in dataloader:
                    optimizer.zero_grad()
                    outputs = model(batch_embeddings).squeeze()
                    
                    if outputs.dim() == 0:
                        outputs = outputs.unsqueeze(0)
                    if batch_labels.dim() == 0:
                        batch_labels = batch_labels.unsqueeze(0)
                    
                    loss = criterion(outputs, batch_labels)
                    loss.backward()
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                    num_batches += 1
                
                avg_loss = epoch_loss / num_batches if num_batches > 0 else 0
                progress = int((epoch + 1) / epochs * 100)
                
                # Early stopping check
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    
                if patience_counter >= patience and suggestions.get("early_stopping"):
                    print(f"Early stopping at epoch {epoch + 1}")
                    break
                
                # Update job status
                self.jobs[job_id]["progress"] = progress
                self.jobs[job_id]["metrics"].append({
                    "epoch": epoch + 1,
                    "loss": avg_loss,
                    "best_loss": best_loss,
                    "timestamp": datetime.now().isoformat()
                })
                
                if epoch % 10 == 0:
                    print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")
                
                await asyncio.sleep(0.01)
            
            # CALCULATE GENE IMPORTANCE AFTER TRAINING COMPLETES
            # Check for both old single context and new multi-context
            contexts_to_analyze = []
            # Get unique proteins that were actually used in training
            unique_training_proteins = list(set(all_valid_proteins))
            
            # Get config dict to check for selected_contexts
            config_dict = config.dict()
            
            # Handle multi-select contexts
            if 'selected_contexts' in config_dict and config_dict['selected_contexts']:
                contexts_to_analyze = config_dict['selected_contexts']
                print(f"Found selected_contexts: {contexts_to_analyze}")
            # Fall back to single context for backward compatibility
            elif config.context:
                contexts_to_analyze = [config.context]
                print(f"Using single context: {config.context}")
            
    
            if contexts_to_analyze:
                context_importance = {}
                
                for context in contexts_to_analyze:
                    # Get embeddings for proteins in this specific context
                    context_id = self.data_service.parse_context(context)
                    
                    # Get embeddings for the proteins that were used in training
                    context_embeddings, valid_indices = self.data_service.get_embeddings_for_proteins(
                        unique_training_proteins, context_id
                    )
                    
                    # Get the valid proteins for this context
                    valid_proteins = [unique_training_proteins[i] for i in valid_indices]
        
                    # Calculate importance with context-specific embeddings
                    importance_data = self.calculate_gene_importance(
                        model, 
                        protein_names=valid_proteins,
                        context_names=[context],
                        embeddings=context_embeddings  # Pass the context-specific embeddings!
                    )
        
                    context_importance[context] = importance_data
                
                # Store in job results
                self.jobs[job_id]["gene_importance"] = context_importance
                print(f"Gene importance calculated for contexts: {contexts_to_analyze}")
            else:
                print("No contexts found for gene importance calculation")
            
            # Save trained model
            model_id = f"model_{job_id}"
            self.trained_models[model_id] = {
                'model': model,
                'contexts_used': contexts_to_use,
                'hyperparameters': {
                    'batch_size': batch_size,
                    'learning_rate': learning_rate,
                    'epochs': epoch + 1,  # Actual epochs trained
                    'weight_decay': weight_decay,
                    'architecture': suggestions.get("architecture"),
                    'model_type': suggestions.get("model_type", "default")
                }
            }
            
            # Update job status
            self.jobs[job_id]["status"] = "completed"
            self.jobs[job_id]["model_id"] = model_id
            self.jobs[job_id]["final_loss"] = avg_loss
            self.jobs[job_id]["best_loss"] = best_loss
            self.jobs[job_id]["actual_epochs"] = epoch + 1
            
            print(f"Training completed for job {job_id}, final loss: {avg_loss:.4f}")
        
        except Exception as e:
            self.jobs[job_id]["status"] = "failed"
            self.jobs[job_id]["error"] = str(e)
            print(f"Training error for job {job_id}: {e}")
            import traceback
            traceback.print_exc()
    
    def calculate_gene_importance(self, model, protein_names, context_names=None, embeddings=None):
        """
        Calculate gene importance scores from trained MLP
        Using interaction between model weights and context-specific embeddings
        """
        # Get first layer - handle both Sequential and single layer models
        if isinstance(model, nn.Sequential):
            first_layer = None
            for layer in model:
                if isinstance(layer, nn.Linear):
                    first_layer = layer
                    break
        else:
            first_layer = model
        
        if first_layer is None:
            return {'protein_scores': {}, 'ranked_proteins': []}
        
        # Get weights
        first_layer_weights = first_layer.weight.data.cpu().numpy()  # shape: (hidden_units, input_dim)
        
        # If embeddings provided, calculate context-specific importance
        if embeddings is not None:
            # embeddings should be shape: (n_proteins, input_dim)
            if not isinstance(embeddings, np.ndarray):
                embeddings = embeddings.cpu().numpy() if hasattr(embeddings, 'cpu') else np.array(embeddings)
            
            # Calculate importance as the magnitude of activation for each protein
            # This captures how strongly each protein's embedding activates the first layer
            importance_scores = []
            for i in range(len(protein_names)):
                if i < len(embeddings):
                    # Get this protein's embedding
                    protein_embedding = embeddings[i]
                    
                    # Calculate how this embedding interacts with the model weights
                    # This is the pre-activation values for this protein
                    activation = np.abs(first_layer_weights @ protein_embedding)
                    
                    # Take mean activation across all hidden units
                    importance = activation.mean()
                    importance_scores.append(importance)
                else:
                    importance_scores.append(0.0)
            
            importance_scores = np.array(importance_scores)
        else:
            # Fallback to weight-only importance (your original method)
            importance_scores = np.abs(first_layer_weights).mean(axis=0)
        
        # Normalize to 0-1 range
        if importance_scores.max() > importance_scores.min():
            importance_scores = (importance_scores - importance_scores.min()) / (importance_scores.max() - importance_scores.min())
        else:
            importance_scores = np.ones_like(importance_scores) * 0.5
        
        # Create protein-score mapping
        protein_importance = {}
        for idx, protein in enumerate(protein_names):
            if idx < len(importance_scores):
                protein_importance[protein] = float(importance_scores[idx])
        
        # Sort by importance
        sorted_proteins = sorted(protein_importance.items(), key=lambda x: x[1], reverse=True)
        
        return {
            'protein_scores': protein_importance,
            'ranked_proteins': sorted_proteins
        }
    
    def get_job(self, job_id: str) -> dict:
        """Get job status"""
        if job_id not in self.jobs:
            raise ValueError("Job not found")
        return self.jobs[job_id]
    
    def list_jobs(self) -> Dict[str, dict]:
        """List all training jobs"""
        return {k: v for k, v in self.jobs.items() if v.get("type") == "training"}
    
    def delete_model(self, model_id: str) -> bool:
        """Delete a trained model"""
        if model_id in self.trained_models:
            del self.trained_models[model_id]
            return True
        return False