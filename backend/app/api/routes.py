# app/api/routes.py
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, BackgroundTasks
from typing import Dict, List, Optional

from app.core.dependencies import get_data_service, get_training_service, get_prediction_service
from app.models.schemas import (
    TrainingRequest, PredictionRequest, JobResponse, 
    DataUploadResponse, ProteinPrediction, SimilarProtein,
    ProteinContextResponse, ContextInfo
)
from app.services.data_service import DataService
from app.services.training_service import TrainingService
from app.services.prediction_service import PredictionService

router = APIRouter()

@router.get("/contexts")
async def get_contexts(data_service: DataService = Depends(get_data_service)):
    """Get available biological contexts"""
    try:
        context_names = data_service.get_context_names()
        
        # Convert integer keys to strings for JSON serialization
        context_mapping_str = {str(k): v for k, v in data_service.context_names_map.items()}
        
        return {
            "contexts": context_names,
            "context_mapping": context_mapping_str,
            "total_contexts": len(context_names)
        }
    except Exception as e:
        print(f"Error in get_contexts: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(500, f"Internal server error: {str(e)}")

@router.get("/proteins")
async def get_proteins(
    search: Optional[str] = None,
    limit: int = 100,
    data_service: DataService = Depends(get_data_service)
):
    """Get available protein IDs with optional search"""
    return data_service.get_proteins(search, limit)

@router.post("/upload-training-data", response_model=DataUploadResponse)
async def upload_training_data(
    positive_file: UploadFile = File(...),
    negative_file: UploadFile = File(...),
    training_service: TrainingService = Depends(get_training_service)
):
    """Upload positive and negative protein JSON files (PINNACLE format)"""
    try:
        pos_contents = await positive_file.read()
        neg_contents = await negative_file.read()
        
        result = await training_service.upload_training_data(pos_contents, neg_contents)
        
        return DataUploadResponse(**result)
        
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(400, f"Error processing files: {str(e)}")

# ===== ADD THESE NEW ENDPOINTS HERE =====

@router.post("/training/analyze/{data_id}")
async def analyze_dataset(
    data_id: str,
    training_service: TrainingService = Depends(get_training_service)
):
    """Get parameter suggestions for uploaded data"""
    try:
        return await training_service.analyze_dataset_and_suggest_params(data_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/training/start")
async def start_training_with_suggestions(
    request: TrainingRequest, 
    data_id: str,
    training_service: TrainingService = Depends(get_training_service)
):
    """Start training with optional custom parameters"""
    try:
        job_id = await training_service.start_training(request, data_id)
        return {"job_id": job_id, "status": "Training started"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ===== END OF NEW ENDPOINTS =====

@router.post("/train", response_model=Dict[str, str])
async def start_training(
    request: TrainingRequest,
    background_tasks: BackgroundTasks,
    data_id: str,
    training_service: TrainingService = Depends(get_training_service)
):
    """Start MLP fine-tuning job (legacy endpoint - use /training/start instead)"""
    try:
        job_id = await training_service.start_training(request, data_id)
        return {"job_id": job_id, "status": "Training started"}
    except ValueError as e:
        raise HTTPException(400, str(e))

@router.get("/jobs/{job_id}", response_model=JobResponse)
async def get_job_status(
    job_id: str,
    training_service: TrainingService = Depends(get_training_service)
):
    """Get training job status"""
    try:
        job = training_service.get_job(job_id)
        return JobResponse(job_id=job_id, **job)
    except ValueError:
        raise HTTPException(404, "Job not found")

@router.get("/jobs", response_model=Dict[str, Dict])
async def list_jobs(training_service: TrainingService = Depends(get_training_service)):
    """List all training jobs"""
    return {"jobs": training_service.list_jobs()}

@router.post("/predict", response_model=Dict[str, List[ProteinPrediction]])
async def predict(
    request: PredictionRequest,
    training_service: TrainingService = Depends(get_training_service),
    prediction_service: PredictionService = Depends(get_prediction_service)
):
    """Make predictions using trained model"""
    try:
        predictions = prediction_service.predict(request, training_service.trained_models)
        return {"predictions": predictions}
    except ValueError as e:
        raise HTTPException(400, str(e))

@router.delete("/models/{model_id}")
async def delete_model(
    model_id: str,
    training_service: TrainingService = Depends(get_training_service)
):
    """Delete a trained model"""
    if training_service.delete_model(model_id):
        return {"message": "Model deleted"}
    raise HTTPException(404, "Model not found")

@router.get("/embeddings/similar-proteins")
async def find_similar_proteins(
    protein_id: str,
    context: str,
    top_k: int = 10,
    data_service: DataService = Depends(get_data_service)
):
    """Find most similar proteins to a query protein in a given context"""
    try:
        context_id = data_service.parse_context(context)
        similar = data_service.find_similar_proteins(protein_id, context_id, top_k)
        
        return {
            "query_protein": protein_id,
            "context": data_service.context_names_map[context_id],
            "similar_proteins": similar
        }
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, str(e))

@router.get("/embeddings/protein-contexts", response_model=ProteinContextResponse)
async def get_protein_contexts(
    protein_id: str,
    data_service: DataService = Depends(get_data_service)
):
    """Get all contexts where a protein appears"""
    contexts = data_service.get_protein_contexts(protein_id)
    
    return ProteinContextResponse(
        protein_id=protein_id,
        contexts=[ContextInfo(**c) for c in contexts],
        total_contexts=len(contexts)
    )

@router.get("/embeddings/context-stats")
async def get_context_stats(data_service: DataService = Depends(get_data_service)):
    """Get statistics about embeddings per context"""
    stats = []
    
    for context_id in data_service.context_ids:
        context_name = data_service.context_names_map.get(context_id)
        embeddings_shape = data_service.pinnacle_embeddings_dict[context_id].shape
        
        stats.append({
            "context_id": context_id,
            "context_name": context_name,
            "protein_count": embeddings_shape[0],
            "embedding_dim": embeddings_shape[1]
        })
    
    return {
        "contexts": stats,
        "total_contexts": len(stats),
        "embedding_dimension": data_service.embedding_dim
    }