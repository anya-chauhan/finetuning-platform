# app/core/dependencies.py
from fastapi import Request
from app.services.data_service import DataService
from app.services.training_service import TrainingService
from app.services.prediction_service import PredictionService

def get_data_service(request: Request) -> DataService:
    """Dependency to get data service"""
    return request.app.state.data_service

def get_training_service(request: Request) -> TrainingService:
    """Dependency to get training service"""
    if not hasattr(request.app.state, 'training_service'):
        request.app.state.training_service = TrainingService(request.app.state.data_service)
    return request.app.state.training_service

def get_prediction_service(request: Request) -> PredictionService:
    """Dependency to get prediction service"""
    if not hasattr(request.app.state, 'prediction_service'):
        request.app.state.prediction_service = PredictionService(request.app.state.data_service)
    return request.app.state.prediction_service