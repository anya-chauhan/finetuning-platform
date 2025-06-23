# main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from app.api.routes import router
from app.core.config import settings
from app.services.data_service import DataService

app = FastAPI(title="PINNACLE Fine-tuning API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes without prefix to maintain compatibility
app.include_router(router)

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    print("Initializing PINNACLE data service...")
    try:
        data_service = DataService()
        await data_service.initialize()
        app.state.data_service = data_service
        print(f"✅ Data service initialized successfully")
        print(f"   - Proteins: {data_service.num_proteins}")
        print(f"   - Contexts: {data_service.num_contexts}")
        print(f"   - Embedding dimension: {data_service.embedding_dim}")
    except Exception as e:
        print(f"❌ Failed to initialize data service: {e}")
        import traceback
        traceback.print_exc()
        raise

@app.get("/")
async def root():
    """API root endpoint"""
    data_service = app.state.data_service
    return {
        "message": "PINNACLE Fine-tuning API",
        "version": "1.0.0",
        "status": "running",
        "num_proteins": data_service.num_proteins,
        "num_contexts": data_service.num_contexts,
        "embedding_dim": data_service.embedding_dim
    }

if __name__ == "__main__":
    print("Starting PINNACLE Fine-tuning API...")
    print("API will be available at: http://localhost:8000")
    print("API documentation at: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)