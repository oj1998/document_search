from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
import os
import logging
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development. Restrict this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from app.models import (
    QueryRequest, 
    QueryResponse, 
    DocumentInput
)
from app.database import (
    initialize_db_pool,
    initialize_embeddings_model,
    match_documents_in_db,
    add_document_to_db,
    close_db_pool,
    delete_document_from_db,
    get_health_status
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Startup and shutdown events manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize database and embeddings model
    try:
        await initialize_db_pool()
        initialize_embeddings_model()
        logger.info("Application initialized successfully")
    except Exception as e:
        logger.error(f"Startup error: {str(e)}")
        raise
    
    yield
    
    # Cleanup on shutdown
    await close_db_pool()

# Initialize FastAPI
app = FastAPI(
    title="Document Matcher API",
    description="Match document descriptions to existing documents with semantic search",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return await get_health_status()

@app.post("/match-documents", response_model=QueryResponse)
async def match_documents(request: QueryRequest):
    """Match documents based on description query"""
    return await match_documents_in_db(request)

@app.post("/add-document")
async def add_document(document: DocumentInput):
    """Add a document to the database with embeddings"""
    return await add_document_to_db(document)

@app.post("/bulk-add-documents")
async def bulk_add_documents(documents: list[DocumentInput]):
    """Add multiple documents to the database with embeddings"""
    results = []
    for document in documents:
        result = await add_document_to_db(document)
        results.append(result)
    
    return {
        "status": "success",
        "added_count": len(documents)
    }

@app.delete("/document/{document_id}")
async def delete_document(document_id: str, by_custom_id: bool = True):
    """Delete a document by ID"""
    return await delete_document_from_db(document_id, by_custom_id)
