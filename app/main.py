# main.py

from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
import os
import logging
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware

from app.models import (
    QueryRequest, 
    QueryResponse, 
    DocumentInput
)
from app.database import (
    initialize_embeddings_model,
    initialize_vector_store,
    match_documents_in_db,
    add_document_to_db,
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
    # Initialize embeddings model and vector store
    try:
        logger.info("Initializing application components...")
        initialize_embeddings_model()
        initialize_vector_store()
        logger.info("Application initialized successfully")
    except Exception as e:
        logger.error(f"Startup error: {str(e)}")
        raise
    
    yield
    
    # No specific cleanup needed since PGVector doesn't have a close method
    logger.info("Application shutting down")

# Initialize FastAPI
app = FastAPI(
    title="Document Matcher API",
    description="Match document descriptions to existing documents with semantic search",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development. Restrict this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return await get_health_status()

@app.post("/match-documents", response_model=QueryResponse)
async def match_documents(request: QueryRequest):
    """Match documents based on description query"""
    # Log the request parameters
    logger.info(f"Match request - Query: '{request.query}'")
    logger.info(f"Match request - Min confidence: {request.min_confidence}")
    logger.info(f"Match request - Max results: {request.max_results}")
    
    # Use LangChain-based document matching
    result = await match_documents_in_db(request)
    
    # Log the results
    logger.info(f"Match results - Found {len(result.matches)} matches")
    
    return result

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
        "added_count": len(documents),
        "documents": results
    }

@app.delete("/document/{document_id}")
async def delete_document(document_id: str, by_custom_id: bool = True):
    """Delete a document by ID"""
    return await delete_document_from_db(document_id, by_custom_id)

@app.post("/debug/match-low-threshold")
async def match_documents_low_threshold(request: QueryRequest):
    """Match documents with a very low confidence threshold for debugging"""
    # Create a copy of the request with a lower confidence threshold
    debug_request = QueryRequest(
        query=request.query,
        max_results=20,  # Return more results for debugging
        min_confidence=0.01,  # Very low threshold
        metadata_filter=request.metadata_filter  # Keep original filters
    )
    
    logger.info(f"Debug match with low threshold for query: {debug_request.query}")
    return await match_documents_in_db(debug_request)
