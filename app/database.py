# database.py - Updated with LangChain components

import os
import asyncio
import logging
from datetime import datetime
from typing import Optional, List, Dict, Any

from fastapi import HTTPException
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector
from langchain.schema.document import Document
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import EmbeddingsFilter

from app.models import (
    DocumentMatch, 
    QueryRequest, 
    QueryResponse, 
    DocumentInput,
    HealthResponse
)

# Setup logging
logger = logging.getLogger(__name__)

# Global components
embeddings_model: Optional[OpenAIEmbeddings] = None
vector_store: Optional[PGVector] = None

# Get environment variables
CONNECTION_STRING = os.getenv('POSTGRES_CONNECTION_STRING')
COLLECTION_NAME = os.getenv('COLLECTION_NAME', 'document_collection')

def initialize_embeddings_model():
    """Initialize the OpenAI embeddings model"""
    global embeddings_model
    
    if not os.getenv('OPENAI_API_KEY'):
        raise ValueError("OPENAI_API_KEY environment variable is required")
    
    logger.info("Initializing OpenAI embeddings model...")
    try:
        embeddings_model = OpenAIEmbeddings(
            model="text-embedding-3-small",
            dimensions=1536
        )
        
        logger.info("Embeddings model initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize embeddings model: {str(e)}")
        raise

def initialize_vector_store():
    """Initialize the PGVector store"""
    global vector_store, embeddings_model
    
    if embeddings_model is None:
        raise ValueError("Embeddings model must be initialized first")
    
    if not CONNECTION_STRING:
        raise ValueError("POSTGRES_CONNECTION_STRING environment variable is required")
    
    try:
        logger.info(f"Initializing PGVector with collection name: {COLLECTION_NAME}")
        
        # Check the format of CONNECTION_STRING
        # Ensure it's in the right format: postgresql+psycopg://user:pass@host:port/dbname
        # If it starts with postgresql:// we need to convert it
        conn_string = CONNECTION_STRING
        if conn_string.startswith('postgresql://'):
            # Replace postgresql:// with postgresql+psycopg://
            conn_string = conn_string.replace('postgresql://', 'postgresql+psycopg://')
            logger.info(f"Converted connection string format for compatibility")
            
        # Updated PGVector initialization based on newer API
        vector_store = PGVector(
            connection_string=conn_string,
            embedding_function=embeddings_model,
            collection_name=COLLECTION_NAME,
            use_jsonb=True  # Use JSONB for metadata storage
        )
        
        logger.info("Vector store initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize vector store: {str(e)}")
        raise

def check_db_connection():
    """Check if the database connection is active"""
    global vector_store
    try:
        if vector_store is not None:
            # A simple check that the store exists
            return True
        return False
    except Exception:
        return False

async def get_health_status() -> HealthResponse:
    """Get the health status of the application"""
    db_healthy = check_db_connection()
    
    return HealthResponse(
        status="healthy" if db_healthy and embeddings_model else "unhealthy",
        database="connected" if db_healthy else "disconnected",
        embeddings_model="available" if embeddings_model else "unavailable",
        timestamp=datetime.utcnow().isoformat()
    )

async def match_documents_in_db(request: QueryRequest) -> QueryResponse:
    """Match documents in the database using LangChain's retriever"""
    global vector_store
    
    if not vector_store or not embeddings_model:
        raise HTTPException(status_code=503, detail="Service not fully initialized")
    
    start_time = datetime.utcnow()
    
    try:
        # Log query parameters
        logger.info(f"Query: '{request.query}'")
        logger.info(f"Metadata filter: {request.metadata_filter}")
        logger.info(f"Min confidence: {request.min_confidence}")
        logger.info(f"Max results: {request.max_results}")
        
        # Set up the search parameters
        search_kwargs = {
            "k": request.max_results * 2,  # Get more results than needed and filter later
            "fetch_k": request.max_results * 4,  # Consider more candidates
            "lambda_mult": 0.5  # Balance between relevance and diversity
        }
        
        # If we have metadata filters, add them to the search parameters
        if request.metadata_filter:
            # Format the filter dict to match LangChain's expectations
            formatted_filter = {}
            for key, value in request.metadata_filter.items():
                if value is not None:
                    formatted_filter[key] = value
            
            if formatted_filter:
                search_kwargs["filter"] = formatted_filter
        
        # Create a retriever with the specified parameters
        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs=search_kwargs
        )
        
        # Add a relevance filter to ensure minimum similarity
        if request.min_confidence > 0:
            embeddings_filter = EmbeddingsFilter(
                embeddings=embeddings_model,
                similarity_threshold=request.min_confidence
            )
            retriever = ContextualCompressionRetriever(
                base_retriever=retriever,
                doc_compressor=embeddings_filter
            )
        
        # Using run_in_executor because LangChain's retriever is not async
        docs = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: retriever.get_relevant_documents(request.query)
        )
        
        logger.info(f"Found {len(docs)} matching documents")
        
        # Convert LangChain documents to our DocumentMatch format
        matches = []
        suggested_folders = set()
        
        for doc in docs:
            # Skip if we don't have the necessary info
            if not hasattr(doc, 'metadata') or not hasattr(doc, 'page_content'):
                continue
            
            metadata = doc.metadata or {}
            
            # Determine document ID from metadata
            document_id = metadata.get('document_id', metadata.get('custom_id', ''))
            if not document_id:
                # Use UUID if available
                document_id = metadata.get('uuid', f"doc_{len(matches)}")
            
            # Extract folder information if available
            folder = metadata.get('folder')
            if folder:
                suggested_folders.add(folder)
            
            # Extract or calculate confidence score
            # LangChain 0.603+ provides relevance_score in metadata
            confidence = metadata.get('relevance_score', 0.95)
            
            # Skip if below confidence threshold
            if confidence < request.min_confidence:
                continue
            
            # Create a snippet from the content
            content = doc.page_content
            snippet = content[:200] + "..." if len(content) > 200 else content
            
            matches.append(DocumentMatch(
                document_id=str(document_id),
                content_snippet=snippet,
                confidence=round(float(confidence), 4),
                metadata=metadata
            ))
        
        # Sort by confidence and limit to max_results
        matches = sorted(matches, key=lambda x: x.confidence, reverse=True)[:request.max_results]
        
        # Calculate query time
        query_time_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        # Create suggested documents list
        suggested_docs = [match.document_id for match in matches]
        
        return QueryResponse(
            matches=matches,
            query_time_ms=round(query_time_ms, 2),
            total_candidates=len(docs),
            suggested_folders=list(suggested_folders),
            suggested_documents=suggested_docs
        )
        
    except Exception as e:
        logger.error(f"Error matching documents: {str(e)}")
        logger.exception("Full traceback:")
        raise HTTPException(status_code=500, detail=str(e))

async def add_document_to_db(document: DocumentInput) -> Dict[str, Any]:
    """Add a document to the database using LangChain's vector store"""
    global vector_store
    
    if not vector_store or not embeddings_model:
        raise HTTPException(status_code=503, detail="Service not fully initialized")
    
    try:
        # Create a LangChain Document from our input
        doc = Document(
            page_content=document.content,
            metadata=document.metadata.copy()
        )
        
        # Add custom_id to metadata if provided
        if document.custom_id:
            doc.metadata['custom_id'] = document.custom_id
        
        # Add the document to the vector store (run in executor since it's synchronous)
        doc_ids = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: vector_store.add_documents([doc])
        )
        
        # Get the assigned ID (should be UUID format)
        doc_id = doc_ids[0] if doc_ids else None
        
        return {
            "status": "success", 
            "document_id": document.custom_id or doc_id,
            "uuid": doc_id
        }
        
    except Exception as e:
        logger.error(f"Error adding document: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

async def delete_document_from_db(document_id: str, by_custom_id: bool = True) -> Dict[str, Any]:
    """Delete a document from the database"""
    global vector_store
    
    if not vector_store:
        raise HTTPException(status_code=503, detail="Vector store not initialized")
    
    try:
        # Convert to the filter format expected by LangChain
        filter_dict = {}
        if by_custom_id:
            filter_dict["custom_id"] = document_id
        else:
            filter_dict["id"] = document_id
        
        # Delete documents matching the filter
        await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: vector_store.delete(filter=filter_dict)
        )
        
        return {"status": "success", "document_id": document_id}
        
    except Exception as e:
        logger.error(f"Error deleting document: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
