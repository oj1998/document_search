import os
import asyncio
import json
import uuid
import logging
from datetime import datetime
from typing import Optional, List, Dict, Any

import asyncpg
from fastapi import HTTPException
from langchain_openai import OpenAIEmbeddings

from app.models import (
    DocumentMatch, 
    QueryRequest, 
    QueryResponse, 
    DocumentInput,
    HealthResponse
)

# Setup logging
logger = logging.getLogger(__name__)

# Global connections
pool: Optional[asyncpg.Pool] = None
embeddings_model: Optional[OpenAIEmbeddings] = None

# Get environment variables
CONNECTION_STRING = os.getenv('POSTGRES_CONNECTION_STRING')
COLLECTION_NAME = os.getenv('COLLECTION_NAME', 'document_collection')

async def initialize_db_pool():
    """Initialize the database connection pool"""
    global pool
    
    if not CONNECTION_STRING:
        raise ValueError("POSTGRES_CONNECTION_STRING environment variable is required")
    
    logger.info("Initializing database pool...")
    pool = await asyncpg.create_pool(
        CONNECTION_STRING,
        min_size=3,
        max_size=10,
        statement_cache_size=0  # Add this line to disable prepared statements cache
    )
    logger.info("Database pool created successfully")
    
    # Verify database has pgvector extension
    async with pool.acquire() as conn:
        logger.info("Verifying pgvector extension...")
        await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        
        # Verify table exists
        table_exists = await conn.fetchval(
            "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = $1)",
            "langchain_pg_embedding"
        )
        
        if not table_exists:
            logger.error("Required table 'langchain_pg_embedding' does not exist")
            raise Exception("Required table does not exist")
            
        logger.info("Database setup verified")

def initialize_embeddings_model():
    """Initialize the OpenAI embeddings model"""
    global embeddings_model
    
    if not os.getenv('OPENAI_API_KEY'):
        raise ValueError("OPENAI_API_KEY environment variable is required")
    
    logger.info("Initializing OpenAI embeddings model...")
    embeddings_model = OpenAIEmbeddings()
    logger.info("Embeddings model initialized successfully")

async def close_db_pool():
    """Close the database connection pool"""
    global pool
    if pool:
        await pool.close()
        logger.info("Database pool closed")

async def get_health_status() -> HealthResponse:
    """Get the health status of the application"""
    try:
        # Check database connection
        db_healthy = False
        if pool:
            try:
                async with pool.acquire() as conn:
                    await conn.execute('SELECT 1')
                    db_healthy = True
            except Exception as e:
                logger.warning(f"Database health check issue: {e}")
        
        return HealthResponse(
            status="healthy" if db_healthy and embeddings_model else "unhealthy",
            database="connected" if db_healthy else "disconnected",
            embeddings_model="available" if embeddings_model else "unavailable",
            timestamp=datetime.utcnow().isoformat()
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="unhealthy",
            database="error",
            embeddings_model="error",
            timestamp=datetime.utcnow().isoformat()
        )

async def match_documents_in_db(request: QueryRequest) -> QueryResponse:
    """Match documents in the database"""
    if not pool or not embeddings_model:
        raise HTTPException(status_code=503, detail="Service not fully initialized")
    
    start_time = datetime.utcnow()
    
    try:
        # Generate embedding for the query
        query_embedding = await asyncio.get_event_loop().run_in_executor(
            None, 
            lambda: embeddings_model.embed_query(request.query)
        )
        
        # Prepare the SQL query with pgvector's cosine distance
        # Note: Using explicit column names to avoid confusion
        sql = """
        SELECT 
            uuid, 
            custom_id,
            document, 
            cmetadata, 
            1 - (embedding <=> $1::vector) as similarity
        FROM 
            langchain_pg_embedding
        """
        
        # Add metadata filter if provided
        params = [query_embedding]
        
        if request.metadata_filter:
            conditions = []
            for i, (key, value) in enumerate(request.metadata_filter.items(), 2):
                conditions.append(f"cmetadata->>{key} = ${i}")
                params.append(str(value))
            
            if conditions:
                sql += " WHERE " + " AND ".join(conditions)
        
        # Add order by and limit
        sql += """
        ORDER BY 
            embedding <=> $1::vector
        LIMIT $2
        """
        params.append(request.max_results * 2)  # Get more for filtering
        
        # Execute the query
        async with pool.acquire() as conn:
            # Register vector type with asyncpg if necessary
            # This tells asyncpg how to handle the vector data type
            try:
                await conn.execute("SELECT NULL::vector")
            except asyncpg.exceptions.UndefinedObjectError:
                # Vector type not registered, register it now
                await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            
            rows = await conn.fetch(sql, params[0], params[-1])
        
        # Process results
        matches = []
        for row in rows:
            # Convert row to dict - using actual column names from the query
            document_id = row['custom_id'] or str(row['uuid'])
            metadata = row['cmetadata'] if row['cmetadata'] else {}
            
            # Calculate confidence (similarity score)
            confidence = float(row['similarity'])
            
            # Skip if below threshold
            if confidence < request.min_confidence:
                continue
                
            # Get content snippet - using the actual column name 'document' instead of 'content'
            content = row['document']
            snippet = content[:200] + "..." if len(content) > 200 else content
            
            matches.append(DocumentMatch(
                document_id=document_id,
                content_snippet=snippet,
                confidence=round(confidence, 4),
                metadata=metadata
            ))
        
        # Sort by confidence and limit to max_results
        matches = sorted(matches, key=lambda x: x.confidence, reverse=True)[:request.max_results]
        
        # Calculate query time
        query_time_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        return QueryResponse(
            matches=matches,
            query_time_ms=round(query_time_ms, 2),
            total_candidates=len(rows)
        )
        
    except Exception as e:
        logger.error(f"Error matching documents: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

async def add_document_to_db(document: DocumentInput) -> Dict[str, Any]:
    """Add a document to the database"""
    if not pool or not embeddings_model:
        raise HTTPException(status_code=503, detail="Service not fully initialized")
    
    try:
        # Generate embedding for the document
        embedding = await asyncio.get_event_loop().run_in_executor(
            None, 
            lambda: embeddings_model.embed_documents([document.content])[0]
        )
        
        # Generate UUID if custom_id not provided
        doc_uuid = str(uuid.uuid4())
        custom_id = document.custom_id or None
        collection_id = str(uuid.uuid4())  # Generate a unique collection_id
        
        # Insert into database
        async with pool.acquire() as conn:
            await conn.execute("""
            INSERT INTO langchain_pg_embedding 
                (uuid, collection_id, embedding, document, cmetadata, custom_id)
            VALUES 
                ($1, $2, $3, $4, $5, $6)
            """, 
                doc_uuid, 
                collection_id,  # Each document gets its own collection_id
                embedding,
                document.content,
                json.dumps(document.metadata),
                custom_id
            )
        
        return {
            "status": "success", 
            "document_id": custom_id or doc_uuid,
            "uuid": doc_uuid
        }
        
    except Exception as e:
        logger.error(f"Error adding document: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

async def delete_document_from_db(document_id: str, by_custom_id: bool = True) -> Dict[str, Any]:
    """Delete a document from the database"""
    if not pool:
        raise HTTPException(status_code=503, detail="Database not initialized")
    
    try:
        async with pool.acquire() as conn:
            if by_custom_id:
                result = await conn.execute("""
                DELETE FROM langchain_pg_embedding 
                WHERE custom_id = $1
                """, document_id)
            else:
                # Assume it's a UUID
                result = await conn.execute("""
                DELETE FROM langchain_pg_embedding 
                WHERE uuid = $1
                """, document_id)
        
        # Parse the DELETE result (e.g., "DELETE 1")
        rows_deleted = int(result.split()[1]) if result else 0
        
        if rows_deleted == 0:
            return {"status": "not_found", "document_id": document_id}
        
        return {"status": "success", "document_id": document_id}
        
    except Exception as e:
        logger.error(f"Error deleting document: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
