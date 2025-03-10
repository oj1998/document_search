import os
import asyncio
import json
import uuid
import logging
from datetime import datetime
from typing import Optional, List, Dict, Any

import numpy as np
import asyncpg
from asyncpg import Connection
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

def validate_embedding(embedding):
    """Validate that an embedding is properly formatted for pgvector"""
    if not embedding:
        return False
        
    if not isinstance(embedding, list):
        return False
        
    if len(embedding) == 0:
        return False
        
    # Check that all elements are numbers
    if not all(isinstance(x, (int, float)) for x in embedding):
        return False
        
    return True

async def register_vector_type(conn: Connection):
    """Register the vector type with asyncpg"""
    await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    
    # Define the vector type
    # This tells asyncpg how to convert between Python and PostgreSQL
    try:
        await conn.set_type_codec(
            'vector',
            encoder=lambda vector: np.array(vector, dtype=np.float32).tobytes(),
            decoder=lambda data: np.frombuffer(data, dtype=np.float32).tolist(),
            format='binary'
        )
        logger.debug("Vector type codec registered successfully")
    except Exception as e:
        logger.error(f"Error registering vector type codec: {str(e)}")
        raise

async def prepare_database():
    """Prepare the database by registering custom types and creating tables if needed"""
    global pool
    
    if not pool:
        raise ValueError("Database pool must be initialized first")
    
    async with pool.acquire() as conn:
        # Register the vector type with asyncpg
        await register_vector_type(conn)

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
    
    # Register vector type with the pool
    await prepare_database()
    logger.info("Vector type registered with database")

def initialize_embeddings_model():
    """Initialize the OpenAI embeddings model"""
    global embeddings_model
    
    if not os.getenv('OPENAI_API_KEY'):
        raise ValueError("OPENAI_API_KEY environment variable is required")
    
    logger.info("Initializing OpenAI embeddings model...")
    try:
        # Specify dimensions to ensure compatibility with pgvector
        # text-embedding-3-small has 1536 dimensions
        embeddings_model = OpenAIEmbeddings(
            model="text-embedding-3-small",  # This model has 1536 dimensions, well below pgvector's limit
            dimensions=1536  # Explicitly set dimensions
        )
        
        # Test the embeddings model
        test_embedding = embeddings_model.embed_query("This is a test query")
        logger.info(f"Test embedding generated successfully with {len(test_embedding)} dimensions")
        
        logger.info(f"Embeddings model initialized successfully with dimensions=1536")
    except Exception as e:
        logger.error(f"Failed to initialize embeddings model: {str(e)}")
        raise

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
    """Match documents in the database using direct SQL"""
    if not pool or not embeddings_model:
        raise HTTPException(status_code=503, detail="Service not fully initialized")
    
    start_time = datetime.utcnow()
    
    try:
        # Generate embedding for the query
        query_embedding = await asyncio.get_event_loop().run_in_executor(
            None, 
            lambda: embeddings_model.embed_query(request.query)
        )
        
        # Validate the embedding
        if not validate_embedding(query_embedding):
            logger.error(f"Invalid embedding generated for query: {request.query}")
            raise HTTPException(status_code=500, detail="Failed to generate valid embedding for query")

        logger.info(f"Generated valid embedding with {len(query_embedding)} dimensions")
        
        # Convert embedding to a string representation for direct SQL
        embedding_str = "[" + ",".join(str(x) for x in query_embedding) + "]"
        
        # Construct a direct SQL query string
        sql = f"""
        SELECT 
            uuid, 
            custom_id,
            document, 
            cmetadata, 
            1 - (embedding <=> '{embedding_str}'::vector) as similarity
        FROM 
            langchain_pg_embedding
        """
        
        # Add metadata filter if provided
        where_clauses = []
        if request.metadata_filter:
            for key, value in request.metadata_filter.items():
                # Escape the value to prevent SQL injection
                escaped_value = str(value).replace("'", "''")
                where_clauses.append(f"cmetadata->>'{ key }' = '{ escaped_value }'")
        
        if where_clauses:
            sql += " WHERE " + " AND ".join(where_clauses)
        
        # Add order by and limit
        sql += f"""
        ORDER BY 
            embedding <=> '{embedding_str}'::vector
        LIMIT {request.max_results * 2}
        """
        
        # Execute the query as direct SQL
        async with pool.acquire() as conn:
            logger.info(f"Executing direct SQL query")
            rows = await conn.fetch(sql)
            logger.info(f"Query returned {len(rows)} rows")
        
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

logger.info(f"Query: '{request.query}'")
logger.info(f"Metadata filter: {request.metadata_filter}")
logger.info(f"Min confidence: {request.min_confidence}")
logger.info(f"Max results: {request.max_results}")

# After executing the SQL query
logger.info(f"Raw SQL query: {sql}")
logger.info(f"Query returned {len(rows)} rows before confidence filtering")

# After filtering by confidence
logger.info(f"Documents filtered out by confidence threshold: {len(rows) - len(matches)}")
logger.info(f"Final matches returned: {len(matches)}")

# Debug info for the first few raw results if available
if rows and len(rows) > 0:
    for i, row in enumerate(rows[:3]):  # Log first 3 rows
        logger.info(f"Row {i}: id={row['custom_id'] or row['uuid']}, similarity={row['similarity']}")

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
        
        # Validate the embedding
        if not validate_embedding(embedding):
            logger.error(f"Invalid embedding generated for document")
            raise HTTPException(status_code=500, detail="Failed to generate valid embedding for document")
        
        # Generate UUID if custom_id not provided
        doc_uuid = str(uuid.uuid4())
        custom_id = document.custom_id or None
        collection_id = str(uuid.uuid4())  # Generate a unique collection_id
        
        # Insert into database
        async with pool.acquire() as conn:
            # Make sure vector type is registered with this connection
            await register_vector_type(conn)
            
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

# Add this function to database.py
async def find_document_by_content(content_snippet: str, limit: int = 5):
    """Find documents containing specific text content"""
    if not pool:
        raise ValueError("Database pool not initialized")
    
    try:
        async with pool.acquire() as conn:
            # Search for documents containing the content snippet
            rows = await conn.fetch("""
            SELECT uuid, custom_id, document, cmetadata
            FROM langchain_pg_embedding
            WHERE document ILIKE $1
            LIMIT $2
            """, f"%{content_snippet}%", limit)
            
            results = []
            for row in rows:
                doc_id = row['custom_id'] or str(row['uuid'])
                content = row['document']
                snippet = content[:100] + "..." if len(content) > 100 else content
                
                results.append({
                    "document_id": doc_id,
                    "content_snippet": snippet,
                    "metadata": row['cmetadata'] if row['cmetadata'] else {}
                })
            
            return {
                "found": len(results) > 0,
                "count": len(results),
                "documents": results
            }
    except Exception as e:
        logger.error(f"Error finding document by content: {str(e)}")
        raise
