from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime

class DocumentMatch(BaseModel):
    document_id: str
    document_name: str  # Add this field
    folder: str  # Add this field
    last_updated: str  # Add this field
    confidence: float
    # You can keep or remove content_snippet depending on whether you still want it
    content_snippet: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

class QueryRequest(BaseModel):
    query: str
    max_results: int = 5
    min_confidence: float = 0.01
    metadata_filter: Optional[Dict[str, Any]] = None

class QueryResponse(BaseModel):
    matches: List[DocumentMatch]
    query_time_ms: float
    total_candidates: int
    suggested_documents: List[str] = Field(default_factory=list)
    suggested_folders: List[str] = Field(default_factory=list)

class DocumentInput(BaseModel):
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    custom_id: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    database: str
    embeddings_model: str
    timestamp: str
