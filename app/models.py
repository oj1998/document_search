from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime

class DocumentMatch(BaseModel):
    document_id: str
    content_snippet: str
    confidence: float
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

class DocumentInput(BaseModel):
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    custom_id: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    database: str
    embeddings_model: str
    timestamp: str
