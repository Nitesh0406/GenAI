"""
models.py - Pydantic models for request/response
"""
from pydantic import BaseModel
from typing import Optional

class QueryRequest(BaseModel):
    question: str
    thread_id: Optional[str] = None
    user_id: Optional[str] = None

class QueryResponse(BaseModel):
    status: str
    messages: list
    visual_outputs: Optional[list] = []
    follow_up: Optional[list] = []

