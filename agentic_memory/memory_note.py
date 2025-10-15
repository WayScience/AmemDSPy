import uuid
from pydantic import BaseModel, Field
from datetime import datetime
from typing import List

def now_ymdhm() -> str:
    return datetime.now().strftime("%Y%m%d%H%M")


class MemoryNote(BaseModel):
    """
    A memory note that represents a single unit of information
    in the memory system.
    """

    content: str = Field(
        ..., description="The main text content of the memory")
    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()), 
        description="Unique identifier for the memory")
    keywords: List[str] = Field(
        default_factory=list, 
        description="Key terms extracted from the content")
    retrieval_count: int = Field(
        0, description="Number of times this memory has been accessed")
    timestamp: str = Field(
        default_factory=now_ymdhm, 
        description="Creation time in format YYYYMMDDHHMM")
    last_accessed: str = Field(
        default_factory=now_ymdhm, 
        description="Last access time in format YYYYMMDDHHMM")
    context: str = Field(
        "General", description="The broader context or domain of the memory")
    category: str = Field(
        "Uncategorized", description="Classification category")
    tags: List[str] = Field(default_factory=list, description="Additional info")
