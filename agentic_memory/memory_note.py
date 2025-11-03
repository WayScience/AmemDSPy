import uuid
from pydantic import BaseModel, Field
from datetime import datetime, timezone
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
    # TODO: Placeholder for future LM backed optimization of memory
    # currently the LM optimization is removed. 
    keywords: List[str] = Field(
        default_factory=list, 
        description="Key terms extracted from the content")
    retrieval_count: int = Field(
        0, description="Number of times this memory has been accessed")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Creation timestamp (UTC)")
    last_accessed: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Last access timestamp (UTC)")
    # TODO: Another placeholder for future context-aware memory management
    context: str = Field(
        "General", description="The broader context or domain of the memory")
    category: str = Field(
        "Uncategorized", description="Classification category")
    tags: List[str] = Field(default_factory=list, description="Additional info")
