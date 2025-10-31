import uuid

from datetime import datetime, timezone
from typing import List, Dict

from pydantic import BaseModel, Field


def now_ymdhm() -> str:
    return datetime.now().strftime("%Y%m%d%H%M")


class MemoryNote(BaseModel):
    """
    A memory note that represents a single unit of information
        in the memory system.
    Absorbs any unknown fields into the `extras` dictionary, intended
        for arbitrary keyword-value metadata that can be used for filtering
        searches later. 
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
    memory_update_history: List[str] = Field(
        default_factory=list, 
        description="History of updates made to this memory")
    memory_update_context: List[str] = Field(
        default_factory=list, 
        description="Contextual notes about updates made")
    extras: Dict[str, str] = Field(
        default_factory=dict,
        description="Arbitrary key-value pairs for additional metadata")

    def __init__(self, **data):
        """Custom init to automatically absorb unknown kwargs into extras."""
    
        # Class-level constant for known fields
        known_fields = list(self.__class__.model_fields.keys())
        known_fields.remove('extras')
        
        known_data = {}
        extras = data.pop('extras', {}).copy()
        
        for key, value in list(data.items()):
            if key in known_fields:
                known_data[key] = value
            else:
                # Move unknown fields to extras
                extras[key] = str(value)
        
        known_data['extras'] = extras
        super().__init__(**known_data)
