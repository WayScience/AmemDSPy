import uuid

from datetime import datetime, timezone
from typing import List, Dict, Any

from pydantic import BaseModel, Field

from .field_handlers import memory_note_field_registry


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

        # do not use specified history and context tracking fields
        _ = data.pop('memory_update_history', None)
        _ = data.pop('memory_update_context', None)

        # exclude the extra from known fields
        known_fields.remove('extras')
        
        # Sort data into known fields and extras
        known_data = {}        
        extras = data.pop('extras', {}).copy()
        for key, value in list(data.items()):
            if key in known_fields:
                known_data[key] = value
            else:
                # Move unknown fields to extras
                extras[key] = str(value)
        
        # Assign extras back to known_data
        known_data['extras'] = extras
        # Call the BaseModel init with known data
        super().__init__(**known_data)

        # Automatically initialize memory_update_history and memory_update_context
        self.memory_update_history.append(self.content)
        self.memory_update_context.append('Initial memory creation.')
    
    def serialize_for_storage(self) -> Dict[str, Any]:
        """
        Serialize the MemoryNote for storage using the field registry.
        
        :return: Dictionary with serialized field values
        """
        data = self.model_dump()
        return memory_note_field_registry.serialize_all(data)
    
    @classmethod
    def deserialize_from_storage(cls, data: Dict[str, Any]) -> 'MemoryNote':
        """
        Deserialize a MemoryNote from storage using the field registry.
        
        :param data: Dictionary with serialized field values
        :return: MemoryNote instance
        """
        deserialized_data = memory_note_field_registry.deserialize_all(data)
        return cls(**deserialized_data)
    
    def update(
        self, 
        content: str, 
        update_context: str = None,
        **kwargs
    ) -> None:
        """
        Update the memory content and track the change in history.
        
        :param content: The new content to update the memory with
        :param update_context: Optional context or reason for the update.
            If not provided, defaults to 'Memory update.'
        :return: None
        """        
        # Append update context
        context_msg = update_context \
            if update_context is not None else 'Memory update.'
        self.memory_update_context.append(context_msg)
        
        # Update the content and append to history
        self.content = content
        self.memory_update_history.append(self.content)
        
        # Update the last_accessed timestamp
        self.last_accessed = datetime.now(timezone.utc)

        # Update any additional fields provided in kwargs
        # prevent updating last_accessed from kwargs
        _ = kwargs.pop('last_accessed', None)
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
