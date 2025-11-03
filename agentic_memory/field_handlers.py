"""
field_handlers.py

Field handlers for MemoryNote fields.
Provides centralized validation, serialization, and deserialization logic.
"""

import json
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, List, Dict, Optional, Callable


class FieldHandler(ABC):
    """Base class for field handlers."""
    
    def __init__(self, validator: Optional[Callable[[Any], bool]] = None):
        """
        Initialize field handler.
        
        :param validator: Optional custom validation function that takes a value
                         and returns True if valid, False otherwise
        """
        self.validator = validator
    
    def validate(self, value: Any) -> bool:
        """
        Validate a field value.
        
        :param value: Value to validate
        :return: True if valid, False otherwise
        """
        if self.validator is not None:
            return self.validator(value)
        return self._default_validate(value)
    
    @abstractmethod
    def _default_validate(self, value: Any) -> bool:
        """Default validation logic. Override in subclasses."""
        pass
    
    @abstractmethod
    def serialize(self, value: Any) -> Any:
        """
        Serialize value for storage in ChromaDB.
        
        :param value: Value to serialize
        :return: Serialized value
        """
        pass
    
    @abstractmethod
    def deserialize(self, value: Any) -> Any:
        """
        Deserialize value from ChromaDB storage.
        
        :param value: Serialized value
        :return: Deserialized value
        """
        pass


class StringFieldHandler(FieldHandler):
    """Handler for string fields - no-op serialization."""
    
    def _default_validate(self, value: Any) -> bool:
        return isinstance(value, str)
    
    def serialize(self, value: str) -> str:
        """No-op: strings are stored as-is."""
        return str(value)
    
    def deserialize(self, value: str) -> str:
        """No-op: strings are retrieved as-is."""
        return str(value)


class IntFieldHandler(FieldHandler):
    """Handler for integer fields."""
    
    def _default_validate(self, value: Any) -> bool:
        return isinstance(value, int)
    
    def serialize(self, value: int) -> str:
        """Convert int to string for storage."""
        return str(value)
    
    def deserialize(self, value: Any) -> int:
        """Convert string back to int."""
        if isinstance(value, int):
            return value
        return int(value)


class ListFieldHandler(FieldHandler):
    """Handler for list fields (typically List[str])."""
    
    def _default_validate(self, value: Any) -> bool:
        return isinstance(value, list)
    
    def serialize(self, value: List) -> str:
        """Serialize list to JSON string."""
        return json.dumps(value)
    
    def deserialize(self, value: Any) -> List:
        """Deserialize JSON string back to list."""
        if isinstance(value, list):
            return value
        if isinstance(value, str):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return []
        return []


class DateTimeFieldHandler(FieldHandler):
    """Handler for datetime fields."""
    
    def _default_validate(self, value: Any) -> bool:
        return isinstance(value, datetime)
    
    def serialize(self, value: datetime) -> str:
        """Serialize datetime to ISO format string."""
        if isinstance(value, datetime):
            return value.isoformat()
        return str(value)
    
    def deserialize(self, value: Any) -> datetime:
        """Deserialize ISO format string back to datetime."""
        if isinstance(value, datetime):
            return value
        if isinstance(value, str):
            try:
                return datetime.fromisoformat(value)
            except ValueError:
                # Fallback to current time if parsing fails
                from datetime import timezone
                return datetime.now(timezone.utc)
        return value


class DictFieldHandler(FieldHandler):
    """Handler for dictionary fields (typically Dict[str, str])."""
    
    def _default_validate(self, value: Any) -> bool:
        return isinstance(value, dict)
    
    def serialize(self, value: Dict) -> str:
        """Serialize dict to JSON string."""
        return json.dumps(value)
    
    def deserialize(self, value: Any) -> Dict:
        """Deserialize JSON string back to dict."""
        if isinstance(value, dict):
            return value
        if isinstance(value, str):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return {}
        return {}


class FieldRegistry:
    """
    Registry that maps field names to their handlers.
    Provides centralized access to serialization, deserialization, and validation.
    """
    
    def __init__(self):
        """Initialize the field registry."""
        self._handlers: Dict[str, FieldHandler] = {}
    
    def register(
        self, 
        field_name: str, 
        handler: FieldHandler
    ) -> None:
        """
        Register a field handler.
        
        :param field_name: Name of the field
        :param handler: FieldHandler instance for this field
        """
        self._handlers[field_name] = handler
    
    def get_handler(self, field_name: str) -> Optional[FieldHandler]:
        """
        Get the handler for a field.
        
        :param field_name: Name of the field
        :return: FieldHandler instance or None if not registered
        """
        return self._handlers.get(field_name)
    
    def serialize(self, field_name: str, value: Any) -> Any:
        """
        Serialize a field value.
        
        :param field_name: Name of the field
        :param value: Value to serialize
        :return: Serialized value
        """
        handler = self.get_handler(field_name)
        if handler is None:
            # Default: convert to string
            return str(value)
        return handler.serialize(value)
    
    def deserialize(self, field_name: str, value: Any) -> Any:
        """
        Deserialize a field value.
        
        :param field_name: Name of the field
        :param value: Serialized value
        :return: Deserialized value
        """
        handler = self.get_handler(field_name)
        if handler is None:
            # Default: return as-is
            return value
        return handler.deserialize(value)
    
    def validate(self, field_name: str, value: Any) -> bool:
        """
        Validate a field value.
        
        :param field_name: Name of the field
        :param value: Value to validate
        :return: True if valid, False otherwise
        """
        handler = self.get_handler(field_name)
        if handler is None:
            # No handler = accept any value
            return True
        return handler.validate(value)
    
    def serialize_all(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Serialize all fields in a dictionary.
        
        :param data: Dictionary of field names to values
        :return: Dictionary with serialized values
        """
        return {
            field_name: self.serialize(field_name, value)
            for field_name, value in data.items()
        }
    
    def deserialize_all(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Deserialize all fields in a dictionary.
        
        :param data: Dictionary of field names to serialized values
        :return: Dictionary with deserialized values
        """
        return {
            field_name: self.deserialize(field_name, value)
            for field_name, value in data.items()
        }


# Create the global field registry for MemoryNote
memory_note_field_registry = FieldRegistry()

# Register handlers for all MemoryNote fields
memory_note_field_registry.register('content', StringFieldHandler())
memory_note_field_registry.register('id', StringFieldHandler())
memory_note_field_registry.register('keywords', ListFieldHandler())
memory_note_field_registry.register('retrieval_count', IntFieldHandler())
memory_note_field_registry.register('timestamp', DateTimeFieldHandler())
memory_note_field_registry.register('last_accessed', DateTimeFieldHandler())
memory_note_field_registry.register('context', StringFieldHandler())
memory_note_field_registry.register('category', StringFieldHandler())
memory_note_field_registry.register('tags', ListFieldHandler())
memory_note_field_registry.register('memory_update_history', ListFieldHandler())
memory_note_field_registry.register('memory_update_context', ListFieldHandler())
memory_note_field_registry.register('extras', DictFieldHandler())
