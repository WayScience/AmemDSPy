import pytest
import json
from datetime import datetime, timezone

from agentic_memory.memory_note import MemoryNote
from agentic_memory.memory_system import AgenticMemorySystem
from agentic_memory.retrievers import ChromaRetriever


class TestMemoryNote:
    """Test suite for MemoryNote class."""
    
    def test_memory_note_creation_with_defaults(self):
        """Test creating a MemoryNote with minimal parameters."""
        note = MemoryNote(content="Test content")
        
        assert note.content == "Test content"
        assert note.id is not None
        assert isinstance(note.id, str)
        assert note.keywords == []
        assert note.retrieval_count == 0
        assert note.context == "General"
        assert note.category == "Uncategorized"
        assert note.tags == []
        assert note.timestamp is not None
        assert note.last_accessed is not None
    
    def test_memory_note_creation_with_all_fields(self):
        """Test creating a MemoryNote with all parameters."""
        custom_timestamp = datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)
        custom_last_accessed = datetime(2024, 1, 1, 1, 0, tzinfo=timezone.utc)
        
        note = MemoryNote(
            content="Full content",
            id="custom-id",
            keywords=["key1", "key2"],
            retrieval_count=5,
            timestamp=custom_timestamp,
            last_accessed=custom_last_accessed,
            context="Custom Context",
            category="Custom Category",
            tags=["tag1", "tag2"]
        )
        
        assert note.content == "Full content"
        assert note.id == "custom-id"
        assert note.keywords == ["key1", "key2"]
        assert note.retrieval_count == 5
        assert note.timestamp == custom_timestamp
        assert note.last_accessed == custom_last_accessed
        assert note.context == "Custom Context"
        assert note.category == "Custom Category"
        assert note.tags == ["tag1", "tag2"]


class TestAgenticMemorySystemInit:
    """Test suite for AgenticMemorySystem initialization."""
    
    def test_init_with_default_retriever(self, retriever):
        """Test initialization with default retriever."""
        system = AgenticMemorySystem(retriever=retriever)
        
        assert system.memories == {}
        assert isinstance(system.retriever, ChromaRetriever)
        assert system._llm_processor is None
    
    def test_init_with_invalid_retriever(self):
        """Test that invalid retriever raises TypeError."""
        with pytest.raises(TypeError):
            AgenticMemorySystem(retriever="invalid")


class TestAgenticMemorySystemCRUD:
    """Test suite for CRUD operations."""
    
    def test_add_note_minimal(self, memory_system):
        """Test adding a note with minimal parameters."""
        note_id = memory_system.add_note(content="Test content")
        
        assert note_id is not None
        assert note_id in memory_system.memories
        assert memory_system.memories[note_id].content == "Test content"
    
    def test_add_note_with_metadata(self, memory_system):
        """Test adding a note with full metadata."""
        note_id = memory_system.add_note(
            content="Rich content",
            keywords=["key1", "key2"],
            context="Test Context",
            category="Test Category",
            tags=["tag1", "tag2"]
        )
        
        note = memory_system.memories[note_id]
        assert note.content == "Rich content"
        assert note.keywords == ["key1", "key2"]
        assert note.context == "Test Context"
        assert note.category == "Test Category"
        assert note.tags == ["tag1", "tag2"]
    
    def test_add_note_custom_timestamp(self, memory_system):
        """Test adding a note with custom timestamp."""
        custom_time = datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)
        note_id = memory_system.add_note(
            content="Timestamped content",
            timestamp=custom_time
        )
        
        note = memory_system.memories[note_id]
        assert note.timestamp == custom_time
    
    def test_update_existing_note(self, memory_system):
        """Test updating an existing note."""
        note_id = memory_system.add_note(
            content="Original content",
            keywords=["old"],
            tags=["old_tag"]
        )
        
        success = memory_system.update(
            note_id,
            content="Updated content",
            keywords=["new"],
            tags=["new_tag"]
        )
        
        assert success is True
        note = memory_system.memories[note_id]
        assert note.content == "Updated content"
        assert note.keywords == ["new"]
        assert note.tags == ["new_tag"]
    
    def test_update_nonexistent_note(self, memory_system):
        """Test updating a non-existent note returns False."""
        success = memory_system.update(
            "nonexistent-id",
            content="New content"
        )
        assert success is False


class TestAgenticMemorySystemSearch:
    """Test suite for search functionality."""
    
    def test_search_empty_system(self, memory_system):
        """Test searching in an empty system."""
        results = memory_system.search("test query")
        assert results == []
    
    def test_search_returns_results(self, populated_memory_system):
        """Test that search returns relevant results."""
        results = populated_memory_system.search("python programming")
        
        assert len(results) > 0
        assert isinstance(results, list)
        assert all(isinstance(r, dict) for r in results)
    
    def test_search_result_structure(self, populated_memory_system):
        """Test that search results have correct structure."""
        results = populated_memory_system.search("machine learning", k=1)
        
        assert len(results) <= 1
        if results:
            result = results[0]
            assert "id" in result
            assert "content" in result
    
    def test_search_respects_k_parameter(self, populated_memory_system):
        """Test that search respects the k parameter."""
        results = populated_memory_system.search("technology", k=2)
        assert len(results) <= 2


class TestAgenticMemorySystemHelpers:
    """Test suite for helper methods."""
    
    def test_serialize_metadata(self, memory_system, sample_memory_note):
        """Test metadata serialization from MemoryNote."""
        metadata = memory_system._serialize_metadata(sample_memory_note)
        ref_metadata = sample_memory_note.model_dump()

        assert isinstance(metadata, dict)

        for key, ref_value in ref_metadata.items():
            assert key in metadata
            if ref_value is None:
                assert metadata[key] is None
            elif isinstance(ref_value, datetime):
                # Datetime objects should be serialized to ISO format strings
                assert metadata[key] == ref_value.isoformat()
                assert isinstance(metadata[key], str)
            elif isinstance(ref_value, (dict, list)):
                assert metadata[key] == json.dumps(ref_value)
                assert json.loads(metadata[key]) == ref_value
            else:
                assert metadata[key] == ref_value
