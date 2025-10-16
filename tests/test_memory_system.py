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
    
    def test_memory_note_with_extras(self):
        """Test that MemoryNote absorbs unknown kwargs into extras."""
        note = MemoryNote(
            content="Test content",
            custom_field1="value1",
            custom_field2="value2"
        )
        
        assert note.content == "Test content"
        assert note.extras == {"custom_field1": "value1", "custom_field2": "value2"}
    
    def test_memory_note_with_explicit_extras(self):
        """Test that explicit extras dict is preserved."""
        note = MemoryNote(
            content="Test content",
            extras={"existing": "value"}
        )
        
        assert note.extras == {"existing": "value"}
    
    def test_memory_note_merges_extras(self):
        """Test that unknown kwargs are merged with explicit extras."""
        note = MemoryNote(
            content="Test content",
            extras={"existing": "value"},
            new_field="new_value"
        )
        
        assert note.extras == {"existing": "value", "new_field": "new_value"}


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
    
    def test_add_note_with_extras(self, memory_system):
        """Test adding a note with extra fields."""
        note_id = memory_system.add_note(
            content="Test content",
            project="project_x",
            author="john_doe"
        )
        
        note = memory_system.memories[note_id]
        assert note.extras == {"project": "project_x", "author": "john_doe"}


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
    
    def test_search_with_keyword_filter(self, keyword_filterable_memory_system):
        """
        Test search with keyword filtering.
        Ensures note matching the filter appears in results
            before notes that do not match.
        """
        memory_system, ids = keyword_filterable_memory_system
        
        # Search with filter
        results = memory_system.search("python", project="project_a")
        
        assert len(results) >= 1

        # If project_b exists in results, it should come after project_a
        id1_index = next(
            (i for i, r in enumerate(results) \
             if r['id'] == ids['project_a']), None)
        id2_index = next(
            (i for i, r in enumerate(results) \
             if r['id'] == ids['project_b']), None)
        
        if id2_index is not None:
            assert id1_index is not None
            assert id1_index < id2_index
    
    def test_search_with_multiple_keyword_filters(self, keyword_filterable_memory_system):
        """Test search with multiple keyword filters."""
        memory_system, ids = keyword_filterable_memory_system
        
        # Filter by both project and author
        results = memory_system.search(
            "learning",
            project="ml_project",
            author="alice"
        )
        
        assert len(results) >= 1
        assert any(r['id'] == ids['ml_alice'] for r in results)
        
        # If ml_bob exists in results, it should come after ml_alice
        id1_index = next(
            (i for i, r in enumerate(results) \
             if r['id'] == ids['ml_alice']), None)
        id2_index = next(
            (i for i, r in enumerate(results) \
             if r['id'] == ids['ml_bob']), None)
        
        if id2_index is not None:
            assert id1_index is not None
            assert id1_index < id2_index


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


class TestAgenticMemorySystemPersistence:
    """Test suite for memory persistence and deserialization."""
    
    def test_load_existing_memories_on_init(self, retriever):
        """Test that existing memories are loaded on initialization."""
        # Create first system and add memories
        system1 = AgenticMemorySystem(retriever=retriever)
        note_id1 = system1.add_note(
            content="First memory",
            keywords=["test", "persistence"],
            tags=["tag1", "tag2"]
        )
        note_id2 = system1.add_note(
            content="Second memory",
            keywords=["test2"],
            tags=["tag3"]
        )
        
        # Create second system with same retriever - should load existing
        system2 = AgenticMemorySystem(retriever=retriever)
        
        # Verify memories were loaded
        assert len(system2.memories) == 2
        assert note_id1 in system2.memories
        assert note_id2 in system2.memories
        assert system2.memories[note_id1].content == "First memory"
        assert system2.memories[note_id2].content == "Second memory"
    
    def test_datetime_deserialization(self, retriever):
        """Test that datetime fields are properly deserialized."""
        # Add note with custom datetime
        custom_timestamp = datetime(2024, 6, 15, 10, 30, 0, tzinfo=timezone.utc)
        system1 = AgenticMemorySystem(retriever=retriever)
        note_id = system1.add_note(
            content="Timestamped memory",
            timestamp=custom_timestamp
        )
        
        # Create new system - load from retriever
        system2 = AgenticMemorySystem(retriever=retriever)
        loaded_note = system2.memories[note_id]
        
        # Verify datetime fields are datetime objects
        assert isinstance(loaded_note.timestamp, datetime)
        assert isinstance(loaded_note.last_accessed, datetime)
        
        # Verify timestamp value is preserved
        assert loaded_note.timestamp == custom_timestamp
    
    def test_list_field_deserialization(self, retriever):
        """Test that list fields (keywords, tags) are properly deserialized."""
        system1 = AgenticMemorySystem(retriever=retriever)
        keywords = ["machine learning", "AI", "neural networks"]
        tags = ["ml", "research", "deep-learning"]
        
        note_id = system1.add_note(
            content="ML research note",
            keywords=keywords,
            tags=tags
        )
        
        # Load from retriever
        system2 = AgenticMemorySystem(retriever=retriever)
        loaded_note = system2.memories[note_id]
        
        # Verify lists are properly deserialized
        assert isinstance(loaded_note.keywords, list)
        assert isinstance(loaded_note.tags, list)
        assert loaded_note.keywords == keywords
        assert loaded_note.tags == tags
    
    def test_all_fields_preserved(self, retriever):
        """Test that all MemoryNote fields are preserved through serialization."""
        system1 = AgenticMemorySystem(retriever=retriever)
        
        # Create note with all fields specified
        custom_timestamp = datetime(2024, 3, 1, 12, 0, 0, tzinfo=timezone.utc)
        note_id = system1.add_note(
            content="Complete memory note",
            keywords=["key1", "key2", "key3"],
            context="TestContext",
            category="TestCategory",
            tags=["tag1", "tag2"],
            timestamp=custom_timestamp
        )
        
        original_note = system1.memories[note_id]
        
        # Load from retriever
        system2 = AgenticMemorySystem(retriever=retriever)
        loaded_note = system2.memories[note_id]
        
        # Verify all fields match
        assert loaded_note.id == original_note.id
        assert loaded_note.content == original_note.content
        assert loaded_note.keywords == original_note.keywords
        assert loaded_note.context == original_note.context
        assert loaded_note.category == original_note.category
        assert loaded_note.tags == original_note.tags
        assert loaded_note.timestamp == original_note.timestamp
        assert loaded_note.retrieval_count == original_note.retrieval_count
        
        # Verify types
        assert isinstance(loaded_note, MemoryNote)
        assert isinstance(loaded_note.timestamp, datetime)
        assert isinstance(loaded_note.last_accessed, datetime)
        assert isinstance(loaded_note.keywords, list)
        assert isinstance(loaded_note.tags, list)
    
    def test_empty_collection_loads_gracefully(self, retriever):
        """Test that loading from empty collection doesn't crash."""
        # Create system with empty retriever
        system = AgenticMemorySystem(retriever=retriever)
        
        # Should have no memories
        assert len(system.memories) == 0
        assert system.memories == {}
    
    @pytest.mark.parametrize("notes_data,expected_count", [
        # Single note with minimal fields
        (
            [{"content": "Single note"}],
            1
        ),
        # Multiple notes with various field combinations
        (
            [
                {
                    "content": "Python programming",
                    "keywords": ["python", "code"],
                    "tags": ["programming"],
                    "category": "Tech"
                },
                {
                    "content": "Machine learning basics",
                    "keywords": ["ml", "ai", "data"],
                    "tags": ["ml", "education"],
                    "context": "Learning"
                },
                {
                    "content": "Database design",
                    "keywords": ["database", "sql", "design"],
                    "tags": ["backend"],
                    "category": "Engineering"
                }
            ],
            3
        ),
        # Notes with datetime fields
        (
            [
                {
                    "content": "Historical event",
                    "timestamp": datetime(
                        2023, 1, 15, 10, 30, 0, tzinfo=timezone.utc),
                    "keywords": ["history"],
                    "tags": ["event"]
                },
                {
                    "content": "Future reminder",
                    "timestamp": datetime(
                        2025, 12, 31, 23, 59, 59, tzinfo=timezone.utc),
                    "last_accessed": datetime(
                        2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
                    "keywords": ["reminder", "future"]
                }
            ],
            2
        ),
        # Empty lists and mix of specified/unspecified fields
        (
            [
                {
                    "content": "Minimal note with empty lists",
                    "keywords": [],
                    "tags": []
                },
                {
                    "content": "Only content and context",
                    "context": "SpecialContext"
                },
                {
                    "content": "Full featured note",
                    "keywords": ["k1", "k2", "k3"],
                    "tags": ["t1", "t2"],
                    "context": "FullContext",
                    "category": "FullCategory"
                }
            ],
            3
        ),
        # Larger number of notes
        (
            [{"content": f"Note number {i}", 
              "keywords": [f"key{i}"], 
              "tags": [f"tag{i}"]} 
             for i in range(10)],
            10
        )
    ])
    def test_multiple_notes_roundtrip(self, retriever, notes_data, expected_count):
        """
        Test multiple notes survive serialization roundtrip with various configurations.
        """
        system1 = AgenticMemorySystem(retriever=retriever)
        
        # Add notes
        note_ids = [system1.add_note(**data) for data in notes_data]
        
        # Load in new system
        system2 = AgenticMemorySystem(retriever=retriever)
        
        # Verify correct number of notes loaded
        assert len(system2.memories) == expected_count
        assert len(system2.memories) == len(notes_data)
        
        # Verify each note
        for note_id, expected_data in zip(note_ids, notes_data):
            loaded_note = system2.memories[note_id]
            
            # Check all specified fields match
            for field, expected_value in expected_data.items():
                actual_value = getattr(loaded_note, field)
                assert actual_value == expected_value, \
                    (
                        f"Field '{field}' mismatch: expected "
                        f"{expected_value}, got {actual_value}"
                    )
            
            # Verify default values for unspecified fields
            if "keywords" not in expected_data:
                assert loaded_note.keywords == []
            if "tags" not in expected_data:
                assert loaded_note.tags == []
            if "context" not in expected_data:
                assert loaded_note.context == "General"
            if "category" not in expected_data:
                assert loaded_note.category == "Uncategorized"
            if "retrieval_count" not in expected_data:
                assert loaded_note.retrieval_count == 0
            
            # Verify critical types are correct
            assert isinstance(loaded_note, MemoryNote)
            assert isinstance(loaded_note.timestamp, datetime)
            assert isinstance(loaded_note.last_accessed, datetime)
            assert isinstance(loaded_note.keywords, list)
            assert isinstance(loaded_note.tags, list)
            assert isinstance(loaded_note.id, str)
            assert isinstance(loaded_note.content, str)
            assert isinstance(loaded_note.context, str)
            assert isinstance(loaded_note.category, str)
            assert isinstance(loaded_note.retrieval_count, int)
