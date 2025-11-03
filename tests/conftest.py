import pytest
import tempfile
import shutil
import json
import copy
from pathlib import Path
from datetime import datetime, timezone

from agentic_memory.retrievers import ChromaRetriever, PersistentChromaRetriever
from agentic_memory.memory_system import AgenticMemorySystem
from agentic_memory.memory_note import MemoryNote


@pytest.fixture
def retriever():
    """Fixture providing a clean ChromaRetriever instance."""
    retriever = ChromaRetriever(collection_name="test_memories")
    yield retriever
    # Cleanup: reset the collection after each test
    retriever.client.reset()


@pytest.fixture
def sample_metadata():
    """Fixture providing sample metadata with various types."""
    return {
        "timestamp": datetime.now(timezone.utc),
        "last_accessed": datetime.now(timezone.utc),
        "tags": ["test", "memory"],
        "context": "unit testing",
        "category": "test_case",
        "retrieval_count": 5,
        "extras": {"source": "unittest", "priority": "high"}
    }

@pytest.fixture
def serialize_metadata():
    """Return a function that serializes a metadata dict without mutating input."""
    def _serialize(d: dict) -> dict:
        out = copy.deepcopy(d)  # avoid cross-test mutation
        for k, v in out.items():
            if isinstance(v, datetime):
                out[k] = v.isoformat()
            elif isinstance(v, (list, dict)):
                out[k] = json.dumps(v)
            elif isinstance(v, (int, float, str, type(None))):
                # leave basic JSON-native scalars as-is
                pass
            else:
                # last-resort stringification
                try:
                    out[k] = str(v)
                except Exception:
                    out[k] = "<unknown>"
        return out
    return _serialize

@pytest.fixture
def serialized_sample_metadata(sample_metadata, serialize_metadata):
    """Fixture providing serialized sample metadata."""
    return serialize_metadata(sample_metadata)

@pytest.fixture
def temp_db_dir():
    """Fixture providing a temporary directory for persistent ChromaDB."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    # Cleanup: remove the temporary directory after test
    shutil.rmtree(temp_dir, ignore_errors=True)

    
@pytest.fixture
def existing_collection(temp_db_dir, serialized_sample_metadata):
    """Fixture that creates a pre-existing collection with data."""
    retriever = PersistentChromaRetriever(
        directory=str(temp_db_dir),
        collection_name="existing_collection"
    )
    retriever.add_document("Existing document", serialized_sample_metadata, "existing_doc")
    return temp_db_dir, "existing_collection"


@pytest.fixture
def memory_system(retriever):
    """Fixture providing a clean AgenticMemorySystem instance."""
    system = AgenticMemorySystem(retriever=retriever)
    yield system
    # Cleanup handled by retriever fixture


@pytest.fixture
def sample_memory_note():
    """Fixture providing a sample MemoryNote instance."""
    return MemoryNote(
        content="This is a test memory about machine learning",
        keywords=["machine learning", "AI", "test"],
        context="Testing",
        category="Technical",
        tags=["test", "ml"]
    )


@pytest.fixture
def populated_memory_system(memory_system):
    """Fixture providing a memory system with pre-populated data."""
    # Add several test memories
    memory_system.add_note(
        content="Python is a programming language",
        keywords=["python", "programming"],
        context="Programming",
        tags=["language", "coding"]
    )
    memory_system.add_note(
        content="Machine learning is a subset of AI",
        keywords=["machine learning", "AI"],
        context="Technology",
        tags=["ml", "ai"]
    )
    memory_system.add_note(
        content="ChromaDB is a vector database",
        keywords=["chromadb", "database", "vector"],
        context="Technology",
        tags=["database", "vector"]
    )
    return memory_system


@pytest.fixture
def keyword_filterable_memory_system(memory_system):
    """
    Fixture providing a memory system with notes that have extras, 
        for testing filtering.
    """
    # Add notes with extras for keyword filtering tests
    ids = {}
    
    ids['project_a'] = memory_system.add_note(
        content="Python programming",
        project="project_a"
    )
    ids['project_b'] = memory_system.add_note(
        content="Python tutorial",
        project="project_b"
    )
    ids['ml_alice'] = memory_system.add_note(
        content="Machine learning",
        project="ml_project",
        author="alice"
    )
    ids['ml_bob'] = memory_system.add_note(
        content="Deep learning",
        project="ml_project",
        author="bob"
    )
    
    return memory_system, ids
