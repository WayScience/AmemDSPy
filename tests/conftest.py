import pytest
import tempfile
import shutil
from pathlib import Path

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
        "timestamp": "2024-01-01T00:00:00",
        "tags": ["test", "memory"],
        "config": {"key": "value"},
        "count": 42,
        "score": 0.95
    }


@pytest.fixture
def temp_db_dir():
    """Fixture providing a temporary directory for persistent ChromaDB."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    # Cleanup: remove the temporary directory after test
    shutil.rmtree(temp_dir, ignore_errors=True)

    
@pytest.fixture
def existing_collection(temp_db_dir, sample_metadata):
    """Fixture that creates a pre-existing collection with data."""
    retriever = PersistentChromaRetriever(
        directory=str(temp_db_dir),
        collection_name="existing_collection"
    )
    retriever.add_document("Existing document", sample_metadata, "existing_doc")
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
