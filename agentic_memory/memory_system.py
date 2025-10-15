from datetime import datetime
from typing import Any, Dict, List, Optional

from .retrievers import ChromaRetriever
from .memory_note import MemoryNote


class AgenticMemorySystem:
    """
    Core memory system that manages memory.    
    Functionalities:
    - Memory creation, retrieval, update, and deletion (CRUD operations)
    - Embedding-based semantic search
    - Metadata management
    - Expandability for integrating LLM-based memory processing
    """
    
    def __init__(
        self, 
        retriever: Optional[ChromaRetriever] = None,
        **kwargs
    ):
        """
        Initialize the memory system.
        
        :param collection_name: Name of the ChromaDB collection
        :param reset_collection: If True, reset the collection on init
        :param kwargs: Additional args for ChromaRetriever
        """
        self.memories = {}
        
        # Initialize ChromaDB retriever
        if retriever is None:
            try:
                self.retriever = ChromaRetriever(**kwargs)
                self.retriever.client.reset()
            except Exception as e:
                raise RuntimeError(
                    f"Error during initializing ChromaDB retriever: {e}")

        elif isinstance(retriever, ChromaRetriever):
            self.retriever = retriever
        else:
            raise TypeError(
                "retriever must be a ChromaRetriever instance, "
                f"got {type(retriever)}"
            )
        
        # for future LLM integration
        self._llm_processor = None

    def add_note(
        self, 
        content: str, 
        **kwargs
    ) -> str:
        """Add a new memory note to the system.
        
        :param kwargs: Additional metadata fields
        :param content: The main text content of the memory
        :param timestamp: Creation time in format YYYYMMDDHHMM
        :param keywords: Key terms extracted from the content
        :param context: The broader context or domain of the memory
        :param category: Classification category
        :param tags: Additional classification tags
        
        :return: The unique ID of the created memory note
        """
        # Create MemoryNote
        note = MemoryNote(
            content=content,
            **kwargs
        )
        
        # Store in local dictionary
        self.memories[note.id] = note
        
        # Add to ChromaDB with complete metadata
        metadata = self._build_metadata(note)
        self.retriever.add_document(note.content, metadata, note.id)
        
        return note.id
    
    def _serialize_metadata(self, note: MemoryNote) -> Dict[str, Any]:
        """
        Build serialized metadata dictionary from a MemoryNote.
        Such that it can be stored in ChromaDB.
        
        :param note: MemoryNote instance
        :return: Metadata dictionary for ChromaDB
        """

        metadata_dump = note.model_dump()               

        return {
            key: json.dumps(value) \
                if isinstance(value, (dict, list)) else value
            for key, value in metadata_dump.items()
        }
    
    def read(self, memory_id: str) -> Optional[MemoryNote]:
        """Retrieve a memory note by its ID.
        
        :param memory_id: ID of the memory to retrieve
        :return: MemoryNote instance or None if not found
        """
        memory = self.memories.get(memory_id)
        if memory:
            # Update access tracking
            memory.last_accessed = datetime.now().strftime("%Y%m%d%H%M")
            memory.retrieval_count += 1
        return memory

    def search(
        self,
        query: str,
        k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Perform a semantic search for memory notes similar to the query.

        :param query: The search query string
        :param k: Number of top results to return
        """
        
        results = self.retriever.search(query, k)
        
        if not results or not results.get('ids'):
            return []

        return [
            self.memories.get(id).model_dump() \
                if id in self.memories else {"id": id, "content": content}
            for id, content in zip(results['ids'][0], results['documents'][0])
        ]
    
    def update(
        self, 
        memory_id: str, 
        **kwargs
    ) -> bool:
        """
        Update a memory note's fields and synchronize with ChromaDB.
        Only allows for update of fields defined in MemoryNote.
        
        :param memory_id: ID of the memory to update
        :param kwargs: Fields to update with new values
        """
        if memory_id not in self.memories:
            return False
        
        note = self.memories[memory_id]

        # Update fields
        for key, value in kwargs.items():
            if hasattr(note, key):
                setattr(note, key, value)
        
        self.retriever.collection.update(
            ids=[memory_id], 
            documents=[note.content], 
            metadatas=[self._build_metadata(note)]
        )
