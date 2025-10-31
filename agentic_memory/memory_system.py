import json
from datetime import datetime, timezone
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
        
        # Load existing memories from the retriever
        self._load_existing_memories()

    def _load_existing_memories(self):
        """
        Load all existing memories from the retriever into self.memories.
        Called during initialization to sync in-memory state with ChromaDB.
        """
        try:
            # Get all documents from the collection
            all_data = self.retriever.collection.get(
                include=["metadatas", "documents"]
            )
            
            if not all_data or not all_data.get('ids'):
                return
            
            # Convert metadata types
            if all_data.get("metadatas"):
                all_data["metadatas"] = self.retriever._convert_metadata_types(
                    [all_data["metadatas"]]
                )[0]
            
            # Reconstruct MemoryNote objects
            for doc_id, content, metadata in zip(
                all_data['ids'], 
                all_data['documents'], 
                all_data['metadatas']
            ):
                # Ensure content is in metadata
                if 'content' not in metadata:
                    metadata['content'] = content
                if 'id' not in metadata:
                    metadata['id'] = doc_id
                    
                note = self._deserialize_metadata(metadata)
                self.memories[doc_id] = note
                
        except AttributeError:
            # Collection may not exist yet - this is fine for new systems
            pass
        except Exception as e:
            # For unexpected errors, warn but don't crash
            import warnings
            warnings.warn(
                f"Failed to load existing memories from retriever: {e}",
                RuntimeWarning
            )

    def add_note(
        self, 
        content: str, 
        **kwargs
    ) -> str:
        """Add a new memory note to the system.
        
        :param kwargs: Additional metadata fields, which can include:
        :param keywords: Key terms extracted from the content
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
        metadata = self._serialize_metadata(note)
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
            key: (
                value.isoformat() if isinstance(value, datetime) 
                else json.dumps(value) if isinstance(value, (dict, list))
                else value
            )
            for key, value in metadata_dump.items()
        }
    
    def _deserialize_metadata(self, metadata: Dict[str, Any]) -> MemoryNote:
        """
        Reconstruct a MemoryNote from deserialized metadata.
        
        :param metadata: Metadata dictionary from ChromaDB (already type-converted)
        :return: MemoryNote instance
        """
        # Remove 'id' from metadata to avoid duplication since it's a separate field
        note_data = metadata.copy()
        
        # Convert any remaining JSON strings if needed
        for key, value in note_data.items():
            if isinstance(value, str) and key in ('keywords', 'tags'):
                try:
                    note_data[key] = json.loads(value)
                except (json.JSONDecodeError, TypeError):
                    pass
        
        return MemoryNote(**note_data)
    
    def read(self, memory_id: str) -> Optional[MemoryNote]:
        """Retrieve a memory note by its ID.
        
        :param memory_id: ID of the memory to retrieve
        :return: MemoryNote instance or None if not found
        """
        memory = self.memories.get(memory_id)
        if memory:
            # Update access tracking
            memory.last_accessed = datetime.now(timezone.utc)
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
        
        try:
            self.retriever.collection.update(
                ids=[memory_id], 
                documents=[note.content], 
                metadatas=[self._serialize_metadata(note)]
            )
        except Exception:
            return False
        
        return True
