"""
RAG-based Memory System for Continual Learning.

This module implements retrieval-augmented generation (RAG) for continual learning,
using vector databases to store and retrieve relevant past experiences.
"""

from typing import List, Dict, Any, Optional
import numpy as np
from pathlib import Path

from loguru import logger

from tau2.continual_learning.buffer import ExperienceBuffer
from tau2.continual_learning.data_model import Experience
from tau2.continual_learning.methods.base import ContinualLearningMethod
from tau2.continual_learning.vector_store import create_vector_store, VectorStore
from tau2.continual_learning.prompt_templates import PromptTemplates
from tau2.continual_learning.enhanced_agent import MemoryEnhancedAgent
from tau2.data_model.message import AssistantMessage, UserMessage, ToolMessage


class RAGMemoryMethod(ContinualLearningMethod):
    """
    RAG-based continual learning with semantic memory retrieval.

    This method stores experiences in a vector database and retrieves
    relevant memories based on semantic similarity to the current task.
    """

    def __init__(
        self,
        buffer: ExperienceBuffer,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize RAG memory method.

        Args:
            buffer: Experience buffer for storage
            config: Configuration dict with:
                - num_memories: Number of memories to retrieve (default 5)
                - vector_store_type: 'inmemory' or 'faiss' (default 'inmemory')
                - embedding_model: Model for embeddings (default 'text-embedding-3-small')
                - embedding_dim: Dimension of embeddings (default 1536)
                - use_openai_embeddings: Whether to use OpenAI API (default True)
                - filter_by_domain: Only retrieve from same domain (default False)
                - min_similarity: Minimum similarity threshold (default 0.0)
        """
        super().__init__(buffer, config)

        # Configuration
        self.num_memories = config.get("num_memories", 5)
        self.vector_store_type = config.get("vector_store_type", "inmemory")
        self.embedding_model = config.get("embedding_model", "text-embedding-3-small")
        self.embedding_dim = config.get("embedding_dim", 1536)
        self.use_openai_embeddings = config.get("use_openai_embeddings", True)
        self.filter_by_domain = config.get("filter_by_domain", False)
        self.min_similarity = config.get("min_similarity", 0.0)

        # Initialize vector store
        if self.vector_store_type == "faiss":
            self.vector_store = create_vector_store("faiss", dimension=self.embedding_dim)
        else:
            self.vector_store = create_vector_store("inmemory")

        # Embedding cache
        self.embeddings_cache: Dict[str, np.ndarray] = {}

        # Current task context
        self.current_task_info: Optional[Dict[str, Any]] = None
        self.retrieved_memories: List[Dict[str, Any]] = []

        logger.info(
            f"RAGMemoryMethod initialized with {self.vector_store_type} store, "
            f"embedding_model={self.embedding_model}"
        )

    def before_task(self, task_idx: int, task_id: str, agent) -> None:
        """
        Retrieve relevant memories before starting a task.

        Args:
            task_idx: Current task index
            task_id: Current task ID
            agent: The agent to enhance with memories
        """
        # Store task info
        self.current_task_info = {
            "task_idx": task_idx,
            "task_id": task_id,
            "domain": getattr(agent, 'domain_policy', None),
        }

        # Retrieve relevant memories if we have any
        if len(self.vector_store) > 0:
            self.retrieved_memories = self._retrieve_memories(task_id, agent)

            # Inject memories into agent if it's a MemoryEnhancedAgent
            if isinstance(agent, MemoryEnhancedAgent):
                agent.set_memories(self.retrieved_memories)
                logger.info(f"Task {task_idx}: Injected {len(self.retrieved_memories)} memories into agent")
            else:
                logger.debug(
                    f"Task {task_idx}: Agent is not MemoryEnhancedAgent, "
                    "memories retrieved but not injected"
                )

        logger.debug(f"Task {task_idx} ({task_id}): RAG memory ready")

    def after_training_step(
        self,
        task_idx: int,
        agent,
        experiences: List[Experience],
        step: int,
    ) -> None:
        """
        Index experiences in vector store after each training step.

        Args:
            task_idx: Current task index
            agent: The agent being trained
            experiences: Experiences from current step
            step: Training step number
        """
        # Add to buffer
        self.buffer.add_batch(experiences)

        # Index in vector store
        for exp in experiences:
            # Create document representation
            doc = self._experience_to_document(exp)

            # Get embedding
            embedding = self._get_embedding(doc['text'])

            # Create unique ID
            exp_id = f"{exp.task_id}_{task_idx}_{step}_{id(exp)}"

            # Add to vector store
            self.vector_store.add(
                id=exp_id,
                embedding=embedding,
                metadata={
                    "task_id": exp.task_id,
                    "task_idx": task_idx,
                    "domain": exp.domain,
                    "success": exp.success,
                    "reward": exp.reward if exp.reward is not None else 0.0,
                    **doc['metadata']
                },
                document=doc['summary'],
            )

        logger.debug(
            f"Task {task_idx}, step {step}: Indexed {len(experiences)} experiences, "
            f"vector store size={len(self.vector_store)}"
        )

    def get_agent_context(self) -> str:
        """
        Get memory context to inject into agent prompts.

        Returns:
            Formatted prompt with retrieved memories
        """
        if not self.retrieved_memories:
            return ""

        return PromptTemplates.retrieved_memories_template(self.retrieved_memories)

    def _retrieve_memories(
        self,
        task_id: str,
        agent,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant memories for a task.

        Args:
            task_id: Current task ID
            agent: Current agent (for domain info)

        Returns:
            List of retrieved memory dicts
        """
        # Create query from task description
        query_text = self._create_task_query(task_id, agent)

        # Get embedding
        query_embedding = self._get_embedding(query_text)

        # Build metadata filter
        metadata_filter = None
        if self.filter_by_domain and hasattr(agent, 'domain_policy'):
            metadata_filter = {"domain": getattr(agent, 'domain', None)}

        # Query vector store
        results = self.vector_store.query(
            query_embedding=query_embedding,
            k=self.num_memories * 2,  # Retrieve more, then filter
            filter=metadata_filter,
        )

        # Filter by similarity threshold
        filtered_results = [
            r for r in results
            if r['similarity'] >= self.min_similarity
        ]

        # Take top k
        top_results = filtered_results[:self.num_memories]

        # Format as memory dicts
        memories = []
        for result in top_results:
            memories.append({
                "task_id": result['metadata'].get('task_id', 'Unknown'),
                "domain": result['metadata'].get('domain', 'Unknown'),
                "success": result['metadata'].get('success', False),
                "similarity": result['similarity'],
                "summary": result['document'],
                "key_actions": result['metadata'].get('key_actions', []),
                "lessons": result['metadata'].get('lessons', ''),
            })

        logger.info(
            f"Retrieved {len(memories)} relevant memories for task {task_id} "
            f"(avg similarity: {np.mean([m['similarity'] for m in memories]):.3f})"
        )

        return memories

    def _create_task_query(self, task_id: str, agent) -> str:
        """
        Create a query string for retrieving relevant memories.

        Args:
            task_id: Task ID
            agent: Current agent

        Returns:
            Query string for embedding
        """
        # Combine task ID and domain policy for richer query
        components = [f"Task: {task_id}"]

        # Add domain policy if available
        if hasattr(agent, 'domain_policy') and agent.domain_policy:
            # Take first few lines of policy as context
            policy_lines = str(agent.domain_policy).split('\n')[:5]
            components.append("Policy: " + " ".join(policy_lines))

        # Add available tools
        if hasattr(agent, 'tools') and agent.tools:
            tool_names = [tool.name for tool in agent.tools[:10]]  # Limit to first 10
            components.append("Tools: " + ", ".join(tool_names))

        return "\n".join(components)

    def _experience_to_document(self, exp: Experience) -> Dict[str, Any]:
        """
        Convert experience to a searchable document.

        Args:
            exp: Experience to convert

        Returns:
            Dict with 'text', 'summary', 'metadata'
        """
        # Extract key information
        actions = self._extract_actions(exp.messages)
        conversation = self._summarize_conversation(exp.messages)

        # Create summary
        outcome = "successful" if exp.success else "failed"
        summary = f"Task {exp.task_id} ({outcome}): {conversation}"

        # Create full text for embedding
        text_components = [
            f"Task: {exp.task_id}",
            f"Domain: {exp.domain}",
            f"Outcome: {outcome}",
            f"Actions: {', '.join(actions)}",
            f"Conversation: {conversation}",
        ]
        text = "\n".join(text_components)

        # Extract lessons learned (from failed experiences)
        lessons = ""
        if not exp.success:
            lessons = self._extract_lessons(exp.messages)

        return {
            "text": text,
            "summary": summary,
            "metadata": {
                "key_actions": actions,
                "lessons": lessons,
                "num_turns": len(exp.messages),
            }
        }

    def _extract_actions(self, messages: List) -> List[str]:
        """Extract tool calls from messages."""
        actions = []
        for msg in messages:
            if isinstance(msg, (AssistantMessage, UserMessage)) and hasattr(msg, 'tool_calls'):
                if msg.tool_calls:
                    for tc in msg.tool_calls:
                        actions.append(tc.name)
        return actions

    def _summarize_conversation(self, messages: List, max_length: int = 200) -> str:
        """Summarize conversation from messages."""
        # Extract text from messages
        texts = []
        for msg in messages:
            if hasattr(msg, 'content') and msg.content:
                texts.append(str(msg.content))

        # Join and truncate
        full_text = " ".join(texts)
        if len(full_text) > max_length:
            full_text = full_text[:max_length] + "..."

        return full_text

    def _extract_lessons(self, messages: List) -> str:
        """Extract lessons from failed experiences."""
        # Look for error messages or user feedback
        lessons = []
        for msg in messages:
            if isinstance(msg, ToolMessage) and hasattr(msg, 'content'):
                # Check for error indicators
                content = str(msg.content).lower()
                if any(word in content for word in ['error', 'invalid', 'failed', 'cannot']):
                    lessons.append(str(msg.content)[:100])

        return " | ".join(lessons) if lessons else "No specific lessons extracted"

    def _get_embedding(self, text: str) -> np.ndarray:
        """
        Get embedding for text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        # Check cache
        if text in self.embeddings_cache:
            return self.embeddings_cache[text]

        # Get embedding
        if self.use_openai_embeddings:
            embedding = self._get_openai_embedding(text)
        else:
            # Fallback to simple hash-based embedding
            embedding = self._get_simple_embedding(text)

        # Cache it
        self.embeddings_cache[text] = embedding

        return embedding

    def _get_openai_embedding(self, text: str) -> np.ndarray:
        """Get embedding from OpenAI API."""
        try:
            from openai import OpenAI
            client = OpenAI()

            response = client.embeddings.create(
                model=self.embedding_model,
                input=text,
            )

            embedding = np.array(response.data[0].embedding)
            return embedding

        except Exception as e:
            logger.warning(f"Failed to get OpenAI embedding: {e}, using fallback")
            return self._get_simple_embedding(text)

    def _get_simple_embedding(self, text: str) -> np.ndarray:
        """Simple hash-based embedding as fallback."""
        # Use hash to create deterministic random embedding
        np.random.seed(hash(text) % (2**32))
        embedding = np.random.randn(self.embedding_dim)
        # Normalize
        embedding = embedding / np.linalg.norm(embedding)
        return embedding

    def save_state(self, path: str) -> None:
        """Save method state to disk."""
        save_dir = Path(path) / "rag_memory_state"
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save vector store
        self.vector_store.save(str(save_dir / "vector_store"))

        # Save embeddings cache
        np.savez(
            save_dir / "embeddings_cache.npz",
            **{text: emb for text, emb in self.embeddings_cache.items()}
        )

        # Save config
        import json
        with open(save_dir / "config.json", 'w') as f:
            json.dump(self.config, f, indent=2)

        logger.info(f"RAGMemoryMethod state saved to {save_dir}")

    def load_state(self, path: str) -> None:
        """Load method state from disk."""
        load_dir = Path(path) / "rag_memory_state"

        if not load_dir.exists():
            logger.warning(f"No saved state found at {load_dir}")
            return

        # Load vector store
        self.vector_store.load(str(load_dir / "vector_store"))

        # Load embeddings cache
        cache_file = load_dir / "embeddings_cache.npz"
        if cache_file.exists():
            cache_data = np.load(cache_file)
            self.embeddings_cache = {key: cache_data[key] for key in cache_data.files}

        # Load config
        config_file = load_dir / "config.json"
        if config_file.exists():
            import json
            with open(config_file, 'r') as f:
                self.config = json.load(f)

        logger.info(f"RAGMemoryMethod state loaded from {load_dir}")
