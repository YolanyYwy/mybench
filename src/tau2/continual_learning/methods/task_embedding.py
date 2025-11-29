"""
Task Embedding for Continual Learning.

This module implements task embedding-based continual learning that computes
task similarities and transfers knowledge from related tasks.
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from pathlib import Path
import json

from loguru import logger

from tau2.continual_learning.buffer import ExperienceBuffer
from tau2.continual_learning.data_model import Experience
from tau2.continual_learning.methods.base import ContinualLearningMethod
from tau2.continual_learning.prompt_templates import PromptTemplates
from tau2.continual_learning.enhanced_agent import PromptEnhancedAgent


class TaskEmbeddingMethod(ContinualLearningMethod):
    """
    Task embedding method for similarity-based knowledge transfer.

    This method embeds tasks based on their descriptions, policies, and tools,
    then transfers experiences from similar past tasks.
    """

    def __init__(
        self,
        buffer: ExperienceBuffer,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize task embedding method.

        Args:
            buffer: Experience buffer for storage
            config: Configuration dict with:
                - num_similar_tasks: Number of similar tasks to retrieve (default 3)
                - embedding_model: Model for task embeddings (default 'text-embedding-3-small')
                - use_openai_embeddings: Use OpenAI API (default True)
                - similarity_threshold: Min similarity for transfer (default 0.3)
                - transfer_ratio: Ratio of transferred experiences (default 0.3)
        """
        super().__init__(buffer, config)

        # Configuration
        self.num_similar_tasks = config.get("num_similar_tasks", 3)
        self.embedding_model = config.get("embedding_model", "text-embedding-3-small")
        self.use_openai_embeddings = config.get("use_openai_embeddings", True)
        self.similarity_threshold = config.get("similarity_threshold", 0.3)
        self.transfer_ratio = config.get("transfer_ratio", 0.3)

        # Task embeddings storage
        self.task_embeddings: Dict[int, np.ndarray] = {}
        self.task_metadata: Dict[int, Dict[str, Any]] = {}

        # Current task context for transfer
        self.similar_task_experiences: List[Experience] = []

        logger.info(
            f"TaskEmbeddingMethod initialized with num_similar={self.num_similar_tasks}"
        )

    def before_task(self, task_idx: int, task_id: str, agent) -> None:
        """
        Compute task embedding and find similar tasks.

        Args:
            task_idx: Current task index
            task_id: Current task ID
            agent: Current agent
        """
        # Compute embedding for current task
        task_embedding = self._embed_task(task_id, agent)
        self.task_embeddings[task_idx] = task_embedding

        # Store task metadata
        self.task_metadata[task_idx] = {
            "task_id": task_id,
            "domain": getattr(agent, 'domain', None),
            "num_tools": len(agent.tools) if hasattr(agent, 'tools') else 0,
        }

        # Find similar previous tasks
        if task_idx > 0:
            similar_tasks = self._find_similar_tasks(task_embedding, task_idx)

            # Load experiences from similar tasks
            self.similar_task_experiences = []
            for sim_task_idx, similarity in similar_tasks:
                # Sample experiences from similar task
                task_exps = self.buffer.sample_by_task_idx(
                    sim_task_idx,
                    n=5  # Take 5 experiences per similar task
                )
                self.similar_task_experiences.extend(task_exps)

            # Inject task context into agent
            if isinstance(agent, PromptEnhancedAgent) and similar_tasks:
                context = self._create_task_context(task_id, agent, similar_tasks)
                agent.add_knowledge(context)
                logger.info(
                    f"Task {task_idx}: Found {len(similar_tasks)} similar tasks, "
                    f"loaded {len(self.similar_task_experiences)} experiences"
                )

        logger.debug(f"Task {task_idx} ({task_id}): Task embedding computed")

    def after_training_step(
        self,
        task_idx: int,
        agent,
        experiences: List[Experience],
        step: int,
    ) -> None:
        """Store experiences in buffer."""
        self.buffer.add_batch(experiences)

        logger.debug(
            f"Task {task_idx}, step {step}: Added {len(experiences)} experiences"
        )

    def get_augmented_experiences(
        self,
        current_experiences: List[Experience],
        task_idx: int,
    ) -> List[Experience]:
        """
        Augment with experiences from similar tasks.

        Args:
            current_experiences: Current task experiences
            task_idx: Current task index

        Returns:
            Augmented experiences with similar task transfers
        """
        if not self.similar_task_experiences:
            return current_experiences

        # Calculate number of transfer experiences
        n_current = len(current_experiences)
        n_transfer = int(n_current * self.transfer_ratio)

        if n_transfer == 0:
            return current_experiences

        # Sample from similar task experiences
        if len(self.similar_task_experiences) <= n_transfer:
            transfer_exps = self.similar_task_experiences
        else:
            # Random sample
            import random
            transfer_exps = random.sample(self.similar_task_experiences, n_transfer)

        # Combine
        augmented = current_experiences + transfer_exps

        logger.debug(
            f"Augmented with {len(transfer_exps)} experiences from similar tasks"
        )

        return augmented

    def _embed_task(self, task_id: str, agent) -> np.ndarray:
        """
        Create embedding for a task.

        Args:
            task_id: Task ID
            agent: Agent with task context

        Returns:
            Task embedding vector
        """
        # Build task description
        components = [f"Task: {task_id}"]

        # Add domain policy snippet
        if hasattr(agent, 'domain_policy') and agent.domain_policy:
            policy_text = str(agent.domain_policy)
            # Take first 500 chars
            policy_snippet = policy_text[:500]
            components.append(f"Policy: {policy_snippet}")

        # Add available tools
        if hasattr(agent, 'tools') and agent.tools:
            tool_names = [tool.name for tool in agent.tools[:20]]
            components.append(f"Available tools: {', '.join(tool_names)}")

        # Combine
        task_description = "\n".join(components)

        # Get embedding
        if self.use_openai_embeddings:
            embedding = self._get_openai_embedding(task_description)
        else:
            embedding = self._get_simple_embedding(task_description)

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
        embedding = np.random.randn(1536)  # Standard dimension
        embedding = embedding / np.linalg.norm(embedding)
        return embedding

    def _find_similar_tasks(
        self,
        query_embedding: np.ndarray,
        current_task_idx: int,
    ) -> List[Tuple[int, float]]:
        """
        Find similar previous tasks.

        Args:
            query_embedding: Current task embedding
            current_task_idx: Current task index

        Returns:
            List of (task_idx, similarity) tuples
        """
        similarities = []

        for task_idx, embedding in self.task_embeddings.items():
            # Only consider previous tasks
            if task_idx >= current_task_idx:
                continue

            # Compute cosine similarity
            similarity = np.dot(query_embedding, embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(embedding) + 1e-10
            )

            # Filter by threshold
            if similarity >= self.similarity_threshold:
                similarities.append((task_idx, float(similarity)))

        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)

        # Return top-k
        return similarities[:self.num_similar_tasks]

    def _create_task_context(
        self,
        task_id: str,
        agent,
        similar_tasks: List[Tuple[int, float]],
    ) -> str:
        """
        Create task context prompt from similar tasks.

        Args:
            task_id: Current task ID
            agent: Current agent
            similar_tasks: List of (task_idx, similarity) tuples

        Returns:
            Formatted context prompt
        """
        # Current task info
        task_info = {
            "domain": getattr(agent, 'domain', 'Unknown'),
            "description": f"Task ID: {task_id}",
        }

        # Similar task info
        similar_task_info = []
        for task_idx, similarity in similar_tasks:
            metadata = self.task_metadata.get(task_idx, {})

            # Get success rate from buffer
            task_exps = self.buffer.sample_by_task_idx(task_idx, n=None)
            if task_exps:
                success_rate = sum(exp.success for exp in task_exps) / len(task_exps)
                approach = self._summarize_task_approach(task_exps)
            else:
                success_rate = 0.0
                approach = "No data available"

            similar_task_info.append({
                "task_id": metadata.get("task_id", f"Task {task_idx}"),
                "domain": metadata.get("domain", "Unknown"),
                "similarity": similarity,
                "success_rate": success_rate,
                "approach": approach,
                "key_insight": f"This task is {similarity:.1%} similar to your current task",
            })

        return PromptTemplates.task_context_template(task_info, similar_task_info)

    def _summarize_task_approach(self, experiences: List[Experience]) -> str:
        """Summarize successful approach from experiences."""
        # Get successful experiences
        successful = [exp for exp in experiences if exp.success]
        if not successful:
            return "No successful examples"

        # Extract common tools
        from collections import Counter
        tools_used = []

        for exp in successful:
            for msg in exp.messages:
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    for tc in msg.tool_calls:
                        tools_used.append(tc.name)

        if tools_used:
            tool_counts = Counter(tools_used)
            top_tools = [tool for tool, _ in tool_counts.most_common(3)]
            return f"Successfully used: {', '.join(top_tools)}"
        else:
            return "Completed through conversation"

    def save_state(self, path: str) -> None:
        """Save method state to disk."""
        save_dir = Path(path)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save embeddings
        embeddings_dict = {
            str(idx): emb.tolist()
            for idx, emb in self.task_embeddings.items()
        }

        state = {
            "config": self.config,
            "task_embeddings": embeddings_dict,
            "task_metadata": {str(k): v for k, v in self.task_metadata.items()},
        }

        with open(save_dir / "task_embedding_state.json", 'w') as f:
            json.dump(state, f, indent=2)

        logger.info(f"TaskEmbeddingMethod state saved to {save_dir}")

    def load_state(self, path: str) -> None:
        """Load method state from disk."""
        load_path = Path(path) / "task_embedding_state.json"

        if not load_path.exists():
            logger.warning(f"No saved state found at {load_path}")
            return

        with open(load_path, 'r') as f:
            state = json.load(f)

        self.config = state.get("config", self.config)

        # Load embeddings (convert back to numpy)
        embeddings_dict = state.get("task_embeddings", {})
        self.task_embeddings = {
            int(k): np.array(v) for k, v in embeddings_dict.items()
        }

        # Load metadata
        self.task_metadata = {
            int(k): v for k, v in state.get("task_metadata", {}).items()
        }

        logger.info(f"TaskEmbeddingMethod state loaded from {load_path}")
