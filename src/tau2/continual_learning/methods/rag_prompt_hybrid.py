"""
Hybrid RAG + Prompt Continual Learning Method.

This module implements a hybrid approach combining RAG-based semantic memory
retrieval with dynamic prompt engineering for maximum effectiveness.
"""

from typing import List, Dict, Any, Optional
from pathlib import Path
import numpy as np

from loguru import logger

from tau2.continual_learning.buffer import ExperienceBuffer
from tau2.continual_learning.data_model import Experience
from tau2.continual_learning.methods.base import ContinualLearningMethod
from tau2.continual_learning.methods.rag_memory import RAGMemoryMethod
from tau2.continual_learning.methods.dynamic_prompting import DynamicPromptingMethod
from tau2.continual_learning.prompt_templates import PromptBuilder
from tau2.continual_learning.enhanced_agent import CLEnhancedAgent


class RAGPromptHybridMethod(ContinualLearningMethod):
    """
    Hybrid method combining RAG memory retrieval with dynamic prompting.

    This method leverages both semantic similarity (RAG) for retrieving
    relevant experiences and task summarization (Dynamic Prompting) for
    accumulating structured knowledge.
    """

    def __init__(
        self,
        buffer: ExperienceBuffer,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize hybrid RAG + Prompt method.

        Args:
            buffer: Experience buffer for storage
            config: Configuration dict with:
                - num_memories: Number of RAG memories (default 5)
                - max_knowledge_items: Max knowledge items from prompting (default 5)
                - use_rag: Enable RAG component (default True)
                - use_prompting: Enable prompting component (default True)
                - embedding_model: Model for embeddings (default 'text-embedding-3-small')
                - vector_store_type: 'inmemory' or 'faiss' (default 'inmemory')
                - replay_ratio: Ratio for experience replay (default 0.3)
        """
        super().__init__(buffer, config)

        # Configuration
        self.use_rag = config.get("use_rag", True)
        self.use_prompting = config.get("use_prompting", True)
        self.replay_ratio = config.get("replay_ratio", 0.3)

        # Initialize sub-methods
        rag_config = {
            "num_memories": config.get("num_memories", 5),
            "embedding_model": config.get("embedding_model", "text-embedding-3-small"),
            "vector_store_type": config.get("vector_store_type", "inmemory"),
            "use_openai_embeddings": config.get("use_openai_embeddings", True),
            "filter_by_domain": config.get("filter_by_domain", False),
            "min_similarity": config.get("min_similarity", 0.0),
        }

        prompting_config = {
            "max_knowledge_items": config.get("max_knowledge_items", 5),
            "summary_mode": config.get("summary_mode", "key_points"),
            "include_errors": config.get("include_errors", True),
            "include_success_patterns": config.get("include_success_patterns", True),
        }

        # Create sub-methods (they share the same buffer)
        self.rag_method = RAGMemoryMethod(buffer, rag_config) if self.use_rag else None
        self.prompt_method = DynamicPromptingMethod(buffer, prompting_config) if self.use_prompting else None

        # Track which experiences came from replay
        self.replay_experiences: List[Experience] = []

        logger.info(
            f"RAGPromptHybridMethod initialized (RAG={self.use_rag}, Prompting={self.use_prompting})"
        )

    def before_task(self, task_idx: int, task_id: str, agent) -> None:
        """
        Prepare agent with both RAG memories and dynamic knowledge.

        Args:
            task_idx: Current task index
            task_id: Current task ID
            agent: Agent to enhance
        """
        # Collect context from both methods
        rag_context = ""
        prompt_context = ""

        # Get RAG memories
        if self.use_rag and self.rag_method:
            self.rag_method.before_task(task_idx, task_id, agent)
            rag_context = self.rag_method.get_agent_context()

        # Get dynamic prompting knowledge
        if self.use_prompting and self.prompt_method:
            self.prompt_method.before_task(task_idx, task_id, agent)
            prompt_context = self.prompt_method.get_agent_context()

        # Combine contexts using PromptBuilder
        if rag_context or prompt_context:
            combined_context = self._combine_contexts(rag_context, prompt_context)

            # Inject into agent
            if isinstance(agent, CLEnhancedAgent):
                agent.update_context(combined_context)
                logger.info(
                    f"Task {task_idx}: Injected hybrid context "
                    f"(RAG={bool(rag_context)}, Prompting={bool(prompt_context)})"
                )

        logger.debug(f"Task {task_idx} ({task_id}): Hybrid method ready")

    def after_training_step(
        self,
        task_idx: int,
        agent,
        experiences: List[Experience],
        step: int,
    ) -> None:
        """
        Process experiences with both sub-methods.

        Args:
            task_idx: Current task index
            agent: Agent being trained
            experiences: Experiences from current step
            step: Training step number
        """
        # Both methods add to the same buffer
        if self.use_rag and self.rag_method:
            self.rag_method.after_training_step(task_idx, agent, experiences, step)

        if self.use_prompting and self.prompt_method:
            self.prompt_method.after_training_step(task_idx, agent, experiences, step)

        # Note: buffer is shared, so experiences are only added once

        logger.debug(
            f"Task {task_idx}, step {step}: Processed {len(experiences)} experiences with hybrid method"
        )

    def after_task(
        self,
        task_idx: int,
        task_result,
        agent,
    ) -> None:
        """
        Update both sub-methods after task completion.

        Args:
            task_idx: Completed task index
            task_result: Task result
            agent: Trained agent
        """
        if self.use_rag and self.rag_method:
            # RAG method doesn't have after_task logic currently
            pass

        if self.use_prompting and self.prompt_method:
            self.prompt_method.after_task(task_idx, task_result, agent)

        logger.info(f"Task {task_idx} completed: Hybrid method updated")

    def get_augmented_experiences(
        self,
        current_experiences: List[Experience],
        task_idx: int,
    ) -> List[Experience]:
        """
        Augment experiences using RAG-based retrieval.

        Args:
            current_experiences: Current task experiences
            task_idx: Current task index

        Returns:
            Augmented experiences with RAG-retrieved examples
        """
        if not self.use_rag or not self.rag_method:
            return current_experiences

        # Use RAG to find similar experiences from buffer
        if len(self.buffer) == 0 or task_idx == 0:
            return current_experiences

        # Calculate number of replay samples
        n_current = len(current_experiences)
        n_replay = int(n_current * self.replay_ratio)

        if n_replay == 0:
            return current_experiences

        # Sample diverse experiences from previous tasks
        previous_task_indices = [
            idx for idx in self.buffer.get_task_indices()
            if idx < task_idx
        ]

        if not previous_task_indices:
            return current_experiences

        # Sample using buffer's diverse sampling
        replay_samples = self.buffer.sample_diverse(
            n=n_replay,
            diversity_key="task_idx",
        )

        # Filter to only previous tasks
        replay_samples = [
            exp for exp in replay_samples
            if exp.task_idx < task_idx
        ]

        # Store for analysis
        self.replay_experiences = replay_samples

        # Combine
        augmented = current_experiences + replay_samples

        logger.debug(
            f"Augmented with {len(replay_samples)} diverse replay experiences"
        )

        return augmented

    def get_agent_context(self) -> str:
        """
        Get combined context from both methods.

        Returns:
            Combined context prompt
        """
        rag_context = ""
        prompt_context = ""

        if self.use_rag and self.rag_method:
            rag_context = self.rag_method.get_agent_context()

        if self.use_prompting and self.prompt_method:
            prompt_context = self.prompt_method.get_agent_context()

        return self._combine_contexts(rag_context, prompt_context)

    def _combine_contexts(self, rag_context: str, prompt_context: str) -> str:
        """
        Intelligently combine RAG and prompting contexts.

        Args:
            rag_context: Context from RAG method
            prompt_context: Context from prompting method

        Returns:
            Combined context
        """
        builder = PromptBuilder()

        # Add prompt context first (structured knowledge)
        if prompt_context:
            builder.add_custom(prompt_context)

        # Add RAG context second (specific examples)
        if rag_context:
            builder.add_custom(rag_context)

        combined = builder.build()

        # Add meta-instruction if both are present
        if rag_context and prompt_context:
            meta_instruction = """
<instructions>
You have access to both structured knowledge from previous tasks and specific relevant examples.
Use the structured knowledge to understand general patterns and the examples for specific guidance.
Prioritize examples when they directly match your current situation.
</instructions>
"""
            combined = meta_instruction + "\n" + combined

        return combined

    def save_state(self, path: str) -> None:
        """Save state of both sub-methods."""
        save_dir = Path(path) / "hybrid_state"
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save config
        import json
        with open(save_dir / "config.json", 'w') as f:
            json.dump(self.config, f, indent=2)

        # Save sub-methods
        if self.use_rag and self.rag_method:
            self.rag_method.save_state(str(save_dir / "rag"))

        if self.use_prompting and self.prompt_method:
            self.prompt_method.save_state(str(save_dir / "prompting"))

        logger.info(f"RAGPromptHybridMethod state saved to {save_dir}")

    def load_state(self, path: str) -> None:
        """Load state of both sub-methods."""
        load_dir = Path(path) / "hybrid_state"

        if not load_dir.exists():
            logger.warning(f"No saved state found at {load_dir}")
            return

        # Load config
        import json
        config_file = load_dir / "config.json"
        if config_file.exists():
            with open(config_file, 'r') as f:
                self.config = json.load(f)

        # Load sub-methods
        if self.use_rag and self.rag_method:
            rag_path = load_dir / "rag"
            if rag_path.exists():
                self.rag_method.load_state(str(rag_path))

        if self.use_prompting and self.prompt_method:
            prompt_path = load_dir / "prompting"
            if prompt_path.exists():
                self.prompt_method.load_state(str(prompt_path))

        logger.info(f"RAGPromptHybridMethod state loaded from {load_dir}")
