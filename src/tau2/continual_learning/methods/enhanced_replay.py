"""
Enhanced Experience Replay with Priority Sampling.

This module implements an improved experience replay method with
priority-based sampling, diversity-aware selection, and importance weighting.
"""

from typing import List, Dict, Any, Optional
import numpy as np

from loguru import logger

from tau2.continual_learning.buffer import ExperienceBuffer
from tau2.continual_learning.data_model import Experience
from tau2.continual_learning.methods.base import ContinualLearningMethod


class EnhancedReplayMethod(ContinualLearningMethod):
    """
    Enhanced Experience Replay with multiple improvements over basic replay.

    Enhancements:
    1. Priority sampling based on task difficulty (reward-based)
    2. Diversity-aware sampling to ensure task coverage
    3. Success rate balancing
    4. Recency bias option
    """

    def __init__(
        self,
        buffer: ExperienceBuffer,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize enhanced replay method.

        Args:
            buffer: Experience buffer for storage
            config: Configuration dict with:
                - replay_ratio: Ratio of replay to current experiences (default 0.5)
                - priority_mode: 'uniform', 'difficulty', 'diversity', 'recent' (default 'difficulty')
                - min_buffer_size: Minimum buffer size before replay (default 10)
                - diversity_key: Key for diversity sampling (default 'task_idx')
                - recent_ratio: Ratio for recent sampling (default 0.7)
                - balance_success: Whether to balance successful/failed experiences (default False)
        """
        super().__init__(buffer, config)

        # Configuration
        self.replay_ratio = config.get("replay_ratio", 0.5)
        self.priority_mode = config.get("priority_mode", "difficulty")
        self.min_buffer_size = config.get("min_buffer_size", 10)
        self.diversity_key = config.get("diversity_key", "task_idx")
        self.recent_ratio = config.get("recent_ratio", 0.7)
        self.balance_success = config.get("balance_success", False)

        # Internal state
        self.experience_priorities: Dict[int, float] = {}
        self.task_difficulties: Dict[int, float] = {}

        logger.info(
            f"EnhancedReplayMethod initialized with mode={self.priority_mode}, "
            f"replay_ratio={self.replay_ratio}"
        )

    def before_task(self, task_idx: int, task_id: str, agent) -> None:
        """Called before starting a new task."""
        logger.debug(f"Task {task_idx} ({task_id}): Enhanced replay ready")

    def after_training_step(
        self,
        task_idx: int,
        agent,
        experiences: List[Experience],
        step: int,
    ) -> None:
        """
        Store experiences and update priorities after each training step.

        Args:
            task_idx: Current task index
            agent: The agent being trained
            experiences: Experiences from current step
            step: Training step number
        """
        # Add experiences to buffer
        self.buffer.add_batch(experiences)

        # Update priorities based on experience outcomes
        for i, exp in enumerate(experiences):
            exp_idx = len(self.buffer) - len(experiences) + i
            priority = self._compute_priority(exp)
            self.experience_priorities[exp_idx] = priority

        logger.debug(
            f"Task {task_idx}, step {step}: Added {len(experiences)} experiences, "
            f"buffer size={len(self.buffer)}"
        )

    def get_augmented_experiences(
        self,
        current_experiences: List[Experience],
        task_idx: int,
    ) -> List[Experience]:
        """
        Augment current experiences with replay samples.

        Args:
            current_experiences: Experiences from current task
            task_idx: Current task index

        Returns:
            Combined list of current and replay experiences
        """
        # Check if buffer has enough experiences
        if len(self.buffer) < self.min_buffer_size:
            logger.debug("Buffer too small for replay, using only current experiences")
            return current_experiences

        # Calculate number of replay samples
        n_current = len(current_experiences)
        n_replay = int(n_current * self.replay_ratio)

        if n_replay == 0:
            return current_experiences

        # Sample replay experiences based on priority mode
        replay_experiences = self._sample_replay(n_replay, task_idx)

        # Combine current and replay
        augmented = current_experiences + replay_experiences

        logger.debug(
            f"Augmented experiences: {n_current} current + {len(replay_experiences)} replay "
            f"= {len(augmented)} total"
        )

        return augmented

    def after_task(
        self,
        task_idx: int,
        task_result,
        agent,
    ) -> None:
        """
        Update task difficulty estimates after task completion.

        Args:
            task_idx: Completed task index
            task_result: Result object from task training
            agent: The agent that was trained
        """
        # Estimate task difficulty from success rate
        # Lower success rate = higher difficulty = higher priority
        success_rate = task_result.train_success_rate
        difficulty = 1.0 - success_rate  # Invert: harder tasks have higher value

        self.task_difficulties[task_idx] = difficulty

        logger.info(
            f"Task {task_idx} completed: success_rate={success_rate:.3f}, "
            f"difficulty={difficulty:.3f}"
        )

    def _sample_replay(self, n: int, current_task_idx: int) -> List[Experience]:
        """
        Sample replay experiences based on priority mode.

        Args:
            n: Number of samples to draw
            current_task_idx: Current task index (to exclude from replay)

        Returns:
            List of sampled experiences
        """
        # Get experiences from previous tasks only
        previous_task_indices = [
            idx for idx in self.buffer.get_task_indices()
            if idx < current_task_idx
        ]

        if not previous_task_indices:
            return []

        # Select sampling strategy
        if self.priority_mode == "uniform":
            samples = self._sample_uniform(n, previous_task_indices)
        elif self.priority_mode == "difficulty":
            samples = self._sample_by_difficulty(n, previous_task_indices)
        elif self.priority_mode == "diversity":
            samples = self._sample_diverse(n, previous_task_indices)
        elif self.priority_mode == "recent":
            samples = self._sample_recent(n, previous_task_indices)
        else:
            logger.warning(f"Unknown priority mode: {self.priority_mode}, using uniform")
            samples = self._sample_uniform(n, previous_task_indices)

        # Apply success balancing if enabled
        if self.balance_success:
            samples = self._balance_success(samples, n)

        return samples

    def _sample_uniform(self, n: int, task_indices: List[int]) -> List[Experience]:
        """Uniform random sampling from previous tasks."""
        return self.buffer.sample_per_task(
            n_per_task=max(1, n // len(task_indices)),
            task_indices=task_indices,
        )[:n]

    def _sample_by_difficulty(self, n: int, task_indices: List[int]) -> List[Experience]:
        """Sample with priority based on task difficulty."""
        # Create weight function based on task difficulty
        def weight_fn(exp: Experience) -> float:
            # Get task difficulty
            task_diff = self.task_difficulties.get(exp.task_idx, 0.5)

            # Boost weight for harder tasks
            # Also boost weight for failed experiences (they're harder)
            failure_boost = 1.5 if not exp.success else 1.0

            return task_diff * failure_boost

        # Sample with priority
        candidates = []
        for task_idx in task_indices:
            candidates.extend(self.buffer.sample_by_task_idx(task_idx, n=None))

        # If we have fewer candidates than needed, return all
        if len(candidates) <= n:
            return candidates

        # Create temporary buffer for priority sampling
        temp_buffer = ExperienceBuffer()
        temp_buffer.add_batch(candidates)

        return temp_buffer.sample_with_priority(n=n, weight_fn=weight_fn)

    def _sample_diverse(self, n: int, task_indices: List[int]) -> List[Experience]:
        """Sample to maximize diversity across tasks."""
        # Get all candidates
        candidates = []
        for task_idx in task_indices:
            candidates.extend(self.buffer.sample_by_task_idx(task_idx, n=None))

        if len(candidates) <= n:
            return candidates

        # Use diversity sampling
        temp_buffer = ExperienceBuffer()
        temp_buffer.add_batch(candidates)

        return temp_buffer.sample_diverse(n=n, diversity_key=self.diversity_key)

    def _sample_recent(self, n: int, task_indices: List[int]) -> List[Experience]:
        """Sample with bias towards recent tasks."""
        # Get all candidates
        candidates = []
        for task_idx in task_indices:
            candidates.extend(self.buffer.sample_by_task_idx(task_idx, n=None))

        if len(candidates) <= n:
            return candidates

        # Use recency-biased sampling
        temp_buffer = ExperienceBuffer()
        temp_buffer.add_batch(candidates)

        return temp_buffer.sample_recent(n=n, recent_ratio=self.recent_ratio)

    def _balance_success(
        self,
        samples: List[Experience],
        target_n: int,
    ) -> List[Experience]:
        """
        Balance successful and failed experiences in samples.

        Args:
            samples: Input samples
            target_n: Target number of samples

        Returns:
            Balanced sample list
        """
        successful = [exp for exp in samples if exp.success]
        failed = [exp for exp in samples if not exp.success]

        # Target 50/50 split
        n_success = min(len(successful), target_n // 2)
        n_fail = min(len(failed), target_n - n_success)

        # If not enough failures, use more successes
        if n_fail < target_n // 2:
            n_success = min(len(successful), target_n - n_fail)

        balanced = []
        if successful:
            balanced.extend(np.random.choice(successful, n_success, replace=False).tolist())
        if failed:
            balanced.extend(np.random.choice(failed, n_fail, replace=False).tolist())

        return balanced

    def _compute_priority(self, exp: Experience) -> float:
        """
        Compute priority for a single experience.

        Args:
            exp: Experience to prioritize

        Returns:
            Priority value (higher = more important)
        """
        # Base priority
        priority = 1.0

        # Boost failed experiences (they're harder to learn)
        if not exp.success:
            priority *= 1.5

        # Boost low-reward experiences
        if exp.reward is not None and exp.reward < 0.5:
            priority *= 1.3

        # Boost rare tasks (based on buffer statistics)
        stats = self.buffer.get_statistics()
        task_count = stats['task_distribution'].get(exp.task_id, 1)
        total = stats['total_experiences']
        if total > 0:
            task_freq = task_count / total
            # Rarer tasks get higher priority
            priority *= (1.0 - task_freq + 0.1)  # +0.1 to avoid zero

        return priority

    def save_state(self, path: str) -> None:
        """Save method state to disk."""
        import json
        from pathlib import Path

        save_path = Path(path) / "enhanced_replay_state.json"
        save_path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            "config": self.config,
            "task_difficulties": self.task_difficulties,
            "experience_priorities": {
                str(k): v for k, v in self.experience_priorities.items()
            },
        }

        with open(save_path, 'w') as f:
            json.dump(state, f, indent=2)

        logger.info(f"EnhancedReplayMethod state saved to {save_path}")

    def load_state(self, path: str) -> None:
        """Load method state from disk."""
        import json
        from pathlib import Path

        load_path = Path(path) / "enhanced_replay_state.json"

        if not load_path.exists():
            logger.warning(f"No saved state found at {load_path}")
            return

        with open(load_path, 'r') as f:
            state = json.load(f)

        self.config = state.get("config", self.config)
        self.task_difficulties = state.get("task_difficulties", {})
        # Convert string keys back to int
        self.experience_priorities = {
            int(k): v for k, v in state.get("experience_priorities", {}).items()
        }

        logger.info(f"EnhancedReplayMethod state loaded from {load_path}")
