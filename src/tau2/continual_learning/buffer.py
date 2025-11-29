"""
Experience buffer for storing and sampling historical experiences.
"""

import random
from typing import Optional

from loguru import logger

from tau2.continual_learning.data_model import Experience


class ExperienceBuffer:
    """
    Buffer for storing experiences from task execution.
    Supports various sampling strategies for continual learning methods.
    """

    def __init__(self, max_size: Optional[int] = None, seed: int = 300):
        """
        Initialize experience buffer.

        Args:
            max_size: Maximum number of experiences to store (None = unlimited)
            seed: Random seed for sampling
        """
        self.max_size = max_size
        self.experiences: list[Experience] = []
        self.seed = seed
        self._rng = random.Random(seed)

    def add(self, experience: Experience) -> None:
        """Add a single experience to the buffer."""
        self.experiences.append(experience)

        # Remove oldest if exceeding max size
        if self.max_size is not None and len(self.experiences) > self.max_size:
            self.experiences.pop(0)
            logger.debug(f"Buffer full, removed oldest experience")

    def add_batch(self, experiences: list[Experience]) -> None:
        """Add multiple experiences to the buffer."""
        for exp in experiences:
            self.add(exp)

    def sample(self, n: int) -> list[Experience]:
        """
        Randomly sample n experiences from the buffer.

        Args:
            n: Number of experiences to sample

        Returns:
            List of sampled experiences
        """
        if n > len(self.experiences):
            logger.warning(
                f"Requested {n} samples but buffer only has {len(self.experiences)}"
            )
            n = len(self.experiences)

        return self._rng.sample(self.experiences, n)

    def sample_by_task(
        self, task_id: str, n: Optional[int] = None
    ) -> list[Experience]:
        """
        Sample experiences from a specific task.

        Args:
            task_id: Task ID to sample from
            n: Number of samples (None = all)

        Returns:
            List of experiences from the task
        """
        task_experiences = [exp for exp in self.experiences if exp.task_id == task_id]

        if n is None or n >= len(task_experiences):
            return task_experiences

        return self._rng.sample(task_experiences, n)

    def sample_by_task_idx(
        self, task_idx: int, n: Optional[int] = None
    ) -> list[Experience]:
        """
        Sample experiences from a specific task index.

        Args:
            task_idx: Task index in the continual learning sequence
            n: Number of samples (None = all)

        Returns:
            List of experiences from the task
        """
        task_experiences = [exp for exp in self.experiences if exp.task_idx == task_idx]

        if n is None or n >= len(task_experiences):
            return task_experiences

        return self._rng.sample(task_experiences, n)

    def sample_successful(self, n: int) -> list[Experience]:
        """Sample only successful experiences."""
        successful = [exp for exp in self.experiences if exp.success]

        if n > len(successful):
            logger.warning(
                f"Requested {n} successful samples but buffer only has {len(successful)}"
            )
            n = len(successful)

        return self._rng.sample(successful, n) if successful else []

    def sample_per_task(
        self, n_per_task: int, task_indices: Optional[list[int]] = None
    ) -> list[Experience]:
        """
        Sample n experiences per task.

        Args:
            n_per_task: Number of experiences to sample per task
            task_indices: Specific task indices to sample from (None = all tasks)

        Returns:
            List of sampled experiences
        """
        if task_indices is None:
            task_indices = list(set(exp.task_idx for exp in self.experiences))

        samples = []
        for task_idx in task_indices:
            task_samples = self.sample_by_task_idx(task_idx, n_per_task)
            samples.extend(task_samples)

        return samples

    def get_all(self) -> list[Experience]:
        """Get all experiences in the buffer."""
        return self.experiences.copy()

    def get_task_ids(self) -> list[str]:
        """Get unique task IDs in the buffer."""
        return list(set(exp.task_id for exp in self.experiences))

    def get_task_indices(self) -> list[int]:
        """Get unique task indices in the buffer."""
        return sorted(set(exp.task_idx for exp in self.experiences))

    def clear(self) -> None:
        """Clear all experiences from the buffer."""
        self.experiences.clear()
        logger.info("Experience buffer cleared")

    def __len__(self) -> int:
        """Return number of experiences in buffer."""
        return len(self.experiences)

    def __repr__(self) -> str:
        """String representation of buffer."""
        return (
            f"ExperienceBuffer(size={len(self)}, "
            f"max_size={self.max_size}, "
            f"tasks={len(self.get_task_ids())})"
        )
