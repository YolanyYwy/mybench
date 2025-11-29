"""
Experience buffer for storing and sampling historical experiences.
"""

import random
from typing import Optional, Dict, Callable
import numpy as np

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

    # ========== Enhanced Sampling Methods for Advanced CL ==========

    def sample_with_priority(
        self,
        n: int,
        weights: Optional[Dict[int, float]] = None,
        weight_fn: Optional[Callable[[Experience], float]] = None,
    ) -> list[Experience]:
        """
        Sample experiences with priority weighting.

        Args:
            n: Number of experiences to sample
            weights: Dict mapping experience index to weight (optional)
            weight_fn: Function to compute weight from experience (optional)

        Returns:
            List of sampled experiences with importance weighting
        """
        if not self.experiences:
            return []

        n = min(n, len(self.experiences))

        # Compute weights
        if weight_fn is not None:
            # Use weight function
            priority_weights = [weight_fn(exp) for exp in self.experiences]
        elif weights is not None:
            # Use provided weights dict
            priority_weights = [weights.get(i, 1.0) for i in range(len(self.experiences))]
        else:
            # Default: uniform weights
            priority_weights = [1.0] * len(self.experiences)

        # Normalize weights
        total_weight = sum(priority_weights)
        if total_weight == 0:
            probabilities = [1.0 / len(self.experiences)] * len(self.experiences)
        else:
            probabilities = [w / total_weight for w in priority_weights]

        # Sample with replacement using probabilities
        indices = self._rng.choices(
            range(len(self.experiences)),
            weights=probabilities,
            k=n
        )

        return [self.experiences[i] for i in indices]

    def sample_diverse(
        self,
        n: int,
        diversity_key: str = "task_idx",
    ) -> list[Experience]:
        """
        Sample experiences to maximize diversity.

        Args:
            n: Number of experiences to sample
            diversity_key: Attribute to use for diversity ('task_idx', 'domain', etc.)

        Returns:
            List of diverse experiences
        """
        if not self.experiences:
            return []

        n = min(n, len(self.experiences))

        # Group by diversity key
        groups = {}
        for exp in self.experiences:
            key_value = getattr(exp, diversity_key, None)
            if key_value not in groups:
                groups[key_value] = []
            groups[key_value].append(exp)

        # Sample evenly from each group
        samples = []
        group_keys = list(groups.keys())
        samples_per_group = max(1, n // len(groups))
        remaining = n

        for key in group_keys:
            group_samples = min(samples_per_group, len(groups[key]), remaining)
            samples.extend(self._rng.sample(groups[key], group_samples))
            remaining -= group_samples
            if remaining <= 0:
                break

        # Fill remaining with random samples if needed
        if len(samples) < n:
            available = [exp for exp in self.experiences if exp not in samples]
            if available:
                additional = self._rng.sample(available, min(n - len(samples), len(available)))
                samples.extend(additional)

        return samples[:n]

    def sample_recent(
        self,
        n: int,
        recent_ratio: float = 0.7,
    ) -> list[Experience]:
        """
        Sample experiences with bias towards recent ones.

        Args:
            n: Number of experiences to sample
            recent_ratio: Ratio of samples from recent half (default 0.7)

        Returns:
            List of experiences biased towards recent ones
        """
        if not self.experiences:
            return []

        n = min(n, len(self.experiences))

        # Split into recent and older
        split_idx = len(self.experiences) // 2
        older_exps = self.experiences[:split_idx]
        recent_exps = self.experiences[split_idx:]

        # Calculate samples from each group
        n_recent = int(n * recent_ratio)
        n_older = n - n_recent

        # Sample
        samples = []
        if recent_exps:
            samples.extend(self._rng.sample(recent_exps, min(n_recent, len(recent_exps))))
        if older_exps:
            samples.extend(self._rng.sample(older_exps, min(n_older, len(older_exps))))

        # Fill if needed
        if len(samples) < n:
            available = [exp for exp in self.experiences if exp not in samples]
            if available:
                samples.extend(self._rng.sample(available, min(n - len(samples), len(available))))

        return samples[:n]

    def sample_by_similarity(
        self,
        query_embedding: np.ndarray,
        n: int,
        embedding_fn: Optional[Callable[[Experience], np.ndarray]] = None,
        embeddings_cache: Optional[Dict[int, np.ndarray]] = None,
    ) -> list[Experience]:
        """
        Sample experiences similar to a query embedding.

        Args:
            query_embedding: Query embedding vector
            n: Number of experiences to sample
            embedding_fn: Function to get embedding from experience
            embeddings_cache: Pre-computed embeddings dict

        Returns:
            List of most similar experiences
        """
        if not self.experiences:
            return []

        n = min(n, len(self.experiences))

        # Get embeddings for all experiences
        if embeddings_cache is not None:
            embeddings = [embeddings_cache.get(i) for i in range(len(self.experiences))]
            # Filter out None values
            valid_indices = [i for i, emb in enumerate(embeddings) if emb is not None]
            embeddings = [embeddings[i] for i in valid_indices]
            experiences = [self.experiences[i] for i in valid_indices]
        elif embedding_fn is not None:
            embeddings = [embedding_fn(exp) for exp in self.experiences]
            valid_indices = [i for i, emb in enumerate(embeddings) if emb is not None]
            embeddings = [embeddings[i] for i in valid_indices]
            experiences = [self.experiences[i] for i in valid_indices]
        else:
            raise ValueError("Either embedding_fn or embeddings_cache must be provided")

        if not embeddings:
            return []

        # Compute similarities
        embeddings_array = np.array(embeddings)
        query_norm = np.linalg.norm(query_embedding)
        embeddings_norms = np.linalg.norm(embeddings_array, axis=1)

        # Cosine similarity
        similarities = np.dot(embeddings_array, query_embedding) / (
            embeddings_norms * query_norm + 1e-10
        )

        # Get top-k indices
        top_k_indices = np.argsort(similarities)[-n:][::-1]

        return [experiences[i] for i in top_k_indices]

    def get_statistics(self) -> Dict[str, any]:
        """
        Get buffer statistics for analysis.

        Returns:
            Dictionary with buffer statistics
        """
        if not self.experiences:
            return {
                "total_experiences": 0,
                "success_rate": 0.0,
                "tasks_represented": 0,
                "domains": [],
                "task_distribution": {},
                "domain_distribution": {},
            }

        successes = [exp.success for exp in self.experiences]
        tasks = [exp.task_id for exp in self.experiences]
        domains = [exp.domain for exp in self.experiences if hasattr(exp, 'domain') and exp.domain]

        # Task distribution
        task_dist = {}
        for task in tasks:
            task_dist[task] = task_dist.get(task, 0) + 1

        # Domain distribution
        domain_dist = {}
        for domain in domains:
            domain_dist[domain] = domain_dist.get(domain, 0) + 1

        return {
            "total_experiences": len(self.experiences),
            "success_rate": np.mean(successes) if successes else 0.0,
            "tasks_represented": len(set(tasks)),
            "domains": list(set(domains)),
            "task_distribution": task_dist,
            "domain_distribution": domain_dist,
            "avg_reward": np.mean([exp.reward for exp in self.experiences if exp.reward is not None]),
        }

    def filter_by_metadata(
        self,
        key: str,
        value: any,
    ) -> list[Experience]:
        """
        Filter experiences by metadata field.

        Args:
            key: Metadata key to filter by
            value: Value to match

        Returns:
            List of experiences matching the filter
        """
        return [
            exp for exp in self.experiences
            if exp.metadata and exp.metadata.get(key) == value
        ]

