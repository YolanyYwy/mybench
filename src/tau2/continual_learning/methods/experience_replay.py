"""
Experience Replay continual learning method.

Classic continual learning approach that stores past experiences
and replays them during training on new tasks.
"""

from typing import Any, Optional

from loguru import logger

from tau2.continual_learning.buffer import ExperienceBuffer
from tau2.continual_learning.data_model import Experience
from tau2.continual_learning.methods.base import ContinualLearningMethod


class ExperienceReplayMethod(ContinualLearningMethod):
    """
    Experience Replay method for continual learning.

    Stores experiences from previous tasks and replays them
    when training on new tasks to mitigate catastrophic forgetting.
    """

    def __init__(
        self,
        buffer: ExperienceBuffer,
        config: Optional[dict[str, Any]] = None,
    ):
        """
        Initialize experience replay method.

        Args:
            buffer: Experience buffer for storing past experiences
            config: Configuration dictionary with the following options:
                - replay_size: Number of experiences to replay per task (default: 10)
                - replay_strategy: How to sample replay data (default: "uniform")
                    - "uniform": Uniformly sample from all previous tasks
                    - "per_task": Sample equally from each previous task
                    - "successful_only": Only replay successful experiences
                    - "recent": Prioritize recent tasks
                - samples_per_task: For "per_task" strategy, samples per task (default: 5)
        """
        default_config = {
            "replay_size": 10,
            "replay_strategy": "uniform",
            "samples_per_task": 5,
        }
        if config:
            default_config.update(config)
        super().__init__(buffer, default_config)

        self.replay_size = self.config["replay_size"]
        self.replay_strategy = self.config["replay_strategy"]
        self.samples_per_task = self.config["samples_per_task"]

    def get_name(self) -> str:
        """Return method name."""
        return f"experience_replay_{self.replay_strategy}"

    def get_augmented_experiences(
        self,
        current_experiences: list[Experience],
        task_idx: int,
    ) -> list[Experience]:
        """
        Augment current experiences with replayed historical experiences.

        Args:
            current_experiences: Experiences from current task
            task_idx: Current task index

        Returns:
            Combined list of current and replayed experiences
        """
        if task_idx == 0:
            # First task, no replay data available
            return current_experiences

        # Sample replay experiences based on strategy
        if self.replay_strategy == "uniform":
            replay_experiences = self._sample_uniform()
        elif self.replay_strategy == "per_task":
            replay_experiences = self._sample_per_task(task_idx)
        elif self.replay_strategy == "successful_only":
            replay_experiences = self._sample_successful()
        elif self.replay_strategy == "recent":
            replay_experiences = self._sample_recent(task_idx)
        else:
            logger.warning(f"Unknown replay strategy: {self.replay_strategy}, using uniform")
            replay_experiences = self._sample_uniform()

        logger.info(
            f"Replaying {len(replay_experiences)} experiences "
            f"(strategy: {self.replay_strategy})"
        )

        # Combine current and replay experiences
        return current_experiences + replay_experiences

    def _sample_uniform(self) -> list[Experience]:
        """Uniformly sample from all previous experiences."""
        return self.buffer.sample(self.replay_size)

    def _sample_per_task(self, current_task_idx: int) -> list[Experience]:
        """Sample equally from each previous task."""
        previous_task_indices = list(range(current_task_idx))
        return self.buffer.sample_per_task(
            self.samples_per_task, previous_task_indices
        )

    def _sample_successful(self) -> list[Experience]:
        """Sample only successful experiences."""
        return self.buffer.sample_successful(self.replay_size)

    def _sample_recent(self, current_task_idx: int) -> list[Experience]:
        """Prioritize recent tasks (weighted sampling)."""
        # Sample more from recent tasks
        # For simplicity, sample proportionally more from recent half
        previous_indices = list(range(current_task_idx))

        if len(previous_indices) <= 2:
            return self._sample_uniform()

        mid_point = len(previous_indices) // 2
        recent_indices = previous_indices[mid_point:]
        old_indices = previous_indices[:mid_point]

        # 70% from recent, 30% from old
        n_recent = int(self.replay_size * 0.7)
        n_old = self.replay_size - n_recent

        recent_samples = self.buffer.sample_per_task(
            max(1, n_recent // len(recent_indices)), recent_indices
        )[:n_recent]

        old_samples = self.buffer.sample_per_task(
            max(1, n_old // len(old_indices)) if old_indices else 0, old_indices
        )[:n_old]

        return recent_samples + old_samples

    def get_method_info(self) -> dict[str, Any]:
        """Get method information."""
        info = super().get_method_info()
        info.update(
            {
                "replay_size": self.replay_size,
                "replay_strategy": self.replay_strategy,
            }
        )
        return info
