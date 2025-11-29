"""
Naive continual learning method (Sequential Fine-tuning).

This serves as a baseline where the agent simply learns tasks sequentially
without any continual learning strategy. It will suffer from catastrophic forgetting.
"""

from typing import Any, Optional

from loguru import logger

from tau2.continual_learning.buffer import ExperienceBuffer
from tau2.continual_learning.methods.base import ContinualLearningMethod


class NaiveMethod(ContinualLearningMethod):
    """
    Naive sequential learning (baseline).

    The agent learns each task sequentially without any special
    continual learning techniques. This serves as a baseline to
    measure catastrophic forgetting.
    """

    def __init__(
        self,
        buffer: ExperienceBuffer,
        config: Optional[dict[str, Any]] = None,
    ):
        """
        Initialize naive method.

        Args:
            buffer: Experience buffer
            config: Configuration (not used for naive method)
        """
        super().__init__(buffer, config or {})
        self.store_experiences = config.get("store_experiences", True) if config else True

    def get_name(self) -> str:
        """Return method name."""
        return "naive"

    def after_training_step(self, task_idx, agent, experiences, step):
        """Store experiences if configured."""
        if self.store_experiences:
            super().after_training_step(task_idx, agent, experiences, step)
        # Otherwise, don't store anything (pure sequential learning)

    def get_augmented_experiences(self, current_experiences, task_idx):
        """No augmentation - just return current experiences."""
        return current_experiences
