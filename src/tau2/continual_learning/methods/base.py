"""
Base class for continual learning methods.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional

from tau2.agent.base import BaseAgent
from tau2.continual_learning.buffer import ExperienceBuffer
from tau2.continual_learning.data_model import Experience, TaskResult


class ContinualLearningMethod(ABC):
    """
    Abstract base class for continual learning methods.

    This class defines the interface that all CL methods must implement.
    Methods are called at different stages of the training process:
    1. before_task: Called before training on a new task
    2. before_training_step: Called before each training step
    3. after_training_step: Called after each training step
    4. after_task: Called after completing a task
    5. get_augmented_experiences: Optionally augment training data with replay
    """

    def __init__(
        self,
        buffer: ExperienceBuffer,
        config: Optional[dict[str, Any]] = None,
    ):
        """
        Initialize the continual learning method.

        Args:
            buffer: Experience buffer for storing/retrieving experiences
            config: Method-specific configuration
        """
        self.buffer = buffer
        self.config = config or {}
        self.current_task_idx: Optional[int] = None

    @abstractmethod
    def get_name(self) -> str:
        """Return the name of this CL method."""
        pass

    def before_task(
        self,
        task_idx: int,
        task_id: str,
        agent: BaseAgent,
    ) -> None:
        """
        Called before starting training on a new task.

        Args:
            task_idx: Index of the task in the continual learning sequence
            task_id: Task ID
            agent: The agent that will be trained
        """
        self.current_task_idx = task_idx

    def before_training_step(
        self,
        task_idx: int,
        agent: BaseAgent,
        step: int,
    ) -> None:
        """
        Called before each training step within a task.

        Args:
            task_idx: Current task index
            agent: The agent being trained
            step: Training step number within the current task
        """
        pass

    def after_training_step(
        self,
        task_idx: int,
        agent: BaseAgent,
        experiences: list[Experience],
        step: int,
    ) -> None:
        """
        Called after each training step within a task.

        Args:
            task_idx: Current task index
            agent: The agent being trained
            experiences: New experiences collected in this step
            step: Training step number within the current task
        """
        # Default: add experiences to buffer
        self.buffer.add_batch(experiences)

    def after_task(
        self,
        task_idx: int,
        task_result: TaskResult,
        agent: BaseAgent,
    ) -> None:
        """
        Called after completing training on a task.

        Args:
            task_idx: Index of the completed task
            task_result: Results from training on this task
            agent: The trained agent
        """
        pass

    def get_augmented_experiences(
        self,
        current_experiences: list[Experience],
        task_idx: int,
    ) -> list[Experience]:
        """
        Optionally augment current training data with historical experiences.

        This is the main method where replay-based CL methods add historical data.

        Args:
            current_experiences: Experiences from the current task
            task_idx: Current task index

        Returns:
            Augmented list of experiences (current + historical)
        """
        # Default: no augmentation, just return current experiences
        return current_experiences

    def get_method_info(self) -> dict[str, Any]:
        """
        Get information about this CL method (for logging/reporting).

        Returns:
            Dictionary with method name and configuration
        """
        return {
            "name": self.get_name(),
            "config": self.config,
            "buffer_size": len(self.buffer),
        }

    def save_state(self, path: str) -> None:
        """
        Save method-specific state (e.g., importance weights for EWC).

        Args:
            path: Directory path to save state
        """
        # Default: nothing to save
        pass

    def load_state(self, path: str) -> None:
        """
        Load method-specific state.

        Args:
            path: Directory path to load state from
        """
        # Default: nothing to load
        pass
