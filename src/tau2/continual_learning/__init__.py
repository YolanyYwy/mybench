"""
Continual Learning framework for tool use agents.

This module provides infrastructure for continual learning experiments
on the tau2 benchmark tasks.
"""

from tau2.continual_learning.buffer import ExperienceBuffer
from tau2.continual_learning.data_model import (
    CLTrainerConfig,
    ContinualLearningResults,
    Experience,
    TaskResult,
)
from tau2.continual_learning.methods.base import ContinualLearningMethod
from tau2.continual_learning.methods.experience_replay import ExperienceReplayMethod
from tau2.continual_learning.methods.naive import NaiveMethod
from tau2.continual_learning.metrics import compute_cl_metrics, print_cl_metrics
from tau2.continual_learning.trainer import (
    ContinualLearningTrainer,
    create_cl_trainer,
)

__all__ = [
    # Core classes
    "ContinualLearningTrainer",
    "ContinualLearningMethod",
    "ExperienceBuffer",
    # Data models
    "CLTrainerConfig",
    "ContinualLearningResults",
    "Experience",
    "TaskResult",
    # Methods
    "NaiveMethod",
    "ExperienceReplayMethod",
    # Utilities
    "create_cl_trainer",
    "compute_cl_metrics",
    "print_cl_metrics",
]
