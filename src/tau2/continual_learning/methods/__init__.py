"""
Methods package for continual learning.
"""

from tau2.continual_learning.methods.base import ContinualLearningMethod
from tau2.continual_learning.methods.experience_replay import ExperienceReplayMethod
from tau2.continual_learning.methods.naive import NaiveMethod

__all__ = [
    "ContinualLearningMethod",
    "NaiveMethod",
    "ExperienceReplayMethod",
]
