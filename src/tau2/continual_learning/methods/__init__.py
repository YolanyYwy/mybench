"""
Methods package for continual learning.
"""

# Base and classic methods
from tau2.continual_learning.methods.base import ContinualLearningMethod
from tau2.continual_learning.methods.naive import NaiveMethod
from tau2.continual_learning.methods.experience_replay import ExperienceReplayMethod

# Advanced memory-based methods
from tau2.continual_learning.methods.enhanced_replay import EnhancedReplayMethod
from tau2.continual_learning.methods.rag_memory import RAGMemoryMethod

# Prompt-based methods
from tau2.continual_learning.methods.dynamic_prompting import DynamicPromptingMethod
from tau2.continual_learning.methods.cot_memory import CoTMemoryMethod

# Meta-learning methods
from tau2.continual_learning.methods.task_embedding import TaskEmbeddingMethod
from tau2.continual_learning.methods.tool_strategy import ToolStrategyMethod

# Hybrid methods
from tau2.continual_learning.methods.rag_prompt_hybrid import RAGPromptHybridMethod

__all__ = [
    # Base and classic
    "ContinualLearningMethod",
    "NaiveMethod",
    "ExperienceReplayMethod",

    # Memory-based
    "EnhancedReplayMethod",
    "RAGMemoryMethod",

    # Prompt-based
    "DynamicPromptingMethod",
    "CoTMemoryMethod",

    # Meta-learning
    "TaskEmbeddingMethod",
    "ToolStrategyMethod",

    # Hybrid
    "RAGPromptHybridMethod",
]
