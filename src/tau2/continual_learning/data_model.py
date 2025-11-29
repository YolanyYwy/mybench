"""
Data models for continual learning framework.
"""

from dataclasses import dataclass, field
from typing import Any, Optional

from tau2.data_model.message import Message
from tau2.data_model.simulation import SimulationRun
from tau2.data_model.tasks import Task


@dataclass
class Experience:
    """A single experience/trajectory from task execution."""

    task_id: str
    task_idx: int  # Index in the continual learning task sequence
    domain: str  # Domain this task belongs to
    trial: int
    messages: list[Message]
    reward: float
    success: bool
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_simulation_run(
        cls, simulation_run: SimulationRun, task_idx: int, domain: str
    ) -> "Experience":
        """Create an Experience from a SimulationRun."""
        return cls(
            task_id=simulation_run.task_id,
            task_idx=task_idx,
            domain=domain,
            trial=simulation_run.trial,
            messages=simulation_run.messages,
            reward=simulation_run.reward_info.reward,
            success=simulation_run.reward_info.reward >= 0.999999,
            metadata={
                "termination_reason": simulation_run.termination_reason,
                "agent_cost": simulation_run.agent_cost,
            },
        )


@dataclass
class TaskResult:
    """Results from training/evaluating on a single task."""

    task_id: str
    task_idx: int
    domain: str  # Domain this task belongs to
    task: Task
    train_experiences: list[Experience]
    eval_experiences: list[Experience]
    avg_train_reward: float
    avg_eval_reward: float
    train_success_rate: float
    eval_success_rate: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ContinualLearningResults:
    """Complete results from a continual learning run."""

    task_sequence: list[str]  # Ordered list of task IDs
    domain_sequence: list[str]  # Ordered list of domains
    task_results: list[TaskResult]
    cl_metrics: dict[str, float]  # Continual learning metrics
    baseline_results: Optional[dict[str, dict[str, float]]] = None  # Per-domain baseline
    method_name: str = ""
    method_config: dict[str, Any] = field(default_factory=dict)
    run_config: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "task_sequence": self.task_sequence,
            "domain_sequence": self.domain_sequence,
            "task_results": [
                {
                    "task_id": tr.task_id,
                    "task_idx": tr.task_idx,
                    "domain": tr.domain,
                    "avg_train_reward": tr.avg_train_reward,
                    "avg_eval_reward": tr.avg_eval_reward,
                    "train_success_rate": tr.train_success_rate,
                    "eval_success_rate": tr.eval_success_rate,
                    "metadata": tr.metadata,
                }
                for tr in self.task_results
            ],
            "cl_metrics": self.cl_metrics,
            "baseline_results": self.baseline_results,
            "method_name": self.method_name,
            "method_config": self.method_config,
            "run_config": self.run_config,
            "metadata": self.metadata,
        }


@dataclass
class CLTrainerConfig:
    """Configuration for continual learning trainer."""

    # Task configuration - MULTI-DOMAIN SUPPORT
    domain_names: list[str] = field(default_factory=lambda: ["airline", "retail", "telecom"])
    task_split_name: str = "train"
    num_tasks_per_domain: Optional[int] = None  # Number of tasks per domain (None = all)
    task_order: str = "sequential"  # 'sequential', 'random', or 'custom'
    domain_order: str = "sequential"  # Order of domains: 'sequential', 'random', 'custom'
    custom_domain_order: Optional[list[str]] = None  # For custom domain ordering

    # Training configuration
    num_train_trials: int = 4  # Trials per task during training
    num_eval_trials: int = 4  # Trials per task during evaluation
    eval_frequency: int = 1  # Evaluate every N tasks
    eval_all_previous: bool = True  # Evaluate on all previous tasks

    # Baseline evaluation
    run_baseline: bool = True  # Run baseline evaluation before continual learning
    baseline_num_trials: int = 4  # Trials for baseline evaluation

    # Agent configuration
    agent_type: str = "llm_agent"
    agent_llm: str = "gpt-4o-mini"
    agent_llm_temperature: float = 0.0
    user_llm: str = "gpt-4o-mini"
    user_llm_temperature: float = 0.0

    # System configuration
    max_steps: int = 200
    max_errors: int = 10
    max_concurrency: int = 3
    seed: int = 300

    # Continual learning method
    cl_method_name: str = "naive"
    cl_method_config: dict[str, Any] = field(default_factory=dict)

    # Save/load paths
    save_dir: Optional[str] = None
    checkpoint_frequency: int = 1  # Save checkpoint every N tasks

    # Backward compatibility
    @property
    def domain_name(self) -> str:
        """Backward compatibility: return first domain if only one domain."""
        return self.domain_names[0] if len(self.domain_names) == 1 else None
