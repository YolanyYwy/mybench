# Continual Learning Framework for Tool Use Agents

This directory contains a continual learning framework built on top of the tau2-bench benchmark. The framework enables training and evaluating agents on sequential tool use tasks while measuring continual learning performance.

## Overview

The continual learning framework provides:

1. **Flexible Training Pipeline**: Train agents on sequences of tool use tasks
2. **CL Method Interface**: Easy-to-extend interface for implementing different continual learning methods
3. **Experience Buffer**: Storage and sampling of historical experiences
4. **CL Metrics**: Comprehensive metrics including forgetting, forward/backward transfer
5. **Built-in Methods**: Naive baseline and Experience Replay implementations

## Architecture

### Core Components

```
continual_learning/
├── __init__.py              # Module exports
├── trainer.py               # Main training orchestrator
├── buffer.py                # Experience storage and sampling
├── data_model.py           # Data structures for CL
├── metrics.py              # CL-specific metrics
├── cli.py                  # Command-line interface
└── methods/                # CL method implementations
    ├── base.py             # Abstract base class
    ├── naive.py            # Sequential learning baseline
    └── experience_replay.py # Experience replay method
```

### Key Classes

- **`ContinualLearningTrainer`**: Orchestrates the training process, manages task sequences, evaluates performance
- **`ContinualLearningMethod`**: Abstract base class for CL methods with hooks at different training stages
- **`ExperienceBuffer`**: Stores and samples experiences with various sampling strategies
- **`CLTrainerConfig`**: Configuration for training experiments

## Quick Start

### 1. Command Line Interface

The easiest way to run continual learning experiments:

```bash
# Naive baseline (sequential learning)
python -m tau2.continual_learning.cli train \
    --domain airline \
    --method naive \
    --num-tasks 5 \
    --save-dir ./results/naive_airline

# Experience Replay
python -m tau2.continual_learning.cli train \
    --domain airline \
    --method experience_replay \
    --num-tasks 5 \
    --replay-size 10 \
    --replay-strategy uniform \
    --save-dir ./results/replay_airline
```

### 2. Python API

For more control, use the Python API:

```python
from tau2.continual_learning import (
    CLTrainerConfig,
    ContinualLearningTrainer,
    ExperienceBuffer,
    ExperienceReplayMethod,
)

# Configure experiment
config = CLTrainerConfig(
    domain_name="airline",
    task_split_name="train",
    num_tasks=10,
    num_train_trials=4,
    num_eval_trials=4,
    agent_llm="gpt-4o-mini",
    save_dir="./results/my_experiment",
)

# Create experience buffer and CL method
buffer = ExperienceBuffer(max_size=1000, seed=300)
cl_method = ExperienceReplayMethod(
    buffer,
    config={
        "replay_size": 10,
        "replay_strategy": "uniform",
    }
)

# Create and run trainer
trainer = ContinualLearningTrainer(config, cl_method)
results = trainer.train()

# View results
print(f"Average Accuracy: {results.cl_metrics['average_accuracy']:.3f}")
print(f"Forgetting: {results.cl_metrics['forgetting']:.3f}")
```

## Implementing Custom CL Methods

The framework is designed for easy extension. Here's how to implement your own continual learning method:

### Step 1: Create a new method class

```python
from tau2.continual_learning.methods.base import ContinualLearningMethod
from tau2.continual_learning.data_model import Experience, TaskResult
from tau2.agent.base import BaseAgent

class MyCustomMethod(ContinualLearningMethod):
    """Your custom continual learning method."""

    def __init__(self, buffer, config=None):
        super().__init__(buffer, config or {})
        # Initialize method-specific state
        self.my_parameter = self.config.get("my_parameter", 1.0)

    def get_name(self) -> str:
        return "my_custom_method"

    def before_task(self, task_idx: int, task_id: str, agent: BaseAgent):
        """Called before training on a new task."""
        super().before_task(task_idx, task_id, agent)
        # Add custom logic here
        print(f"Starting task {task_id}")

    def after_training_step(
        self,
        task_idx: int,
        agent: BaseAgent,
        experiences: list[Experience],
        step: int
    ):
        """Called after each training step."""
        # Store experiences
        self.buffer.add_batch(experiences)
        # Add custom logic (e.g., update importance weights)

    def get_augmented_experiences(
        self,
        current_experiences: list[Experience],
        task_idx: int,
    ) -> list[Experience]:
        """Add replay/augmentation data."""
        if task_idx == 0:
            return current_experiences

        # Sample historical data
        replay_data = self.buffer.sample(10)

        # Return combined data
        return current_experiences + replay_data

    def after_task(
        self,
        task_idx: int,
        task_result: TaskResult,
        agent: BaseAgent,
    ):
        """Called after completing a task."""
        # Add custom logic (e.g., consolidate memory, update regularization)
        pass
```

### Step 2: Use your method

```python
from my_module import MyCustomMethod

buffer = ExperienceBuffer()
my_method = MyCustomMethod(
    buffer,
    config={"my_parameter": 2.0}
)

trainer = ContinualLearningTrainer(config, my_method)
results = trainer.train()
```

## Method Hooks

The `ContinualLearningMethod` base class provides several hooks:

| Hook | When Called | Purpose |
|------|-------------|---------|
| `before_task()` | Before training on a new task | Initialize task-specific state |
| `before_training_step()` | Before each training step | Prepare for training |
| `after_training_step()` | After each training step | Store experiences, update state |
| `get_augmented_experiences()` | During training | Add replay/augmentation data |
| `after_task()` | After completing a task | Consolidate, cleanup |
| `save_state()` / `load_state()` | Checkpoint save/load | Persist method state |

## Continual Learning Metrics

The framework computes several CL metrics:

- **Average Accuracy**: Mean success rate across all tasks
- **Final Performance**: Success rate after training on all tasks
- **Forgetting**: How much performance drops on previous tasks
- **Backward Transfer**: Impact of new tasks on old tasks (negative forgetting)
- **Forward Transfer**: How previous tasks help with new tasks (requires baseline)
- **Learning Curve Area**: Cumulative performance throughout training

## Examples

### Example 1: Compare Methods

```python
from tau2.continual_learning import (
    CLTrainerConfig,
    ContinualLearningTrainer,
    ExperienceBuffer,
    NaiveMethod,
    ExperienceReplayMethod,
)

config = CLTrainerConfig(
    domain_name="airline",
    num_tasks=10,
    save_dir="./results",
)

# Run naive baseline
buffer_naive = ExperienceBuffer()
naive_method = NaiveMethod(buffer_naive)
trainer_naive = ContinualLearningTrainer(config, naive_method)
results_naive = trainer_naive.train()

# Run experience replay
buffer_replay = ExperienceBuffer()
replay_method = ExperienceReplayMethod(buffer_replay, {"replay_size": 20})
trainer_replay = ContinualLearningTrainer(config, replay_method)
results_replay = trainer_replay.train()

# Compare
print(f"Naive Forgetting: {results_naive.cl_metrics['forgetting']:.3f}")
print(f"Replay Forgetting: {results_replay.cl_metrics['forgetting']:.3f}")
```

### Example 2: Custom Task Order

```python
config = CLTrainerConfig(
    domain_name="airline",
    task_order="custom",
    custom_task_order=["0", "5", "10", "2", "8"],
)
```

### Example 3: Different Replay Strategies

```python
strategies = ["uniform", "per_task", "successful_only", "recent"]

for strategy in strategies:
    buffer = ExperienceBuffer()
    method = ExperienceReplayMethod(
        buffer,
        {"replay_size": 10, "replay_strategy": strategy}
    )
    trainer = ContinualLearningTrainer(config, method)
    results = trainer.train()
    print(f"{strategy}: {results.cl_metrics['forgetting']:.3f}")
```

## Configuration Options

### CLTrainerConfig

```python
config = CLTrainerConfig(
    # Task configuration
    domain_name="airline",           # Domain to use
    task_split_name="train",         # train/test/base
    num_tasks=None,                  # Number of tasks (None=all)
    task_order="sequential",         # sequential/random/custom
    custom_task_order=None,          # List of task IDs for custom order

    # Training configuration
    num_train_trials=4,              # Trials per task during training
    num_eval_trials=4,               # Trials per task during evaluation
    eval_frequency=1,                # Evaluate every N tasks
    eval_all_previous=True,          # Evaluate on all previous tasks

    # Agent configuration
    agent_llm="gpt-4o-mini",        # LLM for agent
    agent_llm_temperature=0.0,       # Temperature for agent
    user_llm="gpt-4o-mini",         # LLM for user simulator
    user_llm_temperature=0.0,        # Temperature for user

    # System configuration
    max_steps=200,                   # Max steps per episode
    max_errors=10,                   # Max errors per episode
    max_concurrency=3,               # Parallel execution
    seed=300,                        # Random seed

    # CL method
    cl_method_name="naive",          # Method name (for logging)
    cl_method_config={},             # Method-specific config

    # Save/load
    save_dir=None,                   # Where to save results
    checkpoint_frequency=1,          # Save checkpoint every N tasks
)
```

## Advanced Usage

### Custom Agent Types

To use custom trainable agents (e.g., for actual fine-tuning):

```python
# 1. Implement a trainable agent that extends BaseAgent
# 2. Add update() method for training
# 3. Modify the trainer to call agent.update() with experiences

class MyTrainableAgent(BaseAgent):
    def update(self, experiences: list[Experience]):
        # Implement your training logic
        pass
```

### Multi-Domain Experiments

```python
domains = ["airline", "retail", "telecom"]

for domain in domains:
    config = CLTrainerConfig(domain_name=domain, ...)
    trainer = ContinualLearningTrainer(config, method)
    results = trainer.train()
```

## Tips for Adding New CL Methods

1. **Start with the base class**: Inherit from `ContinualLearningMethod`
2. **Use the buffer**: Leverage `ExperienceBuffer` for storage and sampling
3. **Implement key hooks**: At minimum, implement `get_name()` and `get_augmented_experiences()`
4. **Test incrementally**: Start with simple task sequences
5. **Add configurations**: Make your method flexible through the config dict
6. **Document**: Add docstrings explaining your method

## Common CL Methods to Implement

Here are some classic continual learning methods you could add:

- **EWC (Elastic Weight Consolidation)**: Regularization based on Fisher information
- **PackNet**: Allocate network capacity for each task
- **GEM (Gradient Episodic Memory)**: Constrain gradients using past data
- **A-GEM**: Averaged GEM for efficiency
- **MER (Meta-Experience Replay)**: Meta-learning with replay
- **iCaRL**: Nearest-mean-of-exemplars classifier
- **Progressive Neural Networks**: Lateral connections between task columns

## Troubleshooting

**Q: My method shows high forgetting**
- Check if experiences are being stored correctly in `after_training_step()`
- Verify replay data is being added in `get_augmented_experiences()`
- Try increasing replay buffer size or samples

**Q: Training is slow**
- Reduce `num_train_trials` or `num_eval_trials`
- Reduce `num_tasks` for faster iteration
- Use cheaper LLM models (gpt-4o-mini instead of gpt-4o)

**Q: How do I implement actual model fine-tuning?**
- The current framework focuses on LLM-based agents
- For fine-tuning, you'd need to:
  1. Create a trainable agent class
  2. Implement `update()` method
  3. Modify trainer to call agent updates
  4. Store model checkpoints

## Citation

If you use this continual learning framework, please cite:

```bibtex
@article{tau2bench,
  title={τ²-bench: Tool Use Benchmark},
  author={...},
  year={2024}
}
```

## Contributing

To add new CL methods:

1. Create a new file in `methods/` directory
2. Inherit from `ContinualLearningMethod`
3. Implement required methods
4. Add to `methods/__init__.py`
5. Add tests and documentation
6. Submit a pull request

## License

This framework is part of the tau2-bench project and follows the same license.
