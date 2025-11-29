# Continual Learning Framework - Quick Start Guide

This guide helps you get started with the continual learning framework built on tau2-bench.

## What is This Framework?

This framework extends tau2-bench to evaluate agents on **continual learning** scenarios for tool use tasks. Instead of evaluating agents on individual tasks, we evaluate how well agents can:

1. **Learn sequentially** from a stream of tasks
2. **Retain knowledge** from previous tasks (avoid catastrophic forgetting)
3. **Transfer knowledge** to new tasks

## Installation

The framework is integrated into tau2-bench. Make sure you have the base dependencies installed:

```bash
pip install -e .
```

## Framework Structure

```
src/tau2/continual_learning/
├── trainer.py              # Main training loop
├── buffer.py               # Experience storage
├── data_model.py          # Data structures
├── metrics.py             # CL metrics
├── cli.py                 # Command-line interface
└── methods/               # CL method implementations
    ├── base.py            # Abstract base class
    ├── naive.py           # Sequential learning baseline
    └── experience_replay.py  # Experience replay method
```

## Quick Start: Run Your First Experiment

### Option 1: Command Line

```bash
# Basic experiment with naive baseline
python -m tau2.continual_learning.cli train \
    --domain mock \
    --method naive \
    --num-tasks 3 \
    --num-train-trials 2 \
    --save-dir ./cl_results

# With experience replay
python -m tau2.continual_learning.cli train \
    --domain airline \
    --method experience_replay \
    --num-tasks 5 \
    --replay-size 10 \
    --replay-strategy uniform \
    --save-dir ./cl_results
```

### Option 2: Python Script

```python
from tau2.continual_learning import (
    CLTrainerConfig,
    ContinualLearningTrainer,
    ExperienceBuffer,
    NaiveMethod,
)

# 1. Configure experiment
config = CLTrainerConfig(
    domain_name="airline",
    num_tasks=5,
    num_train_trials=4,
    save_dir="./my_experiment",
)

# 2. Create CL method
buffer = ExperienceBuffer()
method = NaiveMethod(buffer)

# 3. Run training
trainer = ContinualLearningTrainer(config, method)
results = trainer.train()

# 4. View results
print(f"Forgetting: {results.cl_metrics['forgetting']:.3f}")
```

## Understanding the Output

The framework computes these key metrics:

- **Average Accuracy**: Mean success rate across all tasks
- **Forgetting**: How much performance drops on previous tasks (lower is better)
- **Backward Transfer**: Impact of new learning on old tasks (higher is better)
- **Final Performance**: Success rate after all tasks

## Adding Your Own CL Method

### Step 1: Create a new method file

Create `src/tau2/continual_learning/methods/my_method.py`:

```python
from tau2.continual_learning.methods.base import ContinualLearningMethod
from tau2.continual_learning.data_model import Experience

class MyMethod(ContinualLearningMethod):
    def get_name(self) -> str:
        return "my_method"

    def get_augmented_experiences(
        self,
        current_experiences: list[Experience],
        task_idx: int,
    ) -> list[Experience]:
        # Your logic here
        # Example: add replay data
        if task_idx > 0:
            replay = self.buffer.sample(10)
            return current_experiences + replay
        return current_experiences
```

### Step 2: Use your method

```python
from tau2.continual_learning.methods.my_method import MyMethod

buffer = ExperienceBuffer()
my_method = MyMethod(buffer)

trainer = ContinualLearningTrainer(config, my_method)
results = trainer.train()
```

## Common Use Cases

### 1. Compare Multiple Methods

```python
methods = [
    ("Naive", NaiveMethod, {}),
    ("Replay", ExperienceReplayMethod, {"replay_size": 10}),
]

for name, method_class, config in methods:
    buffer = ExperienceBuffer()
    method = method_class(buffer, config)
    trainer = ContinualLearningTrainer(base_config, method)
    results = trainer.train()
    print(f"{name} Forgetting: {results.cl_metrics['forgetting']:.3f}")
```

### 2. Experiment with Replay Strategies

```python
strategies = ["uniform", "per_task", "successful_only", "recent"]

for strategy in strategies:
    method = ExperienceReplayMethod(
        buffer,
        {"replay_size": 10, "replay_strategy": strategy}
    )
    # ... train and evaluate
```

### 3. Custom Task Order

```python
config = CLTrainerConfig(
    domain_name="airline",
    task_order="custom",
    custom_task_order=["0", "5", "10", "2"],  # Specific order
)
```

## CL Method Hooks

When implementing a custom method, you can override these hooks:

```python
class MyMethod(ContinualLearningMethod):
    def before_task(self, task_idx, task_id, agent):
        """Called before starting a new task"""
        pass

    def after_training_step(self, task_idx, agent, experiences, step):
        """Called after each training step"""
        self.buffer.add_batch(experiences)  # Store experiences

    def get_augmented_experiences(self, current_experiences, task_idx):
        """Add replay/augmentation data"""
        return current_experiences + self.buffer.sample(10)

    def after_task(self, task_idx, task_result, agent):
        """Called after completing a task"""
        pass
```

## Configuration Options

Key parameters in `CLTrainerConfig`:

```python
CLTrainerConfig(
    # Task selection
    domain_name="airline",        # Which domain
    num_tasks=10,                 # How many tasks
    task_order="sequential",      # Task order (sequential/random/custom)

    # Training
    num_train_trials=4,           # Trials per task for training
    num_eval_trials=4,            # Trials per task for evaluation
    eval_all_previous=True,       # Eval on all previous tasks

    # Models
    agent_llm="gpt-4o-mini",     # Agent LLM
    user_llm="gpt-4o-mini",      # User simulator LLM

    # Output
    save_dir="./results",         # Where to save
    checkpoint_frequency=1,       # Checkpoint every N tasks
)
```

## Example Workflows

### Workflow 1: Quick Test

```bash
# Small test with mock domain
python -m tau2.continual_learning.cli train \
    --domain mock \
    --method naive \
    --num-tasks 3 \
    --num-train-trials 2 \
    --num-eval-trials 2
```

### Workflow 2: Full Evaluation

```python
# Full experiment on airline domain
config = CLTrainerConfig(
    domain_name="airline",
    task_split_name="train",
    num_tasks=None,  # All tasks
    num_train_trials=4,
    num_eval_trials=4,
    save_dir="./full_eval",
)

# Compare baseline vs replay
for method_name, method_class in [("naive", NaiveMethod), ("replay", ExperienceReplayMethod)]:
    buffer = ExperienceBuffer()
    method = method_class(buffer)
    trainer = ContinualLearningTrainer(config, method)
    results = trainer.train()
```

### Workflow 3: Method Development

```python
# Develop and test your method
class MyNewMethod(ContinualLearningMethod):
    # Your implementation
    pass

# Quick iteration on small subset
config = CLTrainerConfig(
    domain_name="mock",
    num_tasks=3,  # Small for fast iteration
    num_train_trials=1,
)

method = MyNewMethod(ExperienceBuffer())
trainer = ContinualLearningTrainer(config, method)
results = trainer.train()
```

## Troubleshooting

**Q: Training is taking too long**
- Use `domain_name="mock"` for faster testing
- Reduce `num_tasks` and `num_train_trials`
- Use cheaper LLM models

**Q: Out of memory**
- Set `buffer.max_size` to limit buffer size
- Reduce replay size

**Q: How to visualize results?**
- Results are saved as JSON in `save_dir/results.json`
- Checkpoints saved as `checkpoint_task_N.json`
- Load and plot with your favorite tools

## Next Steps

1. **Read the full README**: See `src/tau2/continual_learning/README.md`
2. **Try examples**: Run `examples/continual_learning_example.py`
3. **Implement your method**: Start with the template above
4. **Run experiments**: Compare your method against baselines

## Resources

- Full documentation: `src/tau2/continual_learning/README.md`
- Example scripts: `examples/continual_learning_example.py`
- Base tau2-bench docs: Main repository README

## Support

For questions or issues:
1. Check the full README
2. Look at example scripts
3. Review method implementations in `methods/`
4. Open an issue on GitHub

Happy continual learning! 🚀
