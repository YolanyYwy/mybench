# Cross-Domain Continual Learning - User Guide

## Overview

This framework now **fully meets your requirements**:

✅ **Supports cross-domain continual learning**: airline → retail → telecom
✅ **Automatic Baseline evaluation**: Test each domain's performance independently first
✅ **Learn three domains sequentially**: Train and evaluate in order
✅ **Compare CL vs Baseline**: Automatically calculate performance differences

## Quick Start

### Method 1: Command Line (Recommended for quick testing)

```bash
# Complete cross-domain continual learning experiment
python -m tau2.continual_learning.cli train \
    --domains "airline,retail,telecom" \
    --method naive \
    --num-tasks-per-domain 5 \
    --run-baseline \
    --save-dir ./results/cross_domain_cl

# Using experience replay method
python -m tau2.continual_learning.cli train \
    --domains "airline,retail,telecom" \
    --method experience_replay \
    --replay-size 10 \
    --replay-strategy uniform \
    --num-tasks-per-domain 5 \
    --save-dir ./results/replay_cl
```

### Method 2: Python Script (Recommended for research)

```python
from tau2.continual_learning import (
    CLTrainerConfig,
    ContinualLearningTrainer,
    ExperienceBuffer,
    NaiveMethod,
)

# Configure cross-domain continual learning experiment
config = CLTrainerConfig(
    # Three domains, learn in order
    domain_names=["airline", "retail", "telecom"],

    # Number of tasks per domain
    num_tasks_per_domain=5,  # None = use all tasks

    # Training settings
    num_train_trials=4,
    num_eval_trials=4,

    # ✅ Important: Run baseline evaluation
    run_baseline=True,
    baseline_num_trials=4,

    # Save results
    save_dir="./results/my_experiment",
)

# Create continual learning method
buffer = ExperienceBuffer()
cl_method = NaiveMethod(buffer)

# Run training
trainer = ContinualLearningTrainer(config, cl_method)
results = trainer.train()

# View results
print("Baseline vs CL:")
for domain in config.domain_names:
    baseline = results.baseline_results[domain]['avg_success_rate']
    cl_acc = results.cl_metrics[f'{domain}_avg_accuracy']
    print(f"{domain}: Baseline={baseline:.3f}, CL={cl_acc:.3f}, Diff={cl_acc-baseline:+.3f}")
```

## Experiment Workflow

After running, the framework automatically executes these steps:

### Step 1: Baseline Evaluation (Independent Training)
```
==================================================
Step 1: Baseline Evaluation (per domain)
==================================================

Baseline: airline
  Avg Success Rate: 0.750
  Avg Reward: 0.750

Baseline: retail
  Avg Success Rate: 0.800
  Avg Reward: 0.800

Baseline: telecom
  Avg Success Rate: 0.700
  Avg Reward: 0.700
```

### Step 2: Continual Learning Training
```
==================================================
Step 2: Continual Learning Training
==================================================

Task 1/15: airline/0
Task 2/15: airline/1
...
Task 6/15: retail/0
Task 7/15: retail/1
...
Task 11/15: telecom/0
...
```

### Step 3: Final Results Comparison
```
==================================================
Per-Domain Comparison (CL vs Baseline)
==================================================

airline:
  Baseline:    0.7500
  After CL:    0.7200
  Difference:  -0.0300 ↓

retail:
  Baseline:    0.8000
  After CL:    0.7800
  Difference:  -0.0200 ↓

telecom:
  Baseline:    0.7000
  After CL:    0.7100
  Difference:  +0.0100 ↑
```

## Output Results

### 1. Console Output
- Baseline results (each domain independent)
- Training progress
- Final comparison (CL vs Baseline)
- Continual learning metrics (forgetting, transfer, etc.)

### 2. Saved Files

```
save_dir/
├── results.json              # Complete results
├── checkpoint_task_5.json    # Checkpoints
├── checkpoint_task_10.json
└── checkpoint_task_15.json
```

**results.json content**:
```json
{
  "task_sequence": ["0", "1", "2", ...],
  "domain_sequence": ["airline", "airline", ..., "retail", ..., "telecom", ...],
  "baseline_results": {
    "airline": {"avg_success_rate": 0.75, "num_tasks": 5},
    "retail": {"avg_success_rate": 0.80, "num_tasks": 5},
    "telecom": {"avg_success_rate": 0.70, "num_tasks": 5}
  },
  "cl_metrics": {
    "average_accuracy": 0.733,
    "forgetting": 0.150,
    "airline_avg_accuracy": 0.720,
    "retail_avg_accuracy": 0.780,
    "telecom_avg_accuracy": 0.710,
    "airline_vs_baseline": -0.030,
    "retail_vs_baseline": -0.020,
    "telecom_vs_baseline": +0.010
  }
}
```

## Key Metrics Explained

### Baseline Metrics
- `avg_success_rate`: Average success rate for tasks in this domain (independent training)
- `avg_reward`: Average reward
- `num_tasks`: Number of tasks

### Continual Learning Metrics
- `average_accuracy`: Average accuracy across all tasks
- `forgetting`: Forgetting measure (lower is better)
- `{domain}_avg_accuracy`: Average accuracy for each domain after CL
- `{domain}_vs_baseline`: Performance difference between CL and baseline
  - Positive: CL performs better (positive transfer)
  - Negative: CL performs worse (catastrophic forgetting)

## Common Use Cases

### Scenario 1: Quick Test (Few Tasks)

```bash
python -m tau2.continual_learning.cli train \
    --domains "airline,retail,telecom" \
    --num-tasks-per-domain 3 \
    --num-train-trials 2 \
    --num-eval-trials 2 \
    --save-dir ./test_results
```

### Scenario 2: Complete Evaluation (All Tasks)

```bash
python -m tau2.continual_learning.cli train \
    --domains "airline,retail,telecom" \
    --num-tasks-per-domain None \
    --num-train-trials 4 \
    --num-eval-trials 4 \
    --save-dir ./full_results
```

### Scenario 3: Compare Different CL Methods

```python
methods = ["naive", "experience_replay"]

for method in methods:
    config = CLTrainerConfig(
        domain_names=["airline", "retail", "telecom"],
        num_tasks_per_domain=5,
        save_dir=f"./results/{method}",
    )

    buffer = ExperienceBuffer()
    if method == "naive":
        cl_method = NaiveMethod(buffer)
    else:
        cl_method = ExperienceReplayMethod(buffer, {"replay_size": 10})

    trainer = ContinualLearningTrainer(config, cl_method)
    results = trainer.train()
```

### Scenario 4: Custom Domain Order

```python
config = CLTrainerConfig(
    domain_names=["telecom", "airline", "retail"],  # Custom order
    domain_order="custom",
    custom_domain_order=["telecom", "airline", "retail"],
)
```

## Configuration Parameters Explained

```python
CLTrainerConfig(
    # Domain configuration
    domain_names=["airline", "retail", "telecom"],  # Domains to use
    num_tasks_per_domain=5,                         # Tasks per domain
    task_split_name="train",                        # train/test/base
    task_order="sequential",                        # Task order
    domain_order="sequential",                      # Domain order

    # Training configuration
    num_train_trials=4,                             # Training trials
    num_eval_trials=4,                              # Evaluation trials
    eval_all_previous=True,                         # Evaluate all previous tasks

    # Baseline configuration
    run_baseline=True,                              # Run baseline
    baseline_num_trials=4,                          # Baseline trials

    # LLM configuration
    agent_llm="gpt-4o-mini",
    user_llm="gpt-4o-mini",

    # System configuration
    seed=300,
    save_dir="./results",
)
```

## Example Scripts

Run the provided examples:

```bash
# Basic example
python examples/cross_domain_cl_example.py

# Or directly with Python
cd examples
python cross_domain_cl_example.py
```

## Correspondence with Your Requirements

| Your Requirement | Framework Implementation |
|-----------------|-------------------------|
| Test three domains | ✅ `domain_names=["airline", "retail", "telecom"]` |
| Test baseline first | ✅ `run_baseline=True` automatically evaluates each domain independently |
| Learn sequentially | ✅ Automatically loads and trains in `domain_names` order |
| Compare performance | ✅ Automatically calculates `{domain}_vs_baseline` metrics |
| Evaluate tool use CL capability | ✅ Provides forgetting, transfer, and other CL metrics |

## Core Improvements

Compared to the original version, the framework now:

1. ✅ **Multi-domain support**: `domain_names` accepts a list of domains
2. ✅ **Automatic baseline**: `run_baseline=True` evaluates independently first
3. ✅ **Cross-domain training**: Automatically loads all domains' tasks in order
4. ✅ **Domain-level metrics**: Separately calculates metrics for each domain
5. ✅ **Comparative analysis**: Automatically calculates CL vs baseline differences

## Frequently Asked Questions

**Q: What if I only want to use airline and retail?**
```python
domain_names=["airline", "retail"]
```

**Q: What if I want to change the domain order?**
```python
domain_names=["retail", "airline", "telecom"]  # Train in this order
```

**Q: What if I want to disable baseline?**
```python
run_baseline=False
```

**Q: What if I want to use all tasks?**
```python
num_tasks_per_domain=None  # None means use all
```

## Summary

Your code now **fully meets the requirements**:

1. ✅ Supports cross-domain continual learning (airline → retail → telecom)
2. ✅ Automatic baseline evaluation (test each domain independently)
3. ✅ Sequential multi-domain learning
4. ✅ Automatic CL vs Baseline performance comparison
5. ✅ Complete continual learning metrics

Ready to start your experiments!
