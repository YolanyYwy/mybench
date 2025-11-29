# Cross-Domain Continual Learning - 使用指南

## 概述

这个框架现在**完全符合你的需求**：

✅ **支持跨域持续学习**：airline → retail → telecom
✅ **自动 Baseline 评估**：先独立测试每个域的性能
✅ **依次学习三个域**：按顺序训练并评估
✅ **对比 CL vs Baseline**：自动计算性能差异

## 快速开始

### 方式 1: 命令行（推荐用于快速测试）

```bash
# 完整的跨域持续学习实验
python -m tau2.continual_learning.cli train \
    --domains "airline,retail,telecom" \
    --method naive \
    --num-tasks-per-domain 5 \
    --run-baseline \
    --save-dir ./results/cross_domain_cl

# 使用经验回放方法
python -m tau2.continual_learning.cli train \
    --domains "airline,retail,telecom" \
    --method experience_replay \
    --replay-size 10 \
    --replay-strategy uniform \
    --num-tasks-per-domain 5 \
    --save-dir ./results/replay_cl
```

### 方式 2: Python 脚本（推荐用于研究）

```python
from tau2.continual_learning import (
    CLTrainerConfig,
    ContinualLearningTrainer,
    ExperienceBuffer,
    NaiveMethod,
)

# 配置跨域持续学习实验
config = CLTrainerConfig(
    # 三个域，按顺序学习
    domain_names=["airline", "retail", "telecom"],

    # 每个域使用的任务数
    num_tasks_per_domain=5,  # None = 使用所有任务

    # 训练设置
    num_train_trials=4,
    num_eval_trials=4,

    # ✅ 重要：运行 baseline 评估
    run_baseline=True,
    baseline_num_trials=4,

    # 保存结果
    save_dir="./results/my_experiment",
)

# 创建持续学习方法
buffer = ExperienceBuffer()
cl_method = NaiveMethod(buffer)

# 运行训练
trainer = ContinualLearningTrainer(config, cl_method)
results = trainer.train()

# 查看结果
print("Baseline vs CL:")
for domain in config.domain_names:
    baseline = results.baseline_results[domain]['avg_success_rate']
    cl_acc = results.cl_metrics[f'{domain}_avg_accuracy']
    print(f"{domain}: Baseline={baseline:.3f}, CL={cl_acc:.3f}, Diff={cl_acc-baseline:+.3f}")
```

## 实验流程

运行后，框架会自动执行以下步骤：

### Step 1: Baseline 评估（独立训练）
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

### Step 2: 持续学习训练
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

### Step 3: 最终结果对比
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

## 输出结果

### 1. 控制台输出
- Baseline 结果（每个域独立）
- 训练进度
- 最终对比（CL vs Baseline）
- 持续学习指标（forgetting, transfer, etc.）

### 2. 保存的文件

```
save_dir/
├── results.json              # 完整结果
├── checkpoint_task_5.json    # 检查点
├── checkpoint_task_10.json
└── checkpoint_task_15.json
```

**results.json 内容**：
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

## 关键指标说明

### Baseline 指标
- `avg_success_rate`: 该域任务的平均成功率（独立训练）
- `avg_reward`: 平均奖励
- `num_tasks`: 任务数量

### 持续学习指标
- `average_accuracy`: 所有任务的平均准确率
- `forgetting`: 遗忘度（越低越好）
- `{domain}_avg_accuracy`: 每个域在 CL 后的平均准确率
- `{domain}_vs_baseline`: CL 相比 baseline 的性能差异
  - 正值：CL 表现更好（正向迁移）
  - 负值：CL 表现更差（灾难性遗忘）

## 常见使用场景

### 场景 1: 快速测试（少量任务）

```bash
python -m tau2.continual_learning.cli train \
    --domains "airline,retail,telecom" \
    --num-tasks-per-domain 3 \
    --num-train-trials 2 \
    --num-eval-trials 2 \
    --save-dir ./test_results
```

### 场景 2: 完整评估（所有任务）

```bash
python -m tau2.continual_learning.cli train \
    --domains "airline,retail,telecom" \
    --num-tasks-per-domain None \
    --num-train-trials 4 \
    --num-eval-trials 4 \
    --save-dir ./full_results
```

### 场景 3: 对比不同 CL 方法

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

### 场景 4: 自定义域顺序

```python
config = CLTrainerConfig(
    domain_names=["telecom", "airline", "retail"],  # 自定义顺序
    domain_order="custom",
    custom_domain_order=["telecom", "airline", "retail"],
)
```

## 配置参数详解

```python
CLTrainerConfig(
    # 域配置
    domain_names=["airline", "retail", "telecom"],  # 要使用的域
    num_tasks_per_domain=5,                         # 每个域的任务数
    task_split_name="train",                        # train/test/base
    task_order="sequential",                        # 任务顺序
    domain_order="sequential",                      # 域顺序

    # 训练配置
    num_train_trials=4,                             # 训练试验次数
    num_eval_trials=4,                              # 评估试验次数
    eval_all_previous=True,                         # 评估所有历史任务

    # Baseline 配置
    run_baseline=True,                              # 运行 baseline
    baseline_num_trials=4,                          # baseline 试验次数

    # LLM 配置
    agent_llm="gpt-4o-mini",
    user_llm="gpt-4o-mini",

    # 系统配置
    seed=300,
    save_dir="./results",
)
```

## 示例脚本

运行提供的示例：

```bash
# 基础示例
python examples/cross_domain_cl_example.py

# 或直接用 Python
cd examples
python cross_domain_cl_example.py
```

## 与你需求的对应关系

| 你的需求 | 框架实现 |
|---------|---------|
| 测试三个域 | ✅ `domain_names=["airline", "retail", "telecom"]` |
| 先测 baseline | ✅ `run_baseline=True` 自动独立评估每个域 |
| 依次学习 | ✅ 自动按 `domain_names` 顺序加载和训练 |
| 对比性能 | ✅ 自动计算 `{domain}_vs_baseline` 指标 |
| 评估 tool use CL 能力 | ✅ 提供 forgetting, transfer 等 CL 指标 |

## 核心改进

相比原始版本，现在的框架：

1. ✅ **支持多域**：`domain_names` 接受域列表
2. ✅ **自动 baseline**：`run_baseline=True` 先独立评估
3. ✅ **跨域训练**：自动按顺序加载所有域的任务
4. ✅ **域级指标**：每个域单独计算指标
5. ✅ **对比分析**：自动计算 CL vs baseline 差异

## 常见问题

**Q: 我只想用 airline 和 retail 怎么办？**
```python
domain_names=["airline", "retail"]
```

**Q: 我想改变域的顺序？**
```python
domain_names=["retail", "airline", "telecom"]  # 按这个顺序训练
```

**Q: 我想禁用 baseline？**
```python
run_baseline=False
```

**Q: 我想使用所有任务？**
```python
num_tasks_per_domain=None  # None 表示使用所有
```

## 总结

现在你的代码**完全符合需求**：

1. ✅ 支持跨域持续学习（airline → retail → telecom）
2. ✅ 自动 baseline 评估（独立测试每个域）
3. ✅ 依次学习多个域
4. ✅ 自动对比 CL vs Baseline 性能
5. ✅ 完整的持续学习指标

直接使用即可开始你的实验！
