# 持续学习框架 - 快速入门指南

本指南帮助您快速开始使用基于 tau2-bench 构建的持续学习框架。

## 什么是这个框架？

该框架扩展了 tau2-bench，用于在工具使用任务的**持续学习**场景中评估代理。与评估单个任务不同，我们评估代理如何：

1. **顺序学习** - 从任务流中学习
2. **保留知识** - 避免灾难性遗忘
3. **迁移知识** - 将知识迁移到新任务

## 安装

该框架已集成到 tau2-bench 中。确保已安装基本依赖项：

```bash
pip install -e .
```

## 框架结构

```
src/tau2/continual_learning/
├── trainer.py              # 主训练循环
├── buffer.py               # 经验存储
├── data_model.py          # 数据结构
├── metrics.py             # CL 指标
├── cli.py                 # 命令行接口
└── methods/               # CL 方法实现
    ├── __init__.py
    ├── base.py            # 抽象基类
    ├── naive.py           # 顺序学习基线
    └── experience_replay.py  # 经验回放方法
```

## 快速开始：运行您的第一个实验

### 选项 1：命令行

```bash
# 使用朴素基线的基本实验
python -m tau2.continual_learning.cli train \
    --domain mock \
    --method naive \
    --num-tasks 3 \
    --num-train-trials 2 \
    --save-dir ./cl_results

# 使用经验回放
python -m tau2.continual_learning.cli train \
    --domain airline \
    --method experience_replay \
    --num-tasks 5 \
    --replay-size 10 \
    --replay-strategy uniform \
    --save-dir ./cl_results
```

### 选项 2：Python 脚本

```python
from tau2.continual_learning import (
    CLTrainerConfig,
    ContinualLearningTrainer,
    ExperienceBuffer,
    NaiveMethod,
)

# 1. 配置实验
config = CLTrainerConfig(
    domain_name="airline",
    num_tasks=5,
    num_train_trials=4,
    save_dir="./my_experiment",
)

# 2. 创建 CL 方法
buffer = ExperienceBuffer()
method = NaiveMethod(buffer)

# 3. 运行训练
trainer = ContinualLearningTrainer(config, method)
results = trainer.train()

# 4. 查看结果
print(f"Forgetting: {results.cl_metrics['forgetting']:.3f}")
```

## 理解输出

框架计算这些关键指标：

- **Average Accuracy**: 所有任务的平均成功率
- **Forgetting**: 先前任务性能下降程度（越低越好）
- **Backward Transfer**: 新学习对旧任务的影响（越高越好）
- **Final Performance**: 所有任务后的成功率

## 添加您自己的 CL 方法

### 步骤 1：创建新方法文件

创建 `src/tau2/continual_learning/methods/my_method.py`：

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
        # 您的逻辑在这里
        # 示例：添加回放数据
        if task_idx > 0:
            replay = self.buffer.sample(10)
            return current_experiences + replay
        return current_experiences
```

### 步骤 2：使用您的方法

```python
from tau2.continual_learning.methods.my_method import MyMethod

buffer = ExperienceBuffer()
my_method = MyMethod(buffer)

trainer = ContinualLearningTrainer(config, my_method)
results = trainer.train()
```

## 常见用例

### 1. 比较多种方法

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

### 2. 实验不同回放策略

```python
strategies = ["uniform", "per_task", "successful_only", "recent"]

for strategy in strategies:
    method = ExperienceReplayMethod(
        buffer,
        {"replay_size": 10, "replay_strategy": strategy}
    )
    # ... 训练和评估
```

### 3. 自定义任务顺序

```python
config = CLTrainerConfig(
    domain_name="airline",
    task_order="custom",
    custom_task_order=["0", "5", "10", "2"],  # 特定顺序
)
```

## CL 方法钩子

实现自定义方法时，可以覆盖这些钩子：

```python
class MyMethod(ContinualLearningMethod):
    def before_task(self, task_idx, task_id, agent):
        """在开始新任务之前调用"""
        pass

    def after_training_step(self, task_idx, agent, experiences, step):
        """在每个训练步骤之后调用"""
        self.buffer.add_batch(experiences)  # 存储经验

    def get_augmented_experiences(self, current_experiences, task_idx):
        """添加回放/增强数据"""
        return current_experiences + self.buffer.sample(10)

    def after_task(self, task_idx, task_result, agent):
        """完成任务后调用"""
        pass
```

## 配置选项

`CLTrainerConfig` 中的关键参数：

```python
CLTrainerConfig(
    # 任务选择
    domain_name="airline",        # 哪个域
    num_tasks=10,                 # 多少个任务
    task_order="sequential",      # 任务顺序 (sequential/random/custom)

    # 训练
    num_train_trials=4,           # 每个任务的训练试验次数
    num_eval_trials=4,            # 每个任务的评估试验次数
    eval_all_previous=True,       # 评估所有先前任务

    # 模型
    agent_llm="gpt-4o-mini",     # Agent LLM
    user_llm="gpt-4o-mini",      # 用户模拟器 LLM

    # 输出
    save_dir="./results",         # 保存位置
    checkpoint_frequency=1,       # 每 N 个任务检查点一次
)
```

## 示例工作流程

### 工作流程 1：快速测试

```bash
# 使用 mock 域进行小测试
python -m tau2.continual_learning.cli train \
    --domain mock \
    --method naive \
    --num-tasks 3 \
    --num-train-trials 2 \
    --num-eval-trials 2
```

### 工作流程 2：完整评估

```python
# airline 域上的完整实验
config = CLTrainerConfig(
    domain_name="airline",
    task_split_name="train",
    num_tasks=None,  # 所有任务
    num_train_trials=4,
    num_eval_trials=4,
    save_dir="./full_eval",
)

# 比较基线与回放
for method_name, method_class in [("naive", NaiveMethod), ("replay", ExperienceReplayMethod)]:
    buffer = ExperienceBuffer()
    method = method_class(buffer)
    trainer = ContinualLearningTrainer(config, method)
    results = trainer.train()
```

### 工作流程 3：方法开发

```python
# 开发和测试您的方法
class MyNewMethod(ContinualLearningMethod):
    # 您的实现
    pass

# 在小子集上快速迭代
config = CLTrainerConfig(
    domain_name="mock",
    num_tasks=3,  # 小规模快速迭代
    num_train_trials=1,
)

method = MyNewMethod(ExperienceBuffer())
trainer = ContinualLearningTrainer(config, method)
results = trainer.train()
```

## 故障排除

**问：训练时间太长**
- 使用 `domain_name="mock"` 进行更快测试
- 减少 `num_tasks` 和 `num_train_trials`
- 使用更便宜的 LLM 模型

**问：内存不足**
- 设置 `buffer.max_size` 以限制缓冲区大小
- 减少回放大小

**问：如何可视化结果？**
- 结果保存为 `save_dir/results.json` 中的 JSON
- 检查点保存为 `checkpoint_task_N.json`
- 使用您喜欢的工具加载和绘图

## 下一步

1. **阅读完整 README**：参见 `src/tau2/continual_learning/README.md`
2. **尝试示例**：运行 `examples/continual_learning_example.py`
3. **实现您的方法**：从上面的模板开始
4. **运行实验**：将您的方法与基线进行比较

## 资源

- 完整文档：`src/tau2/continual_learning/README.md`
- 示例脚本：`examples/continual_learning_example.py`
- 基础 tau2-bench 文档：主仓库 README

## 支持

如有问题或问题：
1. 查看完整 README
2. 查看示例脚本
3. 查看 `methods/` 中的方法实现
4. 在 GitHub 上打开 issue

持续学习愉快！ 🚀
