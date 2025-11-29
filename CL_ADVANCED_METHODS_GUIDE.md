# 高级持续学习方法完整指南

本文档介绍新实现的 8 种高级持续学习方法的使用、配置和最佳实践。

## 📋 目录

- [方法概览](#方法概览)
- [安装和依赖](#安装和依赖)
- [基于记忆的方法](#基于记忆的方法)
- [基于提示的方法](#基于提示的方法)
- [基于元学习的方法](#基于元学习的方法)
- [混合方法](#混合方法)
- [性能对比](#性能对比)
- [调试和优化](#调试和优化)

## 方法概览

| 方法类型 | 方法名称 | 核心思想 | 适用场景 |
|---------|---------|---------|---------|
| 记忆 | EnhancedReplayMethod | 优先级采样、难度加权 | 任务难度不均衡 |
| 记忆 | RAGMemoryMethod | 语义检索相关经验 | 任务间有语义相似性 |
| 提示 | DynamicPromptingMethod | 动态知识库构建 | 结构化知识积累 |
| 提示 | CoTMemoryMethod | 推理链学习 | 重视推理过程 |
| 元学习 | TaskEmbeddingMethod | 任务相似度迁移 | 任务相关性强 |
| 元学习 | ToolStrategyMethod | 工具使用模式挖掘 | 工具使用密集 |
| 混合 | RAGPromptHybridMethod | RAG + 动态提示 | 综合场景 |

## 安装和依赖

### 基础安装

```bash
# 克隆仓库并安装
cd tau2-bench
pip install -e .
```

### 可选依赖

```bash
# OpenAI API (用于高质量 embeddings)
pip install openai

# 向量数据库 (FAISS - 更快的相似度搜索)
pip install faiss-cpu  # CPU 版本
# 或
pip install faiss-gpu  # GPU 版本

# 或使用 ChromaDB
pip install chromadb

# 完整安装
pip install numpy scikit-learn openai faiss-cpu
```

### 环境变量

```bash
# 如果使用 OpenAI embeddings
export OPENAI_API_KEY="sk-..."
```

## 基于记忆的方法

### 1. EnhancedReplayMethod - 增强经验回放

#### 核心特性

- ✅ 4 种采样策略: uniform, difficulty, diversity, recent
- ✅ 基于难度的优先级加权
- ✅ 任务覆盖多样性采样
- ✅ 成功/失败样本平衡

#### 使用示例

```python
from tau2.continual_learning import CLTrainerConfig, ContinualLearningTrainer, ExperienceBuffer
from tau2.continual_learning.methods import EnhancedReplayMethod

# 创建 buffer
buffer = ExperienceBuffer(max_size=1000)

# 配置方法
config = {
    "replay_ratio": 0.5,           # 回放比例
    "priority_mode": "difficulty",  # 采样模式
    "min_buffer_size": 10,         # 最小 buffer 大小
    "diversity_key": "task_idx",   # 多样性键
    "recent_ratio": 0.7,           # 近期样本比例
    "balance_success": False,      # 平衡成功/失败
}

method = EnhancedReplayMethod(buffer, config)

# 训练配置
trainer_config = CLTrainerConfig(
    domain_names=["airline", "retail"],
    num_tasks_per_domain=10,
    num_train_trials=4,
    save_dir="./results/enhanced_replay"
)

# 运行训练
trainer = ContinualLearningTrainer(trainer_config, method)
results = trainer.train()

# 查看结果
print(f"Forgetting: {results.cl_metrics['forgetting']:.3f}")
```

#### 采样模式说明

**1. Uniform (均匀采样)**
```python
config = {"priority_mode": "uniform"}
```
- 所有经验等概率采样
- 适合任务难度相似的场景

**2. Difficulty (难度加权)**
```python
config = {"priority_mode": "difficulty"}
```
- 根据任务成功率加权 (难任务优先)
- 失败经验获得 1.5x 权重
- 适合任务难度差异大的场景

**3. Diversity (多样性优先)**
```python
config = {
    "priority_mode": "diversity",
    "diversity_key": "task_idx"  # 或 "domain"
}
```
- 确保各任务均匀采样
- 防止某些任务过度代表
- 适合跨域学习场景

**4. Recent (近期优先)**
```python
config = {
    "priority_mode": "recent",
    "recent_ratio": 0.7  # 70% 来自近期
}
```
- 偏向最近学习的任务
- 缓解灾难性遗忘
- 适合任务序列有时间相关性

#### 高级配置

```python
config = {
    # 基础参数
    "replay_ratio": 0.5,           # 0.0-1.0, 回放/当前比例
    "min_buffer_size": 10,         # 开始回放的最小 buffer 大小

    # 采样策略
    "priority_mode": "difficulty",
    "diversity_key": "task_idx",
    "recent_ratio": 0.7,

    # 样本平衡
    "balance_success": True,       # 平衡成功/失败 (50/50)
}
```

### 2. RAGMemoryMethod - 检索增强记忆

#### 核心特性

- ✅ 语义相似度检索
- ✅ 向量数据库存储 (In-memory / FAISS)
- ✅ OpenAI embeddings 或本地 embeddings
- ✅ 域过滤和相似度阈值

#### 使用示例

```python
from tau2.continual_learning.methods import RAGMemoryMethod

# 使用 OpenAI embeddings (推荐)
config = {
    "num_memories": 5,                    # 检索数量
    "vector_store_type": "inmemory",      # 'inmemory' 或 'faiss'
    "embedding_model": "text-embedding-3-small",
    "embedding_dim": 1536,
    "use_openai_embeddings": True,        # 使用 OpenAI
    "filter_by_domain": False,            # 跨域检索
    "min_similarity": 0.3,                # 最小相似度
}

method = RAGMemoryMethod(buffer, config)
trainer = ContinualLearningTrainer(trainer_config, method)
results = trainer.train()
```

#### 向量存储选项

**In-Memory (适合小规模)**
```python
config = {
    "vector_store_type": "inmemory",
}
```
- 简单、快速启动
- 适合 < 10k 经验
- 无需额外依赖

**FAISS (适合大规模)**
```python
config = {
    "vector_store_type": "faiss",
    "embedding_dim": 1536,
}
```
- 高效相似度搜索
- 适合 > 10k 经验
- 需要 `pip install faiss-cpu`

#### Embedding 选项

**OpenAI Embeddings (高质量)**
```python
config = {
    "use_openai_embeddings": True,
    "embedding_model": "text-embedding-3-small",  # 或 "text-embedding-3-large"
}
```
- 最佳语义质量
- 需要 API key
- 产生 API 费用 (~$0.00002/1K tokens)

**本地 Embeddings (免费)**
```python
config = {
    "use_openai_embeddings": False,
    "embedding_dim": 1536,
}
```
- 基于 hash 的确定性嵌入
- 无 API 费用
- 质量较低但免费

#### 高级配置

```python
config = {
    # 检索参数
    "num_memories": 5,                # 每次检索数量
    "min_similarity": 0.3,            # 相似度阈值
    "filter_by_domain": True,         # 只检索同域经验

    # 向量存储
    "vector_store_type": "faiss",
    "embedding_dim": 1536,

    # Embedding
    "use_openai_embeddings": True,
    "embedding_model": "text-embedding-3-small",
}
```

#### 检索示例

方法会自动检索语义相关的经验:

```
Current Task: "Cancel flight booking and refund"

Retrieved Memories:
1. [Similarity: 0.85] Task: "Modify flight booking"
   → Used tools: search_booking → cancel_booking → process_refund

2. [Similarity: 0.72] Task: "Request refund for canceled flight"
   → Key lesson: Always check refund policy before canceling

3. [Similarity: 0.68] Task: "Change flight date"
   → Successful pattern: Verify customer identity first
```

## 基于提示的方法

### 3. DynamicPromptingMethod - 动态提示工程

#### 核心特性

- ✅ 自动提取任务学习要点
- ✅ 构建累积知识库
- ✅ 3 种摘要模式
- ✅ 包含成功模式和常见错误

#### 使用示例

```python
from tau2.continual_learning.methods import DynamicPromptingMethod

config = {
    "summary_mode": "key_points",      # 摘要模式
    "max_knowledge_items": 5,          # 最多保留任务数
    "include_errors": True,            # 包含错误
    "include_success_patterns": True,  # 包含成功模式
}

method = DynamicPromptingMethod(buffer, config)
trainer = ContinualLearningTrainer(trainer_config, method)
results = trainer.train()
```

#### 摘要模式

**1. Key Points (要点模式)**
```python
config = {"summary_mode": "key_points"}
```
输出示例:
```
Task 1: airline_cancel_booking
- Effective tools: search_booking, cancel_booking, process_refund
- Successful tasks require ~4.5 conversation turns
- Common errors: Invalid booking ID, Refund already processed
```

**2. Brief (简要模式)**
```python
config = {"summary_mode": "brief"}
```
输出示例:
```
Task 1: Mastered - 85% success rate across 20 attempts
```

**3. Detailed (详细模式)**
```python
config = {"summary_mode": "detailed"}
```
输出示例:
```
Task 1: airline_cancel_booking
Success Strategy:
1. First verify booking exists: search_booking
2. Check cancellation policy
3. Execute cancellation: cancel_booking
4. Process refund if applicable: process_refund

Common Failures:
- Attempting to cancel non-existent bookings
- Not checking refund eligibility
```

#### 知识库示例

方法会构建如下知识库并注入到 agent prompt:

```xml
<previous_task_knowledge>
Task 1: airline_book_flight (airline)
Success Rate: 90%
Key Learnings:
- Effective tools: search_flights, book_flight, confirm_booking
- Successful tasks typically require ~5 conversation turns

Common Errors to Avoid:
- Booking without verifying passenger details
- Not checking seat availability

Task 2: airline_cancel_booking (airline)
Success Rate: 85%
Key Learnings:
- Always verify booking exists before canceling
- Check refund policy first

Common Errors to Avoid:
- Invalid booking reference
- Attempting refund after deadline
</previous_task_knowledge>
```

### 4. CoTMemoryMethod - 思维链记忆

#### 核心特性

- ✅ 提取推理链
- ✅ 识别思考模式
- ✅ 少样本示例生成
- ✅ 成功推理模板

#### 使用示例

```python
from tau2.continual_learning.methods import CoTMemoryMethod

config = {
    "num_examples": 3,            # 提供示例数
    "min_success_reward": 0.7,    # 最小奖励
    "extract_mode": "summary",    # 'summary' 或 'full'
}

method = CoTMemoryMethod(buffer, config)
trainer = ContinualLearningTrainer(trainer_config, method)
results = trainer.train()
```

#### 推理链提取

方法会自动识别和提取推理模式:

```
Original Messages:
User: "I need to cancel my flight"
Agent: "Let me first search for your booking to verify it exists"
Agent: [Uses search_booking tool]
Tool: "Booking found: ABC123"
Agent: "Now I'll check the cancellation policy"
Agent: "I can proceed with cancellation. Let me cancel it now"
Agent: [Uses cancel_booking tool]

Extracted Chain:
1. Think: Need to verify booking exists before canceling
2. Act: Use search_booking
3. Observe: Booking ABC123 found
4. Think: Should check policy before proceeding
5. Act: Use cancel_booking
6. Conclusion: Task completed successfully
```

#### 示例注入

提取的推理链会作为 few-shot examples:

```xml
<reasoning_examples>
Example 1: Cancel flight booking
Reasoning Steps:
1. Verify the booking exists using search tools
2. Check cancellation policy and constraints
3. Execute cancellation if allowed
4. Process refund if applicable
Result: ✓ Successful

Example 2: Modify booking date
Reasoning Steps:
1. Retrieve current booking details
2. Check availability for new date
3. Update booking with new details
Result: ✓ Successful
</reasoning_examples>
```

## 基于元学习的方法

### 5. TaskEmbeddingMethod - 任务嵌入

#### 核心特性

- ✅ 任务语义嵌入
- ✅ 相似任务识别
- ✅ 知识迁移
- ✅ 自适应采样

#### 使用示例

```python
from tau2.continual_learning.methods import TaskEmbeddingMethod

config = {
    "num_similar_tasks": 3,           # 检索相似任务数
    "embedding_model": "text-embedding-3-small",
    "use_openai_embeddings": True,
    "similarity_threshold": 0.3,      # 相似度阈值
    "transfer_ratio": 0.3,            # 迁移经验比例
}

method = TaskEmbeddingMethod(buffer, config)
trainer = ContinualLearningTrainer(trainer_config, method)
results = trainer.train()
```

#### 任务相似度计算

方法会基于以下特征嵌入任务:
- 任务描述
- 域策略
- 可用工具列表

示例输出:

```
Current Task: "Cancel hotel reservation"

Similar Past Tasks:
1. [Similarity: 0.82] "Cancel flight booking" (airline)
   → Successfully used: search → cancel → refund
   → Success rate: 85%

2. [Similarity: 0.75] "Modify hotel booking" (retail)
   → Successfully used: lookup → update → confirm
   → Success rate: 78%

3. [Similarity: 0.68] "Request refund" (retail)
   → Successfully used: verify → process_refund
   → Success rate: 92%
```

#### 知识迁移

自动从相似任务迁移经验:

```python
# 如果当前任务与 Task 1 相似度 0.8
# transfer_ratio = 0.3, current_batch = 10

transferred_experiences = sample_from_task1(n=3)  # 10 * 0.3 = 3
augmented_batch = current_batch + transferred_experiences
```

### 6. ToolStrategyMethod - 工具策略学习

#### 核心特性

- ✅ 工具序列挖掘
- ✅ 成功模式识别
- ✅ 域特定策略
- ✅ 频次和成功率加权

#### 使用示例

```python
from tau2.continual_learning.methods import ToolStrategyMethod

config = {
    "min_pattern_support": 3,     # 最小出现次数
    "min_success_rate": 0.6,      # 最小成功率
    "max_sequence_length": 5,     # 最大序列长度
    "num_strategies": 5,          # 推荐策略数
}

method = ToolStrategyMethod(buffer, config)
trainer = ContinualLearningTrainer(trainer_config, method)
results = trainer.train()
```

#### 策略挖掘示例

方法会发现有效的工具使用模式:

```xml
<successful_tool_strategies>
Based on previous experience, the following tool usage patterns
have been effective:

1. search_booking → cancel_booking → process_refund
   Success Rate: 88% (used 15 times)
   Best for: Cancellation with refund tasks

2. search_flights → check_availability → book_flight
   Success Rate: 82% (used 22 times)
   Best for: Flight booking tasks

3. get_customer_info → verify_identity → update_profile
   Success Rate: 95% (used 18 times)
   Best for: Profile update tasks

4. lookup_order → modify_order → send_confirmation
   Success Rate: 79% (used 12 times)
   Best for: Order modification tasks

5. search_booking → get_booking_details
   Success Rate: 92% (used 25 times)
   Best for: Information retrieval tasks
</successful_tool_strategies>
```

#### 模式评分

策略按以下公式排序:
```python
score = success_rate * log(1 + frequency)
```

这样既考虑成功率，也奖励更常用的模式。

## 混合方法

### 7. RAGPromptHybridMethod - 混合方法

#### 核心特性

- ✅ 结合 RAG 和动态提示
- ✅ 结构化知识 + 语义检索
- ✅ 可选启用各组件
- ✅ 智能上下文组合

#### 使用示例

```python
from tau2.continual_learning.methods import RAGPromptHybridMethod

config = {
    # 启用选项
    "use_rag": True,
    "use_prompting": True,

    # RAG 参数
    "num_memories": 5,
    "vector_store_type": "inmemory",
    "embedding_model": "text-embedding-3-small",
    "use_openai_embeddings": True,
    "min_similarity": 0.3,

    # Prompting 参数
    "max_knowledge_items": 5,
    "summary_mode": "key_points",
    "include_errors": True,
    "include_success_patterns": True,

    # Replay 参数
    "replay_ratio": 0.3,
}

method = RAGPromptHybridMethod(buffer, config)
trainer = ContinualLearningTrainer(trainer_config, method)
results = trainer.train()
```

#### 组合策略

方法会智能组合两种上下文:

```xml
<instructions>
You have access to both structured knowledge from previous tasks and
specific relevant examples. Use the structured knowledge to understand
general patterns and the examples for specific guidance.
Prioritize examples when they directly match your current situation.
</instructions>

<previous_task_knowledge>
<!-- 来自 Dynamic Prompting 的结构化知识 -->
Task 1: airline_book_flight
- Effective tools: search_flights, book_flight
- Success rate: 90%
</previous_task_knowledge>

<retrieved_experiences>
<!-- 来自 RAG 的语义相关经验 -->
Relevant Experience 1 (similarity: 0.85)
Task: airline_cancel_booking
Summary: Successfully canceled booking and processed refund
Key Actions: search_booking, cancel_booking, process_refund
</retrieved_experiences>
```

#### 灵活配置

可以选择性启用组件:

```python
# 只用 RAG
config = {"use_rag": True, "use_prompting": False}

# 只用 Prompting
config = {"use_rag": False, "use_prompting": True}

# 两者都用 (推荐)
config = {"use_rag": True, "use_prompting": True}
```

## 性能对比

### 实验设置

```python
# 统一配置
trainer_config = CLTrainerConfig(
    domain_names=["airline", "retail", "telecom"],
    num_tasks_per_domain=10,
    num_train_trials=4,
    num_eval_trials=4,
)

buffer = ExperienceBuffer(max_size=1000)
```

### 预期性能

基于设计预期的性能排序 (实际需测试):

| 方法 | 遗忘度 ↓ | 平均准确率 ↑ | 训练时间 | 内存占用 |
|------|---------|------------|---------|---------|
| RAGPromptHybrid | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 慢 | 高 |
| RAGMemory | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 慢 | 高 |
| TaskEmbedding | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 中 | 中 |
| EnhancedReplay | ⭐⭐⭐ | ⭐⭐⭐⭐ | 快 | 中 |
| DynamicPrompting | ⭐⭐⭐ | ⭐⭐⭐ | 快 | 低 |
| ToolStrategy | ⭐⭐⭐ | ⭐⭐⭐ | 快 | 低 |
| CoTMemory | ⭐⭐ | ⭐⭐⭐ | 快 | 低 |
| Naive (baseline) | ⭐ | ⭐⭐ | 最快 | 最低 |

### 批量测试脚本

```python
from tau2.continual_learning.methods import *

methods = [
    ("Naive", NaiveMethod, {}),
    ("EnhancedReplay", EnhancedReplayMethod, {"priority_mode": "difficulty"}),
    ("RAG", RAGMemoryMethod, {"num_memories": 5}),
    ("DynamicPrompt", DynamicPromptingMethod, {"summary_mode": "key_points"}),
    ("CoT", CoTMemoryMethod, {"num_examples": 3}),
    ("TaskEmbed", TaskEmbeddingMethod, {"num_similar_tasks": 3}),
    ("ToolStrategy", ToolStrategyMethod, {"min_pattern_support": 3}),
    ("Hybrid", RAGPromptHybridMethod, {"use_rag": True, "use_prompting": True}),
]

results_summary = []

for name, method_class, config in methods:
    print(f"\n{'='*50}")
    print(f"Testing: {name}")
    print(f"{'='*50}")

    buffer = ExperienceBuffer(max_size=1000)
    method = method_class(buffer, config)

    trainer = ContinualLearningTrainer(trainer_config, method)
    results = trainer.train()

    results_summary.append({
        "method": name,
        "forgetting": results.cl_metrics['forgetting'],
        "avg_accuracy": results.cl_metrics.get('avg_accuracy', 0),
    })

# 输出对比
import pandas as pd
df = pd.DataFrame(results_summary)
df = df.sort_values('forgetting')
print("\n" + "="*50)
print("RESULTS SUMMARY")
print("="*50)
print(df.to_string(index=False))
```

## 调试和优化

### 启用详细日志

```python
from loguru import logger
import sys

logger.remove()
logger.add(sys.stderr, level="DEBUG")
```

### Buffer 统计

```python
# 查看 buffer 状态
stats = buffer.get_statistics()

print(f"Total experiences: {stats['total_experiences']}")
print(f"Success rate: {stats['success_rate']:.3f}")
print(f"Tasks: {stats['tasks_represented']}")
print(f"Domains: {stats['domains']}")
print(f"Task distribution: {stats['task_distribution']}")
```

### 性能优化

**1. 减少 API 调用 (RAG/TaskEmbedding)**
```python
config = {
    "use_openai_embeddings": False,  # 使用本地 embeddings
}
```

**2. 限制 Buffer 大小**
```python
buffer = ExperienceBuffer(max_size=500)  # 减小 buffer
```

**3. 减少检索数量**
```python
config = {
    "num_memories": 3,        # RAG: 从 5 减到 3
    "num_similar_tasks": 2,   # TaskEmbed: 从 3 减到 2
}
```

**4. 使用更快的向量存储**
```python
config = {
    "vector_store_type": "inmemory",  # 比 FAISS 更快启动
}
```

### 常见问题

**Q: OpenAI API 太贵怎么办?**
```python
config = {
    "use_openai_embeddings": False,  # 使用免费的本地 embeddings
}
```

**Q: 训练太慢?**
```python
# 1. 减少任务数
trainer_config.num_tasks_per_domain = 5  # 从 10 减到 5

# 2. 减少trials
trainer_config.num_train_trials = 2  # 从 4 减到 2

# 3. 使用更快的模型
trainer_config.agent_llm = "gpt-3.5-turbo"  # 而不是 gpt-4
```

**Q: 内存不足?**
```python
# 1. 限制 buffer
buffer = ExperienceBuffer(max_size=300)

# 2. 减少回放
config = {"replay_ratio": 0.2}  # 从 0.5 减到 0.2

# 3. 关闭 RAG
config = {"use_rag": False, "use_prompting": True}
```

**Q: 如何可视化结果?**
```python
import json
import matplotlib.pyplot as plt

# 加载结果
with open("./results/results.json", 'r') as f:
    results_data = json.load(f)

# 绘制学习曲线
tasks = list(range(len(results_data['task_results'])))
accuracies = [r['train_success_rate'] for r in results_data['task_results']]

plt.plot(tasks, accuracies)
plt.xlabel('Task Index')
plt.ylabel('Success Rate')
plt.title('Continual Learning Curve')
plt.savefig('learning_curve.png')
```

## 下一步

1. ✅ **选择方法**: 根据任务特点选择合适的方法
2. ✅ **小规模测试**: 先在 mock 域测试
3. ✅ **调参优化**: 调整配置参数
4. ✅ **大规模评估**: 在完整数据集上评估
5. ✅ **对比分析**: 与 baseline 对比

## 参考资源

- 基础框架: `CONTINUAL_LEARNING_QUICKSTART.md`
- 代码示例: `examples/` 目录
- API 文档: `src/tau2/continual_learning/`

Happy Learning! 🚀
