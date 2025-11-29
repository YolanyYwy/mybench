# Advanced Continual Learning Methods - Complete Guide

Complete guide to using, configuring, and best practices for the 8 newly implemented advanced continual learning methods.

## 📋 Table of Contents

- [Methods Overview](#methods-overview)
- [Installation and Dependencies](#installation-and-dependencies)
- [Memory-Based Methods](#memory-based-methods)
- [Prompt-Based Methods](#prompt-based-methods)
- [Meta-Learning Methods](#meta-learning-methods)
- [Hybrid Methods](#hybrid-methods)
- [Performance Comparison](#performance-comparison)
- [Debugging and Optimization](#debugging-and-optimization)

## Methods Overview

| Method Type | Method Name | Core Idea | Use Case |
|------------|-------------|-----------|----------|
| Memory | EnhancedReplayMethod | Priority sampling, difficulty weighting | Unbalanced task difficulty |
| Memory | RAGMemoryMethod | Semantic retrieval of relevant experiences | Semantic similarity between tasks |
| Prompt | DynamicPromptingMethod | Dynamic knowledge base construction | Structured knowledge accumulation |
| Prompt | CoTMemoryMethod | Reasoning chain learning | Emphasis on reasoning process |
| Meta-learning | TaskEmbeddingMethod | Task similarity transfer | Strong task correlation |
| Meta-learning | ToolStrategyMethod | Tool usage pattern mining | Intensive tool usage |
| Hybrid | RAGPromptHybridMethod | RAG + Dynamic prompting | Comprehensive scenarios |

## Installation and Dependencies

### Basic Installation

```bash
# Clone repository and install
cd tau2-bench
pip install -e .
```

### Optional Dependencies

```bash
# OpenAI API (for high-quality embeddings)
pip install openai

# Vector database (FAISS - faster similarity search)
pip install faiss-cpu  # CPU version
# or
pip install faiss-gpu  # GPU version

# Or use ChromaDB
pip install chromadb

# Complete installation
pip install numpy scikit-learn openai faiss-cpu
```

### Environment Variables

```bash
# If using OpenAI embeddings
export OPENAI_API_KEY="sk-..."
```

## Memory-Based Methods

### 1. EnhancedReplayMethod - Enhanced Experience Replay

#### Core Features

- ✅ 4 sampling strategies: uniform, difficulty, diversity, recent
- ✅ Difficulty-based priority weighting
- ✅ Task coverage diversity sampling
- ✅ Success/failure sample balancing

#### Usage Example

```python
from tau2.continual_learning import CLTrainerConfig, ContinualLearningTrainer, ExperienceBuffer
from tau2.continual_learning.methods import EnhancedReplayMethod

# Create buffer
buffer = ExperienceBuffer(max_size=1000)

# Configure method
config = {
    "replay_ratio": 0.5,           # Replay ratio
    "priority_mode": "difficulty",  # Sampling mode
    "min_buffer_size": 10,         # Min buffer size
    "diversity_key": "task_idx",   # Diversity key
    "recent_ratio": 0.7,           # Recent sample ratio
    "balance_success": False,      # Balance success/failure
}

method = EnhancedReplayMethod(buffer, config)

# Training configuration
trainer_config = CLTrainerConfig(
    domain_names=["airline", "retail"],
    num_tasks_per_domain=10,
    num_train_trials=4,
    save_dir="./results/enhanced_replay"
)

# Run training
trainer = ContinualLearningTrainer(trainer_config, method)
results = trainer.train()

# View results
print(f"Forgetting: {results.cl_metrics['forgetting']:.3f}")
```

#### Sampling Modes Explained

**1. Uniform (Uniform Sampling)**
```python
config = {"priority_mode": "uniform"}
```
- All experiences sampled with equal probability
- Suitable for similar task difficulty

**2. Difficulty (Difficulty Weighting)**
```python
config = {"priority_mode": "difficulty"}
```
- Weighted by task success rate (harder tasks prioritized)
- Failed experiences get 1.5x weight
- Suitable for large task difficulty variations

**3. Diversity (Diversity Priority)**
```python
config = {
    "priority_mode": "diversity",
    "diversity_key": "task_idx"  # or "domain"
}
```
- Ensures uniform sampling across tasks
- Prevents over-representation of certain tasks
- Suitable for cross-domain learning

**4. Recent (Recency Priority)**
```python
config = {
    "priority_mode": "recent",
    "recent_ratio": 0.7  # 70% from recent
}
```
- Biases toward recently learned tasks
- Mitigates catastrophic forgetting
- Suitable for temporally correlated task sequences

#### Advanced Configuration

```python
config = {
    # Basic parameters
    "replay_ratio": 0.5,           # 0.0-1.0, replay/current ratio
    "min_buffer_size": 10,         # Min buffer size to start replay

    # Sampling strategy
    "priority_mode": "difficulty",
    "diversity_key": "task_idx",
    "recent_ratio": 0.7,

    # Sample balancing
    "balance_success": True,       # Balance success/failure (50/50)
}
```

### 2. RAGMemoryMethod - Retrieval-Augmented Memory

#### Core Features

- ✅ Semantic similarity retrieval
- ✅ Vector database storage (In-memory / FAISS)
- ✅ OpenAI embeddings or local embeddings
- ✅ Domain filtering and similarity threshold

#### Usage Example

```python
from tau2.continual_learning.methods import RAGMemoryMethod

# Using OpenAI embeddings (recommended)
config = {
    "num_memories": 5,                    # Number of retrievals
    "vector_store_type": "inmemory",      # 'inmemory' or 'faiss'
    "embedding_model": "text-embedding-3-small",
    "embedding_dim": 1536,
    "use_openai_embeddings": True,        # Use OpenAI
    "filter_by_domain": False,            # Cross-domain retrieval
    "min_similarity": 0.3,                # Min similarity
}

method = RAGMemoryMethod(buffer, config)
trainer = ContinualLearningTrainer(trainer_config, method)
results = trainer.train()
```

#### Vector Storage Options

**In-Memory (Suitable for small scale)**
```python
config = {
    "vector_store_type": "inmemory",
}
```
- Simple, fast startup
- Suitable for < 10k experiences
- No additional dependencies needed

**FAISS (Suitable for large scale)**
```python
config = {
    "vector_store_type": "faiss",
    "embedding_dim": 1536,
}
```
- Efficient similarity search
- Suitable for > 10k experiences
- Requires `pip install faiss-cpu`

#### Embedding Options

**OpenAI Embeddings (High quality)**
```python
config = {
    "use_openai_embeddings": True,
    "embedding_model": "text-embedding-3-small",  # or "text-embedding-3-large"
}
```
- Best semantic quality
- Requires API key
- Incurs API costs (~$0.00002/1K tokens)

**Local Embeddings (Free)**
```python
config = {
    "use_openai_embeddings": False,
    "embedding_dim": 1536,
}
```
- Hash-based deterministic embeddings
- No API costs
- Lower quality but free

#### Advanced Configuration

```python
config = {
    # Retrieval parameters
    "num_memories": 5,                # Number per retrieval
    "min_similarity": 0.3,            # Similarity threshold
    "filter_by_domain": True,         # Only retrieve same domain experiences

    # Vector storage
    "vector_store_type": "faiss",
    "embedding_dim": 1536,

    # Embedding
    "use_openai_embeddings": True,
    "embedding_model": "text-embedding-3-small",
}
```

#### Retrieval Example

The method automatically retrieves semantically relevant experiences:

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

## Prompt-Based Methods

### 3. DynamicPromptingMethod - Dynamic Prompting

#### Core Features

- ✅ Automatic task learning extraction
- ✅ Cumulative knowledge base construction
- ✅ 3 summary modes
- ✅ Includes success patterns and common errors

#### Usage Example

```python
from tau2.continual_learning.methods import DynamicPromptingMethod

config = {
    "summary_mode": "key_points",      # Summary mode
    "max_knowledge_items": 5,          # Max tasks to retain
    "include_errors": True,            # Include errors
    "include_success_patterns": True,  # Include success patterns
}

method = DynamicPromptingMethod(buffer, config)
trainer = ContinualLearningTrainer(trainer_config, method)
results = trainer.train()
```

#### Summary Modes

**1. Key Points (Key points mode)**
```python
config = {"summary_mode": "key_points"}
```
Output example:
```
Task 1: airline_cancel_booking
- Effective tools: search_booking, cancel_booking, process_refund
- Successful tasks require ~4.5 conversation turns
- Common errors: Invalid booking ID, Refund already processed
```

**2. Brief (Brief mode)**
```python
config = {"summary_mode": "brief"}
```
Output example:
```
Task 1: Mastered - 85% success rate across 20 attempts
```

**3. Detailed (Detailed mode)**
```python
config = {"summary_mode": "detailed"}
```
Output example:
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

[Continue with remaining sections...]
