"""
Prompt Templates for Continual Learning Methods.

This module provides reusable prompt templates for injecting knowledge,
experiences, strategies, and other CL-specific information into agent prompts.
"""

from typing import List, Dict, Any


class PromptTemplates:
    """Collection of prompt templates for continual learning."""

    @staticmethod
    def knowledge_base_template(knowledge_items: List[Dict[str, Any]]) -> str:
        """
        Create a knowledge base prompt from previous task learnings.

        Args:
            knowledge_items: List of dicts with 'task_id', 'domain', 'learnings'

        Returns:
            Formatted knowledge base prompt
        """
        if not knowledge_items:
            return ""

        sections = ["<previous_task_knowledge>"]
        sections.append(
            "The following knowledge has been accumulated from previous tasks. "
            "Use it to inform your approach when relevant."
        )
        sections.append("")

        for i, item in enumerate(knowledge_items, 1):
            sections.append(f"## Task {i}: {item.get('task_id', 'Unknown')}")
            sections.append(f"Domain: {item.get('domain', 'Unknown')}")

            if 'success_rate' in item:
                sections.append(f"Success Rate: {item['success_rate']:.1%}")

            if 'learnings' in item:
                sections.append("\nKey Learnings:")
                learnings = item['learnings']
                if isinstance(learnings, list):
                    for learning in learnings:
                        sections.append(f"- {learning}")
                else:
                    sections.append(f"- {learnings}")

            if 'common_errors' in item:
                sections.append("\nCommon Errors to Avoid:")
                errors = item['common_errors']
                if isinstance(errors, list):
                    for error in errors:
                        sections.append(f"- {error}")
                else:
                    sections.append(f"- {errors}")

            sections.append("")

        sections.append("</previous_task_knowledge>")
        return "\n".join(sections)

    @staticmethod
    def few_shot_examples_template(examples: List[Dict[str, Any]]) -> str:
        """
        Create few-shot examples prompt.

        Args:
            examples: List of dicts with 'task', 'actions', 'outcome'

        Returns:
            Formatted few-shot examples
        """
        if not examples:
            return ""

        sections = ["<examples>"]
        sections.append(
            "Here are examples of successful task completions. "
            "Use similar reasoning and action patterns when applicable."
        )
        sections.append("")

        for i, example in enumerate(examples, 1):
            sections.append(f"### Example {i}")

            if 'task' in example:
                sections.append(f"Task: {example['task']}")

            if 'reasoning' in example:
                sections.append(f"\nReasoning Process:")
                sections.append(example['reasoning'])

            if 'actions' in example:
                sections.append(f"\nActions Taken:")
                actions = example['actions']
                if isinstance(actions, list):
                    for action in actions:
                        sections.append(f"- {action}")
                else:
                    sections.append(actions)

            if 'outcome' in example:
                sections.append(f"\nOutcome: {example['outcome']}")

            sections.append("")

        sections.append("</examples>")
        return "\n".join(sections)

    @staticmethod
    def tool_strategy_template(strategies: List[Dict[str, Any]]) -> str:
        """
        Create tool usage strategy prompt.

        Args:
            strategies: List of dicts with 'pattern', 'success_rate', 'context'

        Returns:
            Formatted tool strategy guidance
        """
        if not strategies:
            return ""

        sections = ["<successful_tool_strategies>"]
        sections.append(
            "Based on previous experience, the following tool usage patterns "
            "have been effective. Consider using them when appropriate."
        )
        sections.append("")

        for i, strategy in enumerate(strategies, 1):
            pattern = strategy.get('pattern', 'Unknown')
            success_rate = strategy.get('success_rate', 0.0)
            count = strategy.get('count', 0)

            sections.append(f"{i}. {pattern}")
            sections.append(f"   Success Rate: {success_rate:.1%} (used {count} times)")

            if 'context' in strategy:
                sections.append(f"   Best for: {strategy['context']}")

            if 'description' in strategy:
                sections.append(f"   {strategy['description']}")

            sections.append("")

        sections.append("</successful_tool_strategies>")
        return "\n".join(sections)

    @staticmethod
    def retrieved_memories_template(memories: List[Dict[str, Any]]) -> str:
        """
        Create prompt from retrieved episodic memories.

        Args:
            memories: List of memory dicts with 'task_id', 'summary', 'relevance'

        Returns:
            Formatted retrieved memories prompt
        """
        if not memories:
            return ""

        sections = ["<retrieved_experiences>"]
        sections.append(
            "The following relevant experiences from past tasks may help inform your approach:"
        )
        sections.append("")

        for i, memory in enumerate(memories, 1):
            relevance = memory.get('similarity', memory.get('relevance', 0.0))
            sections.append(f"### Relevant Experience {i} (similarity: {relevance:.2f})")

            if 'task_id' in memory:
                sections.append(f"Task: {memory['task_id']}")

            if 'domain' in memory:
                sections.append(f"Domain: {memory['domain']}")

            if 'success' in memory:
                status = "Successful" if memory['success'] else "Failed"
                sections.append(f"Status: {status}")

            if 'summary' in memory:
                sections.append(f"\nSummary: {memory['summary']}")

            if 'key_actions' in memory:
                sections.append("\nKey Actions:")
                for action in memory['key_actions']:
                    sections.append(f"- {action}")

            if 'lessons' in memory:
                sections.append(f"\nLessons: {memory['lessons']}")

            sections.append("")

        sections.append("</retrieved_experiences>")
        return "\n".join(sections)

    @staticmethod
    def chain_of_thought_template(cot_examples: List[Dict[str, Any]]) -> str:
        """
        Create prompt with chain-of-thought reasoning examples.

        Args:
            cot_examples: List of dicts with 'task', 'reasoning_chain', 'success'

        Returns:
            Formatted CoT examples
        """
        if not cot_examples:
            return ""

        sections = ["<reasoning_examples>"]
        sections.append(
            "Here are examples of successful reasoning chains from previous tasks. "
            "Apply similar step-by-step thinking to your current task."
        )
        sections.append("")

        for i, example in enumerate(cot_examples, 1):
            sections.append(f"### Reasoning Example {i}")

            if 'task' in example:
                sections.append(f"Task: {example['task']}")

            if 'reasoning_chain' in example:
                sections.append("\nReasoning Steps:")
                chain = example['reasoning_chain']
                if isinstance(chain, list):
                    for j, step in enumerate(chain, 1):
                        sections.append(f"{j}. {step}")
                else:
                    sections.append(chain)

            if 'conclusion' in example:
                sections.append(f"\nConclusion: {example['conclusion']}")

            if 'success' in example:
                status = "✓ Successful" if example['success'] else "✗ Failed"
                sections.append(f"\nResult: {status}")

            sections.append("")

        sections.append("</reasoning_examples>")
        return "\n".join(sections)

    @staticmethod
    def task_context_template(task_info: Dict[str, Any], similar_tasks: List[Dict[str, Any]]) -> str:
        """
        Create task-specific context with similar past tasks.

        Args:
            task_info: Current task information
            similar_tasks: List of similar past tasks

        Returns:
            Formatted task context
        """
        sections = ["<task_context>"]

        # Current task info
        if task_info:
            sections.append("## Current Task")
            if 'domain' in task_info:
                sections.append(f"Domain: {task_info['domain']}")
            if 'description' in task_info:
                sections.append(f"Description: {task_info['description']}")
            sections.append("")

        # Similar tasks
        if similar_tasks:
            sections.append("## Similar Past Tasks")
            sections.append("The following tasks are similar to your current task:")
            sections.append("")

            for i, task in enumerate(similar_tasks, 1):
                similarity = task.get('similarity', 0.0)
                sections.append(f"### Similar Task {i} (similarity: {similarity:.2f})")

                if 'task_id' in task:
                    sections.append(f"Task: {task['task_id']}")

                if 'domain' in task:
                    sections.append(f"Domain: {task['domain']}")

                if 'approach' in task:
                    sections.append(f"Successful Approach: {task['approach']}")

                if 'key_insight' in task:
                    sections.append(f"Key Insight: {task['key_insight']}")

                sections.append("")

        sections.append("</task_context>")
        return "\n".join(sections)

    @staticmethod
    def meta_learning_template(meta_knowledge: Dict[str, Any]) -> str:
        """
        Create meta-learning knowledge prompt.

        Args:
            meta_knowledge: Dict with cross-task patterns and strategies

        Returns:
            Formatted meta-knowledge prompt
        """
        if not meta_knowledge:
            return ""

        sections = ["<meta_knowledge>"]
        sections.append(
            "The following meta-level knowledge has been learned across multiple tasks:"
        )
        sections.append("")

        if 'general_strategies' in meta_knowledge:
            sections.append("## General Strategies")
            for strategy in meta_knowledge['general_strategies']:
                sections.append(f"- {strategy}")
            sections.append("")

        if 'domain_patterns' in meta_knowledge:
            sections.append("## Domain-Specific Patterns")
            for domain, patterns in meta_knowledge['domain_patterns'].items():
                sections.append(f"\n### {domain}")
                if isinstance(patterns, list):
                    for pattern in patterns:
                        sections.append(f"- {pattern}")
                else:
                    sections.append(f"- {patterns}")
            sections.append("")

        if 'common_pitfalls' in meta_knowledge:
            sections.append("## Common Pitfalls to Avoid")
            for pitfall in meta_knowledge['common_pitfalls']:
                sections.append(f"- {pitfall}")
            sections.append("")

        if 'best_practices' in meta_knowledge:
            sections.append("## Best Practices")
            for practice in meta_knowledge['best_practices']:
                sections.append(f"- {practice}")
            sections.append("")

        sections.append("</meta_knowledge>")
        return "\n".join(sections)

    @staticmethod
    def combine_templates(*template_outputs: str) -> str:
        """
        Combine multiple template outputs into a single prompt.

        Args:
            *template_outputs: Variable number of template output strings

        Returns:
            Combined prompt with proper spacing
        """
        # Filter out empty strings
        non_empty = [t for t in template_outputs if t.strip()]

        if not non_empty:
            return ""

        # Join with double newlines for readability
        return "\n\n".join(non_empty)


class PromptBuilder:
    """Helper class for building complex prompts incrementally."""

    def __init__(self):
        """Initialize the prompt builder."""
        self.sections = []

    def add_knowledge_base(self, knowledge_items: List[Dict[str, Any]]) -> 'PromptBuilder':
        """Add knowledge base section."""
        prompt = PromptTemplates.knowledge_base_template(knowledge_items)
        if prompt:
            self.sections.append(prompt)
        return self

    def add_few_shot_examples(self, examples: List[Dict[str, Any]]) -> 'PromptBuilder':
        """Add few-shot examples section."""
        prompt = PromptTemplates.few_shot_examples_template(examples)
        if prompt:
            self.sections.append(prompt)
        return self

    def add_tool_strategies(self, strategies: List[Dict[str, Any]]) -> 'PromptBuilder':
        """Add tool strategies section."""
        prompt = PromptTemplates.tool_strategy_template(strategies)
        if prompt:
            self.sections.append(prompt)
        return self

    def add_retrieved_memories(self, memories: List[Dict[str, Any]]) -> 'PromptBuilder':
        """Add retrieved memories section."""
        prompt = PromptTemplates.retrieved_memories_template(memories)
        if prompt:
            self.sections.append(prompt)
        return self

    def add_cot_examples(self, cot_examples: List[Dict[str, Any]]) -> 'PromptBuilder':
        """Add chain-of-thought examples section."""
        prompt = PromptTemplates.chain_of_thought_template(cot_examples)
        if prompt:
            self.sections.append(prompt)
        return self

    def add_task_context(
        self,
        task_info: Dict[str, Any],
        similar_tasks: List[Dict[str, Any]]
    ) -> 'PromptBuilder':
        """Add task context section."""
        prompt = PromptTemplates.task_context_template(task_info, similar_tasks)
        if prompt:
            self.sections.append(prompt)
        return self

    def add_meta_knowledge(self, meta_knowledge: Dict[str, Any]) -> 'PromptBuilder':
        """Add meta-learning knowledge section."""
        prompt = PromptTemplates.meta_learning_template(meta_knowledge)
        if prompt:
            self.sections.append(prompt)
        return self

    def add_custom(self, custom_prompt: str) -> 'PromptBuilder':
        """Add custom prompt section."""
        if custom_prompt.strip():
            self.sections.append(custom_prompt)
        return self

    def build(self) -> str:
        """Build the final combined prompt."""
        return "\n\n".join(self.sections)

    def clear(self) -> 'PromptBuilder':
        """Clear all sections."""
        self.sections = []
        return self

    def __str__(self) -> str:
        """Return the built prompt."""
        return self.build()
