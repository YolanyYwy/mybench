"""
Dynamic Prompting for Continual Learning.

This module implements dynamic prompt engineering that accumulates knowledge
from previous tasks and injects it into agent prompts for future tasks.
"""

from typing import List, Dict, Any, Optional
from pathlib import Path
import json

from loguru import logger

from tau2.continual_learning.buffer import ExperienceBuffer
from tau2.continual_learning.data_model import Experience
from tau2.continual_learning.methods.base import ContinualLearningMethod
from tau2.continual_learning.prompt_templates import PromptTemplates, PromptBuilder
from tau2.continual_learning.enhanced_agent import PromptEnhancedAgent
from tau2.data_model.message import AssistantMessage, UserMessage


class DynamicPromptingMethod(ContinualLearningMethod):
    """
    Dynamic prompting method that builds and updates prompts based on task history.

    This method maintains a knowledge base of learnings from previous tasks
    and injects relevant knowledge into agent prompts.
    """

    def __init__(
        self,
        buffer: ExperienceBuffer,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize dynamic prompting method.

        Args:
            buffer: Experience buffer for storage
            config: Configuration dict with:
                - summary_mode: 'detailed', 'brief', or 'key_points' (default 'key_points')
                - max_knowledge_items: Max number of past tasks to include (default 5)
                - include_errors: Whether to include common errors (default True)
                - include_success_patterns: Include successful patterns (default True)
        """
        super().__init__(buffer, config)

        # Configuration
        self.summary_mode = config.get("summary_mode", "key_points")
        self.max_knowledge_items = config.get("max_knowledge_items", 5)
        self.include_errors = config.get("include_errors", True)
        self.include_success_patterns = config.get("include_success_patterns", True)

        # Task summaries storage
        self.task_summaries: Dict[int, Dict[str, Any]] = {}

        logger.info(
            f"DynamicPromptingMethod initialized with summary_mode={self.summary_mode}"
        )

    def before_task(self, task_idx: int, task_id: str, agent) -> None:
        """
        Build and inject dynamic knowledge prompt before task starts.

        Args:
            task_idx: Current task index
            task_id: Current task ID
            agent: Agent to enhance with knowledge
        """
        if task_idx == 0:
            logger.debug("First task, no previous knowledge to inject")
            return

        # Build knowledge base from previous tasks
        knowledge_prompt = self._build_knowledge_prompt(task_idx)

        # Inject into agent if it's a PromptEnhancedAgent
        if isinstance(agent, PromptEnhancedAgent):
            agent.add_knowledge(knowledge_prompt)
            logger.info(f"Task {task_idx}: Injected knowledge from {len(self.task_summaries)} previous tasks")
        else:
            logger.debug(
                f"Task {task_idx}: Agent is not PromptEnhancedAgent, "
                "knowledge prepared but not injected"
            )

        logger.debug(f"Task {task_idx} ({task_id}): Dynamic prompting ready")

    def after_training_step(
        self,
        task_idx: int,
        agent,
        experiences: List[Experience],
        step: int,
    ) -> None:
        """Store experiences in buffer."""
        self.buffer.add_batch(experiences)

        logger.debug(
            f"Task {task_idx}, step {step}: Added {len(experiences)} experiences"
        )

    def after_task(
        self,
        task_idx: int,
        task_result,
        agent,
    ) -> None:
        """
        Create task summary after task completion.

        Args:
            task_idx: Completed task index
            task_result: Task training result
            agent: Trained agent
        """
        # Create comprehensive task summary
        summary = self._create_task_summary(task_idx, task_result)

        # Store summary
        self.task_summaries[task_idx] = summary

        logger.info(
            f"Task {task_idx} summary created: "
            f"success_rate={summary['success_rate']:.3f}, "
            f"learnings={len(summary.get('key_learnings', []))}"
        )

    def get_agent_context(self) -> str:
        """
        Get knowledge context to inject into agent prompts.

        Returns:
            Formatted knowledge base prompt
        """
        if not self.task_summaries:
            return ""

        # Get recent task summaries (limited by max_knowledge_items)
        recent_task_indices = sorted(self.task_summaries.keys())[-self.max_knowledge_items:]
        knowledge_items = [self.task_summaries[idx] for idx in recent_task_indices]

        return PromptTemplates.knowledge_base_template(knowledge_items)

    def _build_knowledge_prompt(self, current_task_idx: int) -> str:
        """
        Build knowledge base prompt from previous tasks.

        Args:
            current_task_idx: Current task index

        Returns:
            Formatted knowledge prompt
        """
        if not self.task_summaries:
            return ""

        # Get summaries of previous tasks
        previous_indices = [idx for idx in self.task_summaries.keys() if idx < current_task_idx]

        # Limit to most recent tasks
        if len(previous_indices) > self.max_knowledge_items:
            previous_indices = previous_indices[-self.max_knowledge_items:]

        # Build knowledge items
        knowledge_items = []
        for idx in previous_indices:
            summary = self.task_summaries[idx]

            # Format for template
            knowledge_item = {
                "task_id": summary["task_id"],
                "domain": summary["domain"],
                "success_rate": summary["success_rate"],
            }

            # Add learnings based on mode
            if self.summary_mode == "detailed":
                knowledge_item["learnings"] = summary.get("detailed_learnings", [])
            elif self.summary_mode == "brief":
                knowledge_item["learnings"] = summary.get("brief_summary", "")
            else:  # key_points
                knowledge_item["learnings"] = summary.get("key_learnings", [])

            # Add errors if configured
            if self.include_errors:
                knowledge_item["common_errors"] = summary.get("common_errors", [])

            knowledge_items.append(knowledge_item)

        return PromptTemplates.knowledge_base_template(knowledge_items)

    def _create_task_summary(
        self,
        task_idx: int,
        task_result,
    ) -> Dict[str, Any]:
        """
        Create a comprehensive summary of task performance.

        Args:
            task_idx: Task index
            task_result: Task result object

        Returns:
            Summary dictionary
        """
        # Basic info
        summary = {
            "task_idx": task_idx,
            "task_id": task_result.task_id,
            "domain": task_result.domain,
            "success_rate": task_result.train_success_rate,
        }

        # Get task experiences
        task_experiences = self.buffer.sample_by_task_idx(task_idx, n=None)

        # Extract key learnings
        summary["key_learnings"] = self._extract_key_learnings(task_experiences)
        summary["detailed_learnings"] = self._extract_detailed_learnings(task_experiences)
        summary["brief_summary"] = self._create_brief_summary(task_result, task_experiences)

        # Extract common errors if enabled
        if self.include_errors:
            summary["common_errors"] = self._extract_common_errors(task_experiences)

        # Extract success patterns if enabled
        if self.include_success_patterns:
            summary["success_patterns"] = self._extract_success_patterns(task_experiences)

        return summary

    def _extract_key_learnings(self, experiences: List[Experience]) -> List[str]:
        """Extract key learnings from experiences."""
        learnings = []

        # Analyze successful experiences
        successful = [exp for exp in experiences if exp.success]
        if successful:
            # Common tools used in successful cases
            tool_usage = self._analyze_tool_usage(successful)
            if tool_usage:
                top_tools = list(tool_usage.keys())[:3]
                learnings.append(f"Effective tools: {', '.join(top_tools)}")

            # Average number of turns
            avg_turns = sum(len(exp.messages) for exp in successful) / len(successful)
            learnings.append(f"Successful tasks typically require ~{avg_turns:.1f} conversation turns")

        # Analyze failures
        failed = [exp for exp in experiences if not exp.success]
        if failed:
            # Common failure patterns
            failure_tools = self._analyze_tool_usage(failed)
            if failure_tools:
                risky_tools = list(failure_tools.keys())[:2]
                learnings.append(f"Be cautious with: {', '.join(risky_tools)}")

        return learnings

    def _extract_detailed_learnings(self, experiences: List[Experience]) -> List[str]:
        """Extract detailed learnings with more context."""
        detailed = []

        # Successful strategies
        successful = [exp for exp in experiences if exp.success]
        if successful:
            tool_sequences = self._extract_tool_sequences(successful)
            if tool_sequences:
                top_sequence = tool_sequences[0]
                detailed.append(
                    f"Successful approach: Use tools in sequence: {' → '.join(top_sequence)}"
                )

        # Failure analysis
        failed = [exp for exp in experiences if not exp.success]
        if failed:
            error_messages = self._extract_error_messages(failed)
            if error_messages:
                detailed.append(f"Common failure reasons: {'; '.join(error_messages[:2])}")

        return detailed

    def _create_brief_summary(self, task_result, experiences: List[Experience]) -> str:
        """Create a brief one-line summary."""
        success_rate = task_result.train_success_rate
        num_exps = len(experiences)

        if success_rate > 0.8:
            outcome = "Mastered"
        elif success_rate > 0.5:
            outcome = "Partially successful"
        else:
            outcome = "Challenging"

        return f"{outcome} - {success_rate:.1%} success rate across {num_exps} attempts"

    def _extract_common_errors(self, experiences: List[Experience]) -> List[str]:
        """Extract common errors from failed experiences."""
        failed = [exp for exp in experiences if not exp.success]
        if not failed:
            return []

        errors = self._extract_error_messages(failed)
        # Count and return most common
        from collections import Counter
        error_counts = Counter(errors)
        return [error for error, _ in error_counts.most_common(3)]

    def _extract_success_patterns(self, experiences: List[Experience]) -> List[str]:
        """Extract patterns from successful experiences."""
        successful = [exp for exp in experiences if exp.success]
        if not successful:
            return []

        patterns = []

        # Tool usage patterns
        tool_sequences = self._extract_tool_sequences(successful)
        if tool_sequences:
            patterns.append(f"Effective tool sequence: {' → '.join(tool_sequences[0])}")

        # Conversation patterns
        avg_turns = sum(len(exp.messages) for exp in successful) / len(successful)
        patterns.append(f"Optimal conversation length: ~{avg_turns:.0f} turns")

        return patterns

    def _analyze_tool_usage(self, experiences: List[Experience]) -> Dict[str, int]:
        """Analyze tool usage frequency."""
        from collections import Counter
        tools = []

        for exp in experiences:
            for msg in exp.messages:
                if isinstance(msg, (AssistantMessage, UserMessage)) and hasattr(msg, 'tool_calls'):
                    if msg.tool_calls:
                        for tc in msg.tool_calls:
                            tools.append(tc.name)

        return dict(Counter(tools).most_common(10))

    def _extract_tool_sequences(self, experiences: List[Experience]) -> List[List[str]]:
        """Extract common tool sequences."""
        from collections import Counter

        sequences = []
        for exp in experiences:
            seq = []
            for msg in exp.messages:
                if isinstance(msg, (AssistantMessage, UserMessage)) and hasattr(msg, 'tool_calls'):
                    if msg.tool_calls:
                        for tc in msg.tool_calls:
                            seq.append(tc.name)
            if seq:
                sequences.append(tuple(seq))  # Use tuple for hashability

        # Return most common sequences
        if not sequences:
            return []

        seq_counts = Counter(sequences)
        return [list(seq) for seq, _ in seq_counts.most_common(3)]

    def _extract_error_messages(self, experiences: List[Experience]) -> List[str]:
        """Extract error messages from experiences."""
        from tau2.data_model.message import ToolMessage

        errors = []
        for exp in experiences:
            for msg in exp.messages:
                if isinstance(msg, ToolMessage) and hasattr(msg, 'content'):
                    content = str(msg.content).lower()
                    # Look for error indicators
                    if any(word in content for word in ['error', 'invalid', 'failed', 'cannot']):
                        # Extract first 100 chars
                        error_text = str(msg.content)[:100]
                        errors.append(error_text)

        return errors

    def save_state(self, path: str) -> None:
        """Save method state to disk."""
        save_dir = Path(path)
        save_dir.mkdir(parents=True, exist_ok=True)

        state = {
            "config": self.config,
            "task_summaries": self.task_summaries,
        }

        with open(save_dir / "dynamic_prompting_state.json", 'w') as f:
            json.dump(state, f, indent=2)

        logger.info(f"DynamicPromptingMethod state saved to {save_dir}")

    def load_state(self, path: str) -> None:
        """Load method state from disk."""
        load_path = Path(path) / "dynamic_prompting_state.json"

        if not load_path.exists():
            logger.warning(f"No saved state found at {load_path}")
            return

        with open(load_path, 'r') as f:
            state = json.load(f)

        self.config = state.get("config", self.config)
        # Convert string keys back to int
        self.task_summaries = {
            int(k): v for k, v in state.get("task_summaries", {}).items()
        }

        logger.info(f"DynamicPromptingMethod state loaded from {load_path}")
