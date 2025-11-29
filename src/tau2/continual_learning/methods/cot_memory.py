"""
Chain-of-Thought (CoT) Memory for Continual Learning.

This module implements CoT-based continual learning that extracts reasoning chains
from successful experiences and uses them as few-shot examples for future tasks.
"""

from typing import List, Dict, Any, Optional
from pathlib import Path
import json
import re

from loguru import logger

from tau2.continual_learning.buffer import ExperienceBuffer
from tau2.continual_learning.data_model import Experience
from tau2.continual_learning.methods.base import ContinualLearningMethod
from tau2.continual_learning.prompt_templates import PromptTemplates
from tau2.continual_learning.enhanced_agent import PromptEnhancedAgent
from tau2.data_model.message import AssistantMessage, UserMessage, ToolMessage


class CoTMemoryMethod(ContinualLearningMethod):
    """
    Chain-of-Thought Memory method that learns from reasoning chains.

    This method extracts step-by-step reasoning from successful task completions
    and provides them as few-shot examples to guide future problem-solving.
    """

    def __init__(
        self,
        buffer: ExperienceBuffer,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize CoT memory method.

        Args:
            buffer: Experience buffer for storage
            config: Configuration dict with:
                - num_examples: Number of CoT examples to provide (default 3)
                - min_success_reward: Min reward to consider example (default 0.7)
                - extract_mode: 'full' or 'summary' (default 'summary')
                - similarity_threshold: Min similarity for retrieval (default 0.0)
        """
        super().__init__(buffer, config)

        # Configuration
        self.num_examples = config.get("num_examples", 3)
        self.min_success_reward = config.get("min_success_reward", 0.7)
        self.extract_mode = config.get("extract_mode", "summary")
        self.similarity_threshold = config.get("similarity_threshold", 0.0)

        # Storage for reasoning chains
        self.reasoning_chains: List[Dict[str, Any]] = []

        logger.info(
            f"CoTMemoryMethod initialized with num_examples={self.num_examples}, "
            f"extract_mode={self.extract_mode}"
        )

    def before_task(self, task_idx: int, task_id: str, agent) -> None:
        """
        Provide CoT examples before task starts.

        Args:
            task_idx: Current task index
            task_id: Current task ID
            agent: Agent to enhance with CoT examples
        """
        if task_idx == 0 or not self.reasoning_chains:
            logger.debug("No CoT examples available yet")
            return

        # Select relevant CoT examples
        relevant_examples = self._select_relevant_examples(task_id, agent)

        # Inject into agent if it's a PromptEnhancedAgent
        if isinstance(agent, PromptEnhancedAgent):
            agent.add_examples(relevant_examples)
            logger.info(
                f"Task {task_idx}: Injected {len(relevant_examples)} CoT examples"
            )
        else:
            logger.debug(
                f"Task {task_idx}: Agent is not PromptEnhancedAgent, "
                "examples prepared but not injected"
            )

    def after_training_step(
        self,
        task_idx: int,
        agent,
        experiences: List[Experience],
        step: int,
    ) -> None:
        """
        Extract and store reasoning chains from experiences.

        Args:
            task_idx: Current task index
            agent: Agent being trained
            experiences: Experiences from current step
            step: Training step number
        """
        # Add to buffer
        self.buffer.add_batch(experiences)

        # Extract reasoning chains from successful experiences
        for exp in experiences:
            if self._is_high_quality_example(exp):
                chain = self._extract_reasoning_chain(exp)
                if chain:
                    self.reasoning_chains.append({
                        "task_idx": task_idx,
                        "task_id": exp.task_id,
                        "domain": exp.domain,
                        "chain": chain,
                        "success": exp.success,
                        "reward": exp.reward,
                    })

        logger.debug(
            f"Task {task_idx}, step {step}: Extracted reasoning chains, "
            f"total chains={len(self.reasoning_chains)}"
        )

    def get_agent_context(self) -> str:
        """
        Get CoT examples to inject into agent prompts.

        Returns:
            Formatted CoT examples prompt
        """
        if not self.reasoning_chains:
            return ""

        # Take most recent high-quality chains
        recent_chains = self.reasoning_chains[-self.num_examples:]

        # Format for template
        cot_examples = []
        for chain_data in recent_chains:
            cot_examples.append({
                "task": chain_data["task_id"],
                "reasoning_chain": chain_data["chain"]["steps"],
                "conclusion": chain_data["chain"].get("conclusion", ""),
                "success": chain_data["success"],
            })

        return PromptTemplates.chain_of_thought_template(cot_examples)

    def _select_relevant_examples(
        self,
        task_id: str,
        agent,
    ) -> List[Dict[str, Any]]:
        """
        Select relevant CoT examples for current task.

        Args:
            task_id: Current task ID
            agent: Current agent

        Returns:
            List of relevant example dicts
        """
        # Filter high-quality chains
        quality_chains = [
            c for c in self.reasoning_chains
            if c["success"] and c.get("reward", 0) >= self.min_success_reward
        ]

        # Simple relevance: prefer same domain
        domain = getattr(agent, 'domain', None)
        if domain:
            domain_chains = [c for c in quality_chains if c.get("domain") == domain]
            if domain_chains:
                quality_chains = domain_chains

        # Take most recent
        recent_chains = quality_chains[-self.num_examples:]

        # Format as examples
        examples = []
        for chain_data in recent_chains:
            examples.append({
                "task": chain_data["task_id"],
                "reasoning": self._format_reasoning(chain_data["chain"]),
                "outcome": "Success" if chain_data["success"] else "Failed",
            })

        return examples

    def _is_high_quality_example(self, exp: Experience) -> bool:
        """
        Check if experience is high-quality enough to extract.

        Args:
            exp: Experience to check

        Returns:
            True if high quality
        """
        # Must be successful
        if not exp.success:
            return False

        # Must have good reward
        if exp.reward is not None and exp.reward < self.min_success_reward:
            return False

        # Must have reasonable conversation length (not too short)
        if len(exp.messages) < 3:
            return False

        return True

    def _extract_reasoning_chain(self, exp: Experience) -> Optional[Dict[str, Any]]:
        """
        Extract reasoning chain from experience messages.

        Args:
            exp: Experience to extract from

        Returns:
            Dict with reasoning chain or None
        """
        steps = []
        tool_calls = []

        for i, msg in enumerate(exp.messages):
            # Extract assistant reasoning
            if isinstance(msg, AssistantMessage):
                if msg.content and self._contains_reasoning(msg.content):
                    # Extract reasoning text
                    reasoning_text = self._clean_text(msg.content)
                    if reasoning_text:
                        steps.append(f"Think: {reasoning_text}")

                # Extract tool calls as action steps
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    for tc in msg.tool_calls:
                        tool_calls.append(tc.name)
                        steps.append(f"Act: Use {tc.name}")

            # Extract user responses as observations
            elif isinstance(msg, UserMessage) and msg.content:
                # Only include if contains meaningful info
                if len(str(msg.content)) > 10:
                    obs_text = self._clean_text(msg.content[:200])
                    steps.append(f"Observe: {obs_text}")

        if not steps:
            return None

        # Create chain summary
        chain = {
            "steps": steps,
            "tool_sequence": tool_calls,
            "num_steps": len(steps),
        }

        # Add conclusion if successful
        if exp.success:
            chain["conclusion"] = "Task completed successfully"

        return chain

    def _contains_reasoning(self, text: str) -> bool:
        """Check if text contains reasoning indicators."""
        if not text:
            return False

        # Look for reasoning patterns
        reasoning_patterns = [
            r'\bthink\b', r'\breason\b', r'\bconsider\b',
            r'\banalyze\b', r'\bevaluate\b', r'\bdetermine\b',
            r'\bfirst\b', r'\bnext\b', r'\bthen\b', r'\bfinally\b',
            r'\bbecause\b', r'\bsince\b', r'\btherefore\b',
        ]

        text_lower = text.lower()
        return any(re.search(pattern, text_lower) for pattern in reasoning_patterns)

    def _clean_text(self, text: str) -> str:
        """Clean and truncate text."""
        if not text:
            return ""

        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', str(text))

        # Truncate if too long
        max_length = 150 if self.extract_mode == 'summary' else 300
        if len(text) > max_length:
            text = text[:max_length] + "..."

        return text.strip()

    def _format_reasoning(self, chain: Dict[str, Any]) -> str:
        """Format reasoning chain for display."""
        steps = chain.get("steps", [])

        if self.extract_mode == 'summary':
            # Summarize into key steps only
            think_steps = [s for s in steps if s.startswith("Think:")]
            act_steps = [s for s in steps if s.startswith("Act:")]

            formatted = []
            if think_steps:
                formatted.append(f"Reasoning: {think_steps[0]}")
            if act_steps:
                formatted.append(f"Actions: {', '.join([s.replace('Act: ', '') for s in act_steps])}")

            return "\n".join(formatted)
        else:
            # Full chain
            return "\n".join([f"{i+1}. {step}" for i, step in enumerate(steps)])

    def get_augmented_experiences(
        self,
        current_experiences: List[Experience],
        task_idx: int,
    ) -> List[Experience]:
        """
        Optionally augment with synthetic CoT teaching experiences.

        Args:
            current_experiences: Current task experiences
            task_idx: Current task index

        Returns:
            Augmented experiences (for now, just return current)
        """
        # For now, just return current experiences
        # Could implement synthetic example generation here
        return current_experiences

    def save_state(self, path: str) -> None:
        """Save method state to disk."""
        save_dir = Path(path)
        save_dir.mkdir(parents=True, exist_ok=True)

        state = {
            "config": self.config,
            "reasoning_chains": self.reasoning_chains,
        }

        with open(save_dir / "cot_memory_state.json", 'w') as f:
            json.dump(state, f, indent=2)

        logger.info(f"CoTMemoryMethod state saved to {save_dir}")

    def load_state(self, path: str) -> None:
        """Load method state from disk."""
        load_path = Path(path) / "cot_memory_state.json"

        if not load_path.exists():
            logger.warning(f"No saved state found at {load_path}")
            return

        with open(load_path, 'r') as f:
            state = json.load(f)

        self.config = state.get("config", self.config)
        self.reasoning_chains = state.get("reasoning_chains", [])

        logger.info(f"CoTMemoryMethod state loaded from {load_path}")
