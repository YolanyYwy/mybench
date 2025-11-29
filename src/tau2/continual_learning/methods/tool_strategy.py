"""
Tool Strategy Learning for Continual Learning.

This module implements tool usage pattern mining and strategy recommendation
based on historical tool call sequences and their outcomes.
"""

from typing import List, Dict, Any, Optional, Tuple
from collections import Counter, defaultdict
from pathlib import Path
import json

from loguru import logger

from tau2.continual_learning.buffer import ExperienceBuffer
from tau2.continual_learning.data_model import Experience
from tau2.continual_learning.methods.base import ContinualLearningMethod
from tau2.continual_learning.prompt_templates import PromptTemplates
from tau2.continual_learning.enhanced_agent import PromptEnhancedAgent
from tau2.data_model.message import AssistantMessage, UserMessage


class ToolStrategyMethod(ContinualLearningMethod):
    """
    Tool strategy learning method that mines successful tool usage patterns.

    This method learns which tool sequences and strategies are most effective
    for different tasks and provides recommendations to agents.
    """

    def __init__(
        self,
        buffer: ExperienceBuffer,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize tool strategy method.

        Args:
            buffer: Experience buffer for storage
            config: Configuration dict with:
                - min_pattern_support: Min occurrences for pattern (default 3)
                - min_success_rate: Min success rate for strategy (default 0.6)
                - max_sequence_length: Max tool sequence length (default 5)
                - num_strategies: Number of strategies to recommend (default 5)
        """
        super().__init__(buffer, config)

        # Configuration
        self.min_pattern_support = config.get("min_pattern_support", 3)
        self.min_success_rate = config.get("min_success_rate", 0.6)
        self.max_sequence_length = config.get("max_sequence_length", 5)
        self.num_strategies = config.get("num_strategies", 5)

        # Tool usage tracking
        self.tool_sequences: Dict[str, Dict[str, List[bool]]] = defaultdict(
            lambda: defaultdict(list)
        )  # domain -> sequence -> [success/fail]

        self.tool_success_rates: Dict[str, Dict[str, float]] = defaultdict(dict)
        self.domain_strategies: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

        logger.info(
            f"ToolStrategyMethod initialized with min_support={self.min_pattern_support}"
        )

    def before_task(self, task_idx: int, task_id: str, agent) -> None:
        """
        Provide tool strategy recommendations before task.

        Args:
            task_idx: Current task index
            task_id: Current task ID
            agent: Agent to enhance with strategies
        """
        if task_idx == 0:
            logger.debug("First task, no strategies available yet")
            return

        # Get domain
        domain = getattr(agent, 'domain', 'unknown')

        # Get strategies for this domain
        strategies = self._get_domain_strategies(domain)

        # Inject into agent
        if isinstance(agent, PromptEnhancedAgent) and strategies:
            strategy_prompt = PromptTemplates.tool_strategy_template(strategies)
            agent.add_knowledge(strategy_prompt)
            logger.info(
                f"Task {task_idx}: Injected {len(strategies)} tool strategies for domain {domain}"
            )

        logger.debug(f"Task {task_idx} ({task_id}): Tool strategies ready")

    def after_training_step(
        self,
        task_idx: int,
        agent,
        experiences: List[Experience],
        step: int,
    ) -> None:
        """
        Learn tool usage patterns from experiences.

        Args:
            task_idx: Current task index
            agent: Agent being trained
            experiences: Experiences from current step
            step: Training step number
        """
        # Add to buffer
        self.buffer.add_batch(experiences)

        # Extract and record tool patterns
        for exp in experiences:
            domain = exp.domain if exp.domain else 'unknown'
            tool_sequence = self._extract_tool_sequence(exp)

            if tool_sequence:
                # Record full sequence
                seq_str = " → ".join(tool_sequence)
                self.tool_sequences[domain][seq_str].append(exp.success)

                # Record sub-sequences
                for length in range(2, min(len(tool_sequence) + 1, self.max_sequence_length + 1)):
                    for i in range(len(tool_sequence) - length + 1):
                        subseq = tool_sequence[i:i+length]
                        subseq_str = " → ".join(subseq)
                        self.tool_sequences[domain][subseq_str].append(exp.success)

        logger.debug(
            f"Task {task_idx}, step {step}: Recorded tool patterns from {len(experiences)} experiences"
        )

    def after_task(
        self,
        task_idx: int,
        task_result,
        agent,
    ) -> None:
        """
        Update strategy database after task completion.

        Args:
            task_idx: Completed task index
            task_result: Task result
            agent: Trained agent
        """
        domain = task_result.domain if task_result.domain else 'unknown'

        # Compute success rates for all sequences
        self._compute_success_rates(domain)

        # Extract top strategies
        self._update_domain_strategies(domain)

        logger.info(
            f"Task {task_idx} completed: Updated strategies for domain {domain}"
        )

    def _extract_tool_sequence(self, exp: Experience) -> List[str]:
        """Extract sequence of tool calls from experience."""
        tools = []

        for msg in exp.messages:
            if isinstance(msg, (AssistantMessage, UserMessage)) and hasattr(msg, 'tool_calls'):
                if msg.tool_calls:
                    for tc in msg.tool_calls:
                        tools.append(tc.name)

        return tools

    def _compute_success_rates(self, domain: str) -> None:
        """Compute success rates for tool sequences in a domain."""
        for seq, outcomes in self.tool_sequences[domain].items():
            if len(outcomes) >= self.min_pattern_support:
                success_rate = sum(outcomes) / len(outcomes)
                self.tool_success_rates[domain][seq] = success_rate

    def _update_domain_strategies(self, domain: str) -> None:
        """Update strategy recommendations for a domain."""
        strategies = []

        for seq, success_rate in self.tool_success_rates[domain].items():
            if success_rate >= self.min_success_rate:
                count = len(self.tool_sequences[domain][seq])

                # Compute strategy score (success rate * log frequency)
                import math
                score = success_rate * math.log(1 + count)

                strategies.append({
                    "pattern": seq,
                    "success_rate": success_rate,
                    "count": count,
                    "score": score,
                    "context": self._infer_context(seq),
                })

        # Sort by score
        strategies.sort(key=lambda x: x["score"], reverse=True)

        # Keep top strategies
        self.domain_strategies[domain] = strategies[:self.num_strategies * 2]

    def _infer_context(self, sequence: str) -> str:
        """Infer when this tool sequence is useful."""
        # Simple heuristic based on tool names
        tools = sequence.split(" → ")

        if any("search" in t.lower() for t in tools):
            return "Information retrieval tasks"
        elif any("update" in t.lower() or "modify" in t.lower() for t in tools):
            return "Data modification tasks"
        elif any("create" in t.lower() or "add" in t.lower() for t in tools):
            return "Data creation tasks"
        elif any("get" in t.lower() or "fetch" in t.lower() for t in tools):
            return "Data retrieval tasks"
        else:
            return "General tasks"

    def _get_domain_strategies(self, domain: str) -> List[Dict[str, Any]]:
        """Get top strategies for a domain."""
        strategies = self.domain_strategies.get(domain, [])

        # Format for template
        formatted = []
        for strategy in strategies[:self.num_strategies]:
            formatted.append({
                "pattern": strategy["pattern"],
                "success_rate": strategy["success_rate"],
                "count": strategy["count"],
                "context": strategy["context"],
                "description": f"This pattern has been successful in {strategy['count']} cases",
            })

        return formatted

    def get_agent_context(self) -> str:
        """
        Get tool strategy context for agent prompts.

        Returns:
            Formatted tool strategy prompt
        """
        # Aggregate strategies across all domains
        all_strategies = []
        for domain, strategies in self.domain_strategies.items():
            for strategy in strategies[:self.num_strategies]:
                all_strategies.append({
                    "pattern": f"{strategy['pattern']} (domain: {domain})",
                    "success_rate": strategy["success_rate"],
                    "count": strategy["count"],
                    "context": strategy["context"],
                })

        # Sort by success rate and take top
        all_strategies.sort(key=lambda x: x["success_rate"], reverse=True)
        top_strategies = all_strategies[:self.num_strategies]

        if not top_strategies:
            return ""

        return PromptTemplates.tool_strategy_template(top_strategies)

    def save_state(self, path: str) -> None:
        """Save method state to disk."""
        save_dir = Path(path)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Convert defaultdicts to regular dicts for JSON serialization
        state = {
            "config": self.config,
            "tool_sequences": {
                domain: {seq: outcomes for seq, outcomes in seqs.items()}
                for domain, seqs in self.tool_sequences.items()
            },
            "tool_success_rates": {
                domain: dict(rates)
                for domain, rates in self.tool_success_rates.items()
            },
            "domain_strategies": {
                domain: strategies
                for domain, strategies in self.domain_strategies.items()
            },
        }

        with open(save_dir / "tool_strategy_state.json", 'w') as f:
            json.dump(state, f, indent=2)

        logger.info(f"ToolStrategyMethod state saved to {save_dir}")

    def load_state(self, path: str) -> None:
        """Load method state from disk."""
        load_path = Path(path) / "tool_strategy_state.json"

        if not load_path.exists():
            logger.warning(f"No saved state found at {load_path}")
            return

        with open(load_path, 'r') as f:
            state = json.load(f)

        self.config = state.get("config", self.config)

        # Load tool sequences
        self.tool_sequences = defaultdict(lambda: defaultdict(list))
        for domain, seqs in state.get("tool_sequences", {}).items():
            for seq, outcomes in seqs.items():
                self.tool_sequences[domain][seq] = outcomes

        # Load success rates
        self.tool_success_rates = defaultdict(dict)
        for domain, rates in state.get("tool_success_rates", {}).items():
            self.tool_success_rates[domain] = rates

        # Load strategies
        self.domain_strategies = defaultdict(list)
        for domain, strategies in state.get("domain_strategies", {}).items():
            self.domain_strategies[domain] = strategies

        logger.info(f"ToolStrategyMethod state loaded from {load_path}")
