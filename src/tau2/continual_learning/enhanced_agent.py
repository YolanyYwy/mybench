"""
Enhanced Agent Wrapper for Continual Learning Methods.

This module provides wrapper classes that enhance agents with continual learning
capabilities, such as dynamic prompt injection, memory retrieval, and context augmentation.
"""

from typing import Optional
from tau2.agent.base import LocalAgent
from tau2.data_model.message import SystemMessage


class CLEnhancedAgent(LocalAgent):
    """
    Agent wrapper that enables continual learning methods to inject additional context.

    This wrapper delegates all agent functionality to a base agent while allowing
    CL methods to augment the agent's context (system prompts, examples, etc.)
    without modifying the base agent implementation.

    Attributes:
        base_agent: The underlying agent to wrap
        cl_method: The continual learning method that provides context
        additional_context: Additional context to inject into system messages
    """

    def __init__(
        self,
        base_agent: LocalAgent,
        cl_method=None,
        additional_context: str = "",
    ):
        """
        Initialize the enhanced agent wrapper.

        Args:
            base_agent: The base agent to wrap
            cl_method: Optional CL method that can provide dynamic context
            additional_context: Static additional context to inject
        """
        self.base_agent = base_agent
        self.cl_method = cl_method
        self.additional_context = additional_context

        # Delegate attributes to base agent
        self.tools = base_agent.tools
        self.domain_policy = base_agent.domain_policy

    def generate_next_message(self, message, state):
        """Generate next message using the base agent."""
        return self.base_agent.generate_next_message(message, state)

    def stop(self, message, state):
        """Check stop condition using the base agent."""
        return self.base_agent.stop(message, state)

    def get_init_state(self, message_history):
        """
        Initialize state with enhanced context from CL method.

        This method retrieves the base agent's initial state and then
        augments it with additional context from the CL method.
        """
        # Get base state
        state = self.base_agent.get_init_state(message_history)

        # Add CL-specific context
        context_parts = []

        # Static additional context
        if self.additional_context:
            context_parts.append(self.additional_context)

        # Dynamic context from CL method
        if self.cl_method and hasattr(self.cl_method, 'get_agent_context'):
            dynamic_context = self.cl_method.get_agent_context()
            if dynamic_context:
                context_parts.append(dynamic_context)

        # Inject combined context as system message
        if context_parts:
            combined_context = "\n\n".join(context_parts)
            state.system_messages.append(
                SystemMessage(role="system", content=combined_context)
            )

        return state

    def set_seed(self, seed: int):
        """Set random seed for the base agent."""
        self.base_agent.set_seed(seed)

    def update_context(self, additional_context: str):
        """
        Update the additional context to inject.

        This allows CL methods to dynamically update agent context
        between tasks or during training.

        Args:
            additional_context: New context to inject
        """
        self.additional_context = additional_context


class PromptEnhancedAgent(CLEnhancedAgent):
    """
    Specialized agent wrapper for prompt-based continual learning methods.

    This variant provides additional utilities for managing prompt templates,
    few-shot examples, and knowledge injection.
    """

    def __init__(
        self,
        base_agent: LocalAgent,
        cl_method=None,
        knowledge_base: Optional[str] = None,
        few_shot_examples: Optional[list] = None,
    ):
        """
        Initialize the prompt-enhanced agent.

        Args:
            base_agent: The base agent to wrap
            cl_method: CL method providing dynamic prompts
            knowledge_base: Static knowledge to inject
            few_shot_examples: List of example interactions
        """
        super().__init__(base_agent, cl_method)
        self.knowledge_base = knowledge_base
        self.few_shot_examples = few_shot_examples or []

    def get_init_state(self, message_history):
        """Initialize state with knowledge and examples."""
        state = super().get_init_state(message_history)

        # Add knowledge base
        if self.knowledge_base:
            state.system_messages.append(
                SystemMessage(
                    role="system",
                    content=f"<knowledge_base>\n{self.knowledge_base}\n</knowledge_base>"
                )
            )

        # Add few-shot examples
        if self.few_shot_examples:
            examples_str = self._format_examples(self.few_shot_examples)
            state.system_messages.append(
                SystemMessage(
                    role="system",
                    content=f"<examples>\n{examples_str}\n</examples>"
                )
            )

        return state

    def _format_examples(self, examples: list) -> str:
        """Format few-shot examples for prompt injection."""
        formatted = []
        for i, example in enumerate(examples, 1):
            if isinstance(example, dict):
                formatted.append(f"Example {i}:")
                formatted.append(f"Task: {example.get('task', 'N/A')}")
                formatted.append(f"Solution: {example.get('solution', 'N/A')}")
                if 'outcome' in example:
                    formatted.append(f"Outcome: {example['outcome']}")
            else:
                formatted.append(f"Example {i}: {example}")
            formatted.append("")  # Blank line between examples

        return "\n".join(formatted)

    def add_knowledge(self, knowledge: str):
        """Add to the knowledge base."""
        if self.knowledge_base:
            self.knowledge_base += f"\n\n{knowledge}"
        else:
            self.knowledge_base = knowledge

    def add_examples(self, examples: list):
        """Add few-shot examples."""
        self.few_shot_examples.extend(examples)

    def clear_examples(self):
        """Clear all few-shot examples."""
        self.few_shot_examples = []


class MemoryEnhancedAgent(CLEnhancedAgent):
    """
    Specialized agent wrapper for memory-based continual learning methods.

    This variant provides utilities for managing episodic memories,
    retrieved experiences, and memory-augmented prompts.
    """

    def __init__(
        self,
        base_agent: LocalAgent,
        cl_method=None,
        memory_context: Optional[str] = None,
    ):
        """
        Initialize the memory-enhanced agent.

        Args:
            base_agent: The base agent to wrap
            cl_method: CL method providing memory retrieval
            memory_context: Initial memory context
        """
        super().__init__(base_agent, cl_method)
        self.memory_context = memory_context
        self.retrieved_memories = []

    def get_init_state(self, message_history):
        """Initialize state with retrieved memories."""
        state = super().get_init_state(message_history)

        # Add memory context
        if self.memory_context:
            state.system_messages.append(
                SystemMessage(
                    role="system",
                    content=f"<memory>\n{self.memory_context}\n</memory>"
                )
            )

        # Add retrieved memories
        if self.retrieved_memories:
            memories_str = self._format_memories(self.retrieved_memories)
            state.system_messages.append(
                SystemMessage(
                    role="system",
                    content=f"<retrieved_experiences>\n{memories_str}\n</retrieved_experiences>"
                )
            )

        return state

    def _format_memories(self, memories: list) -> str:
        """Format retrieved memories for prompt injection."""
        formatted = []
        for i, memory in enumerate(memories, 1):
            formatted.append(f"Memory {i}:")
            if isinstance(memory, dict):
                formatted.append(f"  Task: {memory.get('task_id', 'N/A')}")
                formatted.append(f"  Success: {memory.get('success', 'N/A')}")
                if 'summary' in memory:
                    formatted.append(f"  Summary: {memory['summary']}")
                if 'key_actions' in memory:
                    formatted.append(f"  Key Actions: {', '.join(memory['key_actions'])}")
            else:
                formatted.append(f"  {memory}")
            formatted.append("")

        return "\n".join(formatted)

    def set_memories(self, memories: list):
        """Set the retrieved memories to inject."""
        self.retrieved_memories = memories

    def update_memory_context(self, context: str):
        """Update the memory context."""
        self.memory_context = context

    def clear_memories(self):
        """Clear all retrieved memories."""
        self.retrieved_memories = []
