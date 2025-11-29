"""
Main trainer for continual learning experiments.
"""

import json
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import numpy as np
from loguru import logger

from tau2.continual_learning.buffer import ExperienceBuffer
from tau2.continual_learning.data_model import (
    CLTrainerConfig,
    ContinualLearningResults,
    Experience,
    TaskResult,
)
from tau2.continual_learning.metrics import (
    build_evaluation_matrix,
    compute_cl_metrics,
    print_cl_metrics,
)
from tau2.continual_learning.methods.base import ContinualLearningMethod
from tau2.data_model.simulation import RunConfig, SimulationRun
from tau2.data_model.tasks import Task
from tau2.environment.environment import Environment
from tau2.evaluator.evaluator import EvaluationType, evaluate_simulation
from tau2.orchestrator.orchestrator import Orchestrator
from tau2.registry import registry
from tau2.run import get_tasks
from tau2.user.user_simulator import UserSimulator
from tau2.utils.display import ConsoleDisplay, Text


class ContinualLearningTrainer:
    """
    Main trainer for cross-domain continual learning experiments on tool use tasks.

    This trainer supports:
    1. Multi-domain task sequences (e.g., airline -> retail -> telecom)
    2. Baseline evaluation before continual learning
    3. Per-domain and cross-domain metrics
    """

    def __init__(
        self,
        config: CLTrainerConfig,
        cl_method: ContinualLearningMethod,
    ):
        """
        Initialize the trainer.

        Args:
            config: Trainer configuration
            cl_method: Continual learning method to use
        """
        self.config = config
        self.cl_method = cl_method
        self.buffer = cl_method.buffer

        # Load tasks from multiple domains
        self.tasks, self.task_domains = self._load_and_order_tasks()
        self.task_sequence = [task.id for task in self.tasks]
        self.domain_sequence = self.task_domains

        # Results storage
        self.task_results: list[TaskResult] = []
        self.eval_matrix = np.zeros((len(self.tasks), len(self.tasks)))
        self.baseline_results: Optional[dict[str, dict[str, float]]] = None

        # Create save directory
        if config.save_dir:
            self.save_dir = Path(config.save_dir)
            self.save_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.save_dir = None

        logger.info(f"Initialized Cross-Domain CL Trainer")
        logger.info(f"Domains: {config.domain_names}")
        logger.info(f"Total tasks: {len(self.tasks)}")
        logger.info(f"CL Method: {cl_method.get_name()}")

    def _load_and_order_tasks(self) -> tuple[list[Task], list[str]]:
        """Load and order tasks from multiple domains."""
        # Determine domain order
        domain_names = self.config.domain_names.copy()

        if self.config.domain_order == "random":
            random.seed(self.config.seed)
            random.shuffle(domain_names)
        elif self.config.domain_order == "custom" and self.config.custom_domain_order:
            domain_names = self.config.custom_domain_order

        logger.info(f"Domain order: {' -> '.join(domain_names)}")

        # Load tasks from each domain
        all_tasks = []
        all_domains = []

        for domain_name in domain_names:
            domain_tasks = get_tasks(
                task_set_name=domain_name,
                task_split_name=self.config.task_split_name,
                num_tasks=self.config.num_tasks_per_domain,
            )

            # Optional: shuffle tasks within domain
            if self.config.task_order == "random":
                random.seed(self.config.seed + hash(domain_name))
                random.shuffle(domain_tasks)

            logger.info(f"  {domain_name}: {len(domain_tasks)} tasks")

            all_tasks.extend(domain_tasks)
            all_domains.extend([domain_name] * len(domain_tasks))

        return all_tasks, all_domains

    def train(self) -> ContinualLearningResults:
        """
        Run the full continual learning training process.

        Returns:
            Complete continual learning results
        """
        logger.info("\n" + "=" * 60)
        logger.info("Starting Cross-Domain Continual Learning")
        logger.info("=" * 60)

        # Step 1: Run baseline evaluation if requested
        if self.config.run_baseline:
            logger.info("\n" + "=" * 60)
            logger.info("Step 1: Baseline Evaluation (per domain)")
            logger.info("=" * 60)
            self.baseline_results = self._run_baseline_evaluation()
            self._print_baseline_results()

        # Step 2: Run continual learning
        logger.info("\n" + "=" * 60)
        logger.info("Step 2: Continual Learning Training")
        logger.info("=" * 60)

        for task_idx, (task, domain) in enumerate(zip(self.tasks, self.task_domains)):
            logger.info(f"\n{'='*60}")
            logger.info(f"Task {task_idx + 1}/{len(self.tasks)}: {domain}/{task.id}")
            logger.info(f"{'='*60}")

            # Train on current task
            task_result = self._train_on_task(task_idx, task, domain)
            self.task_results.append(task_result)

            # Evaluate on previous tasks if configured
            if self.config.eval_all_previous and task_idx > 0:
                self._evaluate_on_previous_tasks(task_idx)

            # Save checkpoint
            if self.save_dir and (task_idx + 1) % self.config.checkpoint_frequency == 0:
                self._save_checkpoint(task_idx)

        # Step 3: Compute final CL metrics
        logger.info("\n" + "=" * 60)
        logger.info("Step 3: Computing Continual Learning Metrics")
        logger.info("=" * 60)
        cl_metrics = self._compute_final_metrics()

        # Create final results
        results = ContinualLearningResults(
            task_sequence=self.task_sequence,
            domain_sequence=self.domain_sequence,
            task_results=self.task_results,
            cl_metrics=cl_metrics,
            baseline_results=self.baseline_results,
            method_name=self.cl_method.get_name(),
            method_config=self.cl_method.config,
            run_config=self.config.__dict__,
        )

        # Save final results
        if self.save_dir:
            self._save_results(results)

        # Display results
        self._print_final_summary(results)

        return results

    def _run_baseline_evaluation(self) -> dict[str, dict[str, float]]:
        """
        Run baseline evaluation: train and evaluate on each domain independently.

        Returns:
            Dictionary mapping domain name to metrics
        """
        baseline_results = {}

        for domain_name in self.config.domain_names:
            logger.info(f"\n{'='*40}")
            logger.info(f"Baseline: {domain_name}")
            logger.info(f"{'='*40}")

            # Get tasks for this domain
            domain_tasks_indices = [
                i for i, d in enumerate(self.task_domains) if d == domain_name
            ]
            domain_tasks = [self.tasks[i] for i in domain_tasks_indices]

            if not domain_tasks:
                logger.warning(f"No tasks found for domain {domain_name}")
                continue

            # Evaluate on all tasks in this domain
            domain_results = []
            for task in domain_tasks:
                # Create fresh agent for baseline
                experiences = self._run_task_trials(
                    task=task,
                    domain=domain_name,
                    task_idx=-1,  # Not part of sequence
                    num_trials=self.config.baseline_num_trials,
                    is_training=False,
                )
                success_rate = np.mean([exp.success for exp in experiences])
                avg_reward = np.mean([exp.reward for exp in experiences])
                domain_results.append({
                    "task_id": task.id,
                    "success_rate": success_rate,
                    "reward": avg_reward,
                })

            # Compute domain-level metrics
            avg_success = np.mean([r["success_rate"] for r in domain_results])
            avg_reward = np.mean([r["reward"] for r in domain_results])

            baseline_results[domain_name] = {
                "avg_success_rate": float(avg_success),
                "avg_reward": float(avg_reward),
                "num_tasks": len(domain_tasks),
                "per_task_results": domain_results,
            }

            logger.info(f"  Avg Success Rate: {avg_success:.3f}")
            logger.info(f"  Avg Reward: {avg_reward:.3f}")

        return baseline_results

    def _print_baseline_results(self):
        """Print baseline results in a nice format."""
        if not self.baseline_results:
            return

        logger.info("\n" + "=" * 60)
        logger.info("Baseline Results (Independent Training per Domain)")
        logger.info("=" * 60)

        for domain, results in self.baseline_results.items():
            logger.info(f"\n{domain}:")
            logger.info(f"  Tasks: {results['num_tasks']}")
            logger.info(f"  Avg Success: {results['avg_success_rate']:.3f}")
            logger.info(f"  Avg Reward: {results['avg_reward']:.3f}")

        logger.info("=" * 60)

    def _train_on_task(self, task_idx: int, task: Task, domain: str) -> TaskResult:
        """
        Train on a single task with continual learning method.

        Args:
            task_idx: Index of the task
            task: The task to train on
            domain: Domain this task belongs to

        Returns:
            Task training results
        """
        logger.info(f"Training on task {task.id} (domain: {domain})...")

        # Notify CL method
        agent = self._create_agent(domain)
        self.cl_method.before_task(task_idx, task.id, agent)

        # Collect training experiences
        train_experiences = self._run_task_trials(
            task=task,
            domain=domain,
            task_idx=task_idx,
            num_trials=self.config.num_train_trials,
            is_training=True,
        )

        # Get augmented experiences (CL method may add replay data)
        augmented_experiences = self.cl_method.get_augmented_experiences(
            train_experiences, task_idx
        )

        # Training step (store experiences for future use)
        self.cl_method.after_training_step(
            task_idx, agent, train_experiences, step=0
        )

        # Evaluate on current task
        eval_experiences = self._run_task_trials(
            task=task,
            domain=domain,
            task_idx=task_idx,
            num_trials=self.config.num_eval_trials,
            is_training=False,
        )

        # Compute metrics
        train_rewards = [exp.reward for exp in train_experiences]
        eval_rewards = [exp.reward for exp in eval_experiences]
        train_success = [exp.success for exp in train_experiences]
        eval_success = [exp.success for exp in eval_experiences]

        task_result = TaskResult(
            task_id=task.id,
            task_idx=task_idx,
            domain=domain,
            task=task,
            train_experiences=train_experiences,
            eval_experiences=eval_experiences,
            avg_train_reward=float(np.mean(train_rewards)),
            avg_eval_reward=float(np.mean(eval_rewards)),
            train_success_rate=float(np.mean(train_success)),
            eval_success_rate=float(np.mean(eval_success)),
            metadata={"eval_on_previous_tasks": {}},
        )

        # Update eval matrix
        self.eval_matrix[task_idx, task_idx] = task_result.eval_success_rate

        # Notify CL method
        self.cl_method.after_task(task_idx, task_result, agent)

        logger.info(
            f"Task {task.id} - Train: {task_result.train_success_rate:.3f}, "
            f"Eval: {task_result.eval_success_rate:.3f}"
        )

        return task_result

    def _evaluate_on_previous_tasks(self, current_task_idx: int) -> None:
        """Evaluate current agent on all previous tasks."""
        logger.info(f"Evaluating on previous {current_task_idx} tasks...")

        for prev_task_idx in range(current_task_idx):
            prev_task = self.tasks[prev_task_idx]
            prev_domain = self.task_domains[prev_task_idx]

            # Run evaluation trials
            eval_experiences = self._run_task_trials(
                task=prev_task,
                domain=prev_domain,
                task_idx=prev_task_idx,
                num_trials=self.config.num_eval_trials,
                is_training=False,
            )

            # Compute success rate
            success_rate = float(np.mean([exp.success for exp in eval_experiences]))

            # Update eval matrix
            self.eval_matrix[prev_task_idx, current_task_idx] = success_rate

            # Store in current task result metadata
            self.task_results[current_task_idx].metadata["eval_on_previous_tasks"][
                prev_task_idx
            ] = success_rate

            logger.info(f"  {prev_domain}/{prev_task.id}: {success_rate:.3f}")

    def _run_task_trials(
        self,
        task: Task,
        domain: str,
        task_idx: int,
        num_trials: int,
        is_training: bool,
    ) -> list[Experience]:
        """Run multiple trials on a task."""
        experiences = []

        for trial in range(num_trials):
            simulation_run = self._run_single_trial(
                task=task,
                domain=domain,
                trial=trial,
                seed=self.config.seed + trial + task_idx * 1000,
            )

            experience = Experience.from_simulation_run(
                simulation_run, task_idx, domain
            )
            experiences.append(experience)

        return experiences

    def _run_single_trial(
        self,
        task: Task,
        domain: str,
        trial: int,
        seed: int,
    ) -> SimulationRun:
        """Run a single trial on a task."""
        # Create environment for this domain
        env_constructor = registry.get_env_constructor(domain)
        environment = env_constructor()

        # Create agent
        agent = self._create_agent(domain)

        # Get user tools (may not be available for all domains)
        try:
            user_tools = environment.get_user_tools()
        except Exception:
            user_tools = None

        # Create user
        user = UserSimulator(
            tools=user_tools,
            instructions=str(task.user_scenario),
            llm=self.config.user_llm,
            llm_args={"temperature": self.config.user_llm_temperature},
        )

        # Create orchestrator
        orchestrator = Orchestrator(
            domain=domain,
            agent=agent,
            user=user,
            environment=environment,
            task=task,
            max_steps=self.config.max_steps,
            max_errors=self.config.max_errors,
            seed=seed,
            solo_mode=False,
            validate_communication=False,
        )

        # Run simulation
        simulation = orchestrator.run()

        # Evaluate
        reward_info = evaluate_simulation(
            domain=domain,
            task=task,
            simulation=simulation,
            evaluation_type=EvaluationType.ALL,
            solo_mode=False,
        )

        simulation.reward_info = reward_info
        simulation.trial = trial

        return simulation

    def _create_agent(self, domain: str):
        """Create an agent instance for the specified domain."""
        from tau2.agent.llm_agent import LLMAgent

        env_constructor = registry.get_env_constructor(domain)
        environment = env_constructor()

        agent = LLMAgent(
            tools=environment.get_tools(),
            domain_policy=environment.get_policy(),
            llm=self.config.agent_llm,
            llm_args={"temperature": self.config.agent_llm_temperature},
        )

        return agent

    def _compute_final_metrics(self) -> dict[str, float]:
        """Compute final continual learning metrics."""
        logger.info("\nComputing continual learning metrics...")

        results = ContinualLearningResults(
            task_sequence=self.task_sequence,
            domain_sequence=self.domain_sequence,
            task_results=self.task_results,
            cl_metrics={},
            baseline_results=self.baseline_results,
            method_name=self.cl_method.get_name(),
            method_config=self.cl_method.config,
            run_config=self.config.__dict__,
        )

        cl_metrics = compute_cl_metrics(
            results,
            baseline_results=None,
            eval_matrix=self.eval_matrix,
        )

        # Add per-domain metrics
        for domain in self.config.domain_names:
            domain_indices = [
                i for i, d in enumerate(self.domain_sequence) if d == domain
            ]
            if domain_indices:
                domain_success_rates = [
                    self.task_results[i].eval_success_rate for i in domain_indices
                ]
                cl_metrics[f"{domain}_avg_accuracy"] = float(
                    np.mean(domain_success_rates)
                )

                # Compare with baseline if available
                if self.baseline_results and domain in self.baseline_results:
                    baseline_acc = self.baseline_results[domain]["avg_success_rate"]
                    cl_acc = cl_metrics[f"{domain}_avg_accuracy"]
                    cl_metrics[f"{domain}_vs_baseline"] = cl_acc - baseline_acc

        return cl_metrics

    def _print_final_summary(self, results: ContinualLearningResults):
        """Print comprehensive final summary."""
        logger.info("\n" + "=" * 70)
        logger.info("FINAL RESULTS SUMMARY")
        logger.info("=" * 70)

        # Overall metrics
        print_cl_metrics(results.cl_metrics)

        # Per-domain comparison
        if results.baseline_results:
            logger.info("\n" + "=" * 70)
            logger.info("Per-Domain Comparison (CL vs Baseline)")
            logger.info("=" * 70)

            for domain in self.config.domain_names:
                if domain in results.baseline_results:
                    baseline = results.baseline_results[domain]["avg_success_rate"]
                    cl_key = f"{domain}_avg_accuracy"
                    if cl_key in results.cl_metrics:
                        cl_acc = results.cl_metrics[cl_key]
                        diff = cl_acc - baseline
                        sign = "↑" if diff > 0 else "↓" if diff < 0 else "="

                        logger.info(f"\n{domain}:")
                        logger.info(f"  Baseline:    {baseline:.4f}")
                        logger.info(f"  After CL:    {cl_acc:.4f}")
                        logger.info(f"  Difference:  {diff:+.4f} {sign}")

            logger.info("=" * 70)

    def _save_checkpoint(self, task_idx: int) -> None:
        """Save training checkpoint."""
        if not self.save_dir:
            return

        checkpoint_path = self.save_dir / f"checkpoint_task_{task_idx}.json"

        checkpoint = {
            "task_idx": task_idx,
            "task_sequence": self.task_sequence[: task_idx + 1],
            "domain_sequence": self.domain_sequence[: task_idx + 1],
            "task_results": [
                {
                    "task_id": tr.task_id,
                    "task_idx": tr.task_idx,
                    "domain": tr.domain,
                    "avg_train_reward": tr.avg_train_reward,
                    "avg_eval_reward": tr.avg_eval_reward,
                    "train_success_rate": tr.train_success_rate,
                    "eval_success_rate": tr.eval_success_rate,
                }
                for tr in self.task_results
            ],
            "eval_matrix": self.eval_matrix.tolist(),
            "method_info": self.cl_method.get_method_info(),
            "baseline_results": self.baseline_results,
        }

        with open(checkpoint_path, "w") as f:
            json.dump(checkpoint, f, indent=2)

        logger.info(f"Saved checkpoint to {checkpoint_path}")

    def _save_results(self, results: ContinualLearningResults) -> None:
        """Save final results."""
        if not self.save_dir:
            return

        results_path = self.save_dir / "results.json"

        with open(results_path, "w") as f:
            json.dump(results.to_dict(), f, indent=2)

        logger.info(f"Saved final results to {results_path}")


def create_cl_trainer(
    config: CLTrainerConfig,
    cl_method: ContinualLearningMethod,
) -> ContinualLearningTrainer:
    """
    Factory function to create a continual learning trainer.

    Args:
        config: Trainer configuration
        cl_method: Continual learning method

    Returns:
        Configured trainer
    """
    return ContinualLearningTrainer(config, cl_method)
