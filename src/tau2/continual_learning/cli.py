"""
Command-line interface for continual learning experiments.
"""

import click
from loguru import logger

from tau2.continual_learning import (
    CLTrainerConfig,
    ContinualLearningTrainer,
    ExperienceBuffer,
    ExperienceReplayMethod,
    NaiveMethod,
)


@click.group()
def cl_cli():
    """Continual learning commands for tau2."""
    pass


@cl_cli.command(name="train")
@click.option(
    "--domains",
    type=str,
    default="airline,retail,telecom",
    help="Comma-separated list of domains (e.g., 'airline,retail,telecom')",
)
@click.option(
    "--method",
    type=str,
    default="naive",
    help="CL method (naive, experience_replay)",
)
@click.option(
    "--task-split",
    type=str,
    default="train",
    help="Task split to use (train, test, base)",
)
@click.option(
    "--num-tasks-per-domain",
    type=int,
    default=None,
    help="Number of tasks per domain (None = all)",
)
@click.option(
    "--num-train-trials",
    type=int,
    default=4,
    help="Number of training trials per task",
)
@click.option(
    "--num-eval-trials",
    type=int,
    default=4,
    help="Number of evaluation trials per task",
)
@click.option(
    "--run-baseline/--no-baseline",
    default=True,
    help="Run baseline evaluation before continual learning",
)
@click.option(
    "--agent-llm",
    type=str,
    default="gpt-4o-mini",
    help="LLM for agent",
)
@click.option(
    "--user-llm",
    type=str,
    default="gpt-4o-mini",
    help="LLM for user simulator",
)
@click.option(
    "--replay-size",
    type=int,
    default=10,
    help="Number of experiences to replay (for experience_replay method)",
)
@click.option(
    "--replay-strategy",
    type=str,
    default="uniform",
    help="Replay sampling strategy (uniform, per_task, successful_only, recent)",
)
@click.option(
    "--save-dir",
    type=str,
    default=None,
    help="Directory to save results",
)
@click.option(
    "--seed",
    type=int,
    default=300,
    help="Random seed",
)
def train_cl(
    domains,
    method,
    task_split,
    num_tasks_per_domain,
    num_train_trials,
    num_eval_trials,
    run_baseline,
    agent_llm,
    user_llm,
    replay_size,
    replay_strategy,
    save_dir,
    seed,
):
    """Run cross-domain continual learning training."""

    # Parse domains
    domain_names = [d.strip() for d in domains.split(",")]

    # Create config
    config = CLTrainerConfig(
        domain_names=domain_names,
        task_split_name=task_split,
        num_tasks_per_domain=num_tasks_per_domain,
        num_train_trials=num_train_trials,
        num_eval_trials=num_eval_trials,
        run_baseline=run_baseline,
        agent_llm=agent_llm,
        user_llm=user_llm,
        save_dir=save_dir,
        seed=seed,
        cl_method_name=method,
    )

    # Create buffer
    buffer = ExperienceBuffer(max_size=None, seed=seed)

    # Create CL method
    if method == "naive":
        cl_method = NaiveMethod(buffer)
    elif method == "experience_replay":
        cl_method = ExperienceReplayMethod(
            buffer,
            config={
                "replay_size": replay_size,
                "replay_strategy": replay_strategy,
            },
        )
    else:
        raise ValueError(f"Unknown CL method: {method}")

    # Create trainer
    trainer = ContinualLearningTrainer(config, cl_method)

    # Run training
    logger.info(f"Starting cross-domain continual learning")
    logger.info(f"Domains: {' -> '.join(domain_names)}")
    logger.info(f"Method: {method}")
    results = trainer.train()

    logger.info("\nTraining completed!")
    logger.info(f"Results saved to: {save_dir}")


if __name__ == "__main__":
    cl_cli()
