"""
Example script demonstrating continual learning framework usage.

This script shows how to:
1. Set up a continual learning experiment
2. Compare different CL methods
3. Analyze results
"""

from pathlib import Path

from loguru import logger

from tau2.continual_learning import (
    CLTrainerConfig,
    ContinualLearningTrainer,
    ExperienceBuffer,
    ExperienceReplayMethod,
    NaiveMethod,
    print_cl_metrics,
)


def example_basic_training():
    """Example 1: Basic continual learning training."""
    logger.info("\n" + "=" * 60)
    logger.info("Example 1: Basic Continual Learning Training")
    logger.info("=" * 60)

    # Configure experiment
    config = CLTrainerConfig(
        domain_names=["mock"],  # Using mock domain for faster testing
        task_split_name="base",
        num_tasks_per_domain=3,  # Small number for demo
        num_train_trials=2,
        num_eval_trials=2,
        agent_llm="gpt-4o-mini",
        user_llm="gpt-4o-mini",
        save_dir="./examples/output/basic_training",
        seed=42,
    )

    # Create CL method (using naive baseline)
    buffer = ExperienceBuffer(max_size=1000, seed=42)
    cl_method = NaiveMethod(buffer)

    # Create and run trainer
    trainer = ContinualLearningTrainer(config, cl_method)
    results = trainer.train()

    # Display results
    logger.info("\nFinal Results:")
    print_cl_metrics(results.cl_metrics)

    return results


def example_compare_methods():
    """Example 2: Compare different CL methods."""
    logger.info("\n" + "=" * 60)
    logger.info("Example 2: Comparing CL Methods")
    logger.info("=" * 60)

    # Shared config
    base_config = {
        "domain_names": ["mock"],
        "task_split_name": "base",
        "num_tasks_per_domain": 3,
        "num_train_trials": 2,
        "num_eval_trials": 2,
        "agent_llm": "gpt-4o-mini",
        "user_llm": "gpt-4o-mini",
        "seed": 42,
    }

    methods_to_compare = [
        ("Naive", NaiveMethod, {}),
        (
            "Replay-Uniform",
            ExperienceReplayMethod,
            {"replay_size": 5, "replay_strategy": "uniform"},
        ),
        (
            "Replay-PerTask",
            ExperienceReplayMethod,
            {"replay_size": 10, "replay_strategy": "per_task", "samples_per_task": 2},
        ),
    ]

    results_comparison = {}

    for method_name, method_class, method_config in methods_to_compare:
        logger.info(f"\n{'='*40}")
        logger.info(f"Running: {method_name}")
        logger.info(f"{'='*40}")

        # Create config
        config = CLTrainerConfig(
            **base_config,
            save_dir=f"./examples/output/comparison/{method_name}",
        )

        # Create method
        buffer = ExperienceBuffer(max_size=1000, seed=42)
        cl_method = method_class(buffer, method_config)

        # Train
        trainer = ContinualLearningTrainer(config, cl_method)
        results = trainer.train()

        # Store results
        results_comparison[method_name] = results.cl_metrics

    # Compare results
    logger.info("\n" + "=" * 60)
    logger.info("Comparison Summary")
    logger.info("=" * 60)

    metrics_to_compare = [
        "average_accuracy",
        "forgetting",
        "backward_transfer",
        "final_performance",
    ]

    for metric in metrics_to_compare:
        logger.info(f"\n{metric}:")
        for method_name, metrics in results_comparison.items():
            logger.info(f"  {method_name:.<30} {metrics[metric]:.4f}")

    return results_comparison


def example_custom_task_order():
    """Example 3: Custom task ordering."""
    logger.info("\n" + "=" * 60)
    logger.info("Example 3: Custom Task Ordering")
    logger.info("=" * 60)

    # Define custom task order
    custom_order = ["0", "2", "1"]  # Specific order

    config = CLTrainerConfig(
        domain_names=["mock"],
        task_split_name="base",
        task_order="custom",
        num_tasks_per_domain=3,
        num_train_trials=2,
        num_eval_trials=2,
        agent_llm="gpt-4o-mini",
        user_llm="gpt-4o-mini",
        save_dir="./examples/output/custom_order",
        seed=42,
    )

    buffer = ExperienceBuffer()
    cl_method = NaiveMethod(buffer)

    trainer = ContinualLearningTrainer(config, cl_method)
    results = trainer.train()

    logger.info(f"\nTask sequence used: {results.task_sequence}")
    print_cl_metrics(results.cl_metrics)

    return results


def example_different_replay_strategies():
    """Example 4: Testing different replay strategies."""
    logger.info("\n" + "=" * 60)
    logger.info("Example 4: Different Replay Strategies")
    logger.info("=" * 60)

    strategies = ["uniform", "per_task", "successful_only", "recent"]

    base_config = CLTrainerConfig(
        domain_names=["mock"],
        num_tasks_per_domain=3,
        num_train_trials=2,
        num_eval_trials=2,
        agent_llm="gpt-4o-mini",
        user_llm="gpt-4o-mini",
        seed=42,
    )

    strategy_results = {}

    for strategy in strategies:
        logger.info(f"\nTesting strategy: {strategy}")

        buffer = ExperienceBuffer(seed=42)
        cl_method = ExperienceReplayMethod(
            buffer,
            {
                "replay_size": 5,
                "replay_strategy": strategy,
            },
        )

        base_config.save_dir = f"./examples/output/strategies/{strategy}"
        trainer = ContinualLearningTrainer(base_config, cl_method)
        results = trainer.train()

        strategy_results[strategy] = results.cl_metrics["forgetting"]

    # Display comparison
    logger.info("\n" + "=" * 40)
    logger.info("Forgetting by Replay Strategy")
    logger.info("=" * 40)
    for strategy, forgetting in strategy_results.items():
        logger.info(f"{strategy:.<30} {forgetting:.4f}")

    return strategy_results


def main():
    """Run all examples."""
    logger.info("\n" + "#" * 60)
    logger.info("Continual Learning Framework Examples")
    logger.info("#" * 60)

    # Create output directory
    Path("./examples/output").mkdir(parents=True, exist_ok=True)

    # Run examples
    try:
        # Example 1: Basic training
        example_basic_training()

        # Example 2: Compare methods
        # example_compare_methods()  # Uncomment to run

        # Example 3: Custom task order
        # example_custom_task_order()  # Uncomment to run

        # Example 4: Replay strategies
        # example_different_replay_strategies()  # Uncomment to run

        logger.info("\n" + "#" * 60)
        logger.info("All examples completed successfully!")
        logger.info("#" * 60)

    except Exception as e:
        logger.error(f"Error running examples: {e}")
        raise


if __name__ == "__main__":
    main()
