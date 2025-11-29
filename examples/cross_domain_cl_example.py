"""
Example: Cross-Domain Continual Learning

This example demonstrates the CORRECT usage pattern for your use case:
1. Baseline evaluation on each domain independently
2. Sequential learning across domains (airline -> retail -> telecom)
3. Comparison of CL performance vs baseline
"""

from loguru import logger

from tau2.continual_learning import (
    CLTrainerConfig,
    ContinualLearningTrainer,
    ExperienceBuffer,
    ExperienceReplayMethod,
    NaiveMethod,
)


def cross_domain_continual_learning_example():
    """
    Example: Cross-domain continual learning on airline -> retail -> telecom.

    This matches your requirements:
    1. First, evaluate baseline performance on each domain independently
    2. Then, train sequentially: airline -> retail -> telecom
    3. Compare CL performance with baseline
    """
    logger.info("\n" + "=" * 70)
    logger.info("Cross-Domain Continual Learning Example")
    logger.info("Domains: airline -> retail -> telecom")
    logger.info("=" * 70)

    # Configure for cross-domain CL
    config = CLTrainerConfig(
        # Three domains in order
        domain_names=["airline", "retail", "telecom"],
        task_split_name="train",
        num_tasks_per_domain=3,  # Use 3 tasks per domain for demo

        # Training settings
        num_train_trials=2,  # Reduce for faster demo
        num_eval_trials=2,

        # IMPORTANT: Run baseline evaluation
        run_baseline=True,  # This evaluates each domain independently first
        baseline_num_trials=2,

        # Agent settings
        agent_llm="gpt-4o-mini",
        user_llm="gpt-4o-mini",

        # Save results
        save_dir="./examples/output/cross_domain_cl",
        seed=42,
    )

    # Create CL method
    buffer = ExperienceBuffer(max_size=1000, seed=42)
    cl_method = NaiveMethod(buffer)  # Start with naive baseline

    # Create and run trainer
    trainer = ContinualLearningTrainer(config, cl_method)

    logger.info("\nThe trainer will:")
    logger.info("1. STEP 1: Evaluate each domain independently (baseline)")
    logger.info("2. STEP 2: Train sequentially across domains with CL")
    logger.info("3. STEP 3: Compare CL vs baseline performance\n")

    results = trainer.train()

    # Results interpretation
    logger.info("\n" + "=" * 70)
    logger.info("RESULTS INTERPRETATION")
    logger.info("=" * 70)

    logger.info("\n📊 Baseline Results:")
    if results.baseline_results:
        for domain, metrics in results.baseline_results.items():
            logger.info(f"  {domain}: {metrics['avg_success_rate']:.3f}")

    logger.info("\n📈 After Continual Learning:")
    for domain in config.domain_names:
        key = f"{domain}_avg_accuracy"
        if key in results.cl_metrics:
            logger.info(f"  {domain}: {results.cl_metrics[key]:.3f}")

    logger.info("\n🔍 Performance Change (CL - Baseline):")
    for domain in config.domain_names:
        key = f"{domain}_vs_baseline"
        if key in results.cl_metrics:
            diff = results.cl_metrics[key]
            sign = "✓ Better" if diff > 0 else "✗ Worse" if diff < 0 else "= Same"
            logger.info(f"  {domain}: {diff:+.3f} {sign}")

    logger.info("\n📉 Continual Learning Metrics:")
    logger.info(f"  Forgetting: {results.cl_metrics.get('forgetting', 0):.3f}")
    logger.info(f"  Average Accuracy: {results.cl_metrics.get('average_accuracy', 0):.3f}")

    return results


def compare_cl_methods():
    """
    Example: Compare different CL methods on cross-domain learning.
    """
    logger.info("\n" + "=" * 70)
    logger.info("Comparing CL Methods Across Domains")
    logger.info("=" * 70)

    base_config = CLTrainerConfig(
        domain_names=["airline", "retail", "telecom"],
        num_tasks_per_domain=2,  # Small for quick demo
        num_train_trials=2,
        num_eval_trials=2,
        run_baseline=True,
        agent_llm="gpt-4o-mini",
        user_llm="gpt-4o-mini",
        seed=42,
    )

    methods = [
        ("Naive", NaiveMethod, {}),
        ("Replay-Uniform", ExperienceReplayMethod, {
            "replay_size": 5,
            "replay_strategy": "uniform"
        }),
        ("Replay-PerTask", ExperienceReplayMethod, {
            "replay_size": 10,
            "replay_strategy": "per_task",
            "samples_per_task": 2
        }),
    ]

    results_comparison = {}

    for method_name, method_class, method_config in methods:
        logger.info(f"\n{'='*50}")
        logger.info(f"Running: {method_name}")
        logger.info(f"{'='*50}")

        # Create config with save directory
        base_config.save_dir = f"./examples/output/comparison/{method_name}"

        # Create method
        buffer = ExperienceBuffer(max_size=1000, seed=42)
        cl_method = method_class(buffer, method_config)

        # Train
        trainer = ContinualLearningTrainer(base_config, cl_method)
        results = trainer.train()

        # Store results
        results_comparison[method_name] = results.cl_metrics

    # Compare results
    logger.info("\n" + "=" * 70)
    logger.info("COMPARISON SUMMARY")
    logger.info("=" * 70)

    metrics_to_compare = [
        "average_accuracy",
        "forgetting",
        "airline_vs_baseline",
        "retail_vs_baseline",
        "telecom_vs_baseline",
    ]

    for metric in metrics_to_compare:
        logger.info(f"\n{metric}:")
        for method_name, metrics in results_comparison.items():
            value = metrics.get(metric, float('nan'))
            logger.info(f"  {method_name:.<30} {value:.4f}")

    return results_comparison


def main():
    """Run the cross-domain continual learning example."""

    # Main example: Cross-domain CL with baseline
    logger.info("\n" + "#" * 70)
    logger.info("EXAMPLE 1: Basic Cross-Domain Continual Learning")
    logger.info("#" * 70)

    results = cross_domain_continual_learning_example()

    # Uncomment to run comparison
    # logger.info("\n\n" + "#" * 70)
    # logger.info("EXAMPLE 2: Comparing CL Methods")
    # logger.info("#" * 70)
    # compare_cl_methods()

    logger.info("\n" + "#" * 70)
    logger.info("Examples completed!")
    logger.info("#" * 70)


if __name__ == "__main__":
    main()
