"""
Metrics for evaluating continual learning performance.
"""

from typing import Optional

import numpy as np
from loguru import logger

from tau2.continual_learning.data_model import ContinualLearningResults, TaskResult


def compute_average_accuracy(task_results: list[TaskResult]) -> float:
    """
    Compute average accuracy across all tasks.

    Args:
        task_results: List of task results

    Returns:
        Average evaluation success rate
    """
    if not task_results:
        return 0.0

    accuracies = [tr.eval_success_rate for tr in task_results]
    return float(np.mean(accuracies))


def compute_forgetting(
    results: ContinualLearningResults,
    eval_matrix: Optional[np.ndarray] = None,
) -> float:
    """
    Compute average forgetting metric.

    Forgetting measures how much performance drops on previous tasks
    after learning new tasks.

    Formula: F = (1/N-1) * sum_{i=1}^{N-1} (max_j R_{i,j} - R_{i,N})
    where R_{i,j} is the performance on task i after training on task j.

    Args:
        results: Continual learning results
        eval_matrix: Optional pre-computed evaluation matrix (N_tasks x N_tasks)
                    where entry [i,j] = performance on task i after training on task j

    Returns:
        Average forgetting score (lower is better, 0 = no forgetting)
    """
    if eval_matrix is None:
        # Build eval matrix from results metadata
        n_tasks = len(results.task_sequence)
        eval_matrix = np.zeros((n_tasks, n_tasks))

        for task_result in results.task_results:
            task_idx = task_result.task_idx
            # Current task performance
            eval_matrix[task_idx, task_idx] = task_result.eval_success_rate

            # Performance on previous tasks (if available in metadata)
            if "eval_on_previous_tasks" in task_result.metadata:
                for prev_task_idx, perf in task_result.metadata[
                    "eval_on_previous_tasks"
                ].items():
                    eval_matrix[prev_task_idx, task_idx] = perf

    n_tasks = eval_matrix.shape[0]
    if n_tasks <= 1:
        return 0.0

    forgetting_scores = []
    for i in range(n_tasks - 1):  # For each task except the last
        max_perf = np.max(eval_matrix[i, : i + 2])  # Max performance up to this task
        final_perf = eval_matrix[i, -1]  # Performance after all tasks
        forgetting_scores.append(max_perf - final_perf)

    return float(np.mean(forgetting_scores))


def compute_forward_transfer(
    results: ContinualLearningResults,
    baseline_results: Optional[ContinualLearningResults] = None,
) -> float:
    """
    Compute forward transfer metric.

    Forward transfer measures how learning previous tasks helps with new tasks.

    Formula: FT = (1/N-1) * sum_{i=2}^{N} (R_{i,i} - b_i)
    where R_{i,i} is performance on task i after sequential training,
    and b_i is baseline performance (training on task i from scratch).

    Args:
        results: Continual learning results
        baseline_results: Optional baseline results (training each task independently)

    Returns:
        Average forward transfer (positive = beneficial, negative = harmful)
    """
    if baseline_results is None:
        logger.warning(
            "No baseline results provided for forward transfer computation. "
            "Returning 0.0"
        )
        return 0.0

    n_tasks = len(results.task_sequence)
    if n_tasks <= 1:
        return 0.0

    transfer_scores = []
    for i in range(1, n_tasks):  # Start from second task
        cl_perf = results.task_results[i].eval_success_rate
        baseline_perf = baseline_results.task_results[i].eval_success_rate
        transfer_scores.append(cl_perf - baseline_perf)

    return float(np.mean(transfer_scores))


def compute_backward_transfer(
    results: ContinualLearningResults,
    eval_matrix: Optional[np.ndarray] = None,
) -> float:
    """
    Compute backward transfer metric.

    Backward transfer measures how learning new tasks affects previous tasks.
    This is essentially the negative of forgetting.

    Args:
        results: Continual learning results
        eval_matrix: Optional pre-computed evaluation matrix

    Returns:
        Average backward transfer (positive = beneficial, negative = harmful)
    """
    return -compute_forgetting(results, eval_matrix)


def compute_final_performance(task_results: list[TaskResult]) -> float:
    """
    Compute final performance across all tasks.

    Args:
        task_results: List of task results

    Returns:
        Average success rate on final evaluation
    """
    return compute_average_accuracy(task_results)


def compute_learning_curve_area(task_results: list[TaskResult]) -> float:
    """
    Compute area under the learning curve.

    This measures the cumulative performance throughout training.

    Args:
        task_results: List of task results ordered by task sequence

    Returns:
        Average performance across all tasks during training
    """
    if not task_results:
        return 0.0

    # Average performance at each time step
    performances = []
    for i, task_result in enumerate(task_results):
        # Average performance on all tasks seen so far
        avg_perf = np.mean([tr.eval_success_rate for tr in task_results[: i + 1]])
        performances.append(avg_perf)

    return float(np.mean(performances))


def compute_cl_metrics(
    results: ContinualLearningResults,
    baseline_results: Optional[ContinualLearningResults] = None,
    eval_matrix: Optional[np.ndarray] = None,
) -> dict[str, float]:
    """
    Compute all continual learning metrics.

    Args:
        results: Continual learning results
        baseline_results: Optional baseline results for forward transfer
        eval_matrix: Optional evaluation matrix for forgetting computation

    Returns:
        Dictionary of metric name -> value
    """
    metrics = {
        "average_accuracy": compute_average_accuracy(results.task_results),
        "final_performance": compute_final_performance(results.task_results),
        "forgetting": compute_forgetting(results, eval_matrix),
        "backward_transfer": compute_backward_transfer(results, eval_matrix),
        "learning_curve_area": compute_learning_curve_area(results.task_results),
    }

    if baseline_results is not None:
        metrics["forward_transfer"] = compute_forward_transfer(results, baseline_results)

    return metrics


def build_evaluation_matrix(
    task_results: list[TaskResult],
) -> np.ndarray:
    """
    Build evaluation matrix from task results.

    Matrix[i, j] = performance on task i after training on task j

    Args:
        task_results: List of task results

    Returns:
        Evaluation matrix (N_tasks x N_tasks)
    """
    n_tasks = len(task_results)
    eval_matrix = np.zeros((n_tasks, n_tasks))

    for task_result in task_results:
        task_idx = task_result.task_idx
        eval_matrix[task_idx, task_idx] = task_result.eval_success_rate

        if "eval_on_previous_tasks" in task_result.metadata:
            for prev_idx, perf in task_result.metadata["eval_on_previous_tasks"].items():
                eval_matrix[int(prev_idx), task_idx] = perf

    return eval_matrix


def print_cl_metrics(metrics: dict[str, float]) -> None:
    """Pretty print continual learning metrics."""
    logger.info("\n" + "=" * 50)
    logger.info("Continual Learning Metrics")
    logger.info("=" * 50)

    for metric_name, value in metrics.items():
        logger.info(f"{metric_name:.<30} {value:.4f}")

    logger.info("=" * 50 + "\n")
