"""
ClearML hyperparameter optimization script
"""
import argparse

from clearml import Task
from clearml.automation import (
    HyperParameterOptimizer,
    UniformParameterRange,
    DiscreteParameterRange,
)


def job_complete_callback(
    job_id,                 # type: str
    objective_value,        # type: float
    objective_iteration,    # type: int
    job_parameters,         # type: dict
    top_performance_job_id  # type: str
):
    print('Job completed!', job_id, objective_value, objective_iteration, job_parameters)
    if job_id == top_performance_job_id:
        print('WOOT WOOT we broke the record! Objective reached {}'.format(objective_value))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-task-id", default="a1ab901fc0474a0cabf8f34bffe688f1")
    parser.add_argument("--task-name", default="poisson-scale HPO")
    parser.add_argument("--queue", default="muller")
    parser.add_argument("--max-epochs", type=int, default=32)
    parser.add_argument("--lr-min", type=float, default=1e-5)
    parser.add_argument("--lr-max", type=float, default=1e-2)
    parser.add_argument("--total-jobs", type=int, default=8)
    parser.add_argument("--max-concurrent", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--optimizer", choices=["random", "optuna", "bohb"], default="random",
                        help="Search strategy; optuna (TPE + median pruning) and bohb stop poor trials early")
    parser.add_argument("--min-iters", type=int, default=3,
                        help="Early stopping: minimum epochs before a trial can be pruned")
    parser.add_argument("--poll-min", type=float, default=None,
                        help="Controller polling period in minutes (default 2.0 for random, 0.2 otherwise)")
    return parser.parse_args()


def get_optimizer_settings(args):
    """Return (optimizer_class, extra kwargs) for the chosen search strategy."""
    if args.optimizer == "random":
        from clearml.automation import RandomSearch
        return RandomSearch, {}
    # Early stopping strategies: objective is reported once per epoch (iterations 1..max_epochs).
    # max_iteration_per_job is one past the last epoch so finished trials complete instead of being aborted.
    kwargs = dict(max_iteration_per_job=args.max_epochs + 1, min_iteration_per_job=args.min_iters)
    if args.optimizer == "optuna":
        import optuna
        from clearml.automation.optuna import OptimizerOptuna
        # Median pruning against completed trials at the same epoch
        kwargs["optuna_pruner"] = optuna.pruners.MedianPruner(
            n_startup_trials=2, n_warmup_steps=args.min_iters)
        return OptimizerOptuna, kwargs
    from clearml.automation.hpbandster import OptimizerBOHB
    kwargs["eta"] = 3
    return OptimizerBOHB, kwargs


def main():
    args = parse_args()

    # Initialize the HPO task
    task = Task.init(
        project_name="FNO Tests",
        task_name=args.task_name,
        task_type=Task.TaskTypes.optimizer
    )

    # Base task for the hyperparameter optimization
    base_task = Task.get_task(task_id=args.base_task_id)

    optimizer_class, optimizer_kwargs = get_optimizer_settings(args)
    if args.poll_min is None:
        args.poll_min = 2.0 if args.optimizer == "random" else 0.2

    # Define the hyperparameter search space
    print(f"Constructing HyperParameterOptimizer ({optimizer_class.__name__})")
    optimizer = HyperParameterOptimizer(
        base_task_id=base_task.id,
        hyper_parameters=[
            UniformParameterRange("General/lr", min_value=args.lr_min, max_value=args.lr_max),
            DiscreteParameterRange("General/max_epochs", values=[args.max_epochs]),
        ],
        objective_metric_title="best_val_err",
        objective_metric_series="train",
        objective_metric_sign="min",
        max_number_of_concurrent_tasks=args.max_concurrent,
        total_max_jobs=args.total_jobs,
        save_top_k_tasks_only=args.top_k,
        # The queue to execute the tasks on
        execution_queue=args.queue,
        optimizer_class=optimizer_class,
        pool_period_min=args.poll_min,
        **optimizer_kwargs,
    )

    # Start the optimization process
    print("Starting hyperparameter optimization.")
    optimizer.start(job_complete_callback=job_complete_callback)
    print("Waiting for hyperparameter optimization to complete.")
    optimizer.wait()
    # print the top performing experiments id
    top_exp = optimizer.get_top_experiments(top_k=args.top_k)
    print("Top performing experiments:")
    print([t.id for t in top_exp])
    print("Stopping optimizer")
    optimizer.stop()

    print("All done!")

if __name__ == "__main__":
    main()
