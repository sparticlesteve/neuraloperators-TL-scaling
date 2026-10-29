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
    return parser.parse_args()


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

    # Define the hyperparameter search space
    print("Constructing HyperParameterOptimizer")
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
