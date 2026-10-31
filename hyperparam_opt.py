"""
ClearML hyperparameter optimization script

The search space is given on the command line, so an agent can design it:

    python hyperparam_opt.py --base-task-id <id> \\
        --log-uniform General/lr 1e-4 1e-2 \\
        --choice General/batch_size 64 128 256 \\
        --total-jobs 6 --max-concurrent 3 --tags round-1

Without any --uniform/--log-uniform/--choice option, it searches General/lr
log-uniformly between --lr-min and --lr-max.
"""
import argparse
import json
import math
import random
import time

from clearml import Task
from clearml.automation import (
    HyperParameterOptimizer,
    UniformParameterRange,
    LogUniformParameterRange,
    DiscreteParameterRange,
    RandomSearch,
)
from clearml.automation.parameters import RandomSeed

OBJECTIVE_TITLE = "best_val_err"
OBJECTIVE_SERIES = "train"


def job_complete_callback(
    job_id,                 # type: str
    objective_value,        # type: float
    objective_iteration,    # type: int
    job_parameters,         # type: dict
    top_performance_job_id  # type: str
):
    # The optimizer sometimes reports a trial before it finishes (status still
    # in_progress); the trial keeps running, and the final summary uses its final value.
    if job_parameters.get("status") != "completed":
        print('Trial counted early, still running:', job_id, 'epoch', objective_iteration,
              'objective so far', objective_value)
        return
    print('Job completed!', job_id, objective_value, objective_iteration, job_parameters)
    if job_id == top_performance_job_id:
        print('WOOT WOOT we broke the record! Objective reached {}'.format(objective_value))


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-task-id", required=True,
                        help="task to clone for each trial (e.g. the 2-epoch base in the project)")
    parser.add_argument("--task-name", default="poisson-scale HPO")
    parser.add_argument("--project", default="SC26 demo-dev/FNO")
    parser.add_argument("--queue", default="muller")
    parser.add_argument("--max-epochs", type=int, default=32)
    parser.add_argument("--uniform", nargs=3, action="append", default=[],
                        metavar=("NAME", "MIN", "MAX"), help="float sampled uniformly")
    parser.add_argument("--log-uniform", nargs=3, action="append", default=[],
                        metavar=("NAME", "MIN", "MAX"), help="float sampled log-uniformly")
    parser.add_argument("--choice", nargs="+", action="append", default=[],
                        metavar="NAME VALUE", help="NAME followed by the values to pick from")
    parser.add_argument("--lr-min", type=float, default=1e-5)
    parser.add_argument("--lr-max", type=float, default=1e-2)
    parser.add_argument("--total-jobs", type=int, default=8)
    parser.add_argument("--max-concurrent", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=None,
                        help="archive all but the top k trials (default: keep all)")
    parser.add_argument("--seed", type=int, default=None,
                        help="sampling seed (default: random; ClearML's own default is fixed)")
    parser.add_argument("--tags", nargs="*", default=[])
    return parser.parse_args()


def build_search_space(args):
    space = []
    for name, lo, hi in args.uniform:
        space.append(UniformParameterRange(name, min_value=float(lo), max_value=float(hi)))
    for name, lo, hi in args.log_uniform:
        # LogUniformParameterRange takes exponents
        space.append(LogUniformParameterRange(
            name, min_value=math.log10(float(lo)), max_value=math.log10(float(hi))))
    for name, *values in args.choice:
        space.append(DiscreteParameterRange(name, values=values))
    if not space:
        space.append(LogUniformParameterRange(
            "General/lr", min_value=math.log10(args.lr_min), max_value=math.log10(args.lr_max)))
    space.append(DiscreteParameterRange("General/max_epochs", values=[args.max_epochs]))
    return space


def wait_for_trials(controller_id, poll_s=30, timeout_s=1800):
    """Wait until every trial has finished, so the summary has final values and
    optimizer.stop() aborts nothing. Gives up after timeout_s (a trial can sit in
    "queued" forever if its Slurm job died)."""
    final = {"completed", "failed", "stopped", "closed", "published"}
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        trials = Task.get_tasks(tags=[f"opt: {controller_id}"])
        running = [t.id for t in trials if t.get_status() not in final]
        if not running:
            return
        print(f"Waiting for {len(running)} trial(s) still running: {running}")
        time.sleep(poll_s)
    print("Timed out waiting for trials; summarizing anyway")


def main():
    args = parse_args()
    seed = args.seed if args.seed is not None else random.randrange(2**31)
    RandomSeed.set_random_seed(seed)

    # Initialize the HPO task
    task = Task.init(
        project_name=args.project,
        task_name=args.task_name,
        task_type=Task.TaskTypes.optimizer,
        tags=args.tags,
    )
    task.set_parameter("General/seed_used", seed)
    print(f"Sampling seed: {seed}")

    # Base task for the hyperparameter optimization
    base_task = Task.get_task(task_id=args.base_task_id)

    # Define the hyperparameter search space
    print("Constructing HyperParameterOptimizer")
    optimizer = HyperParameterOptimizer(
        base_task_id=base_task.id,
        hyper_parameters=build_search_space(args),
        objective_metric_title=OBJECTIVE_TITLE,
        objective_metric_series=OBJECTIVE_SERIES,
        objective_metric_sign="min",
        optimizer_class=RandomSearch,
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
    wait_for_trials(task.id)

    # Summarize every trial, best first, and attach it to the HPO task
    details = optimizer.get_top_experiments_details(top_k=args.total_jobs)
    summary = []
    for d in details:
        metric = next(iter(d["metrics"].values()), {})
        summary.append({
            "task_id": d["task_id"],
            "objective": metric.get("value"),
            "hyper_parameters": {k: v for k, v in d["hyper_parameters"].items()
                                 if k != "General/max_epochs"},
        })
    task.upload_artifact("trial_summary", artifact_object=summary)
    print("Trials, best first:")
    print(json.dumps(summary, indent=1))
    print("Stopping optimizer")
    optimizer.stop()

    print("All done!")


if __name__ == "__main__":
    main()
