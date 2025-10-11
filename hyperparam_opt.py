"""
ClearML hyperparameter optimization script
"""
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


def main():
    
    # Initialize the HPO task
    task = Task.init(
        project_name="FNO Tests",
        task_name="poisson-scale HPO",
        task_type=Task.TaskTypes.optimizer
    )

    # Base task for the hyperparameter optimization
    base_task = Task.get_task(task_id="a1ab901fc0474a0cabf8f34bffe688f1")
    #base_task = Task.create(
    #    task_name="poisson-scale-k1-hyperparam",
    #    project_name="FNO Tests",
    #    task_type="training",
    #    repo="https://github.com/sparticlesteve/neuraloperators-TL-scaling.git",
    #    branch="clearml-testing",
    #    binary="/bin/bash",
    #    script="./run_clearml.sh",
    #    # Specify the configuration to use
    #    #args=["--config", "poisson-scale-k1_5"],
    #)

    # Define the hyperparameter search space
    print("Constructing HyperParameterOptimizer")
    optimizer = HyperParameterOptimizer(
        base_task_id=base_task.id,
        hyper_parameters=[
            UniformParameterRange("General/lr", min_value=1e-5, max_value=1e-2),
            DiscreteParameterRange("General/max_epochs", values=[32]),
        ],
        objective_metric_title="val_loss",
        objective_metric_series="train",
        objective_metric_sign="min",
        max_number_of_concurrent_tasks=8,
        total_max_jobs=8,
        # The queue to execute the tasks on
        execution_queue="muller",
    )

    # Start the optimization process
    print("Starting hyperparameter optimization.")
    optimizer.start(job_complete_callback=job_complete_callback)
    print("Waiting for hyperparameter optimization to complete.")
    optimizer.wait()
    # print the top performing experiments id
    top_exp = optimizer.get_top_experiments(top_k=2)
    print("Top performing experiments:")
    print([t.id for t in top_exp])
    print("Stopping optimizer")
    optimizer.stop()

    print("All done!")

if __name__ == "__main__":
    main()
