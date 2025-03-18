import os


datasets = [
    'Heart',
]

seeds = list(range(1))
if os.environ.get('SLURM_ARRAY_TASK_ID') is not None:  # in case of slurm array job, only run one seed per job
    seeds = [int(os.environ["SLURM_ARRAY_TASK_ID"])]


local_laptop = True    # True for local laptop, False for remote (slurm) cluster
logging_level = 'INFO'  # 'CRITICAL', 'ERROR', 'WARNING', 'INFO' or 'DEBUG'

# Params search settings
tune_max_num_epochs = 1000  # max number of epochs for each HPO
tune_eval_freq = 20  # evaluate val acc every n epochs
tune_early_stop_patience = 200  # stop HPO if no improvement in val acc for n epochs. NOTE: turn this off when freeze or prune is used

tune_search_alg = 'Optuna'  # 'Grid' or 'Optuna'; consider reduce search space or increase computing resources if use Grid search
tune_samples = 32  # if use Optuna, the number of trials for each HPO
tune_verbose = 0   # verbosity level for Ray Tune
remove_tune_dir = True  # remove intermediate directories created by tuner.fit() after each HPO

simplify_timeout_second = 1800  # time limit for simplifying model, in seconds


# num folds for HPO cross validation, 1 for just train/val split
def get_hpo_cvn(num_train_samples):
    if num_train_samples >= 5_000:  # more than 5k
        return 1
    if num_train_samples >= 2_000:  # between 2k and 5k
        return 3
    return 4  # less than 2k

# the dtype used for calculating the number of logic gate level operations and the disk space to store parameters
# models were trained with default dtype, which is float32 for most torch models
# using float16 for calculation because this generally does not downgrade model performances significantly
dtype_for_floats = 'float16'
dtype_for_ints = 'int16'
# using 16-bit int to store the feature indices, node IDs, etc.
bits_for_indices = 16


# Resource allocation
total_num_cpus_hpo = 16
total_num_gpus_hpo = 0

total_num_cpus_train = 16
total_num_gpus_train = 0

if local_laptop:
    total_num_cpus_hpo = 8
    total_num_gpus_hpo = 0
    total_num_cpus_train = 8
    total_num_gpus_train = 0

# Based on experiments, using 1 CPU (and 0 GPU because splitting A100 into many fractions is inefficient) per trial
# and setting torch.set_num_threads(1) and num_workers = 0 is most efficient for Ray + Slurm
tune_max_concurrent = total_num_cpus_hpo  # max concurrent trials
train_max_concurrent = total_num_cpus_train

# Get CPU and GPU requirements per trial / run
def get_resource_per_run(dataset, stage):
    if stage == 'hpo':
        cpu_base = total_num_cpus_hpo / tune_max_concurrent
        gpu_base = total_num_gpus_hpo / tune_max_concurrent
    elif stage == 'train':
        cpu_base = total_num_cpus_train / train_max_concurrent
        gpu_base = total_num_gpus_train / train_max_concurrent
    else:
        raise ValueError("Invalid stage. Must be 'hpo' or 'train'.")
    return cpu_base, gpu_base


