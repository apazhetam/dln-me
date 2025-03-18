import argparse
import numpy as np
import pandas as pd
import os
import json
import subprocess
import ray
import time
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.params_search import params_search
from experiments.main import evaluate_model
from experiments.utils import *
from experiments.settings import *

import logging
logging.basicConfig(level=logging_level, format='%(message)s')
os.environ['LOGGING_LEVEL'] = logging_level
logging_level_num = logging.getLevelName(logging_level)
def print_log(msg, msg_log_level):
    msg_log_level_num = logging.getLevelName(msg_log_level) if isinstance(msg_log_level, str) else msg_log_level
    if msg_log_level_num >= logging_level_num:
        print(msg)



def train_batch(datasets, seeds):
    ray.init(ignore_reinit_error=True, log_to_driver=True, include_dashboard=False)
    ray_tasks = []
    for dataset in datasets:
        for seed in seeds:
            params_path = f"{get_params_path(dataset, seed)}/params.json"
            f = open(params_path, 'r')
            params = json.loads(f.read())
            f.close()
            cpu_req, gpu_req = get_resource_per_run(dataset, 'train')
            ray_tasks.append(train_remote.options(num_cpus=cpu_req, num_gpus=gpu_req).remote(params))
    ray.get(ray_tasks)
    time.sleep(1)  # wait for console to display the last print statements, if applicable
    ray.shutdown()


@ray.remote(num_cpus=0, num_gpus=0)  # num_cpus and num_gpus specified in train_batch() fn, put dummy values here to make .options() work
def train_remote(params):
    torch.set_num_threads(1)
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    cmd = f'python {current_dir}/main.py --train_model True '
    cmd += ' '.join(f'--{k} {v}' for k, v in params.items())
    cmd += ' --save_model'  # make sure to add a space before '--flag'
    # cmd += ' --log_training'
    # cmd += ' --simplify_model'
    st = time.time()
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    if process.returncode != 0:
        print_log(f"Training error: {stderr.decode('utf-8').strip()}. Command: {cmd}", 'ERROR')
    else:
        tc = int(time.time() - st)
        print_log(f"Training done. {stdout.decode('utf-8').strip()} "
                     f"{params['dataset'], params['seed']}. "
                     f"Time: {'{}:{:02d}'.format(tc//60, tc%60)}.", 'INFO')



def evaluate_batch(datasets, seeds):
    ray.init(ignore_reinit_error=True, log_to_driver=True, include_dashboard=False)
    ray_tasks = []
    for dataset in datasets:
        for seed in seeds:
            ray_tasks.append(evaluate_remote.remote(dataset, seed))
    results = ray.get(ray_tasks)
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pd.DataFrame(results).to_csv(f"{parent_dir}/results/evaluation.csv", index=False)
    time.sleep(1)  # wait for console to display the last print statements, if applicable
    ray.shutdown()


@ray.remote(num_cpus=2, num_gpus=0)
def evaluate_remote(dataset, seed):
    return evaluate_model(dataset, seed)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--params-search', action='store_true')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--evaluate', action='store_true')
    args = parser.parse_args()

    if args.params_search:
        params_search(datasets, seeds)
    if args.train:
        train_batch(datasets, seeds)
    if args.evaluate:
        evaluate_batch(datasets, seeds)


