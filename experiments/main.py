import argparse
import os
import pandas as pd
import torch
import json
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.data import DataConfig, TabularDataset
from experiments.model import (
    get_model,
    train_epoch,
    get_bl_acc,
    get_loss,
    extract_rule_features_and_labels,
    evaluate_accuracy,
    evaluate_computation,
)
from experiments.train_logger import ResultsLogger, WandbLogger
from experiments.simplify_model import SimplifyLogicNet
from experiments.utils import *
from experiments.settings import *

import logging

logging_level = os.getenv("LOGGING_LEVEL", "DEBUG").upper()
logging.basicConfig(level=logging.getLevelName(logging_level), format="%(message)s")


torch_num_threads = os.environ.get("TORCH_NUM_THREADS")
if torch_num_threads:
    logging.info(f"Setting torch num threads to {torch_num_threads}")
    torch.set_num_threads(int(torch_num_threads))

num_workers = 0
if os.environ.get("NUM_WORKERS"):
    num_workers = int(os.environ.get("NUM_WORKERS"))


def get_parser():
    # to accept bool values in str format
    def str2bool(v):
        return v.lower() in ("true", "1")

    # to accept None or int values
    def int_or_none(value):
        if value is None or value.lower() == "null" or value.lower() == "none":
            return None
        return int(value)

    parser = argparse.ArgumentParser(
        description="Train and/or evaluate differentiable logic network (DLN)."
    )

    # Data
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="The dataset to use. Note: input features should be scaled to [0, 1].",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed (default: 0).")

    parser.add_argument(
        "--train_model",
        type=str2bool,
        default=True,
        help="Whether to train the model (default: True).",
    )
    parser.add_argument(
        "--evaluate_model",
        type=str2bool,
        default=False,
        help="Whether to evaluate the model based on saved results (default: False).",
    )

    # Training
    parser.add_argument(
        "--num_epochs", type=int, default=1000, help="Number of epochs (default: 1000)."
    )
    parser.add_argument(
        "--batch_size", type=int, default=64, help="Batch size (default: 64)."
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.01,
        help="Learning rate (default: 0.01).",
    )
    parser.add_argument(
        "--balance_class_weights",
        type=str2bool,
        default=True,
        help="Whether to adjust weights inversely proportional to class frequencies.",
    )

    # Model
    parser.add_argument(
        "--num_heads",
        type=int,
        default=1,
        help="Number of SumLayer heads (default: 1).",
    )
    parser.add_argument(
        "--num_hidden_layers",
        type=int,
        default=3,
        help="Number of hidden layers (default: 3).",
    )
    parser.add_argument(
        "--first_hl_size",
        type=int,
        default=100,
        help="Number of neurons in the first hidden layer (default: 100).",
    )
    parser.add_argument(
        "--last_hl_size_wrt_first",
        type=float,
        default=0.5,
        help="Number of neurons in the last hidden layer with respect to the first hidden layer size (default: 0.5).",
    )

    parser.add_argument(
        "--tau_out",
        type=float,
        default=1.0,
        help="The temperature for the output layer.",
    )
    parser.add_argument(
        "--grad_factor",
        type=float,
        default=1.0,
        help="The gradient factor for all layers.",
    )

    parser.add_argument(
        "--discretize_strategy",
        type=str,
        default="tree",
        choices=["tree", "uniform", "quantile", "kmeans"],
        help="Strategy for discretizing continuous attributes, for thresholds initialization.",
    )
    parser.add_argument(
        "--continuous_resolution",
        type=int,
        default=4,
        help="Number of thresholds for each continuous attributes discretization.",
    )

    parser.add_argument(
        "--num_phase_rounds",
        type=int_or_none,
        default=None,
        help="Number of full (neuron, link) phase rounds. "
        "If not specified, will default to half of num_epochs, i.e., alternatively update neurons and links.",
    )

    parser.add_argument(
        "--subset_links",
        type=str2bool,
        default=True,
        help="Whether to only search a subset of links in the LogicLayer (default: True).",
    )
    parser.add_argument(
        "--subset_link_num",
        type=int,
        default=8,
        help="Number of subset links to keep if subset_links is True (default: 8).",
    )
    parser.add_argument(
        "--subset_gates",
        type=str2bool,
        default=True,
        help="Whether to only search a subset of logic gates in the LogicLayer (default: True).",
    )
    parser.add_argument(
        "--subset_gate_num",
        type=int,
        default=8,
        help="Number of subset gates to keep if subset_gates is True (default: 8).",
    )

    # When STEs are all True, training mode performance is equivalent to inference mode's
    parser.add_argument(
        "--ste_threshold_layer",
        type=str2bool,
        default=True,
        help="Use the Straight-Through Estimator for ThresholdLayer (default: True).",
    )
    parser.add_argument(
        "--ste_logic_layer",
        type=str2bool,
        default=True,
        help="Use the Straight-Through Estimator for LogicLayer (default: True).",
    )
    parser.add_argument(
        "--ste_sum_layer",
        type=str2bool,
        default=True,
        help="Use the Straight-Through Estimator for SumLayer (default: True).",
    )

    parser.add_argument(
        "--phase_unified",
        type=str2bool,
        default=False,
        help="Perform a unified phase (no separate neuron vs link phase) if True (default: False).",
    )

    parser.add_argument(
        "--gumbel_softmax",
        type=str2bool,
        default=False,
        help="Use Gumbel-Softmax for the LogicLayer gates/links (default: False).",
    )
    parser.add_argument(
        "--gumbel_noise_scale",
        type=float,
        default=0.5,
        help="Noise scale factor for Gumbel-Softmax sampling (default: 0.5).",
    )

    parser.add_argument(
        "--concat_input",
        type=str2bool,
        default=False,
        help="Concatenate a ThresholdLayer to all intermediate (second to second last) LogicLayers.",
    )

    parser.add_argument(
        "--concat_bool",
        type=str2bool,
        default=False,
        help="Concatenate non-trainable boolean inputs ([True, False]) to all LogicLayers.",
    )

    parser.add_argument(
        "--tau_init", type=float, default=1, help="Initial temperature for all layers."
    )
    parser.add_argument(
        "--tau_decay", type=float, default=1, help="Temperature decay factor."
    )
    parser.add_argument("--tau_min", type=float, default=1, help="Minimum temperature.")

    parser.add_argument(
        "--prog_freeze",
        type=str2bool,
        default=False,
        help="Progressively freeze the network during training.",
    )
    parser.add_argument(
        "--freeze_start_frac",
        type=float,
        default=0.8,
        help="Fraction of epochs to start freezing. Freezing always end until after the last epoch.",
    )
    parser.add_argument(
        "--freeze_reverse",
        type=str2bool,
        default=False,
        help="Freeze layers closer to the output first.",
    )

    parser.add_argument(
        "--prog_prune",
        type=str2bool,
        default=False,
        help="Progressively prune the network during training.",
    )
    parser.add_argument(
        "--prune_start_frac",
        type=float,
        default=0.5,
        help="Fraction of epochs to start pruning.",
    )
    parser.add_argument(
        "--prune_end_frac",
        type=float,
        default=0.9,
        help="Fraction of epochs to end pruning.",
    )
    parser.add_argument(
        "--prune_reverse",
        type=str2bool,
        default=False,
        help="Prune layers closer to the output first.",
    )
    parser.add_argument(
        "--prune_threshold",
        type=float,
        default=0.8,
        help="Threshold probability for pruning logic neurons.",
    )
    parser.add_argument(
        "--prune_min_neuron_left",
        type=float,
        default=0.1,
        help="Minimum number (if >= 1) or percentage (if < 1) of neurons left after pruning.",
    )

    # Others
    parser.add_argument(
        "--eval_freq",
        type=int,
        default=10,
        help="Evaluation frequency during training (default: 10).",
    )

    # NOTE: training visualization may consume a lot of memory, reduce the frequency or comment out log_model_weights and log_layer_acts_and_grads to avoid OOM
    parser.add_argument(
        "--log_training",
        action="store_true",
        help="Log and visualize model training process. Use eval_freq to control the frequency of logging.",
    )

    parser.add_argument(
        "--save_model", action="store_true", help="Save the trained model."
    )

    parser.add_argument(
        "--simplify_model",
        action="store_true",
        help="Simplify the trained model to a Python file that prints symbolic expressions. "
        "To visualize the simplified model, run: python experiments/DLN_viz.py sympy_code_path graph_path.",
    )

    parser.add_argument(
        "--get_rule_features",
        action="store_true",
        help="Extract binary rule features from the trained model and save to files.",
    )

    return parser


def main(args):
    if args.train_model:
        train_model(args)
    if args.evaluate_model:
        eval_results = evaluate_model(args.dataset, args.seed)
        results_path = get_results_path(args.dataset, args.seed, make_dir=False)
        with open(f"{results_path}/eval_results.json", "w") as f:
            json.dump(eval_results, f, indent=4)


def train_model(args):
    results_path = get_results_path(args.dataset, args.seed, make_dir=True)
    results = ResultsLogger(path=results_path)
    results.store_args(args)
    wandb_logger = WandbLogger(args.log_training, f"{args.dataset}_{args.seed}")

    set_seed(args.seed)

    args_dict = dict(vars(args))
    data_config = DataConfig(**(filter_dict_to_dataclass_fields(DataConfig, args_dict)))
    tabular_dataset = TabularDataset(data_config)

    train_loader, head_train_loaders, val_loader, test_loader = tabular_dataset.get_dataLoader(
        num_workers, val_split=0.8, split_seed=args.seed, num_heads=args.num_heads
    )
    train_loader_eval = tabular_dataset.get_dataLoader(
        num_workers, isEval=True, val_split=0.8, split_seed=args.seed
    )[
        0
    ]  # for eval mode evaluation

    print(f'len(head_train_loaders): {len(head_train_loaders)}')

    model, criterion, optimizer = get_model(args_dict, tabular_dataset)

    results.store_results({"model_str": str(model)})

    loaders = {"train": train_loader_eval, "val": val_loader, "test": test_loader}
    metric_functions = {"loss": get_loss, "acc": get_bl_acc}
    modes = [True, False]  # True for train, False for eval

    phase1_epochs = args.num_epochs // 2

    best_score = 0
    for cur_epoch in range(1, args.num_epochs + 1):
        train_epoch(model, train_loader, head_train_loaders, criterion, optimizer, cur_epoch, phase1_epochs)

        if (
            cur_epoch == 1
            or cur_epoch % args.eval_freq == 0
            or cur_epoch == args.num_epochs
        ):

            metrics = {}
            for loader_name, loader in loaders.items():
                for metric_name, metric_function in metric_functions.items():
                    for mode in modes:
                        mode_name = "train_mode" if mode else "eval_mode"
                        key = f"{metric_name}_{loader_name}Loader_{mode_name}"
                        if metric_name == "loss":
                            metrics[key] = metric_function(
                                model, loader, criterion, mode=mode
                            )
                        else:
                            metrics[key] = metric_function(model, loader, mode=mode)

            results.store_results(metrics)
            wandb_logger.log_metrics(metrics, cur_epoch)
            # wandb_logger.log_model_weights(model, cur_epoch)
            # wandb_logger.log_layer_acts_and_grads(model, train_loader_eval, criterion, cur_epoch)

            cur_score = (
                metrics["acc_valLoader_eval_mode"]
                + metrics["acc_trainLoader_eval_mode"]
            )
            if cur_score >= best_score:
                best_score = cur_score
                if args.save_model:
                    model.save_model(f"{results_path}/model.pth")

    results.save()
    wandb_logger.log_params(vars(args))
    wandb_logger.finish()

    if args.save_model:
        model.save_model(f"{results_path}/model_last.pth")

    if args.simplify_model:
        # use the best model if available
        if args.save_model and os.path.exists(f"{results_path}/model.pth"):
            model.load_model(f"{results_path}/model.pth")

        sympy_model = SimplifyLogicNet(
            model=model,
            feat_names=tabular_dataset.feat_names,
            scaler_params=tabular_dataset.scaler_params,
            dtype_dict=tabular_dataset.dtype_dict,
        )
        sympy_code = sympy_model.get_sympy_code()

        with open(f"{results_path}/sympy_code.py", "w") as f:
            f.write(sympy_code)

    if args.get_rule_features:
        if args.save_model and os.path.exists(f"{results_path}/model.pth"):
            model.load_model(f"{results_path}/model.pth")

        def make_new_dataset(data_loader, path):
            features, labels = extract_rule_features_and_labels(model, data_loader)
            num_features = len(features[0])
            feature_columns = [f"Rule_{i}" for i in range(num_features)]
            df = pd.DataFrame(features, columns=feature_columns)
            df["Target"] = labels
            df.to_csv(path, index=False, header=True)

        train_loader, _, test_loader = tabular_dataset.get_dataLoader(
            num_workers, isEval=True
        )

        folderpath = get_rule_features_path(args.dataset, args.seed, make_dir=True)
        make_new_dataset(train_loader, f"{folderpath}/train.csv")
        make_new_dataset(test_loader, f"{folderpath}/test.csv")


def evaluate_model(dataset, seed):
    eval_results = {"dataset": dataset, "seed": seed}
    bits_for_values = DTYPE_TO_BITS[dtype_for_floats]

    results_path = get_results_path(dataset, seed, make_dir=False)
    results_json_path = os.path.join(results_path, "results.json")
    if not os.path.exists(results_json_path):
        logging.error(
            f"Training results file not found. dataset: {dataset}, seed: {seed}"
        )
        return eval_results

    with open(results_json_path, "r") as f:
        all_results = json.load(f)
    if "args" not in all_results:
        logging.error(
            f"Training results file is missing 'args' key. dataset: {dataset}, seed: {seed}"
        )
        return eval_results
    params = all_results["args"]

    data_config = DataConfig(**(filter_dict_to_dataclass_fields(DataConfig, params)))
    tabular_dataset = TabularDataset(data_config)
    _, _, test_loader = tabular_dataset.get_dataLoader(
        num_workers=num_workers, isEval=True
    )

    model_path = os.path.join(results_path, "model.pth")
    if not os.path.exists(model_path):
        logging.error(f"Model file not found. dataset: {dataset}, seed: {seed}")
        return eval_results

    model = get_model(params, tabular_dataset)[0]
    model.load_model(model_path)
    model.eval()

    acc_metrics = evaluate_accuracy(model, test_loader)

    # get the simplified model
    if args.simplify_model:
        sympy_code_path = f"{results_path}/sympy_code.py"
        sympy_model = SimplifyLogicNet(
            model=model,
            feat_names=tabular_dataset.feat_names,
            scaler_params=tabular_dataset.scaler_params,
            dtype_dict=tabular_dataset.dtype_dict,
        )
        sympy_code = sympy_model.get_sympy_code()
        with open(sympy_code_path, "w") as f:
            f.write(sympy_code)

        sympy_code_path = os.path.join(results_path, "sympy_code.py")
        graph_path = os.path.join(results_path, "DLN_viz")

        comp_metrics = evaluate_computation(
            model=model,
            sympy_code_path=sympy_code_path,
            graph_path=graph_path,
            bits_for_values=bits_for_values,
            bits_for_indices=bits_for_indices,
            simplify_timeout_second=simplify_timeout_second,
        )

        eval_results.update({**acc_metrics, **comp_metrics})
    else:
        eval_results.update({**acc_metrics})

    return eval_results


if __name__ == "__main__":
    parser = get_parser()
    args, _ = parser.parse_known_args()
    main(args)
