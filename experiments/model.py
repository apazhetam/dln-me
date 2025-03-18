import os
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score
from operator import itemgetter
from itertools import groupby
from dataclasses import dataclass
from typing import List, Optional
from collections import defaultdict
import json
import subprocess
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.data import TabularDataset
from dln_layer.layer import ThresholdLayer, LogicLayer, SumLayer
from experiments.utils import *
import logging
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='sklearn.*')



# --------------------------------------------------------------------
# model

def get_model(
        config: dict,
        dataset: TabularDataset,
):
    assert is_valid_config(config, dataset), f"Invalid config {config} for dataset {dataset}"

    input_dim, output_dim = dataset.input_dim, dataset.num_classes
    hidden_dims = get_hidden_dims(config, output_dim)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config.update({
        'input_dim': input_dim, 'output_dim': output_dim, 'hidden_dims': hidden_dims,
        'device': device,
    })

    threshold_dim, threshold_init = dataset.get_threshold_info()
    config.update({
        'num_con_feat': dataset.num_con_feat,
        'threshold_dim': threshold_dim,
        'threshold_init': threshold_init,
    })

    filtered_dict = filter_dict_to_dataclass_fields(DLNConfig, config)
    modelConfig = DLNConfig(**filtered_dict)
    model = DLN(modelConfig)

    model = model.to(device)

    class_weights = dataset.get_class_weights(config['balance_class_weights'])
    class_weights = class_weights.to(device) if class_weights is not None else None
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])

    return model, criterion, optimizer



@dataclass
class DLNConfig:
    input_dim: int
    threshold_dim: int
    output_dim: int
    hidden_dims: List[int]
    num_epochs: int
    device: str

    num_con_feat: int
    threshold_init: Optional[np.ndarray] = None
    th_slope_init: float = 2  #TODO make it a parameter in HPO?

    sum_link_threshold: float = 0.8  #TODO make it a parameter in HPO?

    num_phase_rounds: Optional[int] = None

    grad_factor: float = 1

    subset_links: bool = True
    subset_link_num: int = 8
    subset_gates: bool = True
    subset_gate_num: int = 8

    ste_threshold_layer: bool = True
    ste_logic_layer: bool = True
    ste_sum_layer: bool = True

    phase_unified: bool = False

    gumbel_softmax: bool = False
    gumbel_noise_scale: float = 0.5

    concat_input: bool = False
    concat_bool: bool = False

    tau_init: float = 1
    tau_decay: float = 1
    tau_min: float = 1
    tau_out: float = 1

    prog_freeze: bool = False
    freeze_start_frac: Optional[float] = None
    freeze_reverse: Optional[bool] = None

    prog_prune: bool = False
    prune_start_frac: Optional[float] = None
    prune_end_frac: Optional[float] = None
    prune_reverse: Optional[bool] = None
    prune_threshold: Optional[float] = None
    prune_min_neuron_left: Optional[float] = None


class DLN(nn.Module):
    def __init__(self, config: DLNConfig):
        super(DLN, self).__init__()
        self.config = config

        self.concat_input = config.concat_input
        concat_input_dim = config.input_dim if config.concat_input else 0

        self.concat_bool = config.concat_bool
        concat_bool_dim = 2 if config.concat_bool else 0
        if self.concat_bool:
            self.bool_constants = nn.Parameter(
                torch.tensor([1.0, 0.0], dtype=torch.float32, device=config.device).view(1, -1),
                requires_grad=False)

        common_args = dict(
            grad_factor=config.grad_factor,
            device=config.device,
            phase_unified=config.phase_unified,
            tau_init=config.tau_init,
            tau_decay=config.tau_decay,
            tau_min=config.tau_min,
        )

        def create_threshold_layer(in_dim):
            return ThresholdLayer(
                in_dim=in_dim,
                threshold_dim=config.threshold_dim,
                threshold_init=config.threshold_init,
                th_slope_init=config.th_slope_init,
                ste_threshold_layer=config.ste_threshold_layer,
                **common_args
            )

        def create_logic_layer(in_dim, out_dim):
            return LogicLayer(
                in_dim=in_dim,
                out_dim=out_dim,
                subset_links=config.subset_links,
                subset_link_num=config.subset_link_num,
                subset_gates=config.subset_gates,
                subset_gate_num=config.subset_gate_num,
                ste_logic_layer=config.ste_logic_layer,
                gumbel=config.gumbel_softmax,
                gumbel_noise_scale=config.gumbel_noise_scale,
                **common_args
            )

        def create_sum_layer(in_dim, out_dim):
            return SumLayer(
                in_dim=in_dim,
                out_dim=out_dim,
                link_threshold=config.sum_link_threshold,
                ste_sum_layer=config.ste_sum_layer,
                tau_out=config.tau_out,
                **common_args
            )

        self.layers = nn.ModuleList()
        self.layers.append(create_threshold_layer(config.input_dim))

        self.layers.append(create_logic_layer(
            in_dim=config.input_dim + concat_bool_dim,
            out_dim=config.hidden_dims[0],
        ))
        for i in range(len(config.hidden_dims) - 1):
            self.layers.append(create_logic_layer(
                in_dim=config.hidden_dims[i] + concat_bool_dim + concat_input_dim,
                out_dim=config.hidden_dims[i+1],
            ))

        self.layers.append(create_sum_layer(
            in_dim=config.hidden_dims[-1],
            out_dim=config.output_dim,
        ))

        self.depth = len(self.layers)
        # layer index in self.layers to the depth of the layer, starting from 1
        # i.e., the 0-th layer (the first ThresholdLayer) has depth 1
        self.idx2depth = list(range(1, self.depth + 1))
        self.idx2next_layer_idx = list(range(1, self.depth)) + [None]

        if self.concat_input:
            for i in range(len(config.hidden_dims) - 1):
                self.layers.append(create_threshold_layer(config.input_dim))
                self.idx2depth.append(i + 2)
                self.idx2next_layer_idx.append(i + 2)

        self.print_layers()
        self.epochs_trained = 0
        self.num_phase_rounds = config.num_phase_rounds or config.num_epochs // 2
        self.phase_schedule = None
        self.in_neuron_phase = True
        self.init_phase()

        self.prog_freeze = config.prog_freeze
        if self.prog_freeze:
            layer_indices = list(range(len(self.layers)))
            self.freeze_schedule = self.get_prog_schedule(
                layer_indices, config.freeze_start_frac, 1, config.num_epochs, config.freeze_reverse)
            logging.debug(f"freeze_schedule: {self.freeze_schedule}")

        self.prog_prune = config.prog_prune
        if self.prog_prune:
            self.prune_threshold = config.prune_threshold
            self.prune_min_neuron_left = config.prune_min_neuron_left
            layer_indices = list(range(len(self.layers)))
            self.prune_schedule = self.get_prog_schedule(
                layer_indices, config.prune_start_frac, config.prune_end_frac, config.num_epochs, config.prune_reverse)
            logging.debug(f"prune_schedule: {self.prune_schedule}")


    def forward(self, x):
        out = x
        for i in range(self.depth):
            layer = self.layers[i]
            if isinstance(layer, LogicLayer):
                if self.concat_input and i > 0 and isinstance(self.layers[i-1], LogicLayer):
                    th_layer = self.layers[self.depth + i - 2]
                    out = torch.cat((out, th_layer(x)), dim=1)
                if self.concat_bool:
                    out = torch.cat((out, self.bool_constants.expand(x.size(0), -1)), dim=1)
            out = layer(out)
        return out


    @torch.no_grad()
    def init_phase(self):
        for layer in self.layers:
            layer.init_phase()

        epochs_per_phase = self.config.num_epochs // (2 * self.num_phase_rounds)
        self.phase_schedule = {i: (i // epochs_per_phase) % 2 == 0
                               for i in range(0, self.config.num_epochs + 1, epochs_per_phase)}
        # logging.debug(f"phase_schedule: {self.phase_schedule}")

    @torch.no_grad()
    def set_phase(self):
        if self.epochs_trained not in self.phase_schedule.keys():
            return
        self.in_neuron_phase = self.phase_schedule[self.epochs_trained]
        # logging.debug(f"\tafter {self.epochs_trained} epochs, setting to {'neuron' if self.in_neuron_phase else 'link'} phase")
        for layer in self.layers:
            if self.in_neuron_phase:
                layer.set_neuron_phase()
            else:
                layer.set_link_phase()


    @torch.no_grad()
    def update_temperature(self):
        for layer in self.layers:
            layer.update_temperature()


    @torch.no_grad()
    def freeze_params(self):
        if not self.prog_freeze or (self.epochs_trained not in self.freeze_schedule.keys()):
            return
        freeze_list = self.freeze_schedule[self.epochs_trained]
        logging.debug(f"\tafter {self.epochs_trained} epochs, freeze_list: {freeze_list}")
        for idx in freeze_list:
            layer = self.layers[idx]
            logging.debug(f"\t\tfreezing layer {idx}, {layer}")
            layer.freeze_params()


    @torch.no_grad()
    def prune(self):
        if not self.prog_prune or (self.epochs_trained not in self.prune_schedule.keys()):
            return

        # reorg the pruning list to make sure to prune in propagation order and to
        # check any concat of Logic and Threshold layers, layers concatenated together have the same next_layer_idx
        prune_list = self.prune_schedule[self.epochs_trained]
        prune_list = [(idx, self.idx2next_layer_idx[idx] or float('inf')) for idx in prune_list]
        prune_list.sort(key=itemgetter(1))
        prune_list = [[x[0] for x in g] for _, g in groupby(prune_list, key=itemgetter(1))]
        prune_list = [sorted(inner_list) for inner_list in prune_list]
        prune_list.sort(key=itemgetter(0))
        logging.debug(f"\tafter {self.epochs_trained} epochs, prune_list: {prune_list}")

        for cur_prune_list in prune_list:
            next_layer_idx = self.idx2next_layer_idx[cur_prune_list[0]]
            next_layer = self.layers[next_layer_idx] if next_layer_idx is not None else None

            mask_list = []
            for idx in cur_prune_list:
                layer = self.layers[idx]
                logging.debug(f"\t\tpruning layer {idx}, {layer}")
                output_mask = layer.prune_neurons(min_neuron_left=self.prune_min_neuron_left, threshold=self.prune_threshold)
                mask_list.append(output_mask)

            if next_layer is not None:
                if self.concat_bool and isinstance(next_layer, LogicLayer):
                    mask_list.append(torch.ones_like(self.bool_constants, dtype=torch.bool).squeeze())
                logging.debug(f"\t\tupdating input mask for layer {next_layer_idx}, {next_layer}")
                next_layer.add_input_mask(torch.cat(mask_list, dim=0))


    def get_prog_schedule(self, layer_indices, start_frac, end_frac, num_epochs, reverse):
        idx_to_schedule = dict()  # {depth: [idx]}
        for idx in layer_indices:
            depth = self.idx2depth[idx]
            idx_to_schedule.setdefault(depth, []).append(idx)

        action_epochs = (np.linspace(
            start_frac, end_frac, len(idx_to_schedule.keys())) * num_epochs).astype(int)
        idx_to_schedule = dict(sorted(idx_to_schedule.items(), reverse=reverse))

        schedule = dict()
        for key, old_k in zip(action_epochs, idx_to_schedule):
            if key not in schedule:
                schedule[int(key)] = idx_to_schedule[old_k]
            else:
                schedule[int(key)] += idx_to_schedule[old_k]
        return schedule


    # returns the total number of neurons before logic simplification
    def get_num_neurons(self):
        return sum(l.get_num_neurons() for l in self.layers if not isinstance(l, ThresholdLayer))

    # returns the total number of values and indices before logic simplification
    def get_num_params(self):
        num_values, num_indices = 0, 0
        for l in self.layers:
            values, indices = l.get_num_params()
            num_values += values
            num_indices += indices
        return num_values, num_indices

    # returns the dict of operations before logic simplification
    def get_ops_dict(self):
        ops_dict = defaultdict(lambda: defaultdict(int))
        for l in self.layers:
            layer_ops_dict = l.get_ops_dict()
            merge_defaultdict_to_another(ops_dict, layer_ops_dict)
        ops_dict = {k: dict(v) for k, v in ops_dict.items()}  # convert defaultdict to dict (for readability)
        return ops_dict


    def save_model(self, filepath):
        state_dict = self.state_dict()
        for i, layer in enumerate(self.layers):
            for name, value in layer.__dict__.items():
                if (name not in ['device'] and
                        not name.startswith('_') and
                        not isinstance(value, torch.Tensor)):
                    state_dict[f'layers.{i}.{name}'] = value
        torch.save(state_dict, filepath)

    def load_model(self, filepath, device=None):
        _device = device or self.config.device
        state_dict = torch.load(filepath, map_location=_device, weights_only=False)
        self.load_state_dict(state_dict, strict=False)
        for i, layer in enumerate(self.layers):
            for name, value in state_dict.items():
                if (name not in ['device'] and name.startswith(f'layers.{i}.') and
                        hasattr(layer, name.split('.')[-1])) and not isinstance(value, torch.Tensor):
                    setattr(layer, name.split('.')[-1], value)
        if device:
            self.to(device)


    def print_layers(self):
        logging.debug('-'*10)
        logging.debug(f"depth: {self.depth}; num layers: {len(self.layers)}")
        for i in range(len(self.layers)):
            logging.debug(f"idx: {i}; layer: {self.layers[i]}, depth: {self.idx2depth[i]}, next_layer_idx: {self.idx2next_layer_idx[i]}")
        logging.debug('-'*10)



def get_hidden_dims(
        config: dict,
        output_dim: int,
) -> list[int]:

    hidden_dims = np.geomspace(config['first_hl_size'],
                               round(config['first_hl_size'] * config['last_hl_size_wrt_first']),
                               num=config['num_hidden_layers']).tolist()
    hidden_dims = list(map(round, hidden_dims))

    # make the last hidden layer's size divisible by the output_dim
    hidden_dims[-1] = ((hidden_dims[-1] + output_dim - 1) // output_dim) * output_dim
    return hidden_dims


def is_valid_config(
        config: dict,
        dataset: TabularDataset,
) -> bool:

    if config.get('num_phase_rounds', None) is not None:
        if config['num_phase_rounds'] <= 0 or config['num_epochs'] < config['num_phase_rounds'] * 2:
            return False

    if config.get('prog_prune', False):
        if config['prune_start_frac'] > config['prune_end_frac']:
            return False

    return True



# --------------------------------------------------------------------
# train and eval

def train_epoch(model, loader, criterion, optimizer):
    model.train()
    device = next(model.parameters()).device
    for data in loader:
        x, y = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        output = model(x)
        # in DLN, some epochs may not require grad (e.g., SumLayer before its prog_freeze and when it's in neuron_phase)
        if output.requires_grad:
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

    if isinstance(model, DLN):
        model.epochs_trained += 1
        model.set_phase()
        model.prune()
        model.freeze_params()
        model.update_temperature()


@torch.no_grad()
def get_bl_acc(model, loader, mode):
    bl_acc = AverageMeter()
    device = next(model.parameters()).device
    orig_mode = model.training
    model.train(mode=mode)
    for data, target in loader:
        output = model(data.to(device))
        pred = output.argmax(1)
        _bl_acc = balanced_accuracy_score(target.cpu().numpy(), pred.cpu().numpy())
        bl_acc.update(_bl_acc, target.size(0))
    model.train(mode=orig_mode)
    return bl_acc.avg

@torch.no_grad()
def get_loss(model, loader, criterion, mode):
    loss = AverageMeter()
    device = next(model.parameters()).device
    orig_mode = model.training
    model.train(mode=mode)
    for data, target in loader:
        output = model(data.to(device))
        _loss = criterion(output, target.to(device)).item()
        loss.update(_loss, target.size(0))
    model.train(mode=orig_mode)
    return loss.avg



def evaluate_accuracy(model, test_loader):
    return {
        'acc_testLoader_train_mode': get_bl_acc(model, test_loader, mode=True),
        'acc_testLoader_eval_mode': get_bl_acc(model, test_loader, mode=False),
    }


def evaluate_computation(model, sympy_code_path, graph_path, bits_for_values, bits_for_indices, simplify_timeout_second):
    # get number of OPs
    # first get the unsimplified model's data, then get the simplified model's data if available
    ops_dict = model.get_ops_dict()
    num_values, num_indices = model.get_num_params()
    num_neurons = model.get_num_neurons()
    simplified = False  # whether the data is from a simplified model

    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    viz_code_path = f"{parent_dir}/experiments/DLN_viz.py"
    cmd = ['python', viz_code_path, sympy_code_path, graph_path, '--only_calc_stats']
    try:
        cmd_return = subprocess.run(cmd, capture_output=True, text=True, timeout=simplify_timeout_second)
        if cmd_return.returncode == 0 and cmd_return.stdout:
            cmd_return = json.loads(cmd_return.stdout)
        else:
            logging.error(f"Error in executing the visualization script for {sympy_code_path}.")
            cmd_return = None
    except subprocess.TimeoutExpired:
        logging.error(f"Timeout in executing the visualization script for {sympy_code_path}.")
        cmd_return = None

    if cmd_return is not None:
        simplified = True
        num_feats = cmd_return['num_feats']
        num_logics = cmd_return['num_logics']
        num_classes = cmd_return['num_classes']
        num_edges = cmd_return['num_edges']
        ops_dict = cmd_return['ops_dict']

        # for each feature, store its index and threshold
        num_values = num_feats
        num_indices = num_feats
        # for each node, store its function if it's a logic neuron, and keep a list of its incoming nodes' indices
        # logic functionality needs 2 bits to store (AND, OR, XOR, NOT)
        num_values += int(np.ceil(num_logics / (bits_for_values // 2)))
        num_indices += num_edges

        num_neurons = num_logics + num_classes

    num_hl_ops, num_bg_ops = get_num_ops_info_from_ops_dict(ops_dict)
    num_params = num_values + num_indices
    num_disk_bits = num_values * bits_for_values + num_indices * bits_for_indices

    return {
        'num_hl_ops': num_hl_ops, 'num_bg_ops': num_bg_ops,
        'num_params': num_params, 'num_disk_bits': num_disk_bits,
        'num_neurons': num_neurons, 'simplified': simplified,
    }



def extract_rule_features_and_labels(model, data_loader):
    orig_mode = model.training
    model.eval()
    device = next(model.parameters()).device
    features_list = []
    labels_list = []

    outputs = []
    def hook_fn(module, input, output):
        outputs.append(output.detach())

    hook = model.layers[-2].register_forward_hook(hook_fn)

    with torch.no_grad():
        for data, target in data_loader:
            data = data.to(device)
            target = target.to(device)

            outputs.clear()
            _ = model(data)

            features_list.extend(outputs[0].cpu().numpy())
            labels_list.extend(target.cpu().numpy())

    hook.remove()
    model.train(mode=orig_mode)
    return features_list, labels_list



class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.count += n
        self.avg = (self.sum / self.count) if self.count > 0 else 0


