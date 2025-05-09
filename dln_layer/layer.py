import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load
from .functional import bin_op_s, GradFactor
import numpy as np
from typing import Optional
import logging
from collections import defaultdict
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.utils import *
from experiments.settings import dtype_for_floats


# Global variable to store the CUDA extension
_logiclayer_cuda = None


def get_cuda_extension():
    global _logiclayer_cuda

    if _logiclayer_cuda is not None:
        return _logiclayer_cuda

    if not torch.cuda.is_available():
        return None

    try:
        cuda_dir = os.path.join(os.path.dirname(__file__), "cuda")
        source_files = [
            os.path.join(cuda_dir, "logic_layer.cpp"),
            os.path.join(cuda_dir, "logic_layer_kernel.cu"),
        ]
        _logiclayer_cuda = load(
            name="logiclayer_cuda",
            sources=source_files,
            # extra_cflags=['-O2'],
            # extra_cuda_cflags=['-O2'],
            verbose=True,
        )
        return _logiclayer_cuda

    except RuntimeError as e:
        logging.warning(f"Failed to load CUDA extension: {e}")
        return None
    except Exception as e:
        logging.error(f"Unexpected error loading CUDA extension: {e}")
        return None


# indices of the gates sorted by completeness and gradient flow, from high to low
# order: NOR (8), NAND (14), XOR (6), XNOR (9), OR (7), AND (1), A (3), B (5),
# ~A (12), ~B (10), IMPLY (11, 13), NIMPLY (2, 4), False (0), True (15)
SUBSET_GATES_PRIORITY = [8, 14, 6, 9, 7, 1, 3, 5, 12, 10, 11, 13, 2, 4, 0, 15]


class BaseLayer(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        device: str,
        grad_factor: float = 1.0,
        phase_unified: bool = False,
        tau_init: float = 1.0,
        tau_decay: float = 1.0,
        tau_min: float = 1.0,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.device = device
        self.grad_factor = grad_factor
        self.phase_unified = phase_unified
        self.tau_init = tau_init
        self.tau_decay = tau_decay
        self.tau_min = tau_min
        self.tau = tau_init

        self.in_neuron_phase = True  # True for neuron phase, False for link phase
        self.freeze = False

    @torch.no_grad()
    def init_phase(self):
        # start with neuron phase
        self.in_neuron_phase = True

    @torch.no_grad()
    def set_neuron_phase(self):
        self.in_neuron_phase = True

    @torch.no_grad()
    def set_link_phase(self):
        self.in_neuron_phase = False

    @torch.no_grad()
    def freeze_params(self):
        self.freeze = True
        for param in self.parameters():
            param.requires_grad = False

    def update_temperature(self):
        raise NotImplementedError("BaseLayer does not implement update_temperature().")

    def forward(self, x):
        raise NotImplementedError("BaseLayer does not implement forward().")

    def prune_neurons(self, *args, **kwargs):
        raise NotImplementedError("Child class must implement prune_neurons().")

    def get_num_neurons(self):
        raise NotImplementedError("Child class must implement get_num_neurons().")

    def get_num_params(self):
        raise NotImplementedError("Child class must implement get_num_params().")

    def get_ops_dict(self):
        raise NotImplementedError("Child class must implement get_ops_dict().")

    def extra_repr(self):
        return f"{self.in_dim}, {self.out_dim}"


class ThresholdLayer(BaseLayer):
    def __init__(
        self,
        in_dim: int,
        threshold_dim: int,
        device: str,
        threshold_init: Optional[np.ndarray] = None,
        th_slope_init: float = 2.0,
        grad_factor: float = 1.0,
        ste_threshold_layer: bool = True,
        phase_unified: bool = False,
        tau_init: float = 1.0,
        tau_decay: float = 1,
        tau_min: float = 1,
    ):
        super().__init__(
            in_dim=in_dim,
            out_dim=in_dim,
            device=device,
            grad_factor=grad_factor,
            phase_unified=phase_unified,
            tau_init=tau_init,
            tau_decay=tau_decay,
            tau_min=tau_min,
        )
        self.threshold_dim = threshold_dim
        self.threshold_init = threshold_init
        self.th_slope_init = th_slope_init
        self.cat_dim = (
            self.in_dim - self.threshold_dim
        )  # number of categorical features
        self.ste = ste_threshold_layer

        self.init_params()

        self.init_phase()

    def init_params(self):
        self.th_bias = nn.Parameter(
            (
                torch.from_numpy(self.threshold_init)
                if self.threshold_init is not None
                else torch.full((self.threshold_dim,), 0.5)
            ).float()
        )

        self.th_slope = nn.Parameter(
            self.th_slope_init * torch.ones(self.threshold_dim)
        )

        output_mask = torch.ones(self.out_dim, dtype=torch.bool, device=self.device)
        self.register_buffer("output_mask", output_mask)

    def forward(self, x):
        x_cat = x[..., : self.cat_dim]
        x_con = x[..., self.cat_dim :]
        x_con_scaled = self.th_slope * (x_con - self.th_bias)

        if (
            self.training
            and not self.freeze
            and (self.in_neuron_phase or self.phase_unified)
        ):
            x_con_act_soft = torch.sigmoid(x_con_scaled / self.tau)
            if self.ste:
                x_con_act_hard = torch.heaviside(x_con_scaled.detach(), x.new_zeros(()))
                x_con_act = (x_con_act_hard - x_con_act_soft).detach() + x_con_act_soft
            else:
                x_con_act = x_con_act_soft

        else:
            x_detached = x_con_scaled.detach()
            x_con_act = torch.heaviside(x_detached, x.new_zeros(()))

        y = torch.cat((x_cat, x_con_act), dim=-1)

        # if self.training and self.tau != 1.:
        #     return GradFactor.apply(y, self.tau)
        if self.training and self.grad_factor != 1.0:
            return GradFactor.apply(y, self.grad_factor)
        return y

    @torch.no_grad()
    # returns output mask: True for keep, False for prune
    def prune_neurons(self, min_neuron_left=1, **kwargs):
        # continuous input values range between 0 and 1, prune nodes with uninformative biases
        margin = 0.1 / (self.th_slope.data.abs() + 1e-8)  # add a small margin
        mask_con = (self.th_bias > 0 - margin) & (self.th_bias < 1 + margin)
        mask_cat = torch.ones(self.cat_dim, dtype=torch.bool, device=self.device)
        mask = torch.cat((mask_cat, mask_con))
        self.output_mask &= mask

        # make sure at least min_neuron_left neurons are kept
        min_neuron_left = int(
            max(
                1,
                min(
                    self.out_dim,
                    (
                        min_neuron_left
                        if min_neuron_left >= 1
                        else self.out_dim * min_neuron_left
                    ),
                ),
            )
        )
        num_remain = self.output_mask.sum().item()
        if num_remain < min_neuron_left:
            shortfall = min_neuron_left - num_remain
            pruned_indices = (~mask_con).nonzero(as_tuple=False).view(-1)
            indices_to_unprune = pruned_indices[
                torch.randperm(pruned_indices.size(0))[:shortfall]
            ]
            self.output_mask[self.cat_dim + indices_to_unprune] = True

        logging.debug(
            f"\t\t\ttotal output num_prune: {(~self.output_mask).sum().item()}"
        )
        return self.output_mask

    @torch.no_grad()
    def update_temperature(self):
        if self.in_neuron_phase or self.phase_unified:
            self.tau = max(self.tau * self.tau_decay, self.tau_min)

    @torch.no_grad()
    def get_num_neurons(self):
        return (self.output_mask[self.cat_dim :]).sum().item()

    @torch.no_grad()
    def get_num_params(self):
        # for each unmasked input, store feature index and bias / equality threshold
        num_unmasked = self.output_mask.sum().item()
        return num_unmasked, num_unmasked

    @torch.no_grad()
    def get_ops_dict(self):
        ops_dict = defaultdict(lambda: defaultdict(int))
        # for each categorical feature, need to do binary equality check
        ops_dict["equality"]["bool"] += (self.output_mask[: self.cat_dim]).sum().item()
        # for each continuous feature, need to do float comparison
        ops_dict["comparison"][dtype_for_floats] += (
            (self.output_mask[self.cat_dim :]).sum().item()
        )
        return ops_dict


class LogicLayer(BaseLayer):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        device: str,
        grad_factor: float = 1.0,
        subset_links: bool = True,
        subset_link_num: int = 8,
        subset_gates: bool = True,
        subset_gate_num: int = 8,
        ste_logic_layer: bool = True,
        phase_unified: bool = False,
        gumbel: bool = False,
        gumbel_noise_scale: float = 0.5,
        tau_init: float = 1.0,
        tau_decay: float = 1,
        tau_min: float = 1,
    ):
        super().__init__(
            in_dim=in_dim,
            out_dim=out_dim,
            device=device,
            grad_factor=grad_factor,
            phase_unified=phase_unified,
            tau_init=tau_init,
            tau_decay=tau_decay,
            tau_min=tau_min,
        )
        self.subset_links = subset_links
        self.subset_link_num = subset_link_num
        self.subset_gates = subset_gates
        self.subset_gate_num = subset_gate_num
        self.ste = ste_logic_layer
        self.gumbel = gumbel
        self.gumbel_noise_scale = gumbel_noise_scale
        self.neuron_tau = tau_init
        self.link_tau = tau_init

        self.init_params()

        self.init_phase()

        self._use_ext_kernels = False
        if "cuda" in self.device:
            self._cuda_ext = get_cuda_extension()
            self._use_ext_kernels = self._cuda_ext is not None
            logging.info(f"{self} CUDA extension loaded: {self._use_ext_kernels}.")

    def init_params(self):
        self.neuron_weights = nn.Parameter(
            nn.init.uniform_(torch.empty(self.out_dim, 16), a=-0.1, b=0.1)
        )
        self.link_weights_a = nn.Parameter(
            nn.init.uniform_(torch.empty(self.out_dim, self.in_dim), a=-0.1, b=0.1)
        )
        self.link_weights_b = nn.Parameter(
            nn.init.uniform_(torch.empty(self.out_dim, self.in_dim), a=-0.1, b=0.1)
        )

        output_mask = torch.ones(self.out_dim, dtype=torch.bool, device=self.device)
        link_weights_a_mask = torch.ones_like(
            self.link_weights_a, dtype=torch.bool, device=self.device
        )
        link_weights_b_mask = torch.ones_like(
            self.link_weights_b, dtype=torch.bool, device=self.device
        )
        self.register_buffer("output_mask", output_mask)
        self.register_buffer("link_weights_a_mask", link_weights_a_mask)
        self.register_buffer("link_weights_b_mask", link_weights_b_mask)

        if self.subset_gates:
            # only search over a subset of the gate types
            num_gates = max(1, min(16, self.subset_gate_num))
            mask = torch.zeros_like(self.neuron_weights, dtype=torch.bool)
            columns_to_keep = SUBSET_GATES_PRIORITY[:num_gates]
            mask[:, columns_to_keep] = True
            self.neuron_weights = nn.Parameter(
                torch.where(mask, self.neuron_weights, torch.tensor(float("-inf")))
            )

        if self.subset_links:
            # only search over a subset of the links
            num_links = max(1, min(self.in_dim, self.subset_link_num))
            self.link_weights_a_mask = torch.zeros_like(
                self.link_weights_a, dtype=torch.bool, device=self.device
            )
            self.link_weights_b_mask = torch.zeros_like(
                self.link_weights_b, dtype=torch.bool, device=self.device
            )
            for i in range(self.out_dim):
                rand_indices = torch.randint(0, self.in_dim, (num_links * 2,))
                self.link_weights_a_mask[i, rand_indices[:num_links]] = True
                self.link_weights_b_mask[i, rand_indices[num_links : num_links * 2]] = (
                    True
                )

    def forward(self, x):
        link_weights_a_masked = self.link_weights_a.masked_fill(
            ~self.link_weights_a_mask, float("-inf")
        )
        link_weights_b_masked = self.link_weights_b.masked_fill(
            ~self.link_weights_b_mask, float("-inf")
        )
        a_hard = x[..., link_weights_a_masked.argmax(-1)]
        b_hard = x[..., link_weights_b_masked.argmax(-1)]
        neuron_weights_hard = F.one_hot(self.neuron_weights.argmax(-1), 16).to(
            self.neuron_weights.dtype
        )

        a, b = a_hard, b_hard
        neuron_weights = neuron_weights_hard

        if self.in_neuron_phase or self.phase_unified:  # neuron search phase
            # trainable: neuron_weights
            if self.training and not self.freeze:
                if not self.gumbel:
                    neuron_weights_soft = F.softmax(
                        self.neuron_weights / self.neuron_tau, dim=-1
                    )
                else:
                    neuron_weights_soft = self.gumbel_softmax(
                        self.neuron_weights, self.neuron_tau, dim=-1
                    )

                if self.ste:
                    neuron_weights = (
                        neuron_weights_hard - neuron_weights_soft
                    ).detach() + neuron_weights_soft
                else:
                    neuron_weights = neuron_weights_soft

        if not self.in_neuron_phase or self.phase_unified:  # link search phase
            # trainable: link_weights_a, link_weights_b
            if self.training and not self.freeze:
                if not self.gumbel:
                    a_soft = torch.matmul(
                        x, F.softmax(link_weights_a_masked / self.link_tau, dim=-1).T
                    )
                    b_soft = torch.matmul(
                        x, F.softmax(link_weights_b_masked / self.link_tau, dim=-1).T
                    )
                else:
                    a_soft = torch.matmul(
                        x,
                        self.gumbel_softmax(
                            link_weights_a_masked,
                            self.link_tau,
                            -1,
                            self.gumbel_noise_scale,
                        ).T,
                    )
                    b_soft = torch.matmul(
                        x,
                        self.gumbel_softmax(
                            link_weights_b_masked,
                            self.link_tau,
                            -1,
                            self.gumbel_noise_scale,
                        ).T,
                    )

                if self.ste:
                    a = (a_hard - a_soft).detach() + a_soft
                    b = (b_hard - b_soft).detach() + b_soft
                else:
                    a, b = a_soft, b_soft

        if not self.training:
            a = a.to(torch.uint8)
            b = b.to(torch.uint8)
            neuron_weights = neuron_weights.to(torch.uint8)

        if self._use_ext_kernels:
            # CUDA kernels expect inputs in (neurons, batch) format instead of (batch, neurons)
            a = a.transpose(0, 1).contiguous()
            b = b.transpose(0, 1).contiguous()
            if self.training:
                y = LogicLayerCudaFunction.apply(a, b, neuron_weights).transpose(0, 1)
            else:
                neuron_weights = self.neuron_weights.argmax(-1).to(torch.uint8)
                with torch.no_grad():
                    y = self._cuda_ext.eval(a, b, neuron_weights).transpose(0, 1)
        else:
            y = bin_op_s(a, b, neuron_weights)

        if not self.training:
            # Convert the output back to the same data type as the input
            # This part could be sped up by using the uint8 data type throughout the evaluation
            y = y.to(x.dtype)

        # if self.training and self.neuron_tau != 1. and self.in_neuron_phase:
        #     y = GradFactor.apply(y, self.neuron_tau)
        # elif self.training and self.link_tau != 1. and not self.in_neuron_phase:
        #     y = GradFactor.apply(y, self.link_tau)
        if self.training and self.grad_factor != 1.0:
            return GradFactor.apply(y, self.grad_factor)
        return y

    @staticmethod
    # can also use torch.nn.functional.gumbel_softmax without the noise scaling
    def gumbel_softmax(logits, tau, dim, gumbel_noise_scale, eps=1e-10):
        U = torch.rand_like(logits)
        gumbel_noise = -torch.log(-torch.log(U + eps) + eps) * gumbel_noise_scale
        y = logits + gumbel_noise
        return F.softmax(y / tau, dim=dim)

    @torch.no_grad()
    # returns output mask: True for keep, False for prune
    def prune_neurons(self, min_neuron_left=1, threshold=0.8):
        w_softmax = F.softmax(self.neuron_weights / self.neuron_tau, dim=-1)
        w_softmax_max = torch.max(w_softmax, dim=-1)[0]
        mask = w_softmax_max >= threshold
        self.output_mask &= mask

        # make sure at least min_neuron_left neurons are kept
        min_neuron_left = int(
            max(
                1,
                min(
                    self.out_dim,
                    (
                        min_neuron_left
                        if min_neuron_left >= 1
                        else self.out_dim * min_neuron_left
                    ),
                ),
            )
        )
        num_remain = self.output_mask.sum().item()
        if num_remain < min_neuron_left:
            shortfall = min_neuron_left - num_remain
            pruned_indices = (~self.output_mask).nonzero(as_tuple=False).view(-1)
            indices_to_unprune = pruned_indices[
                torch.randperm(pruned_indices.size(0))[:shortfall]
            ]
            self.output_mask[indices_to_unprune] = True

        logging.debug(
            f"\t\t\ttotal output num_prune: {(~self.output_mask).sum().item()}"
        )
        return self.output_mask

    @torch.no_grad()
    def add_input_mask(self, mask):  # mask: True for keep, False for prune
        def _add_input_mask(m):
            m &= mask.view(1, -1)
            # if a row becomes all False, randomly select one to be True
            if m.sum(dim=-1).min().item() == 0:
                indices = torch.where(m.sum(dim=-1) == 0)[0]
                m[indices, torch.randint(0, self.in_dim, (indices.size(0),))] = True

        _add_input_mask(self.link_weights_a_mask)
        _add_input_mask(self.link_weights_b_mask)

    @torch.no_grad()
    def update_temperature(self):
        if self.in_neuron_phase or self.phase_unified:
            self.neuron_tau = max(self.neuron_tau * self.tau_decay, self.tau_min)

        if not self.in_neuron_phase or self.phase_unified:
            self.link_tau = max(self.link_tau * self.tau_decay, self.tau_min)

    @torch.no_grad()
    def get_num_neurons(self):
        return self.output_mask.sum().item()

    @torch.no_grad()
    def get_num_params(self):
        # for each unmasked neuron, store the gate type and the indices of the two input neurons
        num_unmasked = self.get_num_neurons()
        num_values = int(np.ceil(num_unmasked / (DTYPE_TO_BITS[dtype_for_floats] // 4)))
        num_indices = 2 * num_unmasked
        return num_values, num_indices

    @torch.no_grad()
    def get_ops_dict(self):
        ops_dict = defaultdict(lambda: defaultdict(int))
        gate_indices = (
            self.neuron_weights.argmax(-1)[self.output_mask].detach().cpu().numpy()
        )
        update_ops_dict_from_gate_indices(ops_dict, gate_indices)
        return ops_dict


class SumLayer(BaseLayer):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        device: str,
        link_threshold: float = 0.8,
        ste_sum_layer: bool = True,
        phase_unified: bool = False,
        tau_out: float = 1.0,
        tau_init: float = 1.0,
        tau_decay: float = 1.0,
        tau_min: float = 1.0,
        num_heads: int = 1,
        **kwargs,
    ):
        super().__init__(
            in_dim=in_dim,
            out_dim=out_dim,
            device=device,
            phase_unified=phase_unified,
            tau_init=tau_init,
            tau_decay=tau_decay,
            tau_min=tau_min,
        )
        self.link_threshold = link_threshold
        self.ste = ste_sum_layer
        self.tau_out = tau_out
        self.num_heads = num_heads

        self.init_params()

        self.init_phase()

    def init_params(self):
        self.sum_weights = nn.ParameterList(
            [
                nn.Parameter(
                    nn.init.uniform_(
                        torch.empty(self.in_dim, self.out_dim), a=-0.1, b=0.1
                    )
                )
                for _ in range(self.num_heads)
            ]
        )
        input_mask = [
            torch.ones(self.in_dim, dtype=torch.bool, device=self.device)
            for _ in range(self.num_heads)
        ]
        self.register_buffer("input_mask", input_mask)

    def forward(self, x):
        # Compute output for each head independently
        outputs = []

        for i in range(self.num_heads):
            link_soft = torch.sigmoid(self.sum_weights[i][self.input_mask[i]] / self.tau)
            link_hard = (link_soft >= self.link_threshold).float()

            if (
                self.training
                and not self.freeze
                and (not self.in_neuron_phase or self.phase_unified)
            ):
                if self.ste:
                    link = (link_hard - link_soft).detach() + link_soft
                else:
                    link = link_soft
            else:
                link = link_hard

            y = torch.matmul(x[..., self.input_mask[i]], link) / self.tau_out
            outputs.append(y)

        return torch.cat(outputs, dim=-1)

    @torch.no_grad()
    # prune input neurons that does not contribute too much to any output neuron
    def prune_neurons(self, **kwargs):
        for i in range(self.num_heads):
            link_soft = torch.sigmoid(self.sum_weights[i] / self.tau)
            link_hard = (link_soft >= self.link_threshold).float()
            mask = link_hard.sum(dim=-1) > 0
            self.input_mask[i] &= mask

            if self.input_mask[i].sum().item() == 0:
                logging.warning(
                    f"\t\t\tall input neurons are pruned for head {i}. Resetting to all True."
                )
                self.input_mask[i].fill_(1)

            logging.debug(f"\t\t\ttotal input num_prune for head {i}: {(~self.input_mask[i]).sum().item()}")

        return None

    @torch.no_grad()
    def add_input_mask(self, mask):  # mask: True for keep, False for prune
        for i in range(self.num_heads):
            self.input_mask[i] &= mask

    @torch.no_grad()
    def update_temperature(self):
        if not self.in_neuron_phase or self.phase_unified:
            self.tau = max(self.tau * self.tau_decay, self.tau_min)

    @torch.no_grad()
    def get_num_neurons(self):
        return self.out_dim * self.num_heads

    @torch.no_grad()
    def get_num_params(self):
        total_params = 0
        for i in range(self.num_heads):
            # for each output neuron, keep a list of input links
            link_soft = torch.sigmoid(self.sum_weights[i][self.input_mask[i]] / self.tau)
            link_hard = (link_soft >= self.link_threshold).float()
            total_params += int(link_hard.sum().item())
        return 0, total_params

    @torch.no_grad()
    def get_ops_dict(self):
        ops_dict = defaultdict(lambda: defaultdict(int))
        for i in range(self.num_heads):
            link_soft = torch.sigmoid(self.sum_weights[i][self.input_mask[i]] / self.tau)
            link_hard = (link_soft >= self.link_threshold).float()
            for j in range(self.out_dim):
                num_inputs_to_sum = int(link_hard[:, j].sum().item())
                ops_dict["aggregation"][num_inputs_to_sum] += 1
        return ops_dict


# Adapted from the "difflogic - A Library for Differentiable Logic Gate Networks" GitHub folder:
# https://github.com/Felix-Petersen/difflogic/blob/main/difflogic/difflogic.py
class LogicLayerCudaFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b, w):
        logiclayer_cuda = get_cuda_extension()
        if logiclayer_cuda is None:
            raise RuntimeError("CUDA extension is not available.")

        ctx.save_for_backward(a, b, w)
        return logiclayer_cuda.forward(a, b, w)

    @staticmethod
    def backward(ctx, grad_y):
        logiclayer_cuda = get_cuda_extension()
        if logiclayer_cuda is None:
            raise RuntimeError("CUDA extension is not available.")

        a, b, w = ctx.saved_tensors
        grad_y = grad_y.contiguous()
        grad_a = grad_b = grad_w = None

        if ctx.needs_input_grad[0] or ctx.needs_input_grad[1]:
            grad_a, grad_b = logiclayer_cuda.backward_ab(a, b, w, grad_y)
            if not ctx.needs_input_grad[0]:
                grad_a = None
            if not ctx.needs_input_grad[1]:
                grad_b = None

        if ctx.needs_input_grad[2]:
            grad_w = logiclayer_cuda.backward_w(a, b, grad_y)

        return grad_a, grad_b, grad_w
