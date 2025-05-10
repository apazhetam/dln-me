import random
import numpy as np
import torch
from dataclasses import fields
import os

from torch.utils.data import DataLoader, SubsetRandomSampler
import random


# --------------------------------------------------------------------
# utility functions


def filter_dict_to_dataclass_fields(dataclass_type, input_dict):
    field_names = {field.name for field in fields(dataclass_type)}
    return {k: v for k, v in input_dict.items() if k in field_names}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_params_path(dataset, seed, make_dir=False, is_ray_tune=False):
    if not is_ray_tune:
        folder_name = "model_params"
    else:  # params search in Ray Tune
        folder_name = "ray_results"
    parent_dir_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    abs_param_path = os.path.join(
        parent_dir_path, f"{folder_name}/{dataset}/seed_{seed}"
    )

    if make_dir and not os.path.exists(abs_param_path):
        os.makedirs(abs_param_path)
    return abs_param_path


def get_results_path(dataset, seed, make_dir=False):
    parent_dir_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_path = f"{parent_dir_path}/results/{dataset}/seed_{seed}"
    if make_dir and not os.path.exists(results_path):
        os.makedirs(results_path)
    return results_path


def get_rule_features_path(dataset, seed, make_dir=False):
    parent_dir_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_path = f"{parent_dir_path}/rule_features/{dataset}/seed_{seed}"
    if make_dir and not os.path.exists(results_path):
        os.makedirs(results_path)
    return results_path


# --------------------------------------------------------------------
# Approximated computation of number of logic gate operations
# NOTE: This is a simplified version of the actual computation, which should be both hardware and software dependent.


# return the number of basic logic gate level operations from higher-level operations of a specific data type
def get_num_basic_gate_ops_from_dtype(op, dtype):
    assert ("float" in dtype) or ("int" in dtype) or (dtype == "bool")

    if "float" in dtype:
        logic_ops_dict = get_basic_gate_ops_dict_from_floats(op, dtype)
    else:
        assert dtype in ["int8", "int16", "int32", "int64", "bool"]
        logic_ops_dict = get_basic_gate_ops_dict_from_inp_bits(op, DTYPE_TO_BITS[dtype])

    return sum([LOGIC_TO_BASIC_GATE_OPS[k] * v for k, v in logic_ops_dict.items()])


# return the dict of 2-input logic gate level operations from higher-level floating point operations
def get_basic_gate_ops_dict_from_floats(op, dtype):
    assert op in [
        "equality",
        "comparison",
        "addition",
        "subtraction",
        "multiplication",
        "division",
        "logarithm",
    ]
    assert dtype in ["float16", "float32", "float64"]

    # get the number of exponent bits and mantissa bits
    if dtype == "float16":
        exponent_bits = 5
        mantissa_bits = 10
    elif dtype == "float32":
        exponent_bits = 8
        mantissa_bits = 23
    else:  # dtype == 'float64'
        exponent_bits = 11
        mantissa_bits = 52

    if op == "equality":
        # check equality in sign, exponents, and mantissa. Worst case is n-bit equality.
        return get_basic_gate_ops_dict_from_inp_bits(
            "equality", exponent_bits + mantissa_bits + 1
        )

    if op == "comparison":
        # compare sign, then exponents, then mantissa. Worst case is n-bit comparison.
        return get_basic_gate_ops_dict_from_inp_bits(
            "comparison", exponent_bits + mantissa_bits + 1
        )

    if op == "addition":
        # first align exponents, which takes 1 e-bit comparison.
        # Then, add or subtract mantissa, which takes 1 (m+1)-bit addition.
        return merge_dicts(
            get_basic_gate_ops_dict_from_inp_bits("comparison", exponent_bits),
            get_basic_gate_ops_dict_from_inp_bits("addition", mantissa_bits + 1),
        )

    if op == "subtraction":
        # subtraction is addition with negation, so same operation count as addition.
        return get_basic_gate_ops_dict_from_floats("addition", dtype)

    if op == "multiplication":
        # XOR signs, add exponents (which takes 1 e-bit addition),
        # then multiply mantissa (which takes 1 (m+1)-bit multiplication).
        return merge_dicts(
            {"XOR": 1},
            merge_dicts(
                get_basic_gate_ops_dict_from_inp_bits("addition", exponent_bits),
                get_basic_gate_ops_dict_from_inp_bits(
                    "multiplication", mantissa_bits + 1
                ),
            ),
        )

    if op == "division":
        # XOR signs, exponents are subtracted, mantissas are divided.
        return merge_dicts(
            {"XOR": 1},
            merge_dicts(
                get_basic_gate_ops_dict_from_inp_bits("subtraction", exponent_bits),
                get_basic_gate_ops_dict_from_inp_bits("division", mantissa_bits + 1),
            ),
        )

    if (
        op == "logarithm"
    ):  # approximated, not exact; actual implementation should be both hardware and software dependent
        # for exponents, may need to convert the base (and use lookup table), which takes 1 e-bit multiplication.
        # then, take logarithm of mantissa, which takes 1 (m+1)-bit logarithm.
        # finally, add the log of exponents and mantissas together, which takes 1 (m+1)-bit addition.
        return merge_dicts(
            get_basic_gate_ops_dict_from_inp_bits("multiplication", exponent_bits),
            merge_dicts(
                get_basic_gate_ops_dict_from_inp_bits("logarithm", mantissa_bits + 1),
                get_basic_gate_ops_dict_from_inp_bits("addition", mantissa_bits + 1),
            ),
        )


# return dict of number of 2-input logic gate level operations from higher-level operations with n bits of precision
def get_basic_gate_ops_dict_from_inp_bits(op, n):
    assert op in [
        "equality",
        "comparison",
        "addition",
        "subtraction",
        "multiplication",
        "division",
        "logarithm",
    ]
    # n represents the number of bits in the input, or the number of bits of precision
    assert isinstance(n, int) and n > 0

    if op == "equality":
        # To test whether A == B, for every bit, do XNOR to check whether they equal.
        # Then, do n-input AND on all XNOR results.
        return {"XNOR": n, "AND": n - 1}

    if op == "comparison":
        # To test whether A > B, for every bit, do P_i = a_i XNOR b_i to check whether they equal.
        # Do G_i = a_i AND !b_i  to see if a_i is 1 and b_i is 0, this should cascade with all previous XNOR results.
        # Result = G_(n-1) OR (G_(n-1) AND P_(n-1)) OR (G_(n-2) AND P_(n-1) AND P_(n-2)) …
        # In summary, n XNOR, (n-1) OR, n(n-1)/2 AND
        return {"XNOR": n, "OR": n - 1, "AND": n * (n - 1) // 2}

    if op == "addition":
        # First bit uses half adder, the rest uses full adder.
        # Half adder takes 5 NAND, full adder takes 9 NAND.
        # In summary, 5 + 9*(n-1) NAND
        return {"NAND": 5 + 9 * (n - 1)}

    if op == "subtraction":
        # Subtraction is addition with negation, so same operation count as addition.
        return get_basic_gate_ops_dict_from_inp_bits("addition", n)

    if op == "multiplication":
        # Every bit in A should multiply with every bit in B, n^2 AND.
        # Then, shift and sum together, needs n half adders and (n-1)^2 full adders.
        # In summary, n^2 AND, (9*n^2 - 13n + 9) NAND.
        return {"AND": n**2, "NAND": 9 * n**2 - 13 * n + 9}

    if op == "division":
        # To keep n bits of precision after division, (n+1) n-bit comparisons, (n+1) n-bit subtractions.
        return merge_dicts(
            scale_dict_values(
                get_basic_gate_ops_dict_from_inp_bits("comparison", n), n + 1
            ),
            scale_dict_values(
                get_basic_gate_ops_dict_from_inp_bits("subtraction", n), n + 1
            ),
        )

    if op == "logarithm":
        # To keep n bits of precision, if uses CORDIC,
        # approximately takes n rounds and each round requires: n n-bit additions, n n-bit multiplications.
        return scale_dict_values(
            merge_dicts(
                scale_dict_values(
                    get_basic_gate_ops_dict_from_inp_bits("addition", n), n
                ),
                scale_dict_values(
                    get_basic_gate_ops_dict_from_inp_bits("multiplication", n), n
                ),
            ),
            n,
        )


# return the number of basic logic gate level operations to sum n 1-bit inputs
def get_num_basic_gate_ops_aggregation(num_binary_inputs):
    # Arrange adders as A_m, ..., A_1, A_0, where A_m is the most significant bit and m >= log2(n)
    # A_0 is half adder, rest are full adders
    # In worst case, A_0 called n times, A_1 called n/2 times, ..., A_m n/(2^m) times
    # In summary, at most n half adders called, n full adders called
    logic_ops_dict = {"NAND": 5 * num_binary_inputs + 9 * num_binary_inputs}
    return sum([LOGIC_TO_BASIC_GATE_OPS[k] * v for k, v in logic_ops_dict.items()])


# return the number of high-level and basic gate level operations from a dictionary of operations
# high-level number of operations is the total number of nodes (feature, logic, class)
# low-level number of operations is the total number of basic gate operations
def get_num_ops_info_from_ops_dict(ops_dict):
    num_hl_ops, num_bg_ops = 0, 0
    for inp_dict in ops_dict.values():
        num_hl_ops += sum(inp_dict.values())

    for op, inp_dict in ops_dict.items():
        # basic gate
        if op in LOGIC_TO_BASIC_GATE_OPS.keys():
            # inp_dict is {str(number of inputs): count}
            if op in ["AND", "OR", "NAND", "NOR"]:
                # n-input needs n-1 gates.
                num_bg_ops += sum(
                    [
                        LOGIC_TO_BASIC_GATE_OPS[op] * (int(k) - 1) * v
                        for k, v in inp_dict.items()
                    ]
                )
            else:
                # other gates' number of operations is fixed.
                num_bg_ops += sum(
                    [LOGIC_TO_BASIC_GATE_OPS[op] * v for k, v in inp_dict.items()]
                )

        elif op in ["aggregation"]:
            # inp_dict is {str(number of binary inputs): count}
            num_bg_ops += sum(
                [
                    get_num_basic_gate_ops_aggregation(int(k)) * v
                    for k, v in inp_dict.items()
                ]
            )

        elif op in [
            "equality",
            "comparison",
            "addition",
            "subtraction",
            "multiplication",
            "division",
            "logarithm",
        ]:
            # inp_dict is {dtype: count}
            num_bg_ops += sum(
                [
                    get_num_basic_gate_ops_from_dtype(op, k) * v
                    for k, v in inp_dict.items()
                ]
            )

        else:
            raise ValueError(f"Unknown operation: {op}")
    return num_hl_ops, num_bg_ops


# update the ops_dict with the gate indices of the logic layer
def update_ops_dict_from_gate_indices(ops_dict, gate_indices):
    for idx in gate_indices:
        op_name = LOGIC_LAYER_IDX_TO_GATE[idx]
        ops_dict[op_name]["2"] += 1  # logic neurons have 2 inputs


# convert 2-input / NOT logic operation to the number of basic gate operations
LOGIC_TO_BASIC_GATE_OPS = {
    "NOT": 0,
    "AND": 1,
    "OR": 1,
    "NAND": 1,
    "NOR": 1,
    "XOR": 3,
    "XNOR": 3,
    "IMPLY": 1,
    "NIMPLY": 1,
    "False": 0,
    "True": 0,
    "PASS": 0,
    "NPASS": 0,
}


# convert index of LogicLayer's gate to its function name
LOGIC_LAYER_IDX_TO_GATE = {
    0: "False",
    1: "AND",
    2: "NIMPLY",
    3: "PASS",
    4: "NIMPLY",
    5: "PASS",
    6: "XOR",
    7: "OR",
    8: "NOR",
    9: "XNOR",
    10: "NPASS",
    11: "IMPLY",
    12: "NPASS",
    13: "IMPLY",
    14: "NAND",
    15: "True",
}


DTYPE_TO_BITS = {
    "float16": 16,
    "float32": 32,
    "float64": 64,
    "int8": 8,
    "int16": 16,
    "int32": 32,
    "int64": 64,
    "bool": 1,
}


# helper function to scale all values in a dictionary
def scale_dict_values(original_dict, factor):
    return {k: v * factor for k, v in original_dict.items()}


# helper function to merge two dictionaries
def merge_dicts(d1, d2):
    merged_dict = d1.copy()
    for key, value in d2.items():
        if key in merged_dict:
            merged_dict[key] += value
        else:
            merged_dict[key] = value
    return merged_dict


# helper function to merge defaultdict(lambda: defaultdict(int)) d2 to another same-typed d1
def merge_defaultdict_to_another(d1, d2):
    for key, inner_dict in d2.items():
        for inner_key, value in inner_dict.items():
            d1[key][inner_key] += value


def get_bootstrapped_loaders(dataset, num_heads, batch_size=64):
    loaders = []
    data_size = len(dataset)
    indices = list(range(data_size))

    for _ in range(num_heads):
        # Sample with replacement
        bootstrap_indices = [random.choice(indices) for _ in range(data_size)]
        sampler = SubsetRandomSampler(bootstrap_indices)
        loader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)
        loaders.append(loader)

    return loaders
