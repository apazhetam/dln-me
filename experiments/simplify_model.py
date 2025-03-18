import numpy as np
import torch
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dln_layer.layer import ThresholdLayer, LogicLayer, SumLayer


ALL_OPERATIONS = [
    "zero",
    "and",
    "not_implies",
    "a",
    "not_implied_by",
    "b",
    "xor",
    "or",
    "not_or",
    "not_xor",
    "not_b",
    "implied_by",
    "not_a",
    "implies",
    "not_and",
    "one",
]


class SimplifyLogicNet:
    def __init__(
            self,
            model: torch.nn.Sequential,
            feat_names=None,
            scaler_params=None,
            dtype_dict=None,
    ):
        super(SimplifyLogicNet, self).__init__()

        self.model = model
        self.feat_names = feat_names
        self.scaler_params = scaler_params
        self.dtype_dict = dtype_dict

        self.rules = []
        th_layers = []
        logic_layers = []
        sum_links = None

        for layer_idx, layer in enumerate(self.model.layers):

            if isinstance(layer, ThresholdLayer):
                th_bias = layer.th_bias.detach().cpu().numpy()
                th_slope = layer.th_slope.detach().cpu().numpy()
                num_cat_feats = layer.cat_dim
                next_layer_idx = self.model.idx2next_layer_idx[layer_idx]
                in_dim = layer.in_dim
                if layer_idx == 0:
                    neuron_idx_offset = 0
                else:  # this layer is concatenated with a Logic layer
                    concat_layer_idx = next_layer_idx - 1
                    neuron_idx_offset = self.model.layers[concat_layer_idx].out_dim
                th_layers.append((next_layer_idx, neuron_idx_offset, th_bias, th_slope, num_cat_feats, in_dim))

            elif isinstance(layer, LogicLayer):
                link_weights_a_masked = layer.link_weights_a.clone().masked_fill(~layer.link_weights_a_mask, float('-inf'))
                link_weights_b_masked = layer.link_weights_b.clone().masked_fill(~layer.link_weights_b_mask, float('-inf'))
                layer_a_idx = link_weights_a_masked.argmax(-1).detach().cpu().numpy()
                layer_b_idx = link_weights_b_masked.argmax(-1).detach().cpu().numpy()
                layer_gate_idx = layer.neuron_weights.argmax(-1).detach().cpu().numpy()
                logic_layers.append((layer_idx, layer_a_idx, layer_b_idx, layer_gate_idx))

            elif isinstance(layer, SumLayer):
                self.num_classes = layer.out_dim
                link_soft = torch.sigmoid(layer.sum_weights / layer.tau)
                sum_links = (link_soft >= layer.link_threshold).float()
                sum_links[~layer.input_mask] = 0
                sum_links = sum_links.detach().cpu().numpy()

            else:
                assert False, f"Error: layer {type(layer)} / {layer} unknown."

        self.th_layers = th_layers
        self.logic_layers = logic_layers
        self.sum_links = sum_links


    def get_gate_code(self, var1, var2, gate_idx):
        operation_name = ALL_OPERATIONS[gate_idx]

        if operation_name == "zero":
            res = "False"
        elif operation_name == "and":
            res = f"And({var1}, {var2})"
        elif operation_name == "not_implies":
            res = f"And({var1}, Not({var2}))"
        elif operation_name == "a":
            res = f"{var1}"
        elif operation_name == "not_implied_by":
            res = f"And({var2}, Not({var1}))"
        elif operation_name == "b":
            res = f"{var2}"
        elif operation_name == "xor":
            res = f"Xor({var1}, {var2})"
        elif operation_name == "or":
            res = f"Or({var1}, {var2})"
        elif operation_name == "not_or":
            res = f"Not(Or({var1}, {var2}))"
        elif operation_name == "not_xor":
            res = f"Not(Xor({var1}, {var2}))"
        elif operation_name == "not_b":
            res = f"Not({var2})"
        elif operation_name == "implied_by":
            res = f"Or(Not({var2}), {var1})"
        elif operation_name == "not_a":
            res = f"Not({var1})"
        elif operation_name == "implies":
            res = f"Or(Not({var1}), {var2})"
        elif operation_name == "not_and":
            res = f"Not(And({var1}, {var2}))"
        elif operation_name == "one":
            res = "True"
        else:
            assert False, f"Operator #.{gate_idx} unknown."

        return res


    def get_logic_layer_code(self, layer_idx, layer_a_idx, layer_b_idx, layer_gate_idx):
        code = []

        if self.model.concat_bool:
            in_dim = self.model.layers[layer_idx].in_dim
            code.append(f"x_{layer_idx-1}_{in_dim-2} = True")
            code.append(f"x_{layer_idx-1}_{in_dim-1} = False")

        for var_id, (a_idx, b_idx, gate_idx) in enumerate(zip(layer_a_idx, layer_b_idx, layer_gate_idx)):

            a = f"x_{layer_idx-1}_{a_idx}"  # indexing the previous layer's output
            b = f"x_{layer_idx-1}_{b_idx}"

            next_layer_idx = self.model.idx2next_layer_idx[layer_idx]
            if next_layer_idx == self.model.depth - 1:  # next layer is the last layer, so the current layer is the last logic layer
                code.append(f"Rule_{var_id} = {self.get_gate_code(a, b, gate_idx)}")
                self.rules.append(f"Rule_{var_id}")
            else:
                code.append(
                    f"x_{layer_idx}_{var_id} = {self.get_gate_code(a, b, gate_idx)}"
                )

        return code


    def get_feat_code(self):
        code = [
            "feats_dict = {}",
            "feats_mapping = {}",
            "def save_feat(expr, feat_name):",
            "    global feats_dict, feats_mapping",
            "    if isinstance(expr, Eq):",
            "        expr_str = f\"{expr.lhs} == {expr.rhs}\"",
            "    else:",
            "        expr_str = str(expr)",
            "    feats_dict[feat_name] = expr_str",
            "    feats_mapping[expr] = feat_name",
            "    feats_mapping[Not(expr)] = f\"~{feat_name}\"",
            "",
        ]

        feats_counter = 0
        for next_layer_idx, neuron_idx_offset, th_bias, th_slope, num_cat_feats, in_dim in self.th_layers:

            for neuron_idx in range(in_dim):
                feat = f"feat_{feats_counter}"
                feats_counter += 1

                feat_name = self.feat_names[neuron_idx]
                feat_dtype = self.dtype_dict[feat_name]
                feat_scaler = self.scaler_params[feat_name]

                if neuron_idx < num_cat_feats:
                    thresh_rule = f"Eq({feat_name}, 1)"

                else:
                    con_idx = neuron_idx - num_cat_feats
                    cutoff = th_bias[con_idx]
                    slope = th_slope[con_idx]

                    if slope == 0:
                        thresh_rule = "False"

                    # For DLN, all continuous features are scaled to [0, 1]
                    # NOTE: remove this True/False cast when not applicable
                    elif cutoff < 0:
                        thresh_rule = "True" if slope > 0 else "False"
                    elif cutoff > 1:
                        thresh_rule = "False" if slope > 0 else "True"

                    else:
                        if 'min' in feat_scaler:  # min-max scaling
                            cutoff = cutoff * (feat_scaler['max'] - feat_scaler['min']) + feat_scaler['min']
                        elif 'mean' in feat_scaler:  # standard scaling
                            cutoff = cutoff * feat_scaler['std'] + feat_scaler['mean']

                        if 'int' in feat_dtype:
                            cutoff = int(round(cutoff))
                        else:
                            cutoff = round(cutoff, 2)
                        cutoff_str = str(cutoff) if isinstance(cutoff, int) else f"{cutoff:.2f}"

                        if slope > 0:
                            thresh_rule = f"{feat_name} > {cutoff_str}"
                        else:
                            thresh_rule = f"{feat_name} <= {cutoff_str}"

                code.append(f'{feat} = {thresh_rule}')
                code.append(f'save_feat({feat}, "{feat}")')
                code.append(f"x_{next_layer_idx-1}_{neuron_idx+neuron_idx_offset} = {feat}")
        return code


    def get_sum_dict(self):
        class_nodes = {i: [] for i in range(self.num_classes)}
        for rule_id in range(len(self.sum_links)):
            if sum(self.sum_links[rule_id]) == self.num_classes:
                continue
            for class_id in np.nonzero(self.sum_links[rule_id])[0]:
                class_nodes[class_id].append(f"Rule_{rule_id}")
        class_dict = {f'Class_{k}': v for k, v in class_nodes.items()}
        return class_dict


    def get_sympy_code(self):

        code = [
            "from sympy import symbols, simplify, simplify_logic, Eq, Ne",
            "from sympy.logic.boolalg import Or, And, Not, Xor",
            "import json",
            "", "",
        ]

        feat_names = list(set(self.feat_names))
        feat_names_str = ' '.join(feat_names)
        feat_names_tuple = ', '.join(feat_names)

        code.append(f"{feat_names_tuple} = symbols('{feat_names_str}')")
        code.append("")
        code.extend(self.get_feat_code())

        code.append("")
        logic_code = []
        for layer_idx, layer_a_idx, layer_b_idx, layer_gate_idx in self.logic_layers:
            logic_code.extend(
                self.get_logic_layer_code(layer_idx, layer_a_idx, layer_b_idx, layer_gate_idx)
            )
            logic_code.append("")
        code.extend(logic_code)

        code.append("")
        # Note: can try running simplification multiple times (e.g., simplify(simplify_logic(simplify(simplify_logic)
        # because sometimes it doesn't simplify all the way
        # Note: if simplification takes too long, set force to False
        code.extend([f"{r}_sym = simplify(simplify_logic({r}, form='dnf', force=True)).subs(feats_mapping)"
                     for r in self.rules])

        code.extend([
            "",
            "rules_dict = {}",
            "feats_counter = max(int(feat.split('_')[1]) for feat in feats_dict.keys()) if feats_dict else 0",
            "def replace_eq_with_feature(expr):",
            "    global feats_counter, feats_mapping",
            "    if isinstance(expr, (Eq, Ne)):",
            "        if expr in feats_mapping:",
            "            return symbols(feats_mapping[expr])",
            "        else:",
            "            feats_counter += 1",
            "            new_feat_name = f\"feat_{feats_counter}\"",
            "            new_feat = symbols(new_feat_name)",
            "            feats_dict[new_feat_name] = f\"{expr.lhs} == {expr.rhs}\" if isinstance(expr, Eq) else f\"{expr.lhs} != {expr.rhs}\"",
            "            feats_mapping[expr] = new_feat_name",
            "            feats_mapping[Not(expr)] = f\"~{new_feat_name}\"",
            "            return new_feat",
            "    elif isinstance(expr, (Or, And, Not, Xor)):",
            "        return expr.func(*[replace_eq_with_feature(arg) for arg in expr.args])",
            "    else:",
            "        return expr",
            "",
            "feats_dict_expr = {v: k for k, v in feats_mapping.items()}",
            "def save_rule(rule_expr, rule_name):",
            "    rule_expr_sub = rule_expr.subs(feats_dict_expr)",
            "    if rule_expr_sub in [True, False]:",
            "        rule_expr = rule_expr_sub",
            "    else:",
            "        rule_expr_str = str(rule_expr)",
            "        if 'Eq' in rule_expr_str or 'Ne' in rule_expr_str:",
            "            rule_expr = replace_eq_with_feature(rule_expr)",
            "    rules_dict[rule_name] = str(rule_expr)",
            "",
        ])
        for r in self.rules:
            code.extend([f"save_rule({r}_sym, '{r}')"])

        class_dict = self.get_sum_dict()
        code.extend([
            "",
            f"class_dict = {class_dict}",
            "true_rules_dict = {}",
            "for c, rule_list in class_dict.items():",
            "    true_rules_dict[c] = [rule for rule in rule_list if rules_dict.get(rule, '') == 'True']",
            "min_true_count = min([len(rules) for rules in true_rules_dict.values()])",
            "for c, rule_list in class_dict.items():",
            "    true_rule_list = true_rules_dict[c]",
            "    class_dict[c] = [rule for rule in rule_list if (rule not in true_rule_list[:min_true_count] and rules_dict.get(rule, '') != 'False')]",
        ])

        code.append("")
        code.append("print(json.dumps({'feats_dict': feats_dict, 'rules_dict': rules_dict, 'class_dict': class_dict}))")
        code.append("")
        return "\n".join(code)


