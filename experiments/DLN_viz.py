import argparse
from sympy.logic.boolalg import Or, And, Not, Xor
from sympy import symbols, simplify_logic, sympify
import subprocess
import json
import os
import sys
import re
from collections import defaultdict
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.settings import dtype_for_floats



def sympy_to_graphviz_nodes(expr, class_node, nodes, edges, feats_dict, edge_counts):
    counter = len(nodes)

    def parse_subexpr(subexpr, parent):
        nonlocal counter
        node_str = str(subexpr)
        node_id = None

        # Check if this sub-expression has already been processed
        for existing_node_id, (existing_node_str, _, _, _) in nodes.items():
            if existing_node_str == node_str:
                node_id = existing_node_id
                break

        if node_id is None:
            if isinstance(subexpr, And):
                node_id = f"AND{counter}"
                nodes[node_id] = (node_str, "AND", 'lightblue', 'diamond')
            elif isinstance(subexpr, Or):
                node_id = f"OR{counter}"
                nodes[node_id] = (node_str, "OR", 'palegreen', 'diamond')
            elif isinstance(subexpr, Not):
                node_id = f"NOT{counter}"
                nodes[node_id] = (node_str, "NOT", 'lightpink', 'diamond')
            elif isinstance(subexpr, Xor):
                node_id = f"XOR{counter}"
                nodes[node_id] = (node_str, "XOR", 'orange', 'diamond')
            else:  # It's a feature
                node_id = node_str
                feat_name = feats_dict.get(node_str, node_str)
                nodes[node_id] = (node_str, feat_name, 'lightyellow', 'box')
            counter += 1

        edges.add((node_id, parent))

        # Recursively process all arguments of the expression
        if isinstance(subexpr, (And, Or, Not, Xor)):
            for arg in subexpr.args:
                parse_subexpr(arg, node_id)

        # Count the number of incoming edges for class nodes
        if 'Class' in parent:
            edge_counts[(node_id, class_node)] = edge_counts.get((node_id, class_node), 0) + 1

    parse_subexpr(expr, class_node)


def simplify_and_plot(rules_dict, class_dict, feats_dict, graph_path, only_calc_stats=False):
    # only_calc_stats: If True, only calculate and print the number of features, nodes, and edges, operations, etc., will not plot the graph

    nodes = {}
    edges = set()
    edge_counts = {}

    # Create nodes for classes
    for class_name in class_dict.keys():
        class_node_id = f"class_{class_name}"
        nodes[class_node_id] = (class_name, class_name, 'lightcoral', 'ellipse')

    # Process each rule and add edges
    for class_name, rule_names in class_dict.items():
        class_node_id = f"class_{class_name}"
        for rule_name in rule_names:
            equation_str = rules_dict[rule_name]
            expr = sympify(equation_str)
            simplified_expr = simplify_logic(expr, deep=True)
            sympy_to_graphviz_nodes(simplified_expr, class_node_id, nodes, edges, feats_dict, edge_counts)

    if only_calc_stats:
        calculate_stats(nodes, edges, edge_counts, feats_dict)
        return

    # Create the graph
    from graphviz import Digraph
    feat_fontsize = '24'
    default_fontsize = '16'
    dot = Digraph(node_attr={'style': 'filled', 'fontsize': default_fontsize})
    dot.attr(ranksep='0.35')  # Smaller values decrease vertical spacing
    dot.attr(nodesep='0.1')  # Smaller values decrease spacing between nodes on the same rank
    dot.attr(splines='true') # Smooth edges
    dot.attr(rankdir='TB')  # TB: top to bottom layout; LR: left to right layout

    # Add nodes and edges to the graph
    for node_id, (node_str, label, color, shape) in nodes.items():
        if node_str in feats_dict.keys() or 'Class' in node_str or 'True' in node_str or 'False' in node_str:
            fontsize = feat_fontsize
        else:
            fontsize = default_fontsize
        dot.node(node_id, label, fillcolor=color, shape=shape, fontsize=fontsize)

    for src, dst in edges:
        count = edge_counts.get((src, dst), 1)
        label = f"*{count}" if count > 1 else ""
        dot.edge(src, dst, label=label, fontsize=feat_fontsize)  # Can use external label (xlabel) to have more control over spacing

    dot.format = 'png'
    dot.render(graph_path, view=False, cleanup=True)


def calculate_stats(nodes, edges, edge_counts, feats_dict):
    # get list of nodes that involved in the decision-making
    feat_nodes = [node for node in nodes.keys() if node in feats_dict.keys() and node != 'True' and node != 'False']
    class_nodes = [node for node in nodes.keys() if 'class_Class_' in node]
    logic_nodes = [node for node in nodes.keys() if node not in feats_dict.keys() and node not in class_nodes and node != 'True' and node != 'False']

    # get a dict of {node_id: # input edges} for logic nodes and class nodes
    input_edge_num = defaultdict(int)
    for (src, dst) in edges:
        if dst in class_nodes:
            input_edge_num[dst] += edge_counts[(src, dst)]
        elif dst in logic_nodes:
            input_edge_num[dst] += 1

    # get a dict of {operation: {op_type: count}}
    ops_dict = defaultdict(lambda: defaultdict(int))

    # feature nodes
    cat_feat_pattern = r'\s*==\s*1\s*|\s*==\s*0\s*'  # pattern for categorical features, e.g., A == 1 or A == 0
    for node in feat_nodes:
        node_str = feats_dict.get(node, node)
        if bool(re.search(cat_feat_pattern, node_str)):
            # binary equality check
            ops_dict['equality']['bool'] += 1
        else:
            # float comparison
            ops_dict['comparison'][dtype_for_floats] += 1

    # logic nodes
    for node in logic_nodes:
        num_input = input_edge_num[node]
        if 'AND' in node:
            ops_dict['AND'][num_input] += 1
        elif 'XOR' in node:  # put XOR before OR
            ops_dict['XOR'][num_input] += 1
        elif 'OR' in node:
            ops_dict['OR'][num_input] += 1
        elif 'NOT' in node:
            ops_dict['NOT'][num_input] += 1

    # class nodes
    for node in class_nodes:
        ops_dict['aggregation'][input_edge_num[node]] += 1

    num_edges = len([e for e in edges if 'class_Class_' not in e[1]]) + sum(edge_counts.values())

    ops_dict = {k: dict(v) for k, v in ops_dict.items()}  # convert defaultdict to dict (for readability)

    print(json.dumps({
        'num_feats': len(feat_nodes),
        'num_logics': len(logic_nodes),
        'num_classes': len(class_nodes),
        'num_edges': num_edges,
        'ops_dict': ops_dict}))


def execute_and_capture_dicts(sympy_code_path):
    result = subprocess.run(['python', sympy_code_path], capture_output=True, text=True)
    if result.stdout:
        return json.loads(result.stdout)
    else:
        raise ValueError("Script did not produce any output.")




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualization for DLN decision making.')
    parser.add_argument("sympy_code_path", help="The .py path of the sympy code, e.g., F/sympy_code.py")
    parser.add_argument("graph_path", help="The path of the graph, will be appended with .png, e.g., F/DLN_viz")
    parser.add_argument('--only_calc_stats', action='store_true',
                        help='Only calculate and prints the graph stats without plotting the graph.')
    args = parser.parse_args()

    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dicts = execute_and_capture_dicts(args.sympy_code_path)
    feats_dict = dicts['feats_dict']
    rules_dict = dicts['rules_dict']
    class_dict = dicts['class_dict']

    for feat in feats_dict.keys():
        exec(f"{feat} = symbols('{feat}')")

    simplify_and_plot(rules_dict, class_dict, feats_dict, args.graph_path, only_calc_stats=args.only_calc_stats)



