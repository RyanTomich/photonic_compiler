import json
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import time
import copy

import dijkstra as dijk
import stacked_graph as sg
import testing as test
import graph_visualization as gv


# read_json_path = '/home/rjtomich/photonic_compiler/model_to_graph/gpt2_graph.json'
# read_json_path = '/home/rjtomich/photonic_compiler/model_to_graph/bert-base-uncased_graph.json'
# read_json_path = '/home/rjtomich/photonic_compiler/Pytorch-LeNet/simple_LeNet_graph.json'
read_json_path = '/home/rjtomich/photonic_compiler/Pytorch-LeNet/simple_LeNet_graph_NoFusion.json'
with open(read_json_path)  as json_file:
    raw_json = json.load(json_file) # returns json file as dict

graph = sg.StackedGraph(raw_json=raw_json)

print("--------------------------------------------")

groups = list(graph.get_node_groups(ASAP = False))
print(f'{groups=}')
print(test.group_validate(graph, groups))

# print(test_group)
# print(groups[11])

# print(graph.stack_list[graph.id_to_idx[test_group[0]]])

test_group = groups[0]

start_stack = sg.StackedNode(0, [], [[]], [[]], opp='start', func_stack=['start'], cost_stack=[0])
first_stack = copy.deepcopy(graph.stack_list[graph.id_to_idx[test_group[0]]])
first_stack.parents = [0]

subgraph_stack_list = [start_stack, first_stack]
for stack_id in test_group[1:]:
    stack = graph.stack_list[graph.id_to_idx[stack_id]]
    new_node = copy.deepcopy(stack)
    new_node.parents = set(new_node.parents) - graph.load_nodes
    subgraph_stack_list.append(new_node)


graph_10 = sg.StackedGraph(stack_list=subgraph_stack_list)
gv.adj_to_graph(graph_10.adj_matrix, save=True)

# print(dijk.branching_stacked_dijkstra(graph_10))
print(dijk.rolling_dijkstra(graph_10))

# for i in dijk.branching_stacked_dijkstra(graph, (0,0)):
#     print(f'{graph.stack_list[i[0]].oppid}: {graph.stack_list[i[0]].opp} --> {graph.stack_list[i[0]].func_stack[i[1]]}')
