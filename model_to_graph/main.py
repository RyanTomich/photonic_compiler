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
import operator_calcs as oc


read_json_path = '/home/rjtomich/photonic_compiler/model_to_graph/gpt2_graph.json'
# read_json_path = '/home/rjtomich/photonic_compiler/model_to_graph/bert-base-uncased_graph.json'
# read_json_path = '/home/rjtomich/photonic_compiler/Pytorch-LeNet/simple_LeNet_graph.json'
# read_json_path = '/home/rjtomich/photonic_compiler/Pytorch-LeNet/simple_LeNet_graph_NoFusion.json'
with open(read_json_path)  as json_file:
    raw_json = json.load(json_file) # returns json file as dict


graph = sg.StackedGraph(raw_json=raw_json)
# gv.adj_to_graph(graph.adj_matrix, save=True, layout = 'spectral')
subgraphs = list(dijk.graph_partition(graph))
dijk.select_nodes(graph, subgraphs)
end_time = dijk.schdeule_nodes(graph, subgraphs)

# for stack in graph.stack_list:
#     print(stack.hardware_selection)
#     print(stack.start_time)




### FORWARD ###

print(f'{end_time}')
print(f'{graph.forward()=}')

# has_ph = set()
# selected_ph = set()
# for stack in graph.stack_list:
#     if len(stack.func_stack) > 1:
#         has_ph.add(stack.oppid)
#     if stack.func_selection == 1:
#         selected_ph.add(stack.oppid)

# print(f'{len(has_ph)=}')
# print(f'{len(selected_ph)=}')

# # print(f"{selected_ph=}")

# new_order, new_layers_list = graph.simple_schedule()

# print(f'{len(new_order)=}')
# print(f'{len(new_layers_list)=}')

# with open('schedule.txt', 'w') as file:
#     file.write(f'{new_order}')
#     for layer in new_layers_list:
#         file.write(f'{layer}\n')
