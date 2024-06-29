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

print(f'{graph.forward()=}')

has_ph = 0
selected_ph = 0
for stack in graph.stack_list:
    if len(stack.func_stack) > 1:
        has_ph += 1
    if stack.func_selection == 1:
        selected_ph +=1

print(f'{has_ph=}')
print(f'{selected_ph=}')

new_order, new_layers_list = graph.schedule()

print(len(new_order))
print(len(new_layers_list))

print(new_layers_list)
