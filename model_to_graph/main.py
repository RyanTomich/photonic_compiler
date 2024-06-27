import json
import numpy as np
import time

import dijkstra as dijk
import stacked_graph as sg


read_json_path = '/home/rjtomich/photonic_compiler/model_to_graph/gpt2_graph.json'
# read_json_path = '/home/rjtomich/photonic_compiler/model_to_graph/bert-base-uncased_graph.json'
# read_json_path = '/home/rjtomich/photonic_compiler/Pytorch-LeNet/simple_LeNet_graph.json'
# read_json_path = '/home/rjtomich/photonic_compiler/Pytorch-LeNet/simple_LeNet_graph_NoFusion.json'
with open(read_json_path)  as json_file:
    raw_json = json.load(json_file) # returns json file as dict


graph = sg.StackedGraph(raw_json=raw_json)
print(graph.stack_list[599])

ap = graph.tarjan_articulation_points()
print(ap)
print(len(ap))
print(list(graph.graph_partition(ap)))

# for i in dijk.branching_stacked_dijkstra(graph, (0,0)):
#     print(f'{graph.stack_list[i[0]].oppid}: {graph.stack_list[i[0]].opp} --> {graph.stack_list[i[0]].func_stack[i[1]]}')
