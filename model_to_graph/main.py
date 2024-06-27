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
# a = {1096, 1097, 1098, 1099, 1100, 1101, 1102, 599}
# for stack in graph.stack_list:
#     if stack.oppid in a:
        # print(stack)
print(graph.stack_list[71])

order, layers_list, layer_count = graph.kahn_topo_sort_working(transpose = True)
print(order)
# print(layers_list)
# print(layer_count)

# print(graph.get_cuts())

# ap = graph.tarjan_articulation_points()
# print(ap)
# print(len(ap))
# print(list(graph.graph_partition(ap)))

# for i in dijk.branching_stacked_dijkstra(graph, (0,0)):
#     print(f'{graph.stack_list[i[0]].oppid}: {graph.stack_list[i[0]].opp} --> {graph.stack_list[i[0]].func_stack[i[1]]}')
