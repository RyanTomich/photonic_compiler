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

# a = {1095, 1096, 1097, 1098, 1099, 1100, 1101, 1102, 1103, 1104, 1105, 1106, 1107}
# a = {1010,1011,1014,1015,1107,1012,1016,1013}

# for node in a:
#     print(graph.stack_list[node])

def validate(group):
    load_instructions = {stack.oppid for stack in graph.stack_list if stack.opp == 'null'}
    included = set(group)
    for stack in group[1:]:
        for parent in graph.stack_list[stack].parents:
            if parent in load_instructions:
                continue
            assert parent in included


groups = list(graph.get_node_groups())
for i,lst in enumerate(groups):
    print(f'    {lst}')
    validate(lst)
    print(f'group {i} passed!')


# for i in dijk.branching_stacked_dijkstra(graph, (0,0)):
#     print(f'{graph.stack_list[i[0]].oppid}: {graph.stack_list[i[0]].opp} --> {graph.stack_list[i[0]].func_stack[i[1]]}')
