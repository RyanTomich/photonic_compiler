import json
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
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

def create_schedule_data(graph):
    data = {
        'hardware': [],
        'start': [],
        'end': [],
        'label': []  # Labels for the blocks
    }

    # for stack in graph.stack_list[16:-6]:
    for stack in graph.stack_list[1:]:
        data['hardware'].append(stack.hardware_selection)
        data['start'].append(stack.start_time)
        data['end'].append(stack.start_time + stack.cost_stack[stack.func_selection])
        data['label'].append(stack.oppid)

    df = pd.DataFrame(data)
    return df


graph = sg.StackedGraph(raw_json=raw_json)
# gv.adj_to_graph(graph.adj_matrix, save=True, layout = 'spectral')
# dijk.scheduling_dijkstra(subgraphs[0])
subgraphs = list(dijk.graph_partition(graph))
dijk.select_nodes(graph, subgraphs)
end_time = dijk.schdeule_nodes(graph, subgraphs)

for stack in graph.stack_list:
    assert stack.hardware_selection != None

print(end_time)


### Schedule Data ###
schedule_data = create_schedule_data(graph)
with open('schedule.txt', 'w') as file:
    for index, row in schedule_data.iterrows():
        file.write(f"{row['label']} --- {row['hardware']} ({row['start']})\n")

# sorted_df = schedule_data.sort_values(by='start')
# print(sorted_df)



### FORWARD ###
# print(f'{end_time}')
# print(f'{graph.forward()=}')

# has_ph = set()
# selected_ph = set()
# for stack in graph.stack_list:
#     if len(stack.func_stack) > 1:
#         has_ph.add(stack.oppid)
#     if stack.func_selection == 1:
#         selected_ph.add(stack.oppid)

# print(f'{len(has_ph)=}')
# print(f'{len(selected_ph)=}')
