import json

import dijkstra as dijk
import stacked_graph as sg
import testing as test
import operator_calcs as oc


JSON_PATH = "/home/rjtomich/photonic_compiler/model_to_graph/gpt2_graph.json"
# JSON_PATH = '/home/rjtomich/photonic_compiler/model_to_graph/bert-base-uncased_graph.json'
# JSON_PATH = '/home/rjtomich/photonic_compiler/Pytorch-LeNet/simple_LeNet_graph.json'
# JSON_PATH = '/home/rjtomich/photonic_compiler/Pytorch-LeNet/simple_LeNet_graph_NoFusion.json'
with open(JSON_PATH, encoding="utf-8") as json_file:
    raw_json = json.load(json_file)  # returns json file as dict
    print("... Json loaded ...")

optimization_variable = 'energy'

graph = sg.StackGraph(raw_json=raw_json, optimization_variable=optimization_variable)
stacked_subgraphs = list(dijk.graph_partition(graph))
flat_subgraphs = dijk.select_nodes(stacked_subgraphs, optimization_variable)

expanded_flat_subgraphs = dijk.expand_nodes(flat_subgraphs)
scheduled_flat_graph, end_time, break_points = dijk.schdeule_nodes(
    graph, expanded_flat_subgraphs
)

schedule_df = scheduled_flat_graph.create_schedule_data(write=True)
empty = test.schedule_validate(schedule_df)
dram, delta_dram, sram, delta_sram = dijk.get_memory_profile(scheduled_flat_graph)


print(end_time)
print(max(sram, key=lambda x: x[1]))







# print(end_time)


# dijk.get_energy_profile(graph, schedule_data)


### Visualization ###
# gv.adj_to_graph(graph.adj_matrix, save=True, layout = 'spectral')

### Schedule Data ###
# schedule_data = create_schedule_data(graph)
# with open('schedule.txt', 'w') as file:
#     for index, row in schedule_data.iterrows():
#         file.write(f"{row['label']} --- {row['hardware']} ({row['start']})\n")

# sorted_df = schedule_data.sort_values(by='start')
# print(sorted_df)

### FORWARD ###
# print(f'{end_time}')
# print(f'{graph.forward()=}')

# has_ph = set()
# selected_ph = set()
# for stack in graph.stack_list:
#     if len(stack.func_stack) > 1:
#         has_ph.add(stack.stack_id)
#     if stack.func_selection == 1:
#         selected_ph.add(stack.stack_id)

# print(f'{len(has_ph)=}')
# print(f'{len(selected_ph)=}')
