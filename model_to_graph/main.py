import json

import dijkstra as dijk
import stacked_graph as sg


JSON_PATH = "/home/rjtomich/photonic_compiler/model_to_graph/gpt2_graph.json"
# JSON_PATH = '/home/rjtomich/photonic_compiler/model_to_graph/bert-base-uncased_graph.json'
# JSON_PATH = '/home/rjtomich/photonic_compiler/Pytorch-LeNet/simple_LeNet_graph.json'
# JSON_PATH = '/home/rjtomich/photonic_compiler/Pytorch-LeNet/simple_LeNet_graph_NoFusion.json'
with open(JSON_PATH, encoding="utf-8") as json_file:
    raw_json = json.load(json_file)  # returns json file as dict
    print('... Json loaded ...')

graph = sg.StackGraph(raw_json=raw_json)
stacked_subgraphs = list(dijk.graph_partition(graph))
flat_graph, flat_subgraphs = dijk.select_nodes(graph, stacked_subgraphs)

# for node in flat_graph.node_list:
#     if node.algorithm in {'dense_phu','pack_phu', 'matmul_phu'}:
#         print(node)

flat_graph, flat_subgraphs = dijk.expand_nodes(flat_graph, flat_subgraphs)
# end_time, break_points = dijk.schdeule_nodes(flat_graph, flat_subgraphs)

# for stack in flat_graph.node_list:
#     assert stack.hardware_selection is not None
# print('... ALL hardware selected ...')


# schedule_data = graph.create_schedule_data()
# # with open('schedule.txt', 'w') as file:
# #     for index, row in schedule_data.iterrows():
# #         file.write(f"{row['label']} --- {row['hardware']} ({row['start']})\n")
# print('... Schedule data made ...')

# dijk.get_memory_profile(graph)
# print('... Memory profile made ...')

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
