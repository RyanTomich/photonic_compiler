"""
Entry to program
"""

import json

import dijkstra as dijk

import stacked_graph as sg
import testing as test
import data_collection as dc


JSON_PATH = "/home/rjtomich/photonic_compiler/model_to_graph/gpt2_graph.json"
# JSON_PATH = '/home/rjtomich/photonic_compiler/model_to_graph/bert-base-uncased_graph.json'
# JSON_PATH = '/home/rjtomich/photonic_compiler/Pytorch-LeNet/simple_LeNet_graph.json'
# JSON_PATH = '/home/rjtomich/photonic_compiler/Pytorch-LeNet/simple_LeNet_graph_NoFusion.json'
with open(JSON_PATH, encoding="utf-8") as json_file:
    raw_json = json.load(json_file)  # returns json file as dict
    # print("... Json loaded ...")

# OPTIMIZATION_VARIABLE = "time"
OPTIMIZATION_VARIABLE = "energy"

graph = sg.StackGraph(raw_json=raw_json, optimization_variable=OPTIMIZATION_VARIABLE)
stacked_subgraphs = list(dijk.graph_partition(graph))
flat_subgraphs = dijk.select_nodes(
    stacked_subgraphs, optimization_variable=OPTIMIZATION_VARIABLE
)
expanded_flat_subgraphs = dijk.expand_nodes(flat_subgraphs)
scheduled_flat_graph, end_time, break_points = dijk.schdeule_nodes(
    graph, expanded_flat_subgraphs
)

for node in scheduled_flat_graph.node_list:
    if node.algorithm == 'get_dram':
        # print (node.output_shapes)
        pass

schedule_df = scheduled_flat_graph.create_schedule_data(write=True)
empty = test.schedule_validate(schedule_df)

dram, delta_dram, sram, delta_sram = dc.get_memory_profile(scheduled_flat_graph)
energy_data, delta_energy, total_energy = dc.get_energy_profile(scheduled_flat_graph)

print()
print("---------- INFO ----------")
print(f'{OPTIMIZATION_VARIABLE=}')
dc.get_photonic(flat_subgraphs)
print(
    dc.get_all_algorithms(flat_subgraphs).symmetric_difference(
        dc.get_all_algorithms(scheduled_flat_graph)
    )
)

print(f'Makespan: {end_time} s')
print(f"Number of Nodes: {len(scheduled_flat_graph.node_list)}")
print(f'Net DRAM: {dram[-1][1]} bits')
print(f'Net SRAM: {sram[-1][1]} bits')
print(f'Total Energy Consumption: {total_energy} pico-joules')
