"""
Entry to program
run using conda (schedule)
"""

import json
import time
from tqdm import tqdm

import hardware as hw
import dijkstra as dijk
import stacked_graph as sg
import testing as test
import data_collection as dc


def forward(relay_path, optimization, available_hardware, profiles = True, get_step_times=True, config = None):

    # progress bar
    def mark_time():
        if get_step_times is True: times.append(time.time())
        progress_bar.update(1)

    times = []

    ops = [
        'open',
        'oringal_graph',
        'stacked_subgraphs',
        'flat_subgraphs',
        'expanded_flat_subgraphs',
        'schdeule_nodes',
        'create_schedule_data',
        'schedule_validate',
        'get_memory_profile',
        'get_energy_profile',
    ]

    if not profiles:
        ops.remove('get_memory_profile')
        ops.remove('get_energy_profile')


    progress_bar = tqdm(total=len(ops), desc="Progress", unit="operation")


    # Calculation
    mark_time()
    with open(relay_path, encoding="utf-8") as json_file:
        raw_json = json.load(json_file)  # returns json file as dict
        # print("... Json loaded ...")
    mark_time()

    WEIGHT_VARIABLE = optimization

    graph = sg.StackGraph(raw_json=raw_json, weight_variable=WEIGHT_VARIABLE)
    dense_time, add_time = dc.get_addmm(graph)

    mark_time()
    stacked_subgraphs = list(dijk.graph_partition(graph))
    mark_time()
    flat_subgraphs = dijk.select_nodes(
        stacked_subgraphs, weight_variable=WEIGHT_VARIABLE, config=config
    )
    mark_time()
    expanded_flat_subgraphs = dijk.expand_nodes(flat_subgraphs)
    mark_time()
    scheduled_flat_graph, end_time, break_points = dijk.schdeule_nodes(
        graph, expanded_flat_subgraphs, available_hardware
    )
    mark_time()

    # print(scheduled_flat_graph.get_node_obj(13))

    schedule_df = scheduled_flat_graph.create_schedule_data(write=True)

    # schedule_df['difference'] = schedule_df['end'] - schedule_df['start']
    # print(schedule_df.loc[schedule_df['difference'].idxmax()])

    mark_time()
    stagnent_time = test.schedule_validate(schedule_df)
    mark_time()

    print("---------- INFO ----------")
    print(f"{WEIGHT_VARIABLE=}")
    dc.get_photonic(flat_subgraphs)
    print(
        dc.get_all_algorithms(flat_subgraphs).symmetric_difference(
            dc.get_all_algorithms(scheduled_flat_graph)
        )
    )

    print(f"Makespan: {end_time} s")
    print(f"Number of Nodes: {len(scheduled_flat_graph.node_list)}")

    if profiles:
        dram, delta_dram, sram, delta_sram = dc.get_memory_profile(scheduled_flat_graph)
        print(f"Net DRAM: {dram[-1][1]} bits")
        print(f"Net SRAM: {sram[-1][1]} bits")
        mark_time()

        energy_data, delta_energy, total_energy = dc.get_energy_profile(
            scheduled_flat_graph
        )
        print(f"Total Energy Consumption: {total_energy} pico-joules")

        mark_time()
        print(f"time_distrabution {dc.get_time_profile(scheduled_flat_graph)} compute seconds ")
        mark_time()

    if get_step_times:
        time_taken = {}
        print(len(times))
        print(len(ops))
        assert len(times)-1 == len(ops)
        for i in range(len(ops)):
            time_taken[ops[i]] = times[i+1] - times[i]
        print(time_taken)

    print("---------- ---- ----------")

    dense_time, add_time = dc.get_addmm(scheduled_flat_graph)



config = None
# config = 'always_cpu'
# config = 'always_phu'

optimization = 'time'
# optimization = 'energy'
# optimizations = ["time", "energy"]


# relay_path = "/home/rjtomich/photonic_compiler/model_to_graph/gpt2_graph.json"
relay_path = '/home/rjtomich/photonic_compiler/model_to_graph/bert-base-uncased_graph.json'
# relay_path = '/home/rjtomich/photonic_compiler/Pytorch-LeNet/simple_LeNet_graph.json'
# relay_path = '/home/rjtomich/photonic_compiler/Pytorch-LeNet/simple_LeNet_graph_NoFusion.json'

# for i in optimizations:
#     forward(relay_path, i)

# available_hardware = hw.initilize_hardware([hw.CPU(10**8, 1), hw.PHU(10**10, 64, 20)])
# available_hardware = hw.initilize_hardware([hw.CPU(10**8, 1)])
# available_hardware = hw.initilize_hardware([hw.PHU(10**10, 64, 20)])

available_hardware = hw.initilize_hardware([hw.CPU(3279973517, 1)])
forward(relay_path, 'time', available_hardware, profiles = True, get_step_times=False, config=config)
