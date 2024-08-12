import os
import sys
import torch
import csv


import Relay_compiler as rc
import tvm_benchmarks as tvm_bench

parent_path_two_up = os.path.dirname(os.path.dirname(os.getcwd()))
target_dir = f'{parent_path_two_up}/model_to_graph'
sys.path.append(target_dir)

import data_collection_shapes as shape_data
import main
import hardware as hw


def collect_func_times(model_name, prompt, write = False):
    model_onnx, input_ids = rc.transformer_torch_to_onnx(model_name, prompt, save=False)
    rc.onnx_to_relay(model_onnx, input_ids, write=True, model_name=model_name, opt_level=0)

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    os.environ["TVM_NUM_THREADS"] = "1"
    torch.set_num_threads(1)


    tvm_func_all = tvm_bench.generate_token_TVM(
        model_name,
        prompt,
        tvm_lib = None,
        benchmark=True,
        function_benchmark=False,
        operator_constants=False,
    )

    # simulation
    relay_path = graph_json_path = f"{model_name}_graph.json"
    optimization = 'time'

    CPU_AVERAGE_CLOCK = 3.208 * 10**9  # 60**9, 6
    PHU_MIN_CLOCK = 10 * 10**9  # 100**9, 10 Ghz

    hardware = []
    hw.Hardware._hardware_reset()
    # hardware.append(hw.CPU(CPU_MAX_CLOCK, 1))
    hardware.append(hw.CPU(CPU_AVERAGE_CLOCK, 1))
    # hardware.append(hw.PHU(PHU_MIN_CLOCK, 1, 20))

    # available_hardware = hw.initilize_hardware([hw.CPU(14792899408, 1)])
    available_hardware = hw.initilize_hardware(hardware)

    main.forward(
        relay_path,
        optimization,
        available_hardware,
        profiles=True,
        get_step_times=False,
        data_collection=True,
        config=None,
    )




    shapes, types = shape_data.print_shapes(f"{model_name}_graph.json")

    if write:
        with open('raw_time_data.csv', 'a', newline='') as file:
            writer = csv.writer(file)
            for tvm_func, time in tvm_func_all.items():
                compiler_name = types[tvm_func]
                shape = shapes[tvm_func]

                writer.writerow([tvm_func, compiler_name, time, shape])

    # file removal
    written_files = [
        f"{model_name}_relay_source.txt",
        f"{model_name}_graph.json",
        f"{model_name}_lib.so",
        f"{model_name}_lib.tar",
        f"{model_name}_params.params",
    ]
    for path in written_files:
        if os.path.isfile(path):
            # Delete the file
            os.remove(path)



model_name = "gpt2"
prompts = [
    # "My ",
    # "My favorite ",
    # "My favorite music ",
    "My favorite music genre ",
    "My favorite music genre has ",
    "My favorite music genre has always been ",
    # "My favorite music genre has always been jazz, especially ",
    # "My favorite music genre has always been jazz, especially the smooth ",
    # "My favorite music genre has always been jazz, especially the smooth sounds of ",
    # "My favorite music genre has always been jazz, especially the smooth sounds of classic artists ",
    # "My favorite music genre has always been jazz, especially the smooth sounds of classic artists like Miles ",
    # "My favorite music genre has always been jazz, especially the smooth sounds of classic artists like Miles Davis and Coltrane",
]

# with open('raw_time_data.csv', 'w', newline='') as file:
#     writer = csv.writer(file)
#     titles = ['tvm_func', 'compiler_name', 'microseconds', 'shape']
#     writer.writerow(titles)

for prompt in prompts:
    print(f'--------------------------{prompt}')

    collect_func_times(model_name, prompt, write = False)
