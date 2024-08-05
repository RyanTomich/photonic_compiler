'''
Run using conda (tvm)
'''

import os
import io
import onnx
import json
import numpy as np

import psutil

import tvm
from tvm import relay
from tvm.contrib import graph_executor

from tvm.contrib.debugger.debug_executor import GraphModuleDebug



import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def generate_token(model_name, prompt):
    model = AutoModelForCausalLM.from_pretrained(model_name, torchscript=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    model = model.eval()  # Change to eval mode

    gen_tokens = model.generate(input_ids, do_sample=False, max_new_tokens=1)

    return int(gen_tokens[0][-1])

def _find_opp(func_name):
    """
    func_name(srt) - whole tvm function name
    returns (srt) - last collection of letters, which is the name
    """
    name_parts = func_name.split("_")
    for part in reversed(name_parts):
        try:
            int(part)
            continue
        except ValueError:
            return part
    return None

def get_trace_data(func_names):
    """Gets average runtimes for a tvm function

    Args:
        func_names (dict): maping function name to list of node names
    """
    trace_path = "/home/rjtomich/photonic_compiler/validation/trace/_tvmdbg_device_CPU_0/_tvmdbg_execution_trace.json"
    with open(trace_path, 'r') as file:
        trace_data = json.load(file)

    events = trace_data.get('traceEvents', [])
    start_times = {}
    end_times = {}
    for event in events:
        name = event.get('name')
        timestamp = event.get('ts')
        phase = event.get('ph')
        process_id = event.get('pid')
        tread_id = event.get('tid')
        if phase == "B":
            start_times[name] = timestamp

        if phase == "E":
            end_times[name] = timestamp

    assert len(start_times) == len(end_times)
    func_duration = {}
    for name, start in start_times.items():
        func_duration[name] = end_times[name] - start

    func_duration_list = {}
    for opp, funcs in func_names.items():
        func_durations = [func_duration[func] for func in funcs]
        func_duration_list[opp] = func_durations

    file_path = 'output.txt'
    cpu_freq = psutil.cpu_freq()

    with open(file_path, 'a') as file:
        file.write(f'threads_used : {tvm.runtime.num_threads()}\n')
        file.write(f"CPU Frequency: {cpu_freq.current} MHz\n")

    for name, durations in func_duration_list.items():
        if name in [
            'tvmgen_default_fused_nn_batch_matmul',
            'tvmgen_default_fused_nn_batch_matmul_1',
            'tvmgen_default_fused_nn_dense',
            'tvmgen_default_fused_nn_dense_1',
            'tvmgen_default_fused_nn_dense_2',
            'tvmgen_default_fused_nn_dense_3',
            'tvmgen_default_fused_nn_dense_4',
            'tvmgen_default_fused_layout_transform_nn_contrib_dense_pack',
            'tvmgen_default_fused_layout_transform_nn_contrib_dense_pack_1',
            'tvmgen_default_fused_layout_transform_nn_contrib_dense_pack_2',
            'tvmgen_default_fused_layout_transform_nn_contrib_dense_pack_3',
        ]:
            formatted_string = f'{name:<40} {sum(durations)/len(durations):>10.2f}'
            print(formatted_string)
            with open(file_path, 'a') as file:
                file.write(formatted_string + '\n')



def generate_token_TVM(modle_name, prompt, benchmark=False, function_benchmark=False, profile=False):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids


    graph_json_path = f"../model_to_graph/{model_name}_graph.json"
    lib_so_path = f"../model_to_graph/{model_name}_lib.so"
    param_bytes_path = f"../model_to_graph/{model_name}_params.params"

    loaded_json = open(graph_json_path).read() # str
    loaded_lib = tvm.runtime.load_module(lib_so_path) # tvm.runtime.module.Module
    loaded_params = bytearray(open(param_bytes_path, "rb").read()) # bytearray

    with open(graph_json_path, 'r') as file:
        json_dict = json.load(file)

    func_names = {}
    for node in json_dict['nodes']:
        if 'attrs' in node:
            func_names.setdefault(node['attrs']['func_name'], []).append(node['name'])

    module = graph_executor.create(loaded_json, loaded_lib, tvm.cpu()) # tvm.contrib.graph_executor.GraphModule
    module.load_params(loaded_params)
    module.set_input("input_ids", input_ids)
    module.run()

    if benchmark:
        benchmark_results = module.benchmark(tvm.cpu(), end_to_end=True)
        print(benchmark_results.results)
        print(benchmark_results)

        cpu_freq = psutil.cpu_freq()
        print (f'threads_used : {tvm.runtime.num_threads()}')
        print(f"CPU Frequency: {cpu_freq.current} MHz")


    if function_benchmark:
        for i in range(10):
            lib = tvm.runtime.load_module(lib_so_path) # tvm.runtime.module.Module

            dev = tvm.cpu()

            # dump_root = "/tmp/tvmdbg"
            dump_root = "trace/"

            m = GraphModuleDebug(
                lib["debug_create"]("default", dev),
                [dev],
                loaded_json,
                dump_root=dump_root,
            )
            m.set_input("input_ids", input_ids)
            m.run()

            get_trace_data(func_names)

        #get resulting token
        tvm_out = m.get_output(0).numpy()
        next_tok = np.argmax(tvm_out[0][-1])
        print(next_tok)

        print (f'threads_used : {tvm.runtime.num_threads()}')
        cpu_freq = psutil.cpu_freq()
        print(f"CPU Frequency: {cpu_freq.current} MHz")

    output = module.get_output(0)
    np_output = output.asnumpy()
    next_tok = np.argmax(np_output[0][-1])
    return next_tok



model_name = "gpt2"
model_name = "bert-base-uncased"
prompt = "my favorite music is"

os.environ["TOKENIZERS_PARALLELISM"] = "false"

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

torch.set_num_threads(1)

real_token = generate_token(model_name, prompt)
tvm_token = generate_token_TVM(model_name, prompt, benchmark=False, function_benchmark=True)
assert real_token == tvm_token
