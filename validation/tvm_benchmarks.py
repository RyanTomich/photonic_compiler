import os
import io
import onnx
import numpy as np

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

def generate_token_TVM(modle_name, prompt, benchmark=False, function_benchmark=False, profile=False):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids


    graph_json_path = f"../model_to_graph/{model_name}_graph.json"
    lib_so_path = f"../model_to_graph/{model_name}_lib.so"
    param_bytes_path = f"../model_to_graph/{model_name}_params.params"

    loaded_json = open(graph_json_path).read() # str
    loaded_lib = tvm.runtime.load_module(lib_so_path) # tvm.runtime.module.Module
    loaded_params = bytearray(open(param_bytes_path, "rb").read()) # bytearray

    module = graph_executor.create(loaded_json, loaded_lib, tvm.cpu()) # tvm.contrib.graph_executor.GraphModule
    module.load_params(loaded_params)
    module.set_input("input_ids", input_ids)
    module.run()

    if benchmark:
        benchmark_results = module.benchmark(tvm.cpu(), end_to_end=True)
        print(benchmark_results.results)
        print(benchmark_results)

        print (f'threads_used : {tvm.runtime.num_threads()}')


    if function_benchmark:
        lib = tvm.runtime.load_module(lib_so_path) # tvm.runtime.module.Module

        dev = tvm.cpu()

        m = GraphModuleDebug(
            lib["debug_create"]("default", dev),
            [dev],
            loaded_json,
            dump_root="/tmp/tvmdbg",
        )
        m.set_input("input_ids", input_ids)
        m.run()
        tvm_out = m.get_output(0).numpy()


    output = module.get_output(0)
    np_output = output.asnumpy()
    next_tok = np.argmax(np_output[0][-1])
    return next_tok



model_name = "gpt2"
prompt = "my favorite music is"

os.environ["TOKENIZERS_PARALLELISM"] = "false"

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

torch.set_num_threads(1)

real_token = generate_token(model_name, prompt)
tvm_token = generate_token_TVM(model_name, prompt, benchmark=False, function_benchmark=True)
assert real_token == tvm_token
