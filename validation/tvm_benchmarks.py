import tvm
from tvm import relay
from tvm.contrib import graph_executor
import onnx
import numpy as np
import os
import io
from tvm.runtime import profiler_vm



import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def generate_token(model_name, prompt):
    model = AutoModelForCausalLM.from_pretrained(model_name, torchscript=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    model = model.eval()  # Change to eval mode

    gen_tokens = model.generate(input_ids, do_sample=False, max_new_tokens=1)

    return int(gen_tokens[0][-1])

def generate_token_TVM(modle_name, prompt, benchmark=False, function_benchmark=False):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids

    graph_json_path = f"../model_to_graph/{model_name}_graph.json"
    lib_so_path = f"../model_to_graph/{model_name}_lib.so"
    param_bytes_path = f"../model_to_graph/{model_name}_params.params"

    loaded_json = open(graph_json_path).read()
    #https://tvm.apache.org/docs/reference/api/python/runtime.html#tvm.runtime.Module
    loaded_lib = tvm.runtime.load_module(lib_so_path) # tvm.runtime.module.Module
    loaded_params = bytearray(open(param_bytes_path, "rb").read())


    module = graph_executor.create(loaded_json, loaded_lib, tvm.cpu()) # tvm.contrib.graph_executor.GraphModule
    module.load_params(loaded_params)

    module = graph_executor.create(loaded_json, loaded_lib, tvm.cpu()) # tvm.contrib.graph_executor.GraphModule
    module.load_params(loaded_params)
    module.set_input("input_ids", input_ids)
    module.run()

    if benchmark:
        benchmark_results = module.benchmark(tvm.cpu(), end_to_end=True)
        print(benchmark_results.results)
        print(benchmark_results)


        # Profiling the execution
        profile_results = module.profile(tvm.cpu())
        print("Profiling Results:")
        for name, time_cost in profile_results.items():
            print(f"{name}: {time_cost} seconds")


    if function_benchmark:
        func_name = 'tvmgen_default_fused_nn_dense_2'

        time_eval_func = loaded_lib.time_evaluator(func_name, tvm.cpu())


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
tvm_token = generate_token_TVM(model_name, prompt, benchmark=True, function_benchmark=False)
assert real_token == tvm_token
