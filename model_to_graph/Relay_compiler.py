"""
compile a generic model in any format to relay IR
"""
import tvm
from tvm import relay
from tvm.contrib import graph_runtime
import onnx
import json
import numpy as np
import os
import io


import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def transformer_torch_to_onnx(model_name, prompt, save = False):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    model = AutoModelForCausalLM.from_pretrained(model_name, torchscript=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    model_onnx = model.eval()  # Change to eval mode

    onnx_model_io = io.BytesIO()

    # Check if the ONNX file already exists
    onnx_file_path = f'{model_name}.onnx'
    print(onnx_file_path)
    if os.path.exists(onnx_file_path):
        print('already a model')
        model_onnx = onnx.load(onnx_file_path)
        onnx.save(model_onnx, onnx_model_io)
    else:
        print('making new model')
        # Convert to ONNX
        input_names = ["input_ids"]
        output_names = ["output"]

        torch.onnx.export(
            model_onnx,
            (input_ids,),
            onnx_model_io,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes={"input_ids": {0: "batch"}},
            opset_version=16
        )
        if save:
            torch.onnx.export(
                model_onnx,
                (input_ids,),
                onnx_file_path,
                input_names=input_names,
                output_names=output_names,
                dynamic_axes={"input_ids": {0: "batch"}},
                opset_version=16
            )


    onnx_model_io.seek(0)
    model_onnx_bytes = onnx_model_io.getvalue()

    model_onnx = onnx.load_model_from_string(model_onnx_bytes)
    return model_onnx, input_ids

def onnx_to_relay(model_onnx, input_ids, model_name = 'model', opt_level = 0, config = {}):

    shape_list = ('input_ids', input_ids.shape) # (name, (1, num_input_ids))
    shape_dict = {'input_ids': input_ids.shape}  # Adjust based on your model's input shape

    onnx.checker.check_model(model_onnx)

    mod, params = relay.frontend.from_onnx(model_onnx, shape_dict)
    # <class 'tvm.ir.module.IRModule'>
    # <class 'dict'>

    # Extract and save Relay function source code
    relay_source_path = f"{model_name}_relay_source.txt"
    with open(relay_source_path, "w") as f:
        f.write(mod.astext(show_meta_data=False))


    target = tvm.target.Target("llvm", host="llvm")
    dev = tvm.cpu(0)
    with tvm.transform.PassContext(opt_level=opt_level, config=config):
        lib = relay.build(mod, target=target, params=params)

    # Save the graph JSON to a file
    graph_json_path = f"{model_name}_graph.json"
    with open(graph_json_path, "w") as f:
        f.write(lib.get_graph_json())

    # Create the function library
    lib.export_library(f"{model_name}_lib.so")

    # Creat paramater library
    param_dict = lib.get_params()
    param_bytes_path = f"{model_name}_params.params"
    with open(param_bytes_path, "wb") as f:
        # f.write(relay.save_param_dict(param_dict).get_bytearray())
        f.write(relay.save_param_dict(param_dict))



def tvm_validation(model_name):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    model = AutoModelForCausalLM.from_pretrained(model_name, torchscript=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    model = model.eval()  # Change to eval mode

    # Real
    print("-----Transformer----:")
    gen_tokens = model.generate(input_ids, do_sample=False, temperature=1, max_length=7)
    print(gen_tokens)
    gen_text = tokenizer.batch_decode(gen_tokens)[0]
    print(gen_text)


    # TVM
    print("-----TVM----:")
    graph_json_path = f"{model_name}_graph.json"
    lib_so_path = f"{model_name}_lib.so"
    param_bytes_path = f"{model_name}_params.params"

    loaded_json = open(graph_json_path).read()
    loaded_lib = tvm.runtime.load_module(lib_so_path)
    loaded_params = bytearray(open(param_bytes_path, "rb").read())

    module = graph_runtime.create(loaded_json, loaded_lib, tvm.cpu())
    module.load_params(loaded_params)
    module.set_input('input_ids', input_ids)
    module.run()

    print('****MY OUTPUT******')

    output = module.get_output(0)
    np_output = output.asnumpy()
    next_tok = np.argmax(np_output[0][-1])
    gen_tokens = np.append(input_ids, next_tok)
    print(gen_tokens)
    gen_text = tokenizer.batch_decode(gen_tokens)
    print(gen_text)


prompt = "my favorite music is"
model_name = "gpt2"

model_onnx, input_ids = transformer_torch_to_onnx(model_name, prompt, save = True)

onnx_to_relay(model_onnx,input_ids, model_name = model_name, opt_level = 3, config = {"relay.FuseOps.max_depth": 5})

# tvm_validation(model_name)

'''
# modles
"gpt2"
"bert-base-uncased"
"google-bert/bert-base-uncased"
https://huggingface.co/docs/transformers/en/model_doc/auto
'''

'''
# module methods
https://tvm.apache.org/docs/reference/api/python/graph_executor.html
print('****MY OUTPUT******')
print(module.benchmark(tvm.cpu()))
print(module.benchmark(tvm.cpu(), end_to_end=True))
benchmark'
'debug_get_output'
'get_input'
'get_input_index'
'get_input_info'
'get_num_inputs'
'get_num_outputs'
'get_output'
    print(module.get_output(0).shape) # of whole model
'load_params'
'run'
'set_input'
'share_params']
'''
