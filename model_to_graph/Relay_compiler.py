"""
compile a generic model in any format to relay IR
"""
import tvm
from tvm import relay
import numpy as np
import onnx
import os
import io
from tvm.contrib import graph_runtime


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
    gen_tokens = model.generate(input_ids, do_sample=False, temperature=1, max_length=6)
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

    output = module.get_output(0)
    np_output = output.asnumpy()
    next_tok = np.argmax(np_output[0][-1])
    gen_tokens = np.append(input_ids, next_tok)
    print(gen_tokens)
    gen_text = tokenizer.batch_decode(gen_tokens)
    print(gen_text)


prompt = "my favorite music is"
model_name = "gpt2"

# model_onnx, input_ids = transformer_torch_to_onnx(model_name, model_name, prompt, save = True)

# onnx_to_relay(model_onnx,input_ids, model_name = model_name, opt_level = 3, config = {"relay.FuseOps.max_depth": 5})

tvm_validation(model_name)

'''
"gpt2"
"bert-base-uncased"
"google-bert/bert-base-uncased"
https://huggingface.co/docs/transformers/en/model_doc/auto
'''


# #### DIRECTLY FROM PYTORCH
# mmodel_name = "gpt2"
# gpt2_model = AutoModelForCausalLM.from_pretrained(model_name)

# # Convert PyTorch model to TorchScript
# input_shape = (1, 10)  # Adjust according to your input shape
# input_name = np.array('test')
# input_example = torch.zeros(input_shape).long()  # Create an example input tensor
# print(input_example)
# print(input_name)
# print(input_example.shape)
# print(input_name.shape)
# # traced_model = torch.jit.trace(gpt2_model, input_example)

# # Convert TorchScript model to Relay IR
# mod, params = relay.frontend.from_pytorch(traced_model, [(input_name, input_shape)])

# # Apply optimizations
# mod = relay.transform.Sequential([relay.transform.RemoveUnusedFunctions(), relay.transform.FoldConstant()]) (mod)

# # Compile Relay IR
# target = 'llvm'  # Change the target according to your preference
# target_host = 'llvm'
# with tvm.transform.PassContext(opt_level=3):
#     graph, lib, params = relay.build(mod, target=target, target_host=target_host, params=params)

# # Write graph, lib, params to files
# graph_path = "compiled_graph.json"
# lib_path = "compiled_lib.tar"
# params_path = "compiled_params.params"

# # Save graph
# with open(graph_path, "w") as f:
#     f.write(graph)

# # Save lib
# lib.export_library(lib_path)

# # Save params
# with open(params_path, "wb") as f:
#     f.write(relay.save_param_dict(params))

# # ERROR: Dictionary inputs to traced functions must have consistent type. Found Tensor and Tuple[Tuple[Tensor, Tensor],
