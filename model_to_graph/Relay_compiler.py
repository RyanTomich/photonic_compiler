"""
compile a generic model in any format to relay IR
"""
import tvm
from tvm import relay
import numpy as np
import onnx
import os
import io

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def transformer_torch_to_onnx(model_name, tokenizer_torch, prompt, save = False):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    model = AutoModelForCausalLM.from_pretrained(model_name, torchscript=True)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_torch)

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


prompt = "my favorite music is"
model = "gpt2"
model_onnx, input_ids = transformer_torch_to_onnx(model, model, prompt, save = True)

print(type(model_onnx))
print(type(input_ids))

# <class 'onnx.onnx_ml_pb2.ModelProto'>
# <class 'torch.Tensor'>

onnx_to_relay(model_onnx,input_ids, model_name = model, opt_level = 1, config = {"relay.FuseOps.max_depth": 0})

'''
"gpt2"
"bert-base-uncased"
"google-bert/bert-base-uncased"
https://huggingface.co/docs/transformers/en/model_doc/auto
'''
