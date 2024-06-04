# https://huggingface.co/docs/transformers/en/model_doc/gpt2
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torchinfo import summary
import torch
import tvm
from tvm import te
from tvm import relay
import onnx
import os

# load model
gpt2 = GPT2LMHeadModel.from_pretrained('gpt2', torchscript=True)# <class 'transformers.models.gpt2.modeling_gpt2.GPT2LMHeadModel'>
gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2') # <class 'transformers.models.gpt2.tokenization_gpt2.GPT2Tokenizer'>

# create input
prompt = "my favorite music is"
input_ids =  gpt2_tokenizer(prompt, return_tensors="pt").input_ids
gpt2 = gpt2.eval() # change to eval mode

# Load ONNX
# Check if file already exists
if not os.path.isfile('./gpt2.onnx'):
    print("File does not exist. Creating ONNX model")

    # convert to ONNX
    input_names = ["input_ids"]
    output_names = ["output"]

    torch.onnx.export( # saves a file
        gpt2,
        (input_ids,),
        "gpt2.onnx",
        input_names=input_names,
        output_names=output_names,
        dynamic_axes={"input_ids": {0: "batch"}}, # dynamic axis for variable batch size
        opset_version=11 # ONNX version to export the model
    )
    print('EXPORTED TO ONNX')
else:
    print("ONNX file already exists. Loading file...")

    onnx_model = onnx.load("gpt2.onnx")

    shape_list = ('input_ids', input_ids.shape) # (name, (1, num_input_ids))
    shape_dict = {'input_ids': input_ids.shape}  # Adjust based on your model's input shape
    onnx.checker.check_model(onnx_model)

    mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)

    target = tvm.target.Target("llvm", host="llvm")
    dev = tvm.cpu(0)
    with tvm.transform.PassContext(opt_level=1):
        lib = relay.build(mod, target=target, params=params)


    file_path = "GPT2_model.so"
    lib.export_library(file_path)

    # Save the graph JSON to a file
    graph_json_path = "GPT2_grap.json"
    with open(graph_json_path, "w") as f:
        f.write(lib.get_graph_json())

    # Save the parameters to a file
    param_dict = lib.get_params()
    param_bytes_path = "GPT2_params.params"
    tvm.relay.save_param_dict(param_dict)
