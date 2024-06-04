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

# infrence
# prompt = "my favorite music is"
# input_ids = gpt2_tokenizer(prompt, return_tensors="pt").input_ids
# gen_tokens = gpt2.generate(input_ids, do_sample=True, temperature=0.9, max_length=100)
# gen_text = gpt2_tokenizer.batch_decode(gen_tokens)[0]
# print(gen_text)

# Load ONNX
prompt = "my favorite music is"
input_ids =  gpt2_tokenizer(prompt, return_tensors="pt").input_ids

gpt2 = gpt2.eval() # change to eval mode


if not os.path.isfile('./gpt2.onnx'):
    print("File does not exist")

    # convert to ONNX
    input_names = ["input_ids"]
    output_names = ["output"]

    torch.onnx.export(
        gpt2, # model to be exported
        (input_ids,), # model inputs
        "gpt2.onnx", # where to save the model
        input_names=input_names, # input names
        output_names=output_names, # output names
        dynamic_axes={"input_ids": {0: "batch"}}, # dynamic axis for variable batch size
        opset_version=11 # ONNX version to export the model
    )
    print('EXPORTED TO ONNX')
else:
    print("File exists")

    shape_list = ('input_ids', input_ids.shape) # (name, (1, num_input_ids))
    shape_dict = {'input_ids': input_ids.shape}  # Adjust based on your model's input shape

    onnx_model = onnx.load("gpt2.onnx")

    print(type(onnx_model))
    onnx.checker.check_model(onnx_model)
    print(type(onnx_model.graph))

    mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)

    target = tvm.target.Target("llvm", host="llvm")
    dev = tvm.cpu(0)
    with tvm.transform.PassContext(opt_level=1):
        lib = relay.build(mod, target=target, params=params) #module


    file_path = "GPT2_model.so"
    lib.export_library(file_path)

    # Save the graph JSON to a file
    graph_json_path = "GPT2_grap.json"
    with open(graph_json_path, "w") as f:
        f.write(lib.get_graph_json())

    # Save the parameters to a file
    param_dict = lib.get_params()  # No need to convert to dictionary
    param_bytes_path = "GPT2_params.params"
    tvm.relay.save_param_dict(param_dict)
