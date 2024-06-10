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
    with tvm.transform.PassContext(opt_level=1, config={"relay.FuseOps.max_depth": 0}):
        lib = relay.build(mod, target=target, params=params)


    # Save the graph JSON to a file
    graph_json_path = "GPT2_graph.json"
    with open(graph_json_path, "w") as f:
        f.write(lib.get_graph_json())

    # Create the function library
    lib.export_library("GPT2_lib.so")

    # Creat paramater library
    param_dict = lib.get_params()
    param_bytes_path = "GPT2_params.params"
    with open(param_bytes_path, "wb") as f:
        f.write(relay.save_param_dict(param_dict).get_bytearray())


    '''
    opt_level=0 no optimizations
    opt_level=1 basic (constant folding)
    opt_level=2 advanced (operator fusion)
    opt_level=3 target-specific

    disabled_pass=
    required_pass=
    trace=my_trace_callback
    instrument

    config={"relay.FuseOps.max_depth": 5}
    config={"relay.backend.use_auto_scheduler": True}):
    config={"relay.backend.parallel_compiler": True}):
    '''

    '''
    # Eliminating fusion for spasific functions
    def disable_fusion_for_function(func):
    """Disable fusion for a specific function in the Relay module."""
    # Traverse the function body and mark each operation as not eligible for fusion
    def disable_fusion(node):
        if isinstance(node, relay.Function):
            return relay.Function(node.params, relay.expr_functor(node.body, disable_fusion), node.ret_type)
        elif isinstance(node, relay.Call):
            # Mark the call as not eligible for fusion
            return relay.Call(disable_fusion(node.op), node.args, node.attrs, node.type_args)
        else:
            return node

    return relay.Function(func.params, disable_fusion(func.body), func.ret_type)
    functions_to_disable_fusion = ["func1", "func2"]

    new_funcs = []
    for func in mod.functions:
        if func.name_hint in functions_to_disable_fusion:
            new_funcs.append(disable_fusion_for_function(func))
        else:
            new_funcs.append(func)

    # Create a new Relay module with modified fusion behavior
    new_mod = relay.Module(new_funcs)

    '''
