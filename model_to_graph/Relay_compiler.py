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
    ''' Takes transformer models to ONNX
    model_name (srt)
    prompt(str)
    save(bool) samve model to files
    '''
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # get model from transformer library
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
    '''
    model_onnx - model in onnx format
    input_ids - tensor with input_ids. Created by the tokenizer
    model_name(srt)
    opt_level(int 0-3)- how much to optimize
    config(dict) - config files
    '''
    shape_dict = {'input_ids': input_ids.shape}  # Adjust based on your model's input shape

    onnx.checker.check_model(model_onnx)

    mod, params = relay.frontend.from_onnx(model_onnx, shape_dict) # <class 'tvm.ir.module.IRModule'>

    # Relay module optimization

    # TODO Figure out how to allow fusion...
    # def my_fuse_pattern(op):
    #     print(op.name)
    #     if isinstance(op, relay.op.op.Op) and op.name in ["dense", "batch_matmul"]:
    #         return False
    #     return True

    # def dense_or_batch_matmul(expr):
    #     # print(dir(op)) <class 'tvm.relay.function.Function'>
    #     print('here!')
    #     func = expr.astext()
    #     for i in  ["dense", "batch_matmul", 'add']:
    #         if i in func:
    #             return False
    #     return True

    # patterns = [
    #     ("fuse_all_except_dense_batch_matmul", relay.dataflow_pattern.wildcard(), dense_or_batch_matmul)
    # ]
    # mod = relay.transform.MergeComposite(patterns)(mod)


    # Apply the custom fusion pattern
    # mod = relay.transform.FuseOps(-1)(mod)

    # Extract and save Relay function source code
    relay_source_path = f"{model_name}_relay_source.txt"
    with open(relay_source_path, "w") as f:
        f.write(mod.astext(show_meta_data=False)) # annotate = func


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
    '''
    model_name(srt)
    must be local files: mod_graph.json, mod_lib.so, mod_params.params
    '''
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

    print(module.benchmark(tvm.cpu()))

    output = module.get_output(0)
    np_output = output.asnumpy()
    next_tok = np.argmax(np_output[0][-1])
    gen_tokens = np.append(input_ids, next_tok)
    print(gen_tokens)
    gen_text = tokenizer.batch_decode(gen_tokens)
    print(gen_text)


prompt = "my favorite music is"
model_name = "gpt2"

# model_onnx, input_ids = transformer_torch_to_onnx(model_name, prompt, save = True)

config = {"relay.backend.use_auto_scheduler": False,
          "relay.FuseOps.max_depth": 0,
          "tir.disable_vectorize": False}
# onnx_to_relay(model_onnx,input_ids, model_name = model_name, opt_level = 0, config = config)

tvm_validation(model_name)


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
'benchmark'
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

'''
<class 'tvm.ir.module.IRModule'>
'astext'
'attrs'
'from_expr'
'functions'
'get_attr'
'get_constructor'
'get_global_type_var'
'get_global_type_vars'
'get_global_var'
'get_global_vars'
'get_type'
'global_type_var_map_'
'global_var_map_'
'handle'
'import_from_std'
'same_as'
'script'
'show'
'source_map'
'type_definitions'
'update'
'update_func'
'with_attr'
'''

'''
Configurate options:

relay.ext.ethos-u.options
relay.ext.ethos-n.options
relay.fallback_device_type
relay.backend.use_meta_schedule
relay.backend.use_auto_scheduler
relay.backend.tir_converter
relay.ToMixedPrecision.keep_orig_output_dtype
relay.FuseOps.max_depth
relay.collage.byoc_max_depth
tir.usmp.use_workspace_io
tir.usmp.enable
tir.UnrollLoop
tir.RemoveNoOp
tir.merge_async_commit_queue_scope
tir.add_lower_pass
relay.ext.cmsisnn.options
tir.disable_assert
tir.LoopPartition
tir.use_async_copy
tir.disable_vectorize
tir.debug_keep_trivial_loop
tir.instrument_bound_checkers
relay.remove_standalone_reshapes.enable
tir.usmp.custom_algorithm
tir.disable_storage_rewrite
tir.vtcm_capacity
tir.ReduceBranchingThroughOvercompute
relay.backend.use_meta_schedule_dispatch
tir.Simplify
tir.disable_cse_tir
tir.is_entry_func
relay.FuseOps.link_params
relay.collage.tvm_max_depth
tir.enable_equiv_terms_in_cse_tir
tir.dma_bypass_cache
tir.noalias
tir.lwp_min_height
tir.instr_siblings
tir.instrument_lwp
testing.immutable_module
tir.contrib.ethos-u.copy_compute_reordering_max_copy_movements
tir.contrib.ethos-u.copy_compute_reordering_reorder_by_cycles
tir.lwp_disable_func_prof
tir.detect_global_barrier
tir.HoistExpression
tir.HoistIfThenElse
tir.InjectDoubleBuffer
tir.usmp.algorithm
tir.lwp_max_depth
tir.reset_start_id
'''