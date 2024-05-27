'''
https://tvm.apache.org/docs/how_to/compile_models/from_tensorflow.html?highlight=tensorflow
'''

import torch.utils._pytree as _torch_pytree
import transformers
from transformers import GPT2Tokenizer, TFGPT2LMHeadModel
import tensorflow as tf
import os
import numpy as np
from transformers import GPT2Tokenizer

import tvm
from tvm import te
from tvm import relay

WRITE_MODEL = False

#### Download pretrained GPT2 Model ####

if WRITE_MODEL:
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = TFGPT2LMHeadModel.from_pretrained(
        "gpt2", from_pt=True, pad_token_id=tokenizer.eos_token_id
    )
    # model.save_pretrained("./tfgpt2model", saved_model=True)

    model_path = "./tfgpt2model"
    tf.saved_model.save(model, model_path)


#### Create Relay IR ####

model_path = "./tfgpt2model"
model = tf.saved_model.load(model_path)

input_name = "input"
output_node = "output_node"
input_shape = [1, 256]  # batch_size, sequence_length

concrete_func = model.signatures[tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
concrete_func = tf.function(concrete_func)

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
sentence = "This is an example sentence."
tokens = tokenizer.encode(sentence, add_special_tokens=True)

# TODO

output = concrete_func(input_ids=padded_tokens, attention_mask=attention_mask)
concrete_func_graph = concrete_func.get_concrete_function(input_ids, attention_mask).graph
frozen_model_graph = tf.graph_util.convert_variables_to_constants(
    tf.compat.v1.Session(graph=tf.Graph()),  # Use tf.compat.v1.Session() for TensorFlow 2.x
    concrete_func_graph.as_graph_def(),
    [output_node])


shape_list = [(input_name, input_shape)]
with tf.Graph().as_default() as tf_graph:
    tf.import_graph_def(frozen_model_graph, name="")
    relay_graph, params = relay.frontend.from_tensorflow(tf_graph, shape=shape_list)


target = tvm.target.Target("llvm", host="llvm")
dev = tvm.cpu(0)
with tvm.transform.PassContext(opt_level=1):
    lib = relay.build(relay_graph, target=target, params=params)

file_path = "GPT2.so"
lib.export_library(file_path)

graph_json_path = "GPT2_graph.json"
with open(graph_json_path, "w") as f:
    f.write(lib.get_graph_json())

param_dict = lib.get_params()  # No need to convert to dictionary
param_bytes_path = "GPT2_params.params"
tvm.relay.save_param_dict(param_dict)
