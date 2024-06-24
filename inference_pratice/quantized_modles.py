import os
import sys
import onnx
import onnxruntime as ort
from onnxruntime import quantization
import numpy as np

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# filename = '../model_to_graph/Relay_compiler.py'
# module_name = 'other_script'

# # Create a module spec
# spec = importlib.util.spec_from_file_location(module_name, filename)

# # Import the module
# relay_compiler = importlib.util.module_from_spec(spec)
# spec.loader.exec_module(relay_compiler)

prompt = "my favorite music is"
model_name = "gpt2"

# model_onnx, input_ids = relay_compiler.transformer_torch_to_onnx(model_name, prompt, save = False)

model_path = '../model_to_graph/gpt2.onnx'
onnx_model = onnx.load(model_path)
onnx.checker.check_model(onnx_model)

ort_provider = ['CPUExecutionProvider']
if torch.cuda.is_available():
    onnx_model.to('cuda')
    ort_provider = ['CUDAExecutionProvider']

ort_sess = ort.InferenceSession(model_path, providers=ort_provider)

tokenizer = AutoTokenizer.from_pretrained(model_name)
input_ids = tokenizer(prompt, return_tensors="pt").input_ids
input_ids_numpy = input_ids.cpu().numpy()

ort_inputs = {ort_sess.get_inputs()[0].name: input_ids_numpy}
ort_outs = ort_sess.run(None, ort_inputs)

logits = torch.tensor(ort_outs[0][0][-1]) # select the last logit
predicted_token_id = torch.argmax(logits, dim=-1) # greedly select the largest proability word
gen_text = tokenizer.batch_decode(np.array([[predicted_token_id]]), skip_special_tokens=True) #
print(gen_text)
