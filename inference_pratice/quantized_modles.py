# import os
# import sys
# import onnx
# import onnxruntime as ort
# from onnxruntime import quantization
# from onnxruntime.quantization import quantize_dynamic, QuantType
# import numpy as np

# import torch
# from transformers import AutoModelForCausalLM, AutoTokenizer

#region ### Creating new ONNX Model
# filename = '../model_to_graph/Relay_compiler.py'
# module_name = 'other_script'

# # Create a module spec
# spec = importlib.util.spec_from_file_location(module_name, filename)

# # Import the module
# relay_compiler = importlib.util.module_from_spec(spec)
# spec.loader.exec_module(relay_compiler)

# prompt = "my favorite music is"
# model_name = "gpt2"

# model_onnx, input_ids = relay_compiler.transformer_torch_to_onnx(model_name, prompt, save = False)
#endregion

#region ### Working with previous ONNX model
'''Quantzation with ONNX is skipping the matmul, which defeats the purposes for this project'''

# def run_model(model_name, model_path, prompt):
#     onnx_model = onnx.load(model_path)
#     onnx.checker.check_model(onnx_model)

#     ort_provider = ['CPUExecutionProvider']
#     if torch.cuda.is_available():
#         onnx_model.to('cuda')
#         ort_provider = ['CUDAExecutionProvider']

#     ort_sess = ort.InferenceSession(model_path, providers=ort_provider)

#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     input_ids = tokenizer(prompt, return_tensors="pt").input_ids
#     input_ids_numpy = input_ids.cpu().numpy()

#     ort_inputs = {ort_sess.get_inputs()[0].name: input_ids_numpy}
#     ort_outs = ort_sess.run(None, ort_inputs)

#     print(torch.argmax(torch.tensor(ort_outs[0]), dim=-1))



#     logits = torch.tensor(ort_outs[0][0][-1]) # select the last logit
#     predicted_token_id = torch.argmax(logits, dim=-1) # greedly select the largest proability word
#     gen_text = tokenizer.batch_decode(np.array([[predicted_token_id]]), skip_special_tokens=True) #
#     return gen_text


# prompt = "my favorite music is"
# model_name = "gpt2"
# model_path = '../model_to_graph/gpt2.onnx'
# print(run_model(model_name, model_path, prompt))

# # Dynamic Quantization
# quant_model_path = 'gpt2_prep.onnx'
# quantize_dynamic(model_path,
#                 quant_model_path,
#                 weight_type=QuantType.QInt8)
# print(f"quantized model saved to:{quant_model_path}")

# print(run_model(model_name, quant_model_path, prompt))

# endregion

#region### my_nlp_architect
'''well cited but appears to have dependancy issues'''
# https://github.com/IntelLabs/nlp-architect
# from transformers import BertTokenizer
# from my_nlp_architect.nlp_architect.models.transformers.quantized_bert import QuantizedBertPreTrainedModel


# model = QuantizedBertPreTrainedModel('quant_bert', from_8bit=True)

# model.save_pretrained('BERTint8')

#endregion

#region ### intel extention for transformers


from transformers import AutoTokenizer, TextStreamer
from intel_extension_for_transformers.transformers import AutoModelForCausalLM
import torch

# # PyTorch Model from Hugging Face
# https://github.com/intel/neural-speed/tree/main?tab=readme-ov-file#pytorch-model-from-hugging-face

# model_name = "Intel/neural-chat-7b-v3-1"     # Hugging Face model_id or local model
model_name = "gpt2"
# prompt = "Once upon a time, there existed a little girl,"
prompt = "Electric Light Orchestra"

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
inputs = tokenizer(prompt, return_tensors="pt").input_ids
streamer = TextStreamer(tokenizer)

model = AutoModelForCausalLM.from_pretrained(model_name, load_in_8bit=True) # <class 'transformers.models.gpt2.modeling_gpt2.GPT2LMHeadModel'>
torch.save(model.state_dict(), 'gpt2_quant.pth')
outputs = model.generate(inputs, streamer=streamer, max_new_tokens=50)


#endregion
