import sys
import os
import numpy as np

from huggingface_transformers.src.transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel
from huggingface_transformers.src.transformers.models.gpt2.tokenization_gpt2 import GPT2Tokenizer
import my_gpt2

gpt2 = GPT2LMHeadModel.from_pretrained('gpt2', output_attentions=True, activation_function = 'gelu') # loading gpt2 from forked hf_transformer library
gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2') # loading gpt2 tokenizer from forked hf_transformer library

prompt = "my favorite music is"
input_ids = gpt2_tokenizer(prompt, return_tensors="pt").input_ids
gen_tokens = gpt2.generate(input_ids, do_sample=True, temperature=0.9, max_length=10)
gen_text = gpt2_tokenizer.batch_decode(gen_tokens)[0]



state_dict = gpt2.state_dict()
parameters = {}
for name, val in state_dict.items():
    parameters[name] = val.numpy().astype(np.float32)

my_model = my_gpt2.MyGPT2(parameters)
my_result = my_model.generate(prompt, gpt2_tokenizer, max_token_len = 100, num_generate = 10)

print(f'MY_GPT2:{my_result}')
print(f'EDITED_TRANSFORMER_GPT2:{gen_text}')
