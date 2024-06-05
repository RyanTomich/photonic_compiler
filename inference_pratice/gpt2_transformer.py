import sys
import os

from huggingface_transformers.src.transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel
from huggingface_transformers.src.transformers.models.gpt2.tokenization_gpt2 import GPT2Tokenizer

gpt2 = GPT2LMHeadModel.from_pretrained('gpt2', output_attentions=True, activation_function = 'gelu') # loading gpt2 from transformers library
gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2') # loading gpt2 tokenizer from transformers library

prompt = "my favorite music is"
input_ids = gpt2_tokenizer(prompt, return_tensors="pt").input_ids
gen_tokens = gpt2.generate(input_ids, do_sample=True, temperature=0.9, max_length=100)
gen_text = gpt2_tokenizer.batch_decode(gen_tokens)[0]

print(gen_text)
