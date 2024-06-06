import numpy as np
import time
from huggingface_transformers.src.transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel
from huggingface_transformers.src.transformers.models.gpt2.tokenization_gpt2 import GPT2Tokenizer

# GPT2 - 12 blocks, 12 heads, emb_size 768
# GPT2-med - 24 blocks, 16 heads, emb_size 1024
# GPT2-large - 36 blocks, 20 heads, emb_size 1280
# GPT2-xl - 48 blocks, 24 heads, emb_size 1600

prompt = "I like music that"
model = 'GPT2-large'

# Out of box Transformer package GPT2
import np_gpt2 as np_gpt2
print("Out of box Transformer")
print(np_gpt2.transformer_gpt2_inference(prompt, *np_gpt2.transformer_gpt2_model('GPT2-large'),max_length = 25))


# # Edited Transformer Package GPT2
# t_gpt2 = GPT2LMHeadModel.from_pretrained('gpt2', output_attentions=True, activation_function = 'gelu') # loading gpt2 from forked hf_transformer library
# t_gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2') # loading gpt2 tokenizer from forked hf_transformer library

# input_ids = t_gpt2_tokenizer(prompt, return_tensors="pt").input_ids

# t_start_time = time.time()
# gen_tokens = t_gpt2.generate(input_ids, do_sample=False, max_length=25)
# t_gen_text = t_gpt2_tokenizer.batch_decode(gen_tokens)[0]
# t_end_time = time.time()


# Custom GPT2
gpt2 = GPT2LMHeadModel.from_pretrained(model, output_attentions=True, activation_function = 'gelu') # loading gpt2 from forked hf_transformer library
gpt2_tokenizer = GPT2Tokenizer.from_pretrained(model) # loading gpt2 tokenizer from forked hf_transformer library
parameters = np_gpt2.get_parameters(gpt2)

np_gpt2 = np_gpt2.NpGPT2(parameters, gpt2_tokenizer, decode_blocks = 36, attn_heads = 20, embedding_size = 1280)

np_start_time = time.time()
np_gen_text = np_gpt2.generate(prompt, max_token_len = 10)
np_end_time = time.time()

print(f'np_GPT2:{np_end_time - np_start_time} \n{np_gen_text}')
