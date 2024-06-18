import numpy as np
import time
from huggingface_transformers.src.transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel
from huggingface_transformers.src.transformers.models.gpt2.tokenization_gpt2 import GPT2Tokenizer

# GPT2 - 12 blocks, 12 heads, emb_size 768
# GPT2-med - 24 blocks, 16 heads, emb_size 1024
# GPT2-large - 36 blocks, 20 heads, emb_size 1280
# GPT2-xl - 48 blocks, 24 heads, emb_size 1600

prompt = "I like music that"
model = 'gpt2'
max_length = 15

# Out of box Transformer package GPT2
import np_gpt2 as np_gpt2
import np_gpt2_int8
# print("Out of box Transformer")
# print(np_gpt2.transformer_gpt2_inference(prompt, *np_gpt2.transformer_gpt2_model(model),max_length = max_length))


# # Edited Transformer Package GPT2
t_gpt2 = GPT2LMHeadModel.from_pretrained(model, output_attentions=True, activation_function = 'gelu') # loading gpt2 from forked hf_transformer library
t_gpt2_tokenizer = GPT2Tokenizer.from_pretrained(model) # loading gpt2 tokenizer from forked hf_transformer library

input_ids = t_gpt2_tokenizer(prompt, return_tensors="pt").input_ids

# t_start_time = time.time()
# gen_tokens = t_gpt2.generate(input_ids, do_sample=False, max_length=25)
# t_gen_text = t_gpt2_tokenizer.batch_decode(gen_tokens)[0]
# t_end_time = time.time()
# print(t_gen_text)

# import model_trace as trace


# Custom GPT2
gpt2 = GPT2LMHeadModel.from_pretrained(model, output_attentions=True, activation_function = 'gelu') # loading gpt2 from forked hf_transformer library
gpt2_tokenizer = GPT2Tokenizer.from_pretrained(model) # loading gpt2 tokenizer from forked hf_transformer library
parameters = np_gpt2.get_parameters(gpt2)

# Charicteristics of the paramaters
# largest_param = 0
# smallest_param = 0
# closest_to_0 = np.inf
# for param in parameters.values():
#     print(type(param[0]))
#     largest = np.max(param)
#     largest_param = max(largest_param, largest)
#     smallest = np.min(param)
#     smallest_param = min(smallest_param, smallest)

#     non_zero_arr = param[param != 0]
#     closest = np.min(np.abs(non_zero_arr))
#     closest_to_0 = min(closest_to_0, closest)

# print(largest_param) #17.4193
# print(smallest_param) #-11.0504
# print(closest_to_0) #1e-04



gpt2_model = np_gpt2.NpGPT2(parameters, gpt2_tokenizer, decode_blocks = 12, attn_heads = 12, embedding_size = 768)

# np_start_time = time.time()
# np_gen_text = gpt2_model.generate(prompt, max_token_len = max_length)
# np_end_time = time.time()

# print(f'np_GPT2:{np_end_time - np_start_time} \n{np_gen_text}')



# int8 GPT2

gpt2_int8 =  np_gpt2_int8.NpGPT2(parameters, gpt2_tokenizer, decode_blocks = 12, attn_heads = 12, embedding_size = 768)
np_start_time = time.time()
np_gen_text = gpt2_int8.generate(prompt, max_token_len = max_length)
np_end_time = time.time()

print(f'np_GPT2_int8:{np_end_time - np_start_time} \n{np_gen_text}')

# graph_len = len(trace.graph.dependancy_graph)
# functions_in = trace.file_write.calls
# print(graph_len)
# print(functions_in)
# # assert graph_len == functions_in + 1 # +1 for START node

# trace.graph.print_instructions()
