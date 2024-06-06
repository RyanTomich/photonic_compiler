import numpy as np
import time


# Edited Transformer Package GPT2
from huggingface_transformers.src.transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel
from huggingface_transformers.src.transformers.models.gpt2.tokenization_gpt2 import GPT2Tokenizer

t_gpt2 = GPT2LMHeadModel.from_pretrained('gpt2', output_attentions=True, activation_function = 'gelu') # loading gpt2 from forked hf_transformer library
t_gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2') # loading gpt2 tokenizer from forked hf_transformer library

prompt = "I like music that"
input_ids = t_gpt2_tokenizer(prompt, return_tensors="pt").input_ids

t_start_time = time.time()
gen_tokens = t_gpt2.generate(input_ids, do_sample=False, max_length=20)
t_gen_text = t_gpt2_tokenizer.batch_decode(gen_tokens)[0]
t_end_time = time.time()



# Custom GPT2
import np_gpt2 as np_gpt2
state_dict = t_gpt2.state_dict()
parameters = {}
for name, val in state_dict.items():
    parameters[name] = np.round(val.numpy().astype(np.float32), 4)

np_gpt2 = np_gpt2.MyGPT2(parameters)
np_start_time = time.time()
np_gen_text = np_gpt2.generate(prompt, t_gpt2_tokenizer, max_token_len = 20)
np_end_time = time.time()

print(f'np_GPT2:{np_gen_text}')
print(f"np_GPT2 exicution time: {np_end_time - np_start_time}")

print(f'EDITED_TRANSFORMER_GPT2:{t_gen_text}')
print(f"np_GPT2 exicution time: {t_end_time - t_start_time}")

assert(t_gen_text == np_gen_text)
