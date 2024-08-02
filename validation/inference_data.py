'''
Run using conda (newest)
'''


import os

import cProfile

import torch
import psutil

from torchsummary import summary
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
from torch.profiler import profile, record_function, ProfilerActivity

def get_transformer_model(model_name):
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids

    return model, input_ids

def model_inference(model, input_ids):
    return model.generate(input_ids, max_new_tokens=1)




def print_inference_data(model, inputs, detailed=False, trace=False):
    """
    https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html
        chrome://tracing/
    """

    print("CPU")
    print(torch.get_num_threads())
    print(f'{psutil.cpu_freq()} MHz')
    with profile(
        activities=[ProfilerActivity.CPU], record_shapes=True, profile_memory=True
    ) as prof:
        with record_function("model_inference"):
            model(inputs)

    if detailed:
        print(prof.key_averages(group_by_input_shape=True).table(sort_by="cpu_time_total"))
    else:
        a = prof.key_averages().table(sort_by="cpu_time_total")
        print(a[-13:])
        # print(prof.key_averages().table(sort_by="cpu_time_total"))


    if trace:
        with profile(activities=[ProfilerActivity.CPU]) as prof:
            model(inputs)

        prof.export_chrome_trace("trace.json")

'''
Run with conda newest
'''

# model_name = "bert-base-uncased"
model_name = "gpt2"
prompt = "my favorite music is"


# os.environ["OMP_NUM_THREADS"] = '1'
# torch.set_num_threads(1)

model, input_ids = get_transformer_model(model_name)

# print_inference_data(model, input_ids, detailed=True, trace=True)




for i in range(1,24,1):

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"

    torch.set_num_threads(1)

    model, input_ids = get_transformer_model(model_name)

    print_inference_data(model, input_ids, detailed=True , trace=False)
