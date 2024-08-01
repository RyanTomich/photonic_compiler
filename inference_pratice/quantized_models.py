import os
import numpy as np

import torch
from torchsummary import summary
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
from torch.profiler import profile, record_function, ProfilerActivity

import psutil



# model stats
def model_param_types(model):
    print(type(model))
    model_weights = model.state_dict()
    types = set()
    for k, v in model_weights.items():
        # print(k)
        types.add(v.dtype)
        # if v.dtype == torch.float32:
        # print(f"{type(v)} -- {v.dtype} -- {k}")
    return types


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
        print(prof.key_averages().table(sort_by="cpu_time_total"))


    if trace:
        with profile(activities=[ProfilerActivity.CPU]) as prof:
            model(inputs)

        prof.export_chrome_trace("trace.json")


def neural_speed_quantize(model):
    """
    https://github.com/intel/neural-speed/tree/main?tab=readme-ov-file#pytorch-model-from-hugging-face
    """
    from intel_extension_for_transformers.transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        load_in_8bit=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.padding_side = "right"
    tokenizer.pad_token = tokenizer.eos_token

    input_ids = tokenizer(prompt, return_tensors="pt", padding=True).input_ids
    # input_ids = tokenizer(prompt, return_tensors="pt").input_ids

    attention_mask = input_ids.ne(tokenizer.pad_token_id).long()

    streamer = TextStreamer(tokenizer)

    output_ids = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        attention_mask=attention_mask,
        pad_token_id=tokenizer.eos_token_id,
        no_repeat_ngram_size=2,
        do_sample=False,
        # streamer = streamer,
    )

    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)


def torch_quantize(model):
    model_int8 = torch.ao.quantization.quantize_dynamic(
        model,
        {
            torch.nn.Linear,
            torch.nn.Conv1d,
        },
        dtype=torch.qint8,
    )
    return model_int8


def archai_quantize(model):

    from archai.quantization.ptq import dynamic_quantization_torch

    model_int8 = dynamic_quantization_torch(model)
    return model_int8


def create_quantiaed_model(model_name, prompt, save=False, profile=False):

    model_fp32 = AutoModelForCausalLM.from_pretrained(
        model_name
    )  # <class 'transformers.models.gpt2.modeling_gpt2.GPT2LMHeadModel'>
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids

    print("Quantized")

    # model_int8 = torch_quantize(model_fp32)
    model_int8 = torch_quantize(model_fp32)

    if profile:
        print_inference_data(model_fp32, input_ids)
        # print_inference_data(model_int8, input_ids, trace=False)

    if save:
        torch.save(model_int8.state_dict(), 'model_state_dict.pth')

    # print(model_int8)

    # for name, module in model_int8.named_modules():
    #     print(module)

    # for name, param in model_int8.named_parameters():

    # state_dict = model_int8.state_dict()
    # for key, value in state_dict.items():
    #     print(f'{type(value)}  -  {value.dtype}')


    # for layer in model_int8.children():
    #     print(layer)


    # summary(model_int8, input_size=(1,4))




# model_name = "bert-base-uncased"
model_name = "gpt2"
prompt = "my favorite music is"

os.environ["OMP_NUM_THREADS"] = "1"
torch.set_num_threads(1)

create_quantiaed_model(model_name, prompt, profile = True)
