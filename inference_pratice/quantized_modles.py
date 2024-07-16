#region ### intel extention for transformers

import inspect
from transformers import AutoTokenizer, TextStreamer
from intel_extension_for_transformers.transformers import AutoModelForCausalLM
import torch
import torchvision.models as modles

# # # PyTorch Model from Hugging Face


def create_model(model_name, path):
    '''
    https://github.com/intel/neural-speed/tree/main?tab=readme-ov-file#pytorch-model-from-hugging-face
    does weights for bert, nothing for GPT2
    '''

    prompt = "Electric Light Orchestra"

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    inputs = tokenizer(prompt, return_tensors="pt").input_ids
    streamer = TextStreamer(tokenizer)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        load_in_8bit=True,
    ) # <class 'transformers.models.gpt2.modeling_gpt2.GPT2LMHeadModel'>

    torch.save(model.state_dict(), path)
    # outputs = model.generate(inputs, streamer=streamer, max_new_tokens=50)
    print('saved')

def model_types(path):
    checkpoint = torch.load(path)
    types = set()
    for k, v in checkpoint.items():
        # print(k)
        types.add(v.dtype)
        if v.dtype == torch.float32:
            print(f'{type(v)} -- {v.dtype} -- {k}')
    print(types)



# model_name = "Intel/neural-chat-7b-v3-1" # no pytorch?
model_name = "gpt2"
# model_name = "gpt2-medium"
# model_name = "bert-base-uncased"
path =  f'{model_name}_quant.pth'

# create_model(model_name, path)
# model_types(path)

#endregion
