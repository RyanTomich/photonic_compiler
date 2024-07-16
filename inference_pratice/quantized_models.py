#region ### intel extention for transformers

from transformers import AutoTokenizer, TextStreamer
from intel_extension_for_transformers.transformers import AutoModelForCausalLM
import torch
import torchvision.models as modles

# # # PyTorch Model from Hugging Face


def create_model(model_name, path, prompt):
    '''
    https://github.com/intel/neural-speed/tree/main?tab=readme-ov-file#pytorch-model-from-hugging-face
    does weights for bert, nothing for GPT2
    '''
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

def model_param_types(path):
    checkpoint = torch.load(path)
    types = set()
    for k, v in checkpoint.items():
        # print(k)
        types.add(v.dtype)
        if v.dtype == torch.float32:
            print(f'{type(v)} -- {v.dtype} -- {k}')
    return types

#endregion
