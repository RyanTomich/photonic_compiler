#region ### intel extention for transformers

from transformers import AutoTokenizer, TextStreamer
from intel_extension_for_transformers.transformers import AutoModelForCausalLM
import torch
import torchvision.models as modles

# # # PyTorch Model from Hugging Face
# # https://github.com/intel/neural-speed/tree/main?tab=readme-ov-file#pytorch-model-from-hugging-face

# model_name = "Intel/neural-chat-7b-v3-1" # no pytorch?
# model_name = "gpt2"
# model_name = "gpt2-medium"
model_name = "bert-base-uncased"
path =  f'{model_name}_quant.pth'

# prompt = "Once upon a time, there existed a little girl,"
prompt = "Electric Light Orchestra"

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
inputs = tokenizer(prompt, return_tensors="pt").input_ids
streamer = TextStreamer(tokenizer)
model = AutoModelForCausalLM.from_pretrained(model_name, load_in_8bit=True) # <class 'transformers.models.gpt2.modeling_gpt2.GPT2LMHeadModel'>
torch.save(model.state_dict(), path)
# outputs = model.generate(inputs, streamer=streamer, max_new_tokens=50)
print('saved')

# # loading model

checkpoint = torch.load(path)
types = set()
for k, v in checkpoint.items():
    types.add(v.dtype)
    # print(f'{type(v)} -- {v.dtype} -- {k}')
print(types)

# model = checkpoint['model']
# model.load_state_dict(checkpoint['state_dict'])
# model.eval()



#endregion
