import torch.utils._pytree as _torch_pytree
import transformers
from transformers import GPT2Tokenizer, TFGPT2LMHeadModel

#### Download pretrained GPT2 Model ####

# tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
# model = TFGPT2LMHeadModel.from_pretrained(
#     "gpt2", from_pt=True, pad_token_id=tokenizer.eos_token_id
# )
# model.save_pretrained("./tfgpt2model", saved_model=True, to_pt=True)

#### Create Relay IR ####
