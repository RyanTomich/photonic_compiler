import sys
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BertTokenizer,
    BertForMaskedLM,
    TextStreamer,
)

import quantized_models as quantized


def original_gpt(model_name, prompt, max_new_tokens, save=False):
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    tokenizer.padding_side = "right"
    tokenizer.pad_token = tokenizer.eos_token

    input_ids = tokenizer(prompt, return_tensors="pt", padding=True).input_ids

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

    if save:
        path = f"neural_chat_original.pth"
        torch.save(model.state_dict(), path)

    return model, generated_text


model_name = "Intel/neural-chat-7b-v3-1" # no pytorch?
# model_name = "gpt2"
# model_name = "gpt2-medium"

prompt = "my favorite music is"

max_new_tokens = 50


print(f'############### {model_name} Original Response ###############')
model, generated = original_gpt(model_name, prompt, max_new_tokens, save=False)
print(generated)
# print(quantized.model_param_types(model))


model, generated = quantized.create_quantized_model(model_name, prompt, max_new_tokens, save=False)
print(f'############### {model_name} Quantized Response ###############')
print(generated)
# print(type(model))

# model_quantize_internal: model size  = 27_625.02 MB
# model_quantize_internal: quant size  =  4316.73 MB

# print(quantized.model_param_types(model))
