import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BertTokenizer, BertForMaskedLM, TextStreamer


import quantized_models as quantized

def original_gpt(model_name, prompt, max_length):
    print(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    tokenizer.padding_side = "right"
    tokenizer.pad_token = tokenizer.eos_token

    input_ids = tokenizer(prompt, return_tensors="pt", padding=True).input_ids

    attention_mask = input_ids.ne(tokenizer.pad_token_id).long()

    output_ids = model.generate(input_ids, max_length=max_length, attention_mask=attention_mask, pad_token_id=tokenizer.eos_token_id, no_repeat_ngram_size=2)

    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    return generated_text


model_name = "Intel/neural-chat-7b-v3-1" # no pytorch?
# model_name = "gpt2"
# model_name = "gpt2-medium"

path =  f'{model_name}_quant.pth'
prompt = "my favorite music is"


# print(original_gpt(model_name, prompt, 50))


print(original_gpt('Intel/neural-chat-7b-v3-1', prompt, 50))
print(original_gpt('gpt2', prompt, 50))
print(original_gpt('gpt2-medium', prompt, 50))
