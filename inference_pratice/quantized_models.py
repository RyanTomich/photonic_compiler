# region ### intel extention for transformers

from transformers import AutoTokenizer, TextStreamer
from intel_extension_for_transformers.transformers import AutoModelForCausalLM
import torch
import torchvision.models as modles

# # # PyTorch Model from Hugging Face


def create_quantized_model(model_name, prompt, max_new_tokens, save):
    """
    https://github.com/intel/neural-speed/tree/main?tab=readme-ov-file#pytorch-model-from-hugging-face
    does weights for bert, nothing for GPT2
    """
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

    if save:
        path = f"neural_chat_quant.pth"
        # print(type(model)) # <class 'neural_speed.Model'>
        # print(f'safing model to {path} ...')
        # print(model.config)
        # print(model.quantization_config)
        # print(model.quant_model)
        # print(model.bin_file)

        # model.save(path)
        # model.save_pretrained(path)
        # torch.save(model.state_dict(), path)

    return model, generated_text


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


# endregion
