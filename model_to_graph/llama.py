import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer, pipeline
from huggingface_hub import login

# Login securely (avoid hardcoding the token)
login()


model_name = "meta-llama/Llama-2-7b-hf"

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Create a text generation pipeline
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Define your input prompt
input_prompt = "Once upon a time"

# Run inference
output = generator(input_prompt, max_length=10, num_return_sequences=1)

# Print the output
print(output[0]['generated_text'])
