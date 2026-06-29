import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "meta-llama/Llama-2-7b-hf"
device = "cuda"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
).to(device)

model.eval()

prompt = "My name is James. What about you?"
inputs = tokenizer(prompt, return_tensors="pt").to(device)

with torch.inference_mode():
    outputs = model.generate(
        **inputs,
        max_new_tokens=69,
        do_sample=False,  # greedy decoding
    )

print(tokenizer.decode(outputs[0], skip_special_tokens=False))
