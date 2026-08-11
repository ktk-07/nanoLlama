import os
import sys
from safetensors import safe_open
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..model.llama import Llama
import transformers
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from time import perf_counter

device = "cuda:0"
device = "cpu"
dtype = torch.float16

# Load the config for Llama 2 model (e.g., the 7B version)
model_name = "meta-llama/Llama-2-7b-hf"
config = AutoConfig.from_pretrained(model_name)
print(config)

tokenizer = AutoTokenizer.from_pretrained(model_name)
print(tokenizer)

old_dtype = torch.get_default_dtype()
print(old_dtype)
torch.set_default_dtype(torch.float16)
new_dtype = torch.get_default_dtype()
print(f"The old_dtype is {old_dtype} the new_type is {new_dtype}")


# Load the Model

#with torch.device("meta"):
#    model = Llama2(config.attention_bias, config.attention_dropout, config.head_dim, config.hidden_size, config.intermediate_size, config.num_hidden_layers, config.num_attention_heads, config.num_key_value_heads, config.rope_parameters["rope_theta"], config.vocab_size)

#print(model)

#total_params = sum(
#    p.numel()
#    for p in model.parameters()
#)

#print(total_params)
#print(config.num_attention_heads, config.num_key_value_heads)
model = Llama(config.attention_bias, config.attention_dropout, config.head_dim, config.hidden_size, config.intermediate_size, config.num_hidden_layers, config.num_attention_heads, config.num_key_value_heads, config.max_position_embeddings, config.mlp_bias, config.rope_parameters["rope_theta"], config.vocab_size)
print(model)

# iF MODEL DOES NTO EXIST IN CACHE DOWNLOAD
working_dir = "/home/buntr"
cache_dir = "/.cache/huggingface/hub/"
model_name = "meta-llama--Llama-2-7b-hf"
file_name0 = "/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-hf/blobs/4ec71fd53e99766de38f24753b30c9e8942630e9e576a1ba27b0ec531e87be41"
file_name1 = "/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-hf/blobs/41780b5dac322ac35598737e99208d90bdc632a1ba3389ebedbb46a1d8385a7f"

for i in range(2):
    full_file_path = working_dir
    if i == 0:
        full_file_path += file_name0
    elif i == 1:
        full_file_path += file_name1
    #with torch.no_grad():
        #with safe_open(full_file_path, framework="pt", device="cpu") as f:
            #for name, param in model.named_parameters():
                #ckpt_tensor = f.get_tensor(name)
                #param.copy_(ckpt_tensor.to(device=device, dtype=dtype))
                #del ckpt_tensor
    break
    with torch.no_grad():
        with safe_open(full_file_path, framework="pt", device="cpu") as f:
            #for name, tensor in list(model.named_parameters()) + list(model.named_buffers()):
                #print(name)
                #src = f.get_tensor(name)
                #tensor.copy_(src.to(device=tensor.device, dtype=tensor.dtype))
                #del src
            available_keys = set(f.keys())
            for name, tensor in list(model.named_parameters()) + list(model.named_buffers()):
                new_name = "model."
                if name != "lm_head.weight":
                    new_name += name
                else:
                    new_name = name

                if new_name in available_keys:
                    src = f.get_tensor(new_name)
                    tensor.copy_(src.to(device=device, dtype=tensor.dtype))
                    del src

loaded = set()
missing = []
shape_mismatch = []

with torch.no_grad():
    for i in range(2):
        full_file_path = working_dir
        if i == 0:
            full_file_path += file_name0
        elif i == 1:
            full_file_path += file_name1
        with safe_open(full_file_path, framework="pt", device="cpu") as f:
            keys = set(f.keys())

            for name, tensor in list(model.named_parameters()) + list(model.named_buffers()):
                new_name = "model."
                if name != "lm_head.weight":
                    new_name += name
                else:
                    new_name = name

                if new_name not in keys:
                    continue

                src = f.get_tensor(new_name)

                if src.shape != tensor.shape:
                    shape_mismatch.append((new_name, tensor.shape, src.shape))
                    continue

                tensor.copy_(src.to(device=tensor.device, dtype=tensor.dtype))
                loaded.add(name)

model_names = set(name for name, _ in list(model.named_parameters()) + list(model.named_buffers()))

print("loaded:", len(loaded))
print("missing:", sorted(model_names - loaded))
print("num missing:", len(model_names - loaded))
print("shape mismatch:", shape_mismatch)

# We can write the inference code here

txt = "My name is James. What about you?"
input_obj = tokenizer.encode(txt) 
input_ids = torch.Tensor(input_obj).to(dtype=torch.long,device=device)
print(input_ids)

def sampling_with_no_temperature(model, input_ids, num_of_tkns_to_generate, device, start_pos=0, use_cache=False):
    
    if not use_cache:
        with torch.no_grad():
            start_pos = start_pos
            original_input_length = input_ids.shape[0]
            num_of_tkn_generated = 0
            num_of_tkns_to_generate = num_of_tkns_to_generate

            output_tkns = torch.Tensor([])

            while num_of_tkn_generated < num_of_tkns_to_generate:

                if num_of_tkn_generated == 0:
                    start_tkns = input_ids.unsqueeze(0)
                else:
                    start_tkns = torch.concat((input_ids.unsqueeze(0),output_tkns),dim=-1)
                    # print("The final start tkns shape is", start_tkns.shape)

                B, L = start_tkns.shape
                attention_mask = (start_tkns > 0).to(dtype=torch.long)
                position_ids =  attention_mask.cumsum(dim=-1) - 1
                position_ids[attention_mask == 0] = 0
                # print(input_ids.shape)
                # print(attention_mask.shape)
                # print(position_ids.shape)
                attention_mask = torch.triu(torch.ones(L,L).to(device=device,dtype=torch.bool),diagonal=1)
                output_logits = model(input_ids=start_tkns.to(device=device),attention_mask=attention_mask.to(device=device),position_ids=position_ids.to(device=device), start_pos=0)
                # Generate a probability distribution from the logits
                model_output_probability_distribution = torch.softmax(output_logits.float(),dim=-1)
                # You always want to take the probability distribution predicted by the final token
                output_probability_distribution = model_output_probability_distribution[0,-1,:] 
                model_output_tkn = torch.argmax(output_probability_distribution, dim=-1)
                #model_output_tkn = torch.multinomial(output_probability_distribution,num_samples=1)[0]
                #model_output_tkn = torch.multinomial(output_probability_distribution, num_samples=1)[0]            
                # print(model_output_tkn, output_probability_distribution[model_output_tkn])

                # print(model_output_probability[0,-1,:])
                # print(model_output_tkn)
                # print(model_output_probability[0,-1,model_output_tkn])
                # print(output.logits.shape)
                # print(model_output_probability.shape)

                if num_of_tkn_generated == 0:
                    output_tkns = model_output_tkn.unsqueeze(0).unsqueeze(0)
                else:
                    output_tkns = torch.concat((output_tkns,model_output_tkn.unsqueeze(0).unsqueeze(0)),dim=-1)

                # print("Output token shape is ", output_tkns.shape) 
                num_of_tkn_generated += 1

            return output_tkns.squeeze(0)
        
    else:
        with torch.no_grad():
            original_input_length = input_ids.shape[0]
            num_of_tkn_generated = 0
            num_of_tkns_to_generate = num_of_tkns_to_generate

            output_tkns = torch.Tensor([])

            # Prefill
            if num_of_tkn_generated == 0:
                start_tkns = input_ids.unsqueeze(0)
            else:
                start_tkns = torch.concat((input_ids.unsqueeze(0),output_tkns),dim=-1)
                # print("The final start tkns shape is", start_tkns.shape)

            B, L = start_tkns.shape
            attention_mask = (start_tkns > 0).to(dtype=torch.long)
            position_ids =  attention_mask.cumsum(dim=-1) - 1
            position_ids[attention_mask == 0] = 0
            # print(input_ids.shape)
            # print(attention_mask.shape)
            # print(position_ids.shape)
            attention_mask = torch.triu(torch.ones(L,L).to(device=device,dtype=torch.bool),diagonal=1)
            output_logits = model(input_ids=start_tkns.to(device=device),attention_mask=attention_mask.to(device=device),position_ids=position_ids.to(device=device),start_pos=0, use_cache=True)
            # Generate a probability distribution from the logits
            model_output_probability_distribution = torch.softmax(output_logits.float(),dim=-1)
            # You always want to take the probability distribution predicted by the final token
            output_probability_distribution = model_output_probability_distribution[0,-1,:] 
            model_output_tkn = torch.argmax(output_probability_distribution, dim=-1)

            if num_of_tkn_generated == 0:
                output_tkns = model_output_tkn.unsqueeze(0).unsqueeze(0)
            else:
                output_tkns = torch.concat((output_tkns,model_output_tkn.unsqueeze(0).unsqueeze(0)),dim=-1)

            num_of_tkn_generated += 1

            # Decode
            input_tkn = model_output_tkn
            starting_pos = L - 1 
            while num_of_tkn_generated < num_of_tkns_to_generate:

                # print("The final start tkns shape is", start_tkns.shape)
                start_pos = starting_pos + num_of_tkn_generated
                start_tkn = torch.Tensor(input_tkn).unsqueeze(0).unsqueeze(0)
                B, L = start_tkn.shape
                attention_mask = (start_tkn > 0).to(dtype=torch.long)
                position_ids =  attention_mask.cumsum(dim=-1) - 1
                position_ids[attention_mask == 0] = 0
                # print(input_ids.shape)
                # print(attention_mask.shape)
                # print(position_ids.shape)
                attention_mask = torch.triu(torch.ones(L,L).to(device=device,dtype=torch.bool),diagonal=1)
                output_logits = model(input_ids=start_tkn.to(device=device),attention_mask=None,position_ids=position_ids.to(device=device),start_pos=start_pos, use_cache=True)
                # Generate a probability distribution from the logits
                model_output_probability_distribution = torch.softmax(output_logits.float(),dim=-1)
                # You always want to take the probability distribution predicted by the final token
                output_probability_distribution = model_output_probability_distribution[0,-1,:] 
                model_output_tkn = torch.argmax(output_probability_distribution, dim=-1)

                if num_of_tkn_generated == 0:
                    output_tkns = model_output_tkn.unsqueeze(0).unsqueeze(0)
                else:
                    output_tkns = torch.concat((output_tkns,model_output_tkn.unsqueeze(0).unsqueeze(0)),dim=-1)

                num_of_tkn_generated += 1
                input_tkn = model_output_tkn

            print(output_tkns)
            return output_tkns.squeeze(0)


print("Starting the sampling")
start = perf_counter()
num_of_tkns_to_generate = 69
use_cache=False
output = sampling_with_no_temperature(model, input_ids, num_of_tkns_to_generate, device, start_pos=0, use_cache=use_cache)
print(input_ids)
print(output)
print(input_ids.tolist() + output.tolist())
print(tokenizer.decode(input_obj + output.tolist()))
end = perf_counter()
print(f"No cache timing: {end - start:.2f}s")
print(f"tkn per second : {(end-start)/ num_of_tkns_to_generate:.4f}")

print(f"Starting the kvcache output\n")
start = perf_counter()
num_of_tkns_to_generate = 69
use_cache=True
output = sampling_with_no_temperature(model, input_ids, num_of_tkns_to_generate, device, start_pos=0, use_cache=use_cache)
print(input_ids)
print(output)
print(input_ids.tolist() + output.tolist())
print(tokenizer.decode(input_obj + output.tolist()))
end = perf_counter()

print(f"KV-Cache timing: {end - start:.4f}s")
print(f"tkn per second : {(end-start)/ num_of_tkns_to_generate:.4f}")
