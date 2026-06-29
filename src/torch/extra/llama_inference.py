import time
# from time import time
import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM

# Load the config for Llama 2 model (e.g., the 7B version)
model_name = "meta-llama/Llama-2-7b-hf"
config = AutoConfig.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

txt = "My name is James. What about you?"
input_obj = tokenizer.encode(txt)
input_ids = torch.Tensor(input_obj)
print(input_ids.shape)

#tokenizer.apply_chat_template(txt)
#chat = [
#  {"role": "user", "content": "Hello, how are you?"},
#  {"role": "assistant", "content": "I'm doing great. How can I help you today?"},
#  {"role": "user", "content": "I'd like to show off how chat templating works!"},
#]
#print(tokenizer.apply_chat_template(txt,tokenise=False))

def sampling_with_no_temperature(model, input_ids, num_of_tkns_to_generate, device):
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

        attention_mask = (start_tkns > 0).to(dtype=torch.long)
        position_ids =  attention_mask.cumsum(dim=-1) - 1
        position_ids[attention_mask == 0] = 0
        # print(input_ids.shape)
        # print(attention_mask.shape)
        # print(position_ids.shape)


        output_logits = model.forward(input_ids=start_tkns.to(device=device),attention_mask=attention_mask.to(device=device),position_ids=position_ids.to(device=device))
        print(output_logits)
        output_logits = output
        # Generate a probability distribution from the logits
        model_output_probability_distribution = torch.softmax(output_logits,dim=-1)
        # You always want to take the probability distribution predicted by the final token
        output_probability_distribution = model_output_probability_distribution[0,-1,:] 
        # model_output_tkn = torch.argmax(output_probability_distribution, dim=-1)

        model_output_tkn = torch.multinomial(output_probability_distribution,num_samples=1)[0]
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


