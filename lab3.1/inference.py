import os
import torch
import time
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
import random
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

model_path='gpt2'
data_path='lab3.1\data.txt'


def cli():
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.padding_side = 'left'
    tokenizer.pad_token = tokenizer.eos_token
    original_model = AutoModelForCausalLM.from_pretrained(model_path, attn_implementation="eager", device_map='cuda')

    while(True):
        prompt=input("Enter a prompt: ")
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        input_ids = input_ids.to('cuda')
        outputs = original_model.generate(input_ids, max_length=100, pad_token_id=tokenizer.eos_token_id)
        print(tokenizer.decode(outputs[0], skip_special_tokens=True))
        
def inference(use_kv_cache=False):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.padding_side = 'left'
    tokenizer.pad_token = tokenizer.eos_token
    original_model = AutoModelForCausalLM.from_pretrained(model_path, attn_implementation="eager", device_map='cuda')
    original_model.eval()
    original_model.to('cuda')
    # plain version: RUN cost 71.25 sec for 8796 tokens, throughput 123.4 tokens/sec 1583-765=818MiB
    # Use KV cache: RUN cost  65.26 sec for 8796 tokens, throughput 134.5 tokens/sec   1487-748=739MiB
    with open(data_path) as f:
        prompt_dataset = [i.strip() for i in f.readlines()]
    output_list=[]
    batchsize=16
    token_nums=0
    start_time = time.time()
    for i in range(0, (len(prompt_dataset) + batchsize - 1) // batchsize):
        batch = prompt_dataset[i * batchsize: (i + 1) * batchsize]
        for prompt in batch:
            input_ids, attention_mask = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=128).input_ids, tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=128).attention_mask
            input_ids = input_ids.to('cuda')
            outputs = original_model.generate(input_ids, max_length=100, pad_token_id=tokenizer.eos_token_id, use_cache=use_kv_cache, attention_mask=attention_mask.to('cuda'))
            output_list.append(tokenizer.decode(outputs[0], skip_special_tokens=True))
    time_taken=time.time()-start_time
    # calculate how many tokens are generated
    for item in output_list:
        token_nums+=len(tokenizer(item, return_tensors="pt").input_ids[0])
    return output_list,time_taken,token_nums
        
def quantize_inference(quantize_bits=3):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.padding_side = 'left'
    tokenizer.pad_token = tokenizer.eos_token
    
    quantize_config = BaseQuantizeConfig(
        bits=quantize_bits,
        group_size=128,
        damp_percent=0.01,
        desc_act=False,
    )
    original_model = AutoGPTQForCausalLM.from_pretrained(model_path, quantize_config)
    original_model.eval()
    original_model.to('cuda')
    with open(data_path) as f:
        prompt_dataset = [i.strip() for i in f.readlines()]
    output_list=[]
    batchsize=16
    
    # quantize on the same dataset
    data=[{
        'input_ids': tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=128).input_ids,
        'attention_mask': tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=128).attention_mask
    } for prompt in prompt_dataset]
    original_model.quantize(data, batch_size=batchsize)
    
    # Int3 quantize: RUN cost  269.1 sec for 8796 tokens, throughput 32.68 tokens/sec   1186-854=332MiB
    # Int4 quantize: RUN cost  133.3 sec for 8702 tokens, throughput 65.26 tokens/sec   1269-748=521MiB
    token_nums=0
    start_time = time.time()
    for i in range(0, (len(prompt_dataset) + batchsize - 1) // batchsize):
        batch = prompt_dataset[i * batchsize: (i + 1) * batchsize]
        for prompt in batch:
            input_ids, attention_mask = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=128).input_ids, tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=128).attention_mask
            input_ids = input_ids.to('cuda')
            outputs = original_model.generate(input_ids=input_ids, max_length=100, pad_token_id=tokenizer.eos_token_id, use_cache=True, attention_mask=attention_mask.to('cuda'))
            output_list.append(tokenizer.decode(outputs[0], skip_special_tokens=True))
    time_taken=time.time()-start_time
    # calculate how many tokens are generated
    for item in output_list:
        token_nums+=len(tokenizer(item, return_tensors="pt").input_ids[0])
    return output_list,time_taken,token_nums
        
        
if __name__ == "__main__":
    print("Inference without KV cache")
    output_list,time_cost, tokens=inference()
    save_path='output_without_kv_cache.txt'
    with open(save_path, 'w') as f:
        for item in output_list:
            f.write("%s\n" % item)
    print(f"Output saved to {save_path}")
    print(f"Time cost: {time_cost} seconds")
    print(f"Total tokens generated: {tokens}")
    print(f"Average tokens per second: {tokens/time_cost}")
    
    print("Inference with KV cache")
    output_list,time_cost, tokens=inference(use_kv_cache=True)
    save_path='output_with_kv_cache.txt'
    with open(save_path, 'w') as f:
        for item in output_list:
            f.write("%s\n" % item)
    print(f"Output saved to {save_path}")
    print(f"Time cost: {time_cost} seconds")
    print(f"Total tokens generated: {tokens}")
    print(f"Average tokens per second: {tokens/time_cost}")
    
    print("Quantize Inference with Int4")
    output_list,time_cost, tokens=quantize_inference(quantize_bits=4)
    save_path='output_4bit.txt'
    with open(save_path, 'w') as f:
        for item in output_list:
            f.write("%s\n" % item)
    print(f"Output saved to {save_path}")
    print(f"Time cost: {time_cost} seconds")
    print(f"Total tokens generated: {tokens}")
    print(f"Average tokens per second: {tokens/time_cost}")
    
    print("Quantize Inference with Int3")
    output_list,time_cost, tokens=quantize_inference(quantize_bits=3)
    save_path='output_3bit.txt'
    with open(save_path, 'w') as f:
        for item in output_list:
            f.write("%s\n" % item)
    print(f"Output saved to {save_path}")
    print(f"Time cost: {time_cost} seconds")
    print(f"Total tokens generated: {tokens}")
    print(f"Average tokens per second: {tokens/time_cost}")