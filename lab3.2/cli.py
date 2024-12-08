from openai import OpenAI
from parallel import openai_api
import numpy as np
import torch
from chating_tamplate import SYSTEM_PROMPT_GSM, USER_PROMPT_GSM_IO, USER_PROMPT_GSM_NAIVE_COT, USER_PROMPT_GSM_ICL, USER_PROMPT_GSM_RFL, USER_PROMPT_GSM_MOD_RFL
from dataset_loaders import gsm_dataset
import os
import hashlib
import argparse

api_key = "sk-0dda3f5b454f4635ae50fb4f6ca9311d"
base_url = "https://api.deepseek.com"
data_path='datasets\\gsm8k\\main\\test-00000-of-00001.parquet'

def cli():
    '''CLI for chatting with the model'''
    client = OpenAI(api_key=api_key, base_url=base_url)
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "Can you tell me a joke?"},
        ],
        stream=False
    )
    print(response.choices[0].message.content)
    
def postprocess(response):
    '''Postprocess the response from the model to get the answer'''
    result={
        'reasoning': None,
        'reflection': None,
        'answer': None
    }
    reasoning=response.split('[[REASONING]]: ')[-1].split('[[END]]')[0]
    answer=response.split('[[ANSWER]]: ')[-1].split('[[END]]')[0]
    reflexion=response.split('[[REFLECTION]]: ')[-1].split('[[END]]')[0]
    try:
        answer=int(answer)
    except:
        answer=None
    result['answer']=answer
    result['reasoning']=reasoning
    result['reflection']=reflexion
    return result

def cache_checker(result: dict) -> bool:
    return result[0]['answer'] is not None

def format_input(data, task_name, method=None):
    input_=None
    if task_name=='GSM':
        if method=='IO':
            input_ = USER_PROMPT_GSM_IO.format(question=data['input'])
        elif method=='NAIVE_COT' or method=='RFL' or method=='MOD_RFL':
            # the first time to call the model, cot,rfl,mod_rfl are the same
            input_ = USER_PROMPT_GSM_NAIVE_COT.format(question=data['input'])
        elif method=='ICL':
            input_ = USER_PROMPT_GSM_ICL.format(question=data['input'])
        else:
            print('Method not supported yet')
    return input_

def format_reflection(data, task_name, method=None):
    reflection=None
    if task_name=='GSM':
        if method=='RFL':
            reflection = USER_PROMPT_GSM_RFL.format(question=data['input'], reasoning=data['reasoning'], answer=data['output'])
        elif method=='MOD_RFL':
            reflection = USER_PROMPT_GSM_MOD_RFL.format(question=data['input'], reasoning=data['reasoning'], answer=data['output'])
        else:
            print('Method not supported yet')
    return reflection
    
def inference(task_name, method, load_num=100, max_repeat=5):
    # Load dataset
    if task_name=='GSM':
        dataset=gsm_dataset(data_path)
        system_prompt=SYSTEM_PROMPT_GSM
    else:
        print('Task not supported yet')
        return
    
    load_num=min(load_num, len(dataset))
    data=[
        {
            'uuid': hashlib.md5(dataset[i]['input'].encode()).hexdigest(),
            'input': dataset[i]['input'],
            'output': None,
            'reasoning': None,
            'reflection': None
        } for i in range(load_num)
    ]
    empty_index = list(range(len(data)))
    ori_repeat=max_repeat
    # init inference
    
    total_called=0
    correct_called=0
    
    # Start chatting
    while len(empty_index) > 0 and max_repeat > 0:
        print(f'Remaining repeat times: {max_repeat}')
        contents = [
            (
                system_prompt,
                format_input(line, task_name, method),
            )
            for line in [data[i] for i in empty_index]
        ]

        results = openai_api(
            contents,
            post_process=postprocess,
            cache_checker=cache_checker,
            num_workers=10,
            cache_dir='cache' if method=='IO' or method=='NAIVE_COT' or method=='ICL' else 'cache_reflection',
            print_interval=10,
        )

        for index, result in zip(empty_index, results):
            data[index]['output'] = result[0]['answer']
            if method=='rfl' or method=='mod_rfl':
                data[index]['reasoning'] = result[0]['reasoning']
                data[index]['reflection'] = result[0]['reflection']
        empty_index = [i for i in empty_index if data[i]['output'] is None]
        print(f'Number of inference failed: {len(empty_index)}')
        max_repeat -= 1
        
    # Perform reflection
    if method=='RFL':
        index_need_reflection = [i for i in range(len(data)) if data[i]['output']!=dataset[i]['answer']]
        print(f'Number of reflection needed: {len(index_need_reflection)}')
    elif method=='MOD_RFL':
        index_need_reflection = range(len(data))
        print(f'Number of reflection needed: {len(index_need_reflection)}')
    
    if method=='RFL' or method=='MOD_RFL':
        empty_index = list(index_need_reflection)
        max_repeat=ori_repeat
        while len(empty_index) > 0 and max_repeat > 0:
            print(f'Remaining repeat times: {max_repeat}')
            contents = [
                (
                    system_prompt,
                    format_reflection(line, task_name, method),
                )
                for line in [data[i] for i in empty_index]
            ]

            results = openai_api(
                contents,
                post_process=postprocess,
                cache_checker=cache_checker,
                num_workers=10,
                cache_dir='cache_reflection',
                print_interval=10,
            )

            for index, result in zip(empty_index, results):
                data[index]['output'] = result[0]['answer']
                data[index]['reasoning'] = result[0]['reasoning']
                data[index]['reflection'] = result[0]['reflection']
            empty_index = [i for i in empty_index if data[i]['output'] is None]
            print(f'Number of reflection failed: {len(empty_index)}')
            max_repeat -= 1

    # Calculate the accuracy
    for i in range(len(data)):
        if data[i]['output'] is not None:
            total_called+=1
            if data[i]['output']==dataset[i]['answer']:
                correct_called+=1
    print(f'Total called: {total_called}')
    print(f'Correct called: {correct_called}')
    print(f'Accuracy: {correct_called/total_called}')
    
def main():
    parser = argparse.ArgumentParser(description='CLI for chatting with the model')
    parser.add_argument('task_name', type=str, default='GSM', help='Task name')
    parser.add_argument('method', type=str, default='IO', help='Method name')
    parser.add_argument('load_num', type=int, default=10, help='Number of data to load')
    parser.add_argument('max_repeat', type=int, default=1, help='Max repeat times')
    args = parser.parse_args()
    inference(args.task_name, args.method, args.load_num, args.max_repeat)

if __name__ == "__main__":
    main()