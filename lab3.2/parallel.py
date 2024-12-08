import hashlib
import logging
import os
from pprint import pprint
import random
import re
import time
from typing import Any, Callable, Literal, Optional
import json
from openai import OpenAI
import ray
from tqdm import tqdm

# This file is modified from an open-source project, is not a totally original work.

api_key='<Your api key here>'
base_url='https://api.deepseek.com'

def generate_hash_uid(text: str) -> str:
    """Generate hash uid"""
    return hashlib.md5(text.encode()).hexdigest()

def chat_api(
    system_content: str,
    user_content: str,
    post_process: Callable = lambda x: x,
):
    """GPT API"""
    client = OpenAI(
        api_key= api_key,
        base_url=base_url,
    )
    while True:
        try:
            all = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": user_content},
                ],
                stream=False
            )
            response = all.choices[0].message.content
            info={
                "input_tokens":all.usage.prompt_tokens,
                "output_tokens":all.usage.completion_tokens,
                "model":all.model
            }
            logging.info(response)
            break
        except Exception as exception:  # pylint: disable=broad-except # noqa: BLE001
            logging.error(exception)
            time.sleep(3)
            continue
    return post_process(response), info


@ray.remote(num_cpus=1)
def _chat_api(
    system_content: str,
    user_content: str,
    post_process: Callable = lambda x: x,
):
    """GPT API"""
    return chat_api(system_content, user_content, post_process)


def openai_api(
    contents: list[str],
    num_workers: int = 10,
    post_process: Callable = lambda x: x,
    cache_dir: Optional[str] = None,
    cache_checker: Callable = lambda _: True,
    print_interval: int = -1,
):
    """API"""
    api_interaction_count = 0
    ray.init()

    contents = list(enumerate(contents))
    content2print = {i: content for i, content in contents if i % print_interval == 0}
    bar = tqdm(total=len(contents))
    results = [None] * len(contents)
    # CHECK content is a tuple or not
    if isinstance(contents[0][1], tuple):
        uids = [generate_hash_uid(content[1][0] + content[1][1]) for content in contents]
    else:
        uids = [generate_hash_uid(content) for content in contents]
    not_finished = []
    while True:
        if len(not_finished) == 0 and len(contents) == 0:
            break
        while len(not_finished) < num_workers and len(contents) > 0:
            index, content = contents.pop()
            uid = uids[index]
            if cache_dir is not None:
                cache_path = os.path.join(cache_dir, f'{uid}.json')
                if os.path.exists(cache_path):
                    with open(cache_path, 'r', encoding='utf-8') as f:
                        try:
                            result = json.load(f)
                        except json.decoder.JSONDecodeError:
                            print(f'JSONDecodeError: {cache_path}')
                            exit()
                    if cache_checker(result):
                        results[index] = result
                        bar.update(1)
                        continue
            system_content, user_content = content
            future = _chat_api.remote(system_content, user_content, post_process)
            
            not_finished.append([index, future])
            api_interaction_count += 1
        if len(not_finished) == 0:
            continue
        # Break into a list of indices and a list of futures
        indices, futures = zip(*not_finished)
        finished, not_finished_futures = ray.wait(list(futures), timeout=1.0)
        # Find the index of completed tasks
        finished_indices = [indices[futures.index(task)] for task in finished]

        for i, task in enumerate(finished):
            results[finished_indices[i]] = ray.get(task)
            uid = uids[finished_indices[i]]
            cache_path = os.path.join(cache_dir, f'{uid}.json')
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(results[finished_indices[i]], f, ensure_ascii=False, indent=4)

            if finished_indices[i] in content2print:
                with tqdm.external_write_mode():
                    print('-' * 80)
                    print(f'Index: {finished_indices[i]}')
                    for content in content2print[finished_indices[i]]:
                        print(content)
                    if isinstance(results[finished_indices[i]], dict):
                        for key, value in results[finished_indices[i]].items():
                            print(f'{key}: {value}')
                    else:
                        print(results[finished_indices[i]])

        # Update the not_finished list to remove completed tasks
        not_finished = [(index, future) for index, future in not_finished if future not in finished]

        bar.update(len(finished))
    bar.close()

    # It is very important to ensure that all results have been collected
    assert all(result is not None for result in results)

    ray.shutdown()
    print(f'API interaction count: {api_interaction_count}')

    return results