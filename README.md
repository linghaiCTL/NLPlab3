# Environment
The main libraries used in this experiments are:  
torch  
transformers  
auto_gptq  
ray  
openai  
tqdm  
pandas  
Please install them before running the code  

## Task1.1
The code file of task 1.1 is ./lab3.1/inference.py. Before running the code, please check whether the path of data and model needs to be modified, because I done my work on windows system, and if the code is running on linux, the path would need a modification.  
To run this file, simply input `python lab3.1\inference.py' in cmd lines.

## Task1.2
The code file of task 1.1 is ./lab3.1/main.py. Before running the code, please check whether the path of data and model needs to be modified, because I done my work on windows system, and if the code is running on linux, the path would need a modification.  
To run this file, simply input `python lab3.1\main.py' in cmd lines.

## Task2
The code file of task 2 is ./lab3.2/cli.py.  To run this file there are several tips needs to be paid attention to:  
### 1
Actually, I write a cache locally to avoid repeated calling. And the content of my cache is uploaded along my code. Therefore you can simply run the code without setting api-key, since the code will find the answer in the cache first and would not call the api.  (Only availible for IO, NAIVE_COT and ICL, RFL and MOD_RFL still needs to call api for some reasoning)
### 2
To modify the api-key, please refer to ./lab3.2/parallel.py and modified the value of the variable `api_key', do not simply pass it by environment variable which is not supported by my code because I run it on windows system and it's really complex to set environment variables under windows system.  

### Running
The code file takes four parameters: task_name, method, load_num, and max_repeat. The only support task_name is 'GSM', and the method should be one of ['IO','NAIVE_COT','ICL','RFL','MOD_RFL'], load_num means how many data points to load, and max_repeat means when haven't get response in wanted formation, repeating calling api upper to how many times (set to 1 most of the times).  
An example should be like: python E:\code\NLP\NLPlab3\lab3.2\cli.py GSM ICL 100 1 