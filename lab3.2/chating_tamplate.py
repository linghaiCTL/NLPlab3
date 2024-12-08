SYSTEM_PROMPT_GSM='''
    You are a helpful assistant for math problems.'''
    
USER_PROMPT_GSM_IO="""
Here is the math problem you need to solve:
[[QUESTION]]: {question} [[END]],

Please provide the answer in the following format, the answer should be a number:
[[ANSWER]]: <Your answer>[[END]]
"""
USER_PROMPT_GSM_NAIVE_COT="""
Here is the math problem you need to solve:
[[QUESTION]]: {question} [[END]],

Please think step by step to sovle this problem, and provide the answer in the following format, the answer should be a number:
[[REASONING]]: <Your reasoning>[[END]],
[[ANSWER]]: <Your answer>[[END]]
"""
USER_PROMPT_GSM_ICL="""
Here is an example of solving a math problem:
[[QUESTION]]: The price of Parmesan cheese is $11 per pound. Mozzarella cheese is $6 per pound.  Amor buys 2 pounds of Parmesan and 3 pounds of mozzarella cheese.  Is she starts with $50 cash, how much money, in dollars, will she have left to buy meat? [[END]],
[[REAONING]]: The cost for buying 2 pounds of parmesan cheese is $11 x 2 = $<<11*2=22>>22.\nThe cost for buying 3 pounds of mozzarella cheese is $6 x 3 = $<<6*3=18>>18.\nThe total amount spent for the 2 kinds of cheese is $22 + $18 = $<<22+18=40>>40.\nSo, Amor still has $50 - $40 = $<<50-40=10>>10 left to buy for meat.[[END]],
[[ANSWER]]: 10[[END]]

Please follow the example to solve the following math problem:
Here is the math problem you need to solve:
[[QUESTION]]: {question} [[END]],

Please think step by step to sovle this problem, and provide the answer in the following format, the answer should be a number:
[[REASONING]]: <Your reasoning>[[END]],
[[ANSWER]]: <Your answer>[[END]]
"""
USER_PROMPT_GSM_RFL="""
Here is the math problem you need to solve:
[[QUESTION]]: {question} [[END]],

You have tried to solve this problem before, the answer you provided is:
[[REASONING]]: {reasoning}[[END]],
[[ANSWER]]: {answer}[[END]]

But the answer is not correct, please try again.

Please reflect on your previous answer, and think step by step to sovle this problem, and provide the answer in the following format, the answer should be a number:
[[REFLECTION]]: <Your reflection>[[END]],
[[REASONING]]: <Your reasoning>[[END]],
[[ANSWER]]: <Your answer>[[END]]
"""
USER_PROMPT_GSM_MOD_RFL="""
Here is the math problem you need to solve:
[[QUESTION]]: {question} [[END]],

You have tried to solve this problem before, the answer you provided is:
[[REASONING]]: {reasoning}[[END]],
[[ANSWER]]: {answer}[[END]]

We don't know whether the answer is correct or not, please review your answer, if it is correct, please provide the answer in the following format; if it is not correct, please rethink and provide the answer in the following format.

Please reflect on your previous answer, and think step by step to sovle this problem, and provide the answer in the following format, the answer should be a number:
[[REFLECTION]]: <Your reflection>[[END]],
[[REASONING]]: <Your reasoning>[[END]],
[[ANSWER]]: <Your answer>[[END]]
"""
