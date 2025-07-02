def get_common_prompt(item, model_name=None):
    question = item['Question']
    if model_name == 'qwq':
        user_prompt = (
            'Please answer the following question step by step.'
            'You should provide your final answer in the format \\boxed{YOUR_ANSWER}.\n\n'
            'Please ensure the truthfulness of the answer. If you find that there is no valid information in the document, please directly state that there is no valid information.'
            f'Question:\n{question}\n\n'
        )
    elif model_name == 'dpsk':
        user_prompt = (
            'Please answer the following question.\n\n'
            'Provide your final answer in the format **ANSWER: {YOUR_ANSWER}**.\n\n'
            f'Question:\n{question}\n\n'
        )
    else:
        user_prompt = (
            'Please answer the following question. You should think step by step to solve it.\n\n'
            'Provide your final answer in the format \\boxed{YOUR_ANSWER}.\n\n'
            f'Question:\n{question}\n\n'
        )
    return user_prompt

def get_code_prompt(item, model_name=None):
    question = item['Question']
    question_title = item.get('question_title', '')
    if model_name == 'qwq':
        user_prompt = (
            'Generate a correct Python program that passes all tests for the given problem. '
            'You should provide your final code within a Python code block using triple backticks (```python\n'
            'YOUR_CODE\n'
            '```).\n\n'
            f'Problem Title: {question_title}\n\n'
            f'Problem Statement:\n{question}\n\n'
        )
    else:
        user_prompt = (
            'You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests. '
            f'You should think step by step to solve it.\n\nQuestion:\n{question}\n\n'
            'Read the inputs from stdin solve the problem and write the answer to stdout (do not directly test on the sample inputs). Enclose your code within delimiters as follows.\n\n'
            "```python\n# YOUR CODE HERE\n```\n\n"
        )
    return user_prompt