import json
import re
import os
import requests
from typing import Dict, List, Tuple
from search.bing_search import bing_web_search, bing_web_search_async, fetch_page_content_async, extract_relevant_info, extract_snippet_with_context
from utils.agent_utils import (
    extract_between, 
)
from utils.generate_utils import generate_response
from prompts.agent_prompts import (
    get_code_o1_instruction, 
    get_code_o1_instruction_withmemory, 
    get_search_o1_instruction, 
    get_search_o1_instruction_withmemory,
    get_multimodal_o1_instruction,
    get_multimodal_o1_instruction_withmemory,
    get_webthinker_instruction_withmemory,
    get_webthinker_instruction,
)
from prompts.naive_rag_prompts import get_naive_rag_instruction, get_naive_rag_instruction_withmemory
from utils.agent_utils import (
    run_python_tool,
    run_multimodal_tool,
    run_search_tool,
    run_deepsearch_tool,
    run_searcho1_tool
)



TOOL_MAP = {
    'python': {
        'begin_call': '<|begin_code_call|>',
        'end_call': '<|end_code_call|>',
        'begin_result': '<|begin_code_result|>',
        'end_result': '<|end_code_result|>',
        'description': "Can execute Python code",
        'short_name': 'python executor',
        'func': run_python_tool
    },
    'multimodal': {
        'begin_call': '<|begin_multimodal_call|>',
        'end_call': '<|end_multimodal_call|>',
        'begin_result': '<|begin_multimodal_result|>',
        'end_result': '<|end_multimodal_result|>',
        'description': "Can analyze image, video, or audio and answer related questions.",
        'short_name': 'multimodal analyzer',
        'func': run_multimodal_tool
    },
    'search': {
        'begin_call': '<|begin_search_query|>',
        'end_call': '<|end_search_query|>',
        'begin_result': '<|begin_search_result|>',
        'end_result': '<|end_search_result|>',
        'description': "Performs single-step web search and summarizes first-page results.",
        'short_name': 'web searcher',
        'func': run_search_tool
    },
    'deepsearch': {
        'begin_call': '<|begin_search_query|>',
        'end_call': '<|end_search_query|>',
        'begin_result': '<|begin_search_result|>',
        'end_result': '<|end_search_result|>',
        'description': "Performs iterative searches, analyzes multiple sources, and synthesizes comprehensive answers through sequential browsing.",
        'short_name': 'deep web explorer',
        'func': run_deepsearch_tool
    },
}

class CustomO1Agent:
    def __init__(
        self,
        agent_name = 'custom_agent',
        max_action_num = 10,
        max_tokens = 60000,
        tool_list = [],
        tool_map = {},
        prompt_func = None, # mapping input data to prompt
        global_vars = {},
        use_boxed = False
    ):
        self.agent_name = agent_name
        self.max_action_num = max_action_num
        self.max_tokens = max_tokens
        self.tool_map = tool_map
        self.description = 'Capable of using multiple specialized tools to solve problems, including web search, code execution, and multimodal analysis.'
        self.use_boxed = use_boxed
        
        if tool_map == {}:
            if tool_list == []:
                tool_map = TOOL_MAP
            else:
                for tool_name in tool_list:
                    tool_map[tool_name] = TOOL_MAP[tool_name]
            self.tool_map = tool_map
        
        tool_names = [k for k in self.tool_map.keys()]
        print(f'Loading {len(tool_names)} tools for {self.agent_name}: {tool_names}')

        self.stop_tokens = [v['end_call'] for v in tool_map.values()]

        if prompt_func is None:
            prompt_func = self.get_flexible_tool_prompt(tool_map)
        self.prompt_func = prompt_func
        self.global_vars = global_vars   
    
    def get_flexible_tool_prompt(self, tool_map) -> str:
        tool_instructions = ""
        for tool_name, tool_info in tool_map.items():
            tool_instructions += (
                f"Tool: {tool_info['short_name']}\n"
                f"Description: {tool_info['description']}\n"
                f"Usage: write {tool_info['begin_call']} your input {tool_info['end_call']}\n"
                f"You will receive results in the format:\n"
                f"  {tool_info['begin_result']} ... result ... {tool_info['end_result']}\n\n"
            )
        
        if self.use_boxed:
            box_message = 'You should provide your final answer in the format \\boxed{{YOUR_ANSWER}}.\n\n'
        else:
            box_message = ''
        prompt_nomemory =  (
            "You are a reasoning assistant with the ability to call tools to help "
            "you answer the user's question accurately. You have special tools:\n\n"
            f"{tool_instructions}"
            "After calling the tool, the system will provide you with result.\n\n"
            "You can repeat the tool calling process multiple times if necessary. The maximum number of tool calling attempts is limited to {max_action_num}.\n\n"
            "Once you have all the information you need, continue your reasoning.\n\n"
            "Remember:\n"
            "- Use the correct tool call to get the information you need.\n"
            "- When done tool calling, continue your reasoning.\n\n"
            'Please answer the following question step by step.'
            f'{box_message}'
            'Question:\n{task}\n\n'
        )

        prompt_withmemory =  (
            "You are a reasoning assistant with the ability to call tools to help "
            "you answer the user's question accurately. You have special tools:\n\n"
            f"{tool_instructions}"
            "After calling the tool, the system will provide you with result.\n\n"
            "You can repeat the tool calling process multiple times if necessary. The maximum number of tool calling attempts is limited to {max_action_num}.\n\n"
            "Once you have all the information you need, continue your reasoning.\n\n"
            "You are provided with the current memory, which contains the information you have already known. Check the current memory before making tool calls.\n\n"
            "Remember:\n"
            "- Use the correct tool call to get the information you need.\n"
            "- When done tool calling, continue your reasoning.\n\n"
            'Please answer the following question step by step.'
            f'{box_message}'
            "Current Memory:\n{current_memory}\n\n"
            'Question:\n{task}\n\n'
        )


        def prompt_func(task_info, current_memory, max_action_num=10):
            if current_memory == '':
                return prompt_nomemory.format(task=task_info, max_action_num=max_action_num)
            else:
                return prompt_withmemory.format(task=task_info, current_memory=current_memory, max_action_num=max_action_num)

        return prompt_func

    async def run(self, input_data: Dict):
        seq = {
            'process_name': self.agent_name,
            'output': '',
            'finished': False,
            'finished_reason': 'Not finished',
            'process': [],
            'action_count': 0,
            'total_tokens': 0
        }
        executed_calls = []
        task_info = input_data.get('task_info', '')
        current_memory = input_data.get('current_memory', '')
        base_model = input_data.get('base_model', 'client')
        
        agent_prompt = self.prompt_func(task_info=task_info, current_memory=current_memory, max_action_num=self.max_action_num)
        agent_prompt = [{"role": "user", "content": agent_prompt}]
        enable_thinking = self.global_vars[base_model]['params'].get('enable_thinking', True)
        if 'qwen3' in self.global_vars[base_model]['model_name'].lower():
            agent_prompt = self.global_vars[base_model]['tokenizer'].apply_chat_template(agent_prompt, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking)
        else:
            agent_prompt = self.global_vars[base_model]['tokenizer'].apply_chat_template(agent_prompt, tokenize=False, add_generation_prompt=True)
        seq['prompt'] = agent_prompt
        seq['total_tokens'] = len(agent_prompt.split())
        input_data['seq'] = seq
        
        while not seq['finished'] and seq['action_count'] < self.max_action_num and seq['total_tokens'] < self.max_tokens:
            response = await generate_response(
                client=self.global_vars[base_model]['instance'],
                tokenizer=self.global_vars[base_model]['tokenizer'],
                model_name=self.global_vars[base_model]['model_name'],
                prompt=seq['prompt'],
                semaphore=self.global_vars['semaphore'],
                max_tokens=self.global_vars['max_tokens'],
                generation_params=self.global_vars[base_model]['params'],
                stop=self.stop_tokens,
                generate_mode="completion"
            )
            response = response.replace('</think>', '').rstrip('\n').rstrip()
            tokens_this_response = len(response.split())
            seq['total_tokens'] += tokens_this_response
            seq['output'] += response
            seq['prompt'] += response

            # 解析response
            if not any([response.endswith(stop_token) for stop_token in self.stop_tokens]):
                seq['finished'] = True
                seq['finished_reason'] = 'Normal finished'
                break

            for tool_name, tool_info in self.tool_map.items():
                begin_call_token, end_call_token = tool_info['begin_call'], tool_info['end_call']
                begin_result_token, end_result_token = tool_info['begin_result'], tool_info['end_result']
                if not response.endswith(end_call_token):
                    continue
                func_call = extract_between(response, begin_call_token, end_call_token)
                break

            # 检测func call
            seq['action_count'] += 1
            if func_call is None:
                func_result = f'\n\nWrong {tool_name} call format. You should put your call between {begin_call_token} and {end_call_token}.\n\n'
                func_output = {}
            elif func_call in executed_calls:
                func_result = f'You have already executed this tool with this call argument. Please use previous found information.'
            else:
                executed_calls.append(func_call)
                tool_func = tool_info['func']
                # func_output: dict, func_result: str
                input_data['func_call'] = func_call
                input_data['reasoning_chain'] = seq['output']
                func_output = await tool_func(data=input_data, global_vars = self.global_vars)
                func_result = func_output.get('result', '')

            seq['process'].append({
                'raw_response': response,
                'func_call': func_call,
                'func_result': func_result
            })

            append_text = f'\n\n{begin_result_token}{func_result}{end_result_token}\n\n'
            seq['prompt'] += append_text
            seq['output'] += append_text
            seq['total_tokens'] += len(append_text.split())
        
        if not seq['finished']:
            if seq['action_count'] >= self.max_action_num:
                warning_message = f'\n\n<warning>You have reached the call limit. Please finish your reasoning and directly give your final answer.</warning>\n\n</think>'
            elif seq['total_tokens'] >= self.max_tokens:
                warning_message = f'\n\n<warning>You have reached the max generation length. Please finish your reasoning and directly give your final answer.</warning>\n\n</think>'
            else:
                warning_message = f'\n\n</think>'
            seq['prompt'] += warning_message
            seq['output'] += warning_message
            response = await generate_response(
                client=self.global_vars[base_model]['instance'],
                tokenizer=self.global_vars[base_model]['tokenizer'],
                model_name=self.global_vars[base_model]['model_name'],
                prompt=seq['prompt'],
                semaphore=self.global_vars['semaphore'],
                max_tokens=self.global_vars['max_tokens'],
                generation_params=self.global_vars[base_model]['params'],
                generate_mode="completion"
            )
            seq['prompt'] += response
            seq['output'] += response
        return seq


class NaiveRAGAgent:
    def __init__(
        self,
        agent_name = 'Naive-RAG-Agent',
        max_tokens = 60000,
        global_vars = {},
        use_boxed = False
    ):
        self.agent_name = agent_name
        self.max_tokens = max_tokens
        self.global_vars = global_vars   
        self.use_boxed = use_boxed
        self.description = 'Performs single-step web search and summarizes first-page results. Optimized for quick fact-checking, web page reading and basic information retrieval, also suitable for web page extracting.'

    async def run(self, input_data: Dict):
        seq = {
            'process_name': self.agent_name,
            'output': '',
            'finished': False,
            'finished_reason': 'Not finished',
            'process': [],
            'total_tokens': 0
        }
        task_info = input_data.get('task_info', '')
        current_memory = input_data.get('current_memory', '')
        base_model = input_data.get('base_model', 'client')

        search_results = await run_search_tool({'func_call': task_info}, self.global_vars)
        seq['search_infos'] = search_results

        if current_memory == '':
            prompt = get_naive_rag_instruction(task_info, search_results, self.use_boxed)
        else:
            prompt = get_naive_rag_instruction_withmemory(task_info, search_results, current_memory, self.use_boxed)
        seq['input_prompt'] = prompt

        output = await generate_response(
            prompt=prompt,
            client=self.global_vars[base_model]['instance'],
            tokenizer=self.global_vars[base_model]['tokenizer'],
            model_name=self.global_vars[base_model]['model_name'],
            generation_params=self.global_vars[base_model]['params'],
            semaphore=self.global_vars['semaphore'],
            generate_mode='chat',
            enable_thinking=True,
        )
        seq['output'] = output
        return seq


class CodeO1(CustomO1Agent):
    def __init__(self, *args, **kwargs):
        kwargs['tool_map'] = {
            'python': {
                'begin_call': '<|begin_code_call|>',
                'end_call': '<|end_code_call|>',
                'begin_result': '<|begin_code_result|>',
                'end_result': '<|end_code_result|>',
                'description': "Can execute Python code",
                'short_name': 'python executor',
                'func': run_python_tool
            },
        }
        super().__init__(*args, **kwargs)
        self.agent_name = 'Code-Agent'
        self.prompt_func = self.get_code_o1_prompt
        self.description = 'Can write **python code** to complete tasks including calculation, programming, special file reading, data analysis, parsing web link, etc. Not suitable for web search and information acquisition.'

        self.use_boxed = kwargs.get('use_boxed', False)

    def get_code_o1_prompt(self, task_info, current_memory, max_action_num=10):
        if current_memory == '':
            prompt = get_code_o1_instruction(task_info, max_action_num, self.use_boxed)
        else:
            prompt = get_code_o1_instruction_withmemory(task_info, current_memory, max_action_num, self.use_boxed)
        return prompt

class SearchO1(CustomO1Agent):
    def __init__(self, *args, **kwargs):
        kwargs['tool_map'] = {
            'search': {
                'begin_call': '<|begin_search_query|>',
                'end_call': '<|end_search_query|>',
                'begin_result': '<|begin_search_result|>',
                'end_result': '<|end_search_result|>',
                'description': "Can search the web for information and get the web page information",
                'short_name': 'web searcher',
                'func': run_searcho1_tool
            }
        }
        super().__init__(*args, **kwargs)
        self.agent_name = 'search_o1_agent'
        self.prompt_func = self.get_search_o1_prompt
        self.description = 'Capable of autonomous thinking and multi-step web exploration. Performs iterative searches, analyzes multiple sources, and synthesizes comprehensive answers through sequential browsing. Best for complex tasks requiring deep analysis.'

        self.use_boxed = kwargs.get('use_boxed', False)

    def get_search_o1_prompt(self, task_info, current_memory, max_action_num=10):
        if current_memory == '':
            prompt = get_search_o1_instruction(task_info, max_action_num, self.use_boxed)
        else:
            prompt = get_search_o1_instruction_withmemory(task_info, current_memory, max_action_num, self.use_boxed)
        return prompt
    

class MultiModalO1(CustomO1Agent):
    def __init__(self, *args, **kwargs):
        kwargs['tool_map'] = {
            'multimodal': {
                'begin_call': '<|begin_multimodal_call|>',
                'end_call': '<|end_multimodal_call|>',
                'begin_result': '<|begin_multimodal_result|>',
                'end_result': '<|end_multimodal_result|>',
                'description': "Can analyze image, video, or audio and answer related questions",
                'short_name': 'multimodal analyzer',
                'func': run_multimodal_tool
            },
        }
        super().__init__(*args, **kwargs)
        self.agent_name = 'Multimodal-Agent'
        self.prompt_func = self.get_multimodal_o1_prompt
        self.description = 'Can use multimodal understanding tools to assist in reasoning and problem-solving, including image, video, and audio. But this agent does not have the ability to explore and obtain webpage information, and needs to provide specific image or video URLs or local file paths to work.'

        self.use_boxed = kwargs.get('use_boxed', False)

    def get_multimodal_o1_prompt(self, task_info, current_memory, max_action_num=10):
        if current_memory == '':
            prompt = get_multimodal_o1_instruction(task_info, max_action_num, self.use_boxed)
        else:
            prompt = get_multimodal_o1_instruction_withmemory(task_info, current_memory, max_action_num, self.use_boxed)
        return prompt


class WebThinker(CustomO1Agent):
    def __init__(self, *args, **kwargs):
        kwargs['tool_map'] = {
            'deepsearch': {
                'begin_call': '<|begin_search_query|>',
                'end_call': '<|end_search_query|>',
                'begin_result': '<|begin_search_result|>',
                'end_result': '<|end_search_result|>',
                'description': "Can search the web multiple times, navigate pages, and extract relevant information",
                'short_name': 'deep web explorer',
                'func': run_deepsearch_tool
            },
        }
        super().__init__(*args, **kwargs)
        self.agent_name = 'Web-Thinker'
        self.prompt_func = self.get_webthinker_prompt
        self.description = 'Capable of autonomous thinking and multi-step web exploration. Performs iterative searches, analyzes multiple sources, and synthesizes comprehensive answers through sequential browsing. Best for complex tasks requiring deep analysis.'

        self.use_boxed = kwargs.get('use_boxed', False)

    def get_webthinker_prompt(self, task_info, current_memory, max_action_num=10):
        if current_memory == '':
            prompt = get_webthinker_instruction(task_info, max_action_num, self.use_boxed)
        else:
            prompt = get_webthinker_instruction_withmemory(task_info, current_memory, max_action_num, self.use_boxed)
        return prompt
    
