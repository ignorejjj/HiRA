import sys
sys.path.append("..")
import re
from typing import Dict, List, Tuple
import os
import subprocess
import asyncio
import aiohttp
from urllib.parse import urlparse, unquote
from typing import Dict, List
import json

from search.bing_search import (
    bing_web_search_async, 
    fetch_page_content_async, 
    extract_relevant_info, 
    extract_snippet_with_context
)
from generate_utils import generate_response
from prompts.agent_prompts import *
from prompts.naive_rag_prompts import *

ERROR_INDICATORS = [
    'limit exceeded',
    'Error fetching',
    'Account balance not enough',
    'Invalid bearer token',
    'HTTP error occurred',
    'Error: Connection error occurred',
    'Error: Request timed out',
    'Unexpected error',
    'Please turn on Javascript',
    'Enable JavaScript',
    'port=443',
    'Please enable cookies',
]

INVALID_SEARCH_QUERIES = [
    "and end with",
    "search query",
    "query",
    "your query here",
    "your query",
    "your search query",
]

def sanitize_filename(filename: str) -> str:
    filename = re.sub(r'[^\w\-\.]', '_', filename)
    if len(filename) > 50:
        name, ext = os.path.splitext(filename)
        filename = name[:50] + ext
    return filename

def get_filename_from_url(url: str) -> str:
    parsed = urlparse(url)
    path = unquote(parsed.path) 
    filename = os.path.basename(path)
    
    if not filename or filename == '/':
        parts = [part for part in parsed.path.split('/') if part]
        filename = f"{parsed.netloc.split('.')[0]}_{parts[-1] if parts else 'resource'}"
    
    return sanitize_filename(filename)


async def download_resource(url: str, file_cache: Dict = {}, cache_dir: str = '') -> str:
    try:
        if url in file_cache:
            return os.path.join(cache_dir, file_cache[url]), ''
        
        os.makedirs(cache_dir, exist_ok=True)
        
        base_filename = get_filename_from_url(url)
        output_name = os.path.splitext(base_filename)[0]
        extension = os.path.splitext(base_filename)[1].lower()
        error_message = ''

        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.webm'}
        is_video = extension in video_extensions

        if is_video:
            try:
                process = await asyncio.create_subprocess_exec(
                    'you-get', '-o', cache_dir, '-O', output_name, url,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                
                stdout, stderr = await process.communicate()
                stderr = stderr.decode('utf-8')
                
                if stderr:
                    error_message = 'Have error in download process: ' + stderr
                
                if process.returncode == 0:
                    downloaded_files = [f for f in os.listdir(cache_dir) if f.startswith(output_name)]
                    if downloaded_files:
                        file_cache[url] = downloaded_files[0]
                        return os.path.join(cache_dir, downloaded_files[0]), error_message
                    else:
                        error_message = 'No file downloaded.'
                        return None, error_message
                
                error_message = f'Download failed with code: {process.returncode}'
                debug_process = await asyncio.create_subprocess_exec(
                    'you-get', '-o', cache_dir, '-O', output_name, '--debug', url,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                _, debug_stderr = await debug_process.communicate()
                error_message = 'Debug mode error: ' + debug_stderr.decode('utf-8')
                return None, error_message
                
            except Exception as e:
                error_message = f'You-get download error: {str(e)}'
                return None, error_message
        
        else:
            import aiohttp
            import asyncio
            
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(url) as response:
                        if response.status == 200:
                            content_type = response.headers.get('content-type', '')
                            if not extension:
                                ext_map = {
                                    'image/jpeg': '.jpg',
                                    'image/png': '.png',
                                    'image/gif': '.gif',
                                    'image/webp': '.webp',
                                    'audio/mpeg': '.mp3',
                                    'audio/wav': '.wav',
                                    'audio/ogg': '.ogg'
                                }
                                extension = ext_map.get(content_type, '')
                                output_name = output_name + extension
                            
                            file_path = os.path.join(cache_dir, output_name)
                            async with aiohttp.ClientSession() as session:
                                async with session.get(url) as response:
                                    if response.status == 200:
                                        content = await response.read()
                                        with open(file_path, 'wb') as f:
                                            f.write(content)
                                        
                                        file_cache[url] = output_name
                                        return file_path, ''
                                    else:
                                        error_message = f'HTTP error: {response.status}'
                                        return None, error_message
                        else:
                            error_message = f'HTTP error: {response.status}'
                            return None, error_message
                            
            except Exception as e:
                error_message = f'Request download error: {str(e)}'
                return None, error_message
    
    except Exception as e:
        error_message = f'General error: {str(e)}'
        return None, error_message


def get_file_type(file_path: str) -> str:
    try:
        info_cmd = ['you-get', '-i', file_path]
        result = subprocess.run(info_cmd, capture_output=True, text=True)
        info_text = result.stdout.lower()
        
        if 'video' in info_text:
            return 'video'
        elif 'audio' in info_text:
            return 'audio'
        elif any(ext in info_text for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.webp']):
            return 'image'
        
        ext = os.path.splitext(file_path)[1].lower()
        if ext in {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}:
            return 'image'
        elif ext in {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.gif'}:
            return 'video'
        elif ext in {'.mp3', '.wav', '.ogg', '.flac', '.m4a'}:
            return 'audio'
            
    except Exception as e:
        print(f"Error in get_file_type: {str(e)}")
    
    return None

def detect_modality(path):
    extension = os.path.splitext(path)[1].lower()
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}
    audio_extensions = {'.mp3', '.wav', '.ogg', '.flac', '.m4a', '.aac'}
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm'}
    
    try:
        if not os.path.exists(path):
            return None
            
        if extension in image_extensions:
            return 'image'
        elif extension in audio_extensions:
            return 'audio'
        elif extension in video_extensions:
            return 'video'
        else:
            return None
            
    except Exception as e:
        print(f"Error in detect_modality: {str(e)}")
        return None
    
async def run_local_omni(omni_api_url, omni_api_key, detected_modality, data_path, question):
    mm_input_dict = {'type': detected_modality, detected_modality: data_path}

    conversation = [
    {
        "role": "system",
        "content": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech.",
    },
    {
        "role": "user",
        "content": [
            mm_input_dict,
            {"type": "text", "text": f"Please provide a detailed analysis and answer to the following question based on the input content. Do not output irrelevant content.\n Question:\n{question}"}
        ],
    }]

    payload = {
        "conversation": conversation,
        "use_audio_in_video": True
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(omni_api_url, json=payload) as response:
            data = await response.json()
            output = data['text']
    return output


async def run_url_omni(omni_api_url,omni_api_key, detected_modality, data_path, question):
    try:
        def encode_file(file_path):
            import base64
            with open(file_path, "rb") as file:
                return base64.b64encode(file.read()).decode("utf-8")
                
        base64_file = encode_file(data_path)
        
        from openai import OpenAI
        client = OpenAI(
            api_key=omni_api_key,
            base_url=omni_api_url,
        )
        
        extension = os.path.splitext(data_path)[1].lower()
        mime_types = {
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.png': 'image/png',
        '.gif': 'image/gif',
        '.bmp': 'image/bmp',
        '.webp': 'image/webp',
        '.mp4': 'video/mp4',
        '.avi': 'video/x-msvideo',
        '.mov': 'video/quicktime',
        '.mp3': 'audio/mpeg',
        '.wav': 'audio/wav',
        '.ogg': 'audio/ogg',
        '.flac': 'audio/flac'
        }
        mime_type = mime_types.get(extension,'')
        if detected_modality == 'video':
            content = {
                    "type": "video_url",
                    "video_url": {"url": f"data:;base64,{base64_file}"},
            }
        elif detected_modality == 'image':
            content = {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime_type};base64,{base64_file}"},
            }
        elif detected_modality == 'audio':
            content = {
                    "type": "input_audio",
                    "input_audio": {
                        "data": f"data:{mime_type};base64,{base64_file}",
                        "format": extension,
                    },
            }
        else:
            return 'Unsupported modality: {}! Only image, video, and audio are supported.'.format(detected_modality)
            
        messages = [{
            "role": "user",
            "content": [
                content,
                {"type": "text", "text": question}
            ]
        }]
        print(f"call omni: {messages}")
        completion = client.chat.completions.create(
            model="qwen-omni-turbo-2025-03-26",
            messages=messages,
            modalities=["text"],  
            stream=True,
            stream_options={"include_usage": True}
        )
        
        full_response = []
        for chunk in completion:
            if chunk.choices:
                delta = chunk.choices[0].delta
                if delta.content:
                    full_response.append(delta.content)
        
        return "".join(full_response)
        
    except Exception as e:
        error_msg = f"Error in run_url_omni: {str(e)}"
        print(error_msg)
        return error_msg
    

async def parse_query_plan(task_info: str, response: str) -> List[str]:
    try:
        match = re.search(r'\{.*\}', response, re.DOTALL)
        if match:
            json_content = json.loads(match.group())
            if 'query_plan' in json_content:
                query_plan = json_content['query_plan'][:3]  
            else:
                query_plan = []
            if 'urls' in json_content:
                urls = json_content['urls'][:3]
            else:
                urls = re.findall(r'https?://(?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b(?:[-a-zA-Z0-9()@:%_\+.~#?&//=]*)', task_info)
                urls = urls[:3]
            return query_plan, urls
    except:
        pass

    return [], []



def extract_between(text, start_marker, end_marker):
    try:
        pattern = re.escape(end_marker[::-1]) + r"(.*?)" + re.escape(start_marker[::-1])
        matches = re.findall(pattern, text[::-1], flags=re.DOTALL)
        if matches:
            text = matches[0][::-1].strip()
            if end_marker in text:
                return None
            else:
                return text
        return None
    except Exception as e:
        print(f"---Error:---\n{str(e)}")
        print(f"-------------------")
        return None



def extract_deep_web_explorer_output(output):
    pattern_info = "**Final Information"
    if "</think>\n" in output:
        extracted_text = output.split("</think>\n")[-1].split("<|begin_click_link|>")[0].replace(pattern_info, "").strip(':**').strip('\n').strip("```").strip()  # 提取</think>后面的内容
    elif pattern_info in output:
        extracted_text = output.split(pattern_info)[-1].split("<|begin_click_link|>")[0].strip('\n').strip(':**').strip("```").strip()  # 提取**Final Information**后面的内容
    else:
        extracted_text = '\n'.join(output.strip().replace("</think>\n", "").replace("\n\n", "\n").split('\n')[-5:])  # 若没提取到，只保留最后5行
    return extracted_text


async def get_code_call_result(code_snippet: str, timeout: int = 30) -> str:
    import asyncio
    import subprocess

    code_snippet = code_snippet.replace('```python', '').replace('```', '')
    
    try:
        process = await asyncio.create_subprocess_exec(
            'python', '-c', code_snippet,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=timeout
            )
        except asyncio.TimeoutError:
            process.kill()
            return f"Code execution timeout after {timeout} seconds"
            
        if process.returncode != 0:
            message = stderr.decode('utf-8').strip()
            message = f"Code execution error:\n{message}"
        else:
            message = stdout.decode('utf-8').strip()
            message = f"Code execution successful. \nCode execution result:\n{message}"
        return message
        
    except Exception as e:
        return f"Code execution error: {str(e)}"
    finally:
        if 'process' in locals() and process.returncode is None:
            process.kill()

async def get_code_call_result_sandbox(code_snippet: str):
    url = "http://0.0.0.0:1000/execute"
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json={"call": code_snippet, "timeout": 30}) as response:
            output = await response.json()
            result = output.get('result', '')
            if result == '' and output.get('error', None) is None:
                result = 'Code exectued successful but no output.'
            elif result == '':
                result = 'Code execution error: ' + output.get("error", "Unknown error")

            if result is None:
                result = 'Code execution error.'
            if len(result) > 2000:
                result = 'Code execution result is too long. The cutted result is: ' + result[:2000]
            return result


async def run_python_tool(data:Dict = {}, global_vars:Dict = {}):
    func_call = data.get('func_call', '')
    path_map = data.get('path_map', {})
    python_prefix_path = global_vars.get('python_prefix_path','')

    try:
        if path_map is not None:
            for k,v in path_map.items():
                func_call = func_call.replace(k, v)
        func_call = func_call.replace('```python', '').replace('```', '')
        func_call_result = await get_code_call_result(func_call)
    except Exception as e:
        func_call_result = f"Error executing code: {e}"
    func_call_result = func_call_result.replace(python_prefix_path, '')

    return {'result': func_call_result}

async def get_multimodal_call_result(call_str, file_cache, api_url, api_key):
    call_str = call_str.strip('\n')
    call_args_list = call_str.split("\n")
    if len(call_args_list) != 2:
        return 'The input format is not correct.'
    data_path = call_args_list[0].replace('data:','').strip()
    question = call_args_list[1].replace('question:','').strip()
    
    if not os.path.exists(data_path):
        if data_path.startswith('http'):
            new_data_path, error_message = await download_resource(data_path, file_cache=file_cache)
            if new_data_path is not None:
                data_path = new_data_path
            else:
                return error_message
        else:
            return 'The data path is not exists! Please give a correct data path.'
        
    detected_modality = detect_modality(data_path)
    if detected_modality is None:
        return 'The input file is not a image/video/audio. Please provide a valid file path.'
    
    try:
        #output = await run_local_omni(detected_modality, data_path, question)
        output = await run_url_omni(api_url, api_key, detected_modality, data_path, question)
    except:
        output = 'Fail to call multimodal tool.'
    return output

async def run_multimodal_tool(data:Dict = {}, global_vars:Dict = {}):
    func_call = data.get('func_call', '')
    file_cache = global_vars.get('file_cache', {})
    path_map = data.get('path_map', {})
    api_url = global_vars.get("omni_api_url", "")
    api_key = global_vars.get("omni_api_key", "")

    try:
        if path_map is not None:
            for k,v in path_map.items():
                func_call = func_call.replace(k, v)
        func_call_result = await get_multimodal_call_result(func_call, file_cache, api_url, api_key)
    except Exception as e:
        func_call_result = f"Error executing multimodal model: {e}"
    return {'result': func_call_result}


async def get_query_plan(func_call: str, aux_client, aux_tokenizer, aux_model_name, aux_generation_params, semaphore):
    plan_prompt_system, plan_prompt_user = get_query_plan_instruction(func_call)
    plan_prompt = [{"role": "system", "content": plan_prompt_system}, {"role": "user", "content": plan_prompt_user}]
    plan_prompt = aux_tokenizer.apply_chat_template(plan_prompt, tokenize=False, add_generation_prompt=True)
    plan_response = await generate_response(
        client=aux_client,  
        tokenizer=aux_tokenizer,
        prompt=plan_prompt, 
        semaphore=semaphore,
        generation_params=aux_generation_params,
        model_name=aux_model_name,
        generate_mode='completion',
        enable_thinking=False
    )
    
    sub_queries, detected_urls = await parse_query_plan(func_call, plan_response)
    if not sub_queries:  
        print("Fail to have sub query")
        sub_queries = [func_call]
    return sub_queries, detected_urls

async def get_formatted_search_result(sub_queries, detected_urls=[], search_cache={}, url_cache={}, bing_subscription_key='', bing_endpoint='', keep_links=True):
    all_results = []
    search_flag_list = []
    for sub_query in sub_queries:
        sub_query = str(sub_query).strip().strip("\"")
        if sub_query in search_cache:
            results = search_cache[sub_query]
            search_flag = 'use cache'
        else:
            try:
                results = await bing_web_search_async(sub_query, bing_subscription_key, bing_endpoint)
                search_cache[sub_query] = results
                search_flag = f'success: {len(results)}'
            except Exception as e:
                print(f"Error during search query '{sub_query}': {e}")
                results = {}
                search_flag = f"error: {str(e)}"
        search_flag_list.append(search_flag)
        relevant_info = extract_relevant_info(results)[:5]  # top-5 for each sub-query
        all_results.extend(relevant_info)
        
    unique_urls = set()
    url_snippets_map = {}

    for url in detected_urls:
        url_snippets_map[url] = ''
        unique_urls.add(url)

    for info in all_results:
        url = info['url']
        snippet = info.get('snippet', "")
        unique_urls.add(url)
        url_snippets_map[url] = snippet
    

    urls_to_fetch = [url for url in unique_urls if url not in url_cache]
    try:
        contents = await fetch_page_content_async(urls_to_fetch, keep_links=keep_links)
        for url, content in contents.items():
            has_error = (any(indicator.lower() in content.lower() for indicator in ERROR_INDICATORS) and len(content.split()) < 64) or len(content) < 50 or len(content.split()) < 20
            if not has_error:
                url_cache[url] = content
    except Exception as e:
        print(f"Error fetching URLs: {e}")

    formatted_documents = ""
    pre_num = 0
    for i, url in enumerate(detected_urls):
        page_content = url_cache.get(url, "")[:15000]
        if page_content == '':
            continue
        pre_num += 1
        formatted_documents += f"**Document {i + 1}:**\n"
        formatted_documents += f"**URL:** {url}\n"
        formatted_documents += f"**Content:** {page_content}\n\n"
    if len(all_results) == 0:
        print("Naive-RAG Agent warning: No relevant information found.")
    for i, doc_info in enumerate(all_results):
        url = doc_info['url']
        snippet = doc_info.get('snippet', "")
        raw_context = url_cache.get(url, "")
        if raw_context == "":
            print(f"Naive-RAG Agent warning: No relevant information found in {url}.")
        success, context = extract_snippet_with_context(raw_context, snippet, context_chars=5000)
        if success:
            context = context
        else:
            context = raw_context[:2 * 5000]

        clean_snippet = re.sub('<[^<]+?>', '', snippet)

        formatted_documents += f"**Document {i + 1 + pre_num}:**\n"
        formatted_documents += f"**Title:** {doc_info.get('title', '').replace('<b>', '').replace('</b>', '')}\n"
        formatted_documents += f"**URL:** {url}\n"
        formatted_documents += f"**Snippet:** {clean_snippet}\n"
        formatted_documents += f"**Content:** {context}\n\n"

    if formatted_documents == '':
        formatted_documents = 'No relevant information found.'
    return formatted_documents


def format_search_results(relevant_info: List[Dict]) -> str:
    formatted_documents = ""
    for i, doc_info in enumerate(relevant_info):
        doc_info['title'] = doc_info['title'].replace('<b>','').replace('</b>','')
        doc_info['snippet'] = doc_info['snippet'].replace('<b>','').replace('</b>','')
        formatted_documents += f"***Web Page {i + 1}:***\n"
        formatted_documents += json.dumps(doc_info, ensure_ascii=False, indent=2) + "\n"
    return formatted_documents


async def run_search_tool(data:Dict = {}, global_vars:Dict = {}):
    func_call = data.get('func_call', '')
    use_query_plan = data.get('use_query_plan', True)

    search_cache = global_vars.get('search_cache', {})
    url_cache = global_vars.get('url_cache', {})
    bing_endpoint = global_vars.get('bing_endpoint', '')
    bing_subscription_key = global_vars.get('bing_subscription_key', '')
    keep_links = global_vars.get('keep_links', True)
    aux_client = global_vars['aux_client']['instance']
    aux_tokenizer = global_vars['aux_client']['tokenizer']
    aux_model_name = global_vars['aux_client']['model_name']
    aux_generation_params = global_vars['aux_client']['params']
    semaphore = global_vars['semaphore']
    
    if use_query_plan:
        sub_queries, detected_urls = await get_query_plan(func_call, aux_client, aux_tokenizer, aux_model_name, aux_generation_params, semaphore)
    else:
        sub_queries = [func_call]
        detected_urls = []
    formatted_documents = await get_formatted_search_result(sub_queries=sub_queries, detected_urls=detected_urls, search_cache=search_cache, url_cache=url_cache, bing_subscription_key=bing_subscription_key, bing_endpoint=bing_endpoint, keep_links=keep_links)
    
    return {'result': formatted_documents, 'sub_queries': sub_queries, 'detected_urls': detected_urls}


async def run_searcho1_tool(data:Dict = {}, global_vars:Dict = {}):
    func_call = data.get('func_call', '')
    reasoning_chain = data.get('reasoning_chain', '')
    use_query_plan = data.get('use_query_plan', True)

    search_cache = global_vars.get('search_cache', {})
    url_cache = global_vars.get('url_cache', {})
    bing_endpoint = global_vars.get('bing_endpoint', '')
    bing_subscription_key = global_vars.get('bing_subscription_key', '')

    aux_client = global_vars['aux_client']['instance']
    aux_tokenizer = global_vars['aux_client']['tokenizer']
    aux_model_name = global_vars['aux_client']['model_name']
    aux_generation_params = global_vars['aux_client']['params']
    semaphore = global_vars['semaphore']

    all_reasoning_steps = reasoning_chain.replace('\n\n', '\n').split("\n")
    truncated_prev_reasoning = ""
    for i, step in enumerate(all_reasoning_steps):
        truncated_prev_reasoning += f"Step {i + 1}: {step}\n\n"

    prev_steps = truncated_prev_reasoning.split('\n\n')
    if len(prev_steps) > 5:
        truncated_prev_reasoning = ''
        for i, step in enumerate(prev_steps):
            if i == 0 or i >= len(prev_steps) - 4 or '<|begin_search_query|>' in step or '<|end_search_query|>' in step or '<|begin_search_result|>' in step or '<|end_search_result|>' in step:
                truncated_prev_reasoning += step + '\n\n'
            else:
                if truncated_prev_reasoning[-len('\n\n...\n\n'):] != '\n\n...\n\n':
                    truncated_prev_reasoning += '...\n\n'
    truncated_prev_reasoning = truncated_prev_reasoning.strip('\n')

    sub_queries = [func_call]
    detected_urls = []
    formatted_documents = await get_formatted_search_result(sub_queries=sub_queries, detected_urls=detected_urls, search_cache=search_cache, url_cache=url_cache, bing_subscription_key=bing_subscription_key, bing_endpoint=bing_endpoint)

    prompt = get_searcho1_summary_instruction(func_call, truncated_prev_reasoning, formatted_documents)
    

    generation_params = aux_generation_params.copy()
    generation_params['enable_thinking'] = True
    raw_output = await generate_response(
        client=aux_client,
        tokenizer=aux_tokenizer,
        prompt=prompt,
        semaphore=semaphore,
        generation_params=generation_params,
        model_name=aux_model_name,
    )
    raw_output = raw_output.split("</think")[-1]
    pattern_info = "**Final Information"
    if "</think>\n" in raw_output:
        extracted_text = raw_output.split("</think>\n")[-1].split("<|begin_click_link|>")[0].replace(pattern_info, "").strip(':**').strip('\n').strip("```").strip()  # 提取</think>后面的内容
        extracted_text = '\n'.join(extracted_text.replace("\n\n", "\n").split('\n')[:5])  # 只保留前5行
    elif pattern_info in raw_output:
        extracted_text = raw_output.split(pattern_info)[-1].split("<|begin_click_link|>")[0].strip('\n').strip(':**').strip("```").strip()  # 提取**Final Information**后面的内容
        extracted_text = '\n'.join(extracted_text.replace("\n\n", "\n").split('\n')[:5])  # 只保留前5行
    else:
        extracted_text = '\n'.join(raw_output.strip().replace("</think>\n", "").replace("\n\n", "\n").split('\n')[-5:])  # 若没提取到，只保留最后5行
    extracted_text = extracted_text[:3000]
    
    return {'result': extracted_text, 'sub_queries': sub_queries, 'detected_urls': detected_urls}


async def generate_deep_web_explorer(
    search_query: str,
    document: str,
    search_intent: str,
    global_vars: Dict
) -> Tuple[str, List[Dict], str]:
    """
    Generate deep web exploration with multiple search and click operations
    Returns the output, list of interaction records, and initial prompt
    """
    client = global_vars['client']['instance']
    tokenizer = global_vars['client']['tokenizer']
    model_name = global_vars['client']['model_name']
    aux_client = global_vars['aux_client']['instance']
    aux_tokenizer = global_vars['aux_client']['tokenizer']
    aux_model_name = global_vars['aux_client']['model_name']
    semaphore = global_vars['semaphore']
    model_params = global_vars['client']['params']
    aux_model_params = global_vars['aux_client']['params']

    enable_thinking = model_params.get('enable_thinking', True)

    BEGIN_SEARCH_QUERY = "<|begin_search_query|>"
    END_SEARCH_QUERY = "<|end_search_query|>"
    BEGIN_SEARCH_RESULT = "<|begin_search_result|>"
    END_SEARCH_RESULT = "<|end_search_result|>"

    BEGIN_CLICK_LINK = "<|begin_click_link|>"
    END_CLICK_LINK = "<|end_click_link|>"
    BEGIN_CLICK_RESULT = "<|begin_click_result|>"
    END_CLICK_RESULT = "<|end_click_result|>"

    search_cache = global_vars.get('search_cache', {})
    url_cache = global_vars.get('url_cache', {})
    bing_endpoint = global_vars.get('bing_endpoint', '')
    bing_subscription_key = global_vars.get('bing_subscription_key', '')
    top_k = global_vars['max_search_limit']
    use_jina = global_vars.get('use_jina', False)
    jina_api_key = global_vars.get('jina_api_key', '')
    keep_links = global_vars.get('keep_links', True)


    prompt = get_deep_web_explorer_instruction(search_query=search_query, search_intent=search_intent, search_result=document)
    output = ""
    original_prompt = ""
    total_tokens = len(prompt.split())
    MAX_TOKENS = 60000
    MAX_INTERACTIONS = 10
    clicked_urls = set()
    executed_search_queries = set()
    total_interactions = 0
    finished = False
    first_generation = True
    
    messages = [{"role": "user", "content": prompt}]
    if 'qwen3' in model_name.lower():
        formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking)
    else:
        formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    while True:
        response = await generate_response(
            client=client,
            tokenizer=tokenizer,
            model_name=model_name,
            prompt=formatted_prompt,
            semaphore=semaphore,
            max_tokens=80000,
            generate_mode="chat" if first_generation else "completion",
            stop=[END_SEARCH_QUERY, END_CLICK_LINK],
            generation_params=model_params,
            enable_thinking=enable_thinking
        )

        if first_generation:
            original_prompt = formatted_prompt
            prompt = formatted_prompt
        
        output += response.replace('</think>\n','')
        total_tokens = len(prompt.split()) + len(response.split())
        first_generation = False

        if total_tokens >= MAX_TOKENS or total_interactions >= MAX_INTERACTIONS:
            break

        # Check for search query
        if response.rstrip().endswith(END_SEARCH_QUERY):
            new_query = extract_between(response, BEGIN_SEARCH_QUERY, END_SEARCH_QUERY)
            total_interactions += 1
            if new_query is None or END_SEARCH_QUERY in new_query or len(new_query) <= 5 or new_query in INVALID_SEARCH_QUERIES:
                continue
            if new_query:
                if new_query in executed_search_queries:
                    # If search query was already executed, append message and continue
                    search_result = f"\n{BEGIN_SEARCH_RESULT}\nYou have already searched for this query. Please use the previously found information.\n{END_SEARCH_RESULT}"
                    output += search_result
                    prompt += output
                    total_tokens += len(search_result.split())
                    continue

                executed_search_queries.add(new_query)  # Add query to executed set
                
                # Execute search
                if new_query in search_cache:
                    results = search_cache[new_query]
                else:
                    try:
                        #results = bing_web_search(new_query, bing_subscription_key, bing_endpoint)
                        results = await bing_web_search_async(new_query, bing_subscription_key, bing_endpoint)
                        search_cache[new_query] = results
                    except Exception as e:
                        print(f"Error during search query '{new_query}': {e}")
                        results = {}
                print('- Searched for:', new_query)

                relevant_info = extract_relevant_info(results)[:top_k]

                formatted_documents = format_search_results(relevant_info)
                
                # Append search results
                search_result = f"\n{BEGIN_SEARCH_RESULT}\n{formatted_documents}\n{END_SEARCH_RESULT}\n"
                output += search_result
                prompt += output
                total_tokens += len(search_result.split())
                
        # Check for click link
        elif response.rstrip().endswith(END_CLICK_LINK):
            url = extract_between(response, BEGIN_CLICK_LINK, END_CLICK_LINK)
            # click_intent = extract_between(response, BEGIN_CLICK_INTENT, END_CLICK_INTENT)
            total_interactions += 1
            click_intent = await generate_response(
                client=aux_client,
                tokenizer=aux_tokenizer,
                model_name=aux_model_name,
                max_tokens=4000,
                generation_params=aux_model_params,
                prompt=get_click_intent_instruction(output),
                semaphore=semaphore,
                enable_thinking=False
            )

            if url and click_intent:
                if url in clicked_urls:
                    # If URL was already clicked, append message
                    click_result = f"\n{BEGIN_CLICK_RESULT}\nYou have already clicked this URL.\n{END_CLICK_RESULT}\n\nOkay,"
                    output += click_result
                    prompt += output
                    total_tokens += len(click_result.split())
                    continue

                clicked_urls.add(url)  # Add URL to clicked set
                print(f"- Clicking on URL: {url} with intent: {click_intent}")
                # Fetch and process page content
                if url not in url_cache:
                    try:
                        content = await fetch_page_content_async(
                            [url], 
                            use_jina=use_jina, 
                            jina_api_key=jina_api_key, 
                            keep_links=keep_links
                        )
                        content = content[url]
                        # Only cache content if it doesn't contain error indicators
                        has_error = (any(indicator.lower() in content.lower() for indicator in ERROR_INDICATORS) and len(content.split()) < 64) or content == ''
                        if not has_error:
                            url_cache[url] = content
                    except Exception as e:
                        print(f"Error fetching URL {url}: {e}")
                        content = ""
                else:
                    content = url_cache[url]

                # Check if content has error indicators
                has_error = any(indicator.lower() in content.lower() for indicator in ERROR_INDICATORS) or content == ''
                
                if has_error:
                    # If content has error, use it directly as summary
                    summary = "Unable to fetch the page content. You can try other links."
                else:
                    # Use web page reader to summarize content
                    reader_prompt = get_web_page_reader_instruction(click_intent, content)
                    summary = await generate_response(
                        client=aux_client,
                        tokenizer=aux_tokenizer,
                        prompt=reader_prompt,
                        semaphore=semaphore,
                        max_tokens=3600,
                        model_name=aux_model_name,
                        generation_params=aux_model_params,
                        enable_thinking=False
                    )

                # Append click results
                click_result = f"\n{BEGIN_CLICK_RESULT}\n{summary}\n{END_CLICK_RESULT}\n"
                output += click_result
                prompt += output
                total_tokens += len(click_result.split())
        
        else:
            finished = True
            break

    # Add max limit message if needed
    if not finished:
        output += f"\n{BEGIN_CLICK_RESULT}\nYou have reached the limit for clicking links.\n{END_CLICK_RESULT}\n\nOK, I will now provide the final information based on my collected information.\n\n**Final Information:**"
        prompt += output
        final_response = await generate_response(
            client=client,
            tokenizer=tokenizer,
            model_name=model_name,
            prompt=prompt,
            semaphore=semaphore,
            max_tokens=80000,
            generate_mode="completion",
            generation_params=model_params,
            enable_thinking=True,
        )
        output += final_response

    return output, original_prompt


async def run_deepsearch_tool(data:Dict = {}, global_vars:Dict = {}):
    search_query = data.get('func_call', '')

    search_cache = global_vars.get('search_cache', {})
    url_cache = global_vars.get('url_cache', {})
    bing_endpoint = global_vars.get('bing_endpoint', '')
    bing_subscription_key = global_vars.get('bing_subscription_key', '')
    use_jina = global_vars.get('use_jina', False)
    jina_api_key = global_vars.get('jina_api_key', '')
    keep_links = global_vars.get('keep_links', True)
    top_k = global_vars.get('max_search_limit', 10)
    
    client = global_vars['client']['instance']
    tokenizer = global_vars['client']['tokenizer']
    model_name = global_vars['client']['model_name']
    model_params = global_vars['client']['params']
    aux_client = global_vars['aux_client']['instance']
    aux_tokenizer = global_vars['aux_client']['tokenizer']
    aux_model_name = global_vars['aux_client']['model_name']
    aux_model_params = global_vars['aux_client']['params']
    semaphore = global_vars['semaphore']
    
    search_intent = await generate_response(
        client=aux_client,
        tokenizer=aux_tokenizer,
        model_name=aux_model_name,
        max_tokens=80000,
        generation_params=aux_model_params,
        prompt=get_search_intent_instruction(data['seq']['output']),
        semaphore=semaphore,
        generate_mode='chat',
        enable_thinking=False
    )

    if search_query in search_cache:
        results = search_cache[search_query]
    else:
        try:
            #results = bing_web_search(search_query, bing_subscription_key, bing_endpoint)
            results = await bing_web_search_async(search_query, bing_subscription_key, bing_endpoint)
            search_cache[search_query] = results
        except Exception as e:
            print(f"Error during search query '{search_query}': {e}")
            results = {}
    print(f'Searched for: "{search_query}"')

    relevant_info = extract_relevant_info(results)[:top_k]

    # Process documents
    urls_to_fetch = []
    for doc_info in relevant_info:
        url = doc_info['url']
        if url not in url_cache:
            urls_to_fetch.append(url)

    if urls_to_fetch:
        try:
            contents = await fetch_page_content_async(
                urls_to_fetch, 
                use_jina=use_jina, 
                jina_api_key=jina_api_key, 
                keep_links=keep_links
            )
            for url, content in contents.items():
                # Only cache content if it doesn't contain error indicators
                has_error = (any(indicator.lower() in content.lower() for indicator in ERROR_INDICATORS) and len(content.split()) < 64) or len(content) < 50 or len(content.split()) < 20
                if not has_error:
                    url_cache[url] = content
                # else:
                #     print(f'---Fetching Error\n{content}')
        except Exception as e:
            print(f"Error fetching URLs: {e}")

    # Get web page information for each result
    for doc_info in relevant_info:
        url = doc_info['url']
        if url not in url_cache:
            raw_content = ""
        else:
            raw_content = url_cache[url]
            is_success, raw_content = extract_snippet_with_context(raw_content, doc_info['snippet'], context_chars=2000)

        # Check if content has error indicators
        has_error = any(indicator.lower() in raw_content.lower() for indicator in ERROR_INDICATORS) or raw_content == ""
    
        if has_error:
            # If content has error, use it directly as summary
            doc_info['page_info'] = "Can not fetch the page content."
        else:
            # Use raw content directly as page info
            doc_info['page_info'] = raw_content
    

    formatted_documents = format_search_results(relevant_info)

    # Generate deep web exploration with interactions
    analysis, explorer_prompt = await generate_deep_web_explorer(
        search_query=search_query,
        search_intent=search_intent,
        document=formatted_documents,
        global_vars=global_vars
    )

    extracted_info = extract_deep_web_explorer_output(analysis)

    return {'result':extracted_info, 'explorer_prompt':explorer_prompt}