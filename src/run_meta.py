import csv
import json
import random
import time
import re
import os
import asyncio
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer
import argparse
from openai import AsyncOpenAI
from typing import List, Dict
from utils.basic_utils import load_caches, save_caches, load_data, prepare_output_dir, load_caches, save_caches
from utils.generate_utils import generate_response
from custom_agent import WebThinker, CodeO1, MultiModalO1, NaiveRAGAgent
from prompts.meta_prompts import (
    get_meta_planner_prompt,
    get_agent_select_prompt,
    get_memory_filter_prompt,
    get_subtask_summary_prompt,
)
from utils.agent_utils import extract_between


def parse_args():
    parser = argparse.ArgumentParser(description="Run Search-o1 for various datasets and models.")
    parser.add_argument(
        "--dataset_name", type=str, required=False, default="custom", help="Name of the dataset to use."
    )
    parser.add_argument("--split", type=str, required=False, default="test", help="Dataset split to use.")
    parser.add_argument(
        "--subset_num", type=int, default=-1, help="Number of examples to process. Defaults to all if not specified."
    )

    parser.add_argument(
        "--experiment_name", type=str, required=False, default="meta", help="Name of the experiment to use."
    )
    parser.add_argument(
        "--use_single_dir", action="store_true", help="Whether to use a single directory for all experiments."
    )
    parser.add_argument("--save_note", type=str, default=None, help="Note to save with the results.")

    parser.add_argument("--no_memory", action="store_true", help="Whether to not use memory.")

    parser.add_argument(
        "--agent_list", nargs="+", required=True, default=["WebThinker", "CodeO1", "MultiModalO1", "NaiveRAG"]
    )
    parser.add_argument("--max_execution_calls", type=int, default=10, help="Maximum number of execution calls.")
    parser.add_argument("--max_bad_calls", type=int, default=3, help="Maximum number of bad calls.")

    parser.add_argument("--api_base_url", type=str, required=True, help="Base URL for the API endpoint")
    parser.add_argument(
        "--aux_api_base_url", type=str, required=True, help="Base URL for the auxiliary model API endpoint"
    )
    parser.add_argument("--model_name", type=str, default="QwQ-32B", help="Name of the model to use")
    parser.add_argument(
        "--aux_model_name", type=str, default="Qwen2.5-72B-Instruct", help="Name of the auxiliary model to use"
    )

    parser.add_argument("--meta_temperature", type=float, default=0.7, help="Sampling temperature.")
    parser.add_argument("--meta_top_p", type=float, default=0.8, help="Top-p sampling parameter.")
    parser.add_argument("--meta_min_p", type=float, default=0.05, help="Minimum p sampling parameter.")
    parser.add_argument("--meta_top_k_sampling", type=int, default=20, help="Top-k sampling parameter.")
    parser.add_argument(
        "--meta_repetition_penalty",
        type=float,
        default=1.05,
        help="Repetition penalty. If not set, defaults based on the model.",
    )
    parser.add_argument(
        "--meta_max_tokens",
        type=int,
        default=80000,
        help="Maximum number of tokens to generate. If not set, defaults based on the model and dataset.",
    )

    parser.add_argument("--temperature", type=float, default=0.6, help="Sampling temperature.")
    parser.add_argument("--top_p", type=float, default=0.95, help="Top-p sampling parameter.")
    parser.add_argument("--min_p", type=float, default=0.05, help="Minimum p sampling parameter.")
    parser.add_argument("--top_k_sampling", type=int, default=20, help="Top-k sampling parameter.")
    parser.add_argument(
        "--repetition_penalty",
        type=float,
        default=1.05,
        help="Repetition penalty. If not set, defaults based on the model.",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=80000,
        help="Maximum number of tokens to generate. If not set, defaults based on the model and dataset.",
    )

    parser.add_argument("--aux_temperature", type=float, default=0.6, help="Sampling temperature.")
    parser.add_argument("--aux_top_p", type=float, default=0.95, help="Top-p sampling parameter.")
    parser.add_argument("--aux_min_p", type=float, default=0.0, help="Minimum p sampling parameter.")
    parser.add_argument("--aux_top_k_sampling", type=int, default=20, help="Top-k sampling parameter.")
    parser.add_argument(
        "--aux_repetition_penalty",
        type=float,
        default=1.0,
        help="Repetition penalty. If not set, defaults based on the model.",
    )
    parser.add_argument(
        "--aux_max_tokens",
        type=int,
        default=60000,
        help="Maximum number of tokens to generate. If not set, defaults based on the model and dataset.",
    )

    parser.add_argument("--max_search_limit", type=int, default=10, help="Maximum number of searches per question.")
    parser.add_argument("--top_k", type=int, default=10, help="Maximum number of search documents to return.")
    parser.add_argument(
        "--keep_links", action="store_true", default=False, help="Whether to keep links in fetched web content"
    )
    parser.add_argument("--bing_subscription_key", type=str, required=True, help="Bing Search API subscription key.")
    parser.add_argument(
        "--bing_endpoint",
        type=str,
        default="https://api.bing.microsoft.com/v7.0/search",
        help="Bing Search API endpoint.",
    )

    parser.add_argument(
        "--omni_api_url",
        type=str,
        default="https://dashscope.aliyuncs.com/compatible-mode/v1",
        help="URL for the Omni API.",
    )
    parser.add_argument("--omni_api_key", type=str, default="", help="API key for the Omni API.")

    parser.add_argument("--cache_dir", type=str, default="./cache", help="Directory to save caches")
    parser.add_argument("--concurrent_limit", type=int, default=32, help="Maximum number of concurrent API calls")
    parser.add_argument("--lora_name", type=str, default=None, help="Name of the LoRA adapter to load")
    parser.add_argument("--lora_path", type=str, default=None, help="Path to the LoRA weights")
    parser.add_argument("--save_dir", type=str, default="./outputs", help="Directory to save outputs")
    return parser.parse_args()


AGENT_MAP = {
    "Web-Thinker": WebThinker,
    "Code-Agent": CodeO1,
    "Multimodal-Agent": MultiModalO1,
    "Naive-RAG-Agent": NaiveRAGAgent,
}

MODEL2PATH = {
    "Qwen2.5-72B-Instruct": "Qwen/Qwen2.5-72B-Instruct",
    "QwQ-32B": "Qwen/QwQ-32B",
}

BEGIN_EXECUTION_CALL = "<|begin_call_subtask|>"
END_EXECUTION_CALL = "<|end_call_subtask|>"
BEGIN_EXECUTION_RESULT = "<|begin_subtask_result|>"
END_EXECUTION_RESULT = "<|end_subtask_result|>"

BEGIN_NOTE_CALL = "<|begin_call_note|>"
END_NOTE_CALL = "<|end_call_note|>"
BEGIN_NOTE_RESULT = "<|begin_note_result|>"
END_NOTE_RESULT = "<|end_note_result|>"


def prepare_global_vars(args):
    # Basic settings
    global_vars = {}
    global_vars["dataset_name"] = args.dataset_name
    global_vars["split"] = args.split
    global_vars["subset_num"] = args.subset_num
    global_vars["experiment_name"] = args.experiment_name
    global_vars["use_single_dir"] = args.use_single_dir
    global_vars["save_note"] = args.save_note
    global_vars["cache_dir"] = args.cache_dir
    global_vars["max_tokens"] = args.max_tokens
    global_vars["agent_list"] = args.agent_list
    global_vars["keep_links"] = args.keep_links
    print(f'global_vars["agent_list"]: {global_vars["agent_list"]}')
    global_vars["gaia_file_dir"] = "./data/GAIA/files"
    global_vars["max_execution_calls"] = args.max_execution_calls
    global_vars["max_bad_calls"] = args.max_bad_calls

    assert all(
        [
            agent in ["WebThinker", "CodeO1", "SearchO1", "MultiModalO1", "NaiveRAG", "CustomO1"]
            for agent in args.agent_list
        ]
    ), f"Agent {args.agent_list} not found"

    global_vars["no_memory"] = args.no_memory

    # Load data
    dataset_name = args.dataset_name.lower()
    data = load_data(dataset_name=dataset_name, split=args.split, subset_num=args.subset_num)
    output_dir = prepare_output_dir(
        model_name=args.model_name,
        dataset_name=dataset_name,
        experiment_name=args.experiment_name,
        use_single_dir=args.use_single_dir,
        save_dir=args.save_dir,
    )
    global_vars["data"] = data
    global_vars["output_dir"] = output_dir
    global_vars["dataset_name"] = dataset_name

    # Load models
    client = AsyncOpenAI(
        api_key="empty",
        base_url=args.api_base_url,
    )
    aux_client = AsyncOpenAI(
        api_key="empty",
        base_url=args.aux_api_base_url,
    )
    client_params = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "top_k_sampling": args.top_k_sampling,
        "repetition_penalty": args.repetition_penalty,
    }
    meta_client_params = {
        "temperature": args.meta_temperature,
        "top_p": args.meta_top_p,
        "max_tokens": args.meta_max_tokens,
        "top_k_sampling": args.meta_top_k_sampling,
        "repetition_penalty": args.meta_repetition_penalty,
    }
    aux_client_params = {
        "temperature": args.aux_temperature,
        "top_p": args.aux_top_p,
        "max_tokens": args.aux_max_tokens,
        "top_k_sampling": args.aux_top_k_sampling,
        "repetition_penalty": args.aux_repetition_penalty,
    }
    if args.model_path is None:
        model_path = MODEL2PATH[args.model_name]
    else:
        model_path = args.model_path
    if args.aux_model_path is None:
        aux_model_path = MODEL2PATH[args.aux_model_name]
    else:
        aux_model_path = args.aux_model_path

    if model_path == aux_model_path:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        aux_tokenizer = tokenizer
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        aux_tokenizer = AutoTokenizer.from_pretrained(aux_model_path)

    global_vars["meta_client"] = {
        "instance": client,
        "model_name": args.model_name,
        "tokenizer": tokenizer,
        "params": meta_client_params,
    }
    global_vars["client"] = {
        "instance": client,
        "model_name": args.model_name,
        "tokenizer": tokenizer,
        "params": client_params,
    }
    global_vars["aux_client"] = {
        "instance": aux_client,
        "model_name": args.aux_model_name,
        "tokenizer": aux_tokenizer,
        "params": aux_client_params,
    }

    global_vars["omni_api_url"] = args.omni_api_url
    global_vars["omni_api_key"] = args.omni_api_key

    # Set semaphore
    global_vars["semaphore"] = asyncio.Semaphore(args.concurrent_limit)

    # Load caches
    search_cache, url_cache, file_cache = load_caches(cache_dir=args.cache_dir, keep_links=args.keep_links)
    global_vars["search_cache"] = search_cache
    global_vars["url_cache"] = url_cache
    global_vars["file_cache"] = file_cache

    global_vars["bing_subscription_key"] = args.bing_subscription_key
    global_vars["bing_endpoint"] = args.bing_endpoint
    global_vars["max_search_limit"] = args.max_search_limit

    return global_vars


async def select_best_agent(detect_sub_task: str, global_vars):
    aux_client = global_vars["aux_client"]["instance"]
    aux_tokenizer = global_vars["aux_client"]["tokenizer"]
    aux_model_name = global_vars["aux_client"]["model_name"]
    aux_client_params = global_vars["aux_client"]["params"]
    semaphore = global_vars["semaphore"]

    agent_list = global_vars["agent_list"]
    AGENT_INFO = {}
    for agent_id in agent_list:
        agent = AGENT_MAP[agent_id](global_vars=global_vars, use_boxed=True)
        description = agent.description
        agent_name = agent.agent_name
        AGENT_INFO[agent_name] = {"agent_name": agent_name, "description": description, "instance": agent}

    formatted_agent_info = ""
    for idx, (agent_name, agent_dict) in enumerate(AGENT_INFO.items()):
        formatted_agent_info += f"Agent {idx+1}: {agent_name}\n"
        formatted_agent_info += f'Description: {agent_dict["description"]}\n\n'
    system_prompt, user_prompt = get_agent_select_prompt(task=detect_sub_task, agent_info=formatted_agent_info)

    prompt = [{"role": "user", "content": system_prompt + "\n\n" + user_prompt}]
    prompt = aux_tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
    output = await generate_response(
        client=aux_client,
        tokenizer=aux_tokenizer,
        prompt=prompt,
        semaphore=semaphore,
        model_name=aux_model_name,
        generation_params=aux_client_params,
        generate_mode="completion",
    )
    output = output.split("</think")[-1].strip()

    try:
        output = output.replace("```json", "").replace("```", "")
        select_agent_name = json.loads(output)["selected_agent_name"]
        reason = json.loads(output)["reason"]
    except Exception as e:
        print(f"Error selecting agent: {e}")
        print(f"output: {output}")
        reason = f"Error selecting agent: {e}"
        select_agent_name = "Web-Thinker"

    if "naive" in select_agent_name.lower():
        select_agent_name = "Naive-RAG-Agent"
    use_agent = AGENT_MAP.get(select_agent_name, AGENT_MAP["Web-Thinker"])

    return reason, select_agent_name, use_agent


async def format_memory(current_memory: list, sub_task: str, select_agent_name: str, global_vars):
    aux_client = global_vars["aux_client"]["instance"]
    aux_tokenizer = global_vars["aux_client"]["tokenizer"]
    aux_model_name = global_vars["aux_client"]["model_name"]
    aux_client_params = global_vars["aux_client"]["params"]
    semaphore = global_vars["semaphore"]

    memory_str = ""
    for idx, memory in enumerate(current_memory):
        if not memory.startswith("Resource:"):
            memory = memory.split("[Source")[0].strip()
        if "Code" in select_agent_name:
            continue
        memory_str += f"Memory Fact {idx+1}: {memory}\n"

    if memory_str == "":
        memory_str = "Currently no memory available."
    else:
        if len(current_memory) > 4:
            prompt = get_memory_filter_prompt(memory_str, sub_task)
            prompt = [{"role": "user", "content": prompt}]
            prompt = aux_tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
            memory_str = await generate_response(
                client=aux_client,
                tokenizer=aux_tokenizer,
                prompt=prompt,
                semaphore=semaphore,
                model_name=aux_model_name,
                generation_params=aux_client_params,
                generate_mode="completion",
            )
            memory_str = memory_str.split("</think>")[-1].strip()
    return memory_str


def update_memory(current_task: str, summary: str, current_memory: list, select_agent_name: str, code_env: dict = {}):
    if select_agent_name in ["WebThinker", "NaiveRAG", "SearchO1"]:
        select_agent_name = "Search Agent"

    conclusion = summary["conclusion"]
    memory_dict = summary["memory"]

    if isinstance(conclusion, dict):
        reasoning_process = conclusion.get("reasoning_process", "")
        final_conclusion = conclusion.get("final_conclusion", "")
        export_conclusion = (
            f"Result from {select_agent_name}:\n{reasoning_process}\nFinal Conclusion:\n{final_conclusion}"
        )
    else:
        final_conclusion = str(conclusion)
        export_conclusion = conclusion

    try:
        discover_list = []
        for fact in memory_dict["fact_memory"]:
            current_memory.append(fact.strip())
            discover_list.append(fact.strip())
    except:
        discover_list = []

    for des, url in memory_dict["resource_memory"].items():
        if (
            "http" in url
            and "https://example" not in url
            and "https://www.example" not in url
            and sum([url in memory for memory in current_memory if memory.startswith("Resource:")]) == 0
        ):
            current_memory.append(f"Resource: {des} - URL: {url}")
            discover_list.append(f"Resource: {des} - URL: {url}")

    memory_conclusion = f"Result for {current_task}:\n{final_conclusion}"
    current_memory.append(memory_conclusion)
    current_memory = list(set(current_memory))

    return current_memory, export_conclusion, discover_list


async def get_subtask_conclusion(
    sub_task: str, select_agent_name: str, reasoning_chain: str, current_memory: list, global_vars
):
    aux_client = global_vars["aux_client"]["instance"]
    aux_tokenizer = global_vars["aux_client"]["tokenizer"]
    aux_model_name = global_vars["aux_client"]["model_name"]
    aux_client_params = global_vars["aux_client"]["params"]
    semaphore = global_vars["semaphore"]

    prompt = get_subtask_summary_prompt(reasoning_chain=reasoning_chain, task_description=sub_task)
    prompt = [{"role": "user", "content": prompt}]
    prompt = aux_tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)

    tokenized_prompt = aux_tokenizer.encode(prompt, add_special_tokens=False)
    if len(tokenized_prompt) > 80000:
        prompt = aux_tokenizer.decode(tokenized_prompt[:40000] + tokenized_prompt[-40000:])
    summary = await generate_response(
        client=aux_client,
        tokenizer=aux_tokenizer,
        prompt=prompt,
        semaphore=semaphore,
        model_name=aux_model_name,
        generation_params=aux_client_params,
        generate_mode="completion",
    )
    summary = summary.split("</think>")[-1].strip()

    try:
        match = re.search(r"\{.*\}", summary, re.DOTALL)
        if match:
            summary = json.loads(match.group())
        else:
            summary = {"conclusion": summary, "memory": {"fact_memory": [], "resource_memory": {}}}
    except:
        summary = {"conclusion": summary, "memory": {"fact_memory": [], "resource_memory": {}}}

    current_memory, conclusion, discover_list = update_memory(sub_task, summary, current_memory, select_agent_name)

    return conclusion, current_memory, discover_list


async def process_meta_sequence(seq, global_vars):
    MAX_EXECUTION_CALLS = global_vars["max_execution_calls"]
    MAX_BAD_CALLS = global_vars["max_bad_calls"]

    item = seq["item"]
    question = seq["question"]
    if "file_name" in item and item["file_name"] != "":
        file_name = item["file_name"]
        file_path = os.path.join(global_vars["gaia_file_dir"], file_name)
        fake_data_path = "input_data.{}".format(file_name.split(".")[-1])

        from utils.basic_utils import get_file_content

        file_string = get_file_content(file_path)
        seq["path_map"] = {fake_data_path: file_path}
        question = f"{question}\n\nAttatched File Path: {fake_data_path}\n\nThe content of the attached file (if loaded successfully):\n{file_string}\n"

    meta_client_name = "meta_client"
    meta_client = global_vars[meta_client_name]["instance"]
    meta_tokenizer = global_vars[meta_client_name]["tokenizer"]
    meta_model_name = global_vars[meta_client_name]["model_name"]
    meta_client_params = global_vars[meta_client_name]["params"]

    meta_prompt = get_meta_planner_prompt(question)
    meta_prompt = [{"role": "user", "content": meta_prompt}]
    meta_prompt = meta_tokenizer.apply_chat_template(meta_prompt, tokenize=False, add_generation_prompt=True)

    seq["memory"] = []
    seq["finished"] = False
    seq["action_count"] = 0
    seq["bad_call_count"] = 0
    seq["meta_prompt"] = meta_prompt
    seq["meta_generation"] = ""
    seq["sub_task_list"] = []
    seq["sub_task_nodes"] = []
    seq["select_reason"] = []
    seq["current_note"] = ""
    seq["total_tokens"] = len(seq["meta_prompt"].split())

    while not seq["finished"] and seq["action_count"] < MAX_EXECUTION_CALLS and seq["bad_call_count"] <= MAX_BAD_CALLS:
        meta_prompt = seq["meta_prompt"]
        response = await generate_response(
            client=meta_client,
            tokenizer=meta_tokenizer,
            model_name=meta_model_name,
            prompt=meta_prompt,
            semaphore=global_vars["semaphore"],
            max_tokens=global_vars["meta_max_tokens"],
            generation_params=meta_client_params,
            stop=[END_EXECUTION_CALL],
            generate_mode="completion",
        )
        response = response.replace("</think>", "").rstrip("\n").rstrip()
        tokens_this_response = len(response.split())
        seq["total_tokens"] += tokens_this_response
        seq["meta_generation"] += response
        seq["meta_prompt"] += response

        detect_sub_task = extract_between(response, BEGIN_EXECUTION_CALL, END_EXECUTION_CALL)
        if detect_sub_task is not None:
            if isinstance(detect_sub_task, str):
                detect_sub_task = detect_sub_task.strip().rstrip("|")
            if detect_sub_task in seq["sub_task_list"]:
                seq["bad_call_count"] += 1
                warning_text = f"\n\n{BEGIN_EXECUTION_RESULT}You have already called this sub-task. Please use the previously found information.{END_EXECUTION_RESULT}\n\n"
                seq["meta_generation"] += warning_text
                seq["meta_prompt"] += warning_text
                continue

            seq["action_count"] += 1
            seq["sub_task_list"].append(detect_sub_task)
            select_reason, select_agent_name, select_agent = await select_best_agent(detect_sub_task, global_vars)
            current_memory = await format_memory(seq["memory"], detect_sub_task, select_agent_name, global_vars)
            seq["select_reason"].append({"reason": select_reason, "select_agent_name": select_agent_name})

            agent_result = await select_agent.run(input_data={"task_info": detect_sub_task, "base_model": "client"})
            sub_task_process = agent_result
            sub_task_result = agent_result["output"]

            conclusion, memory, discover_list = await get_subtask_conclusion(
                detect_sub_task, select_agent_name, sub_task_result, seq["memory"], global_vars
            )
            task_result = f"\n\n{BEGIN_EXECUTION_RESULT}{conclusion}{END_EXECUTION_RESULT}\n\n"
            sub_task_process["conclusion"] = conclusion
            sub_task_process["use_memory"] = current_memory
            sub_task_process["extracted_memory"] = discover_list

            seq["sub_task_nodes"].append(sub_task_process)
            seq["memory"] = memory
            seq["meta_generation"] += task_result
            seq["meta_prompt"] += task_result
        else:
            seq["finished"] = True
            break

    if not seq["finished"]:
        seq[
            "meta_prompt"
        ] += f"\n\n<warning>You have reached the subtask call limit. You are not allowed to call any more sub-tasks. Please organize your reasoning results and directly give your final answer in \\boxed{{}} format.</warning>\n\n"
        seq[
            "meta_generation"
        ] += f"\n\n<warning>You have reached the subtask call limit. You are not allowed to call any more sub-tasks. Please organize your reasoning results and directly give your final answer in \\boxed{{}} format.</warning>\n\n"
        response = await generate_response(
            client=meta_client,
            tokenizer=meta_tokenizer,
            model_name=meta_model_name,
            prompt=meta_prompt,
            semaphore=global_vars["semaphore"],
            max_tokens=global_vars["max_tokens"],
            generation_params=meta_client_params,
            generate_mode="completion",
        )
        seq["meta_generation"] += response
        seq["meta_prompt"] += response
    return seq


async def main_async():
    args = parse_args()
    args.experiment_name = f"{args.experiment_name}_{str(args.agent_list)}"
    global_vars = prepare_global_vars(args)

    data = global_vars["data"]
    active_sequences = []
    for item in data:
        seq = {
            "item": item,
            "question": item["Question"],
            "output": "",
            "finished": False,
        }
        active_sequences.append(seq)

    try:
        tasks = [process_meta_sequence(seq=seq, global_vars=global_vars) for seq in active_sequences]

        with tqdm(total=len(tasks)) as pbar:

            async def track_progress(task):
                result = await task
                pbar.update(1)
                return result

            tracked_tasks = [track_progress(task) for task in tasks]
            completed_sequences = await asyncio.gather(*tracked_tasks)
    except Exception as e:
        print(f"Error: {e}")
        raise e
    # save results
    for item, seq in zip(data, completed_sequences):
        item["Output"] = seq["meta_generation"]
        item["memory"] = seq["memory"]
        item["meta_prompt"] = seq["meta_prompt"]
        item["sub_task_list"] = seq["sub_task_list"]
        item["select_reason"] = seq["select_reason"]
        item["sub_task_nodes"] = []
        item["current_note"] = seq.get("current_note", "")
        item["current_note_list"] = seq.get("current_note_list", [])
        for node in seq["sub_task_nodes"]:
            item["sub_task_nodes"].append(
                {
                    "process_name": node["process_name"],
                    "prompt": node.get("prompt", ""),
                    "output": node.get("output", ""),
                    "process": node.get("process", []),
                    "use_memory": node.get("use_memory", []),
                    "conclusion": node.get("conclusion", ""),
                    "extracted_memory": node.get("extracted_memory", []),
                }
            )

    t = time.localtime()
    save_note = f".{args.save_note}" if args.save_note is not None else ""
    result_json_name = f"{args.split}.{t.tm_mon}.{t.tm_mday},{t.tm_hour}:{t.tm_min}{save_note}.json"
    with open(os.path.join(global_vars["output_dir"], result_json_name), mode="w", encoding="utf-8") as json_file:
        json.dump(data, json_file, indent=4, ensure_ascii=False)

    save_caches(
        cache_dir=global_vars["cache_dir"],
        search_cache=global_vars["search_cache"],
        url_cache=global_vars["url_cache"],
        file_cache=global_vars["file_cache"],
    )
    print("Process completed.")


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
