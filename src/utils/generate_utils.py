import asyncio
from typing import Tuple, List

from openai import AsyncOpenAI


async def generate_response(
    prompt: str,
    client: AsyncOpenAI,
    tokenizer,
    model_name,
    semaphore: asyncio.Semaphore,
    generate_mode: str = "chat",
    retry_limit: int = 3,
    temperature: float = 0.6,
    top_p: float = 0.95,
    max_tokens: int = 32768,
    repetition_penalty: float = 1.0,
    top_k: int = 20,
    min_p: float = 0.0,
    stop: List[str] = [],
    enable_thinking: bool = True,
    remove_thinking: bool = False,
    generation_params: dict = {},
) -> Tuple[str, str]:
    """Generate a single response with retry logic"""
    # process all parameters
    temperature = generation_params.get('temperature', temperature)
    top_p = generation_params.get('top_p', top_p)
    max_tokens = generation_params.get('max_tokens', max_tokens)
    repetition_penalty = generation_params.get('repetition_penalty', repetition_penalty)
    top_k = generation_params.get('top_k', top_k)
    min_p = generation_params.get('min_p', min_p)
    stop = generation_params.get('stop', stop)
    enable_thinking = generation_params.get('enable_thinking', enable_thinking)
    remove_thinking = generation_params.get('remove_thinking', remove_thinking)
    output = ""
    for attempt in range(retry_limit):
        try:
            async with semaphore:
                if generate_mode == "chat":
                    messages = [{"role": "user", "content": prompt}]
                    if 'qwen3' in model_name.lower():
                        formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking)
                    else:
                        formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                else:
                    formatted_prompt = prompt

                response = await client.completions.create(
                    model=model_name,
                    prompt=formatted_prompt,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                    stop=stop,
                    extra_body={
                        'top_k': top_k,
                        'include_stop_str_in_output': True,
                        'repetition_penalty': repetition_penalty,
                        'min_p': min_p
                    },
                    timeout=3000,
                )
                output = response.choices[0].text
                break
        except Exception as e:
            print(f"Generate Response Error occurred: {e}, Starting retry attempt {attempt + 1}")
            if "maximum context length" in str(e).lower():
                # If length exceeds limit, reduce max_tokens by half
                max_tokens = max_tokens // 2
                print(f"Reducing max_tokens to {max_tokens}")
            if attempt == retry_limit - 1:
                print(f"Failed after {retry_limit} attempts: {e}")
            await asyncio.sleep(1 * (attempt + 1))
    if remove_thinking:
        output = output.split("</think>")[-1].strip()
    return output
    