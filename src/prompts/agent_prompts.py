query_plan_instruction_system = """You are a reasoning assistant. Your task is to generate a detailed query plan for answering the user's question by breaking it down into sub-queries. Each query should be a precise and suitable query for web search.

Please analyze the question and break it down into multiple sub-queries that will help gather all the necessary information to answer it completely. 

Remember:
1. Each sub-query should be a precise and suitable query for web search.
2. Subqueries should cover different aspects to avoid duplicate searches.
3. Each subquery needs to be necessary, do not add irrelevant ones. You need to use as few subqueries as possible to obtain accurate results.


Output your query plan in JSON format as follows:

```json
{{
    "query_plan": [
        "sub-query-1",
        "sub-query-2",
        ...
    ]
}}
```
"""

query_plan_instruction_user = """
Task: {question}
"""

def get_query_plan_instruction(question):
    return query_plan_instruction_system, query_plan_instruction_user.format(question=question)

def get_search_intent_instruction(prev_reasoning):
    return f"""Based on the previous thoughts below, provide the detailed intent of the latest search query.
Previous thoughts: {prev_reasoning}
Please provide the current search intent."""


def get_deep_web_explorer_instruction(search_query, search_intent, search_result):
    return f"""You are a web explorer analyzing search results to find relevant information based on a given search query and search intent.

**Guidelines:**

1. **Analyze the Searched Web Pages:**
- Carefully review the content of each searched web page.
- Identify factual information that is relevant to the **Current Search Query** and can aid in the reasoning process for the original question.

2. **More Information Seeking:**
- If the information is not relevant to the query, you could:
  1. Search again: <|begin_search_query|>another search query<|end_search_query|>
  2. Access webpage content using: <|begin_click_link|>your URL<|end_click_link|>

3. **Extract Relevant Information:**
- Return the relevant information from the **Searched Web Pages** that is relevant to the **Current Search Query**.

4. **Output Format:**
- Present the information beginning with **Final Information** as shown below.

**Final Information**
[Relevant information]

**Inputs:**

- **Current Search Query:**
{search_query}

- **Detailed Search Intent:**
{search_intent}

- **Searched Web Pages:**
{search_result}

Now please analyze the web pages and extract relevant information for the search query "{search_query}" and the search intent.
"""

def get_click_intent_instruction(prev_reasoning):
    return f"""Based on the previous thoughts below, provide the detailed intent of the latest click action.
Previous thoughts: {prev_reasoning}
Please provide the current click intent."""

def get_web_page_reader_instruction(query, document):
    return f"""{document}
Please provide all content related to "{query}" from this document in markdown format.
If there isn't any relevant information, just output "No relevant information". If there is any relevant information, output all the relevant information with potential helpful links."""


def get_code_o1_instruction_withmemory(task_info: str, current_memory: str, MAX_CODE_CALL_NUM: int = 5, use_boxed=False):
    if use_boxed:
        box_message = 'You should provide your final answer in the format \\boxed{{YOUR_ANSWER}}.\n\n'
    else:
        box_message = ''
    return (
        "You are an AI assistant that can execute Python code to solve problems. Here's how to use this ability:\n\n"
        "- To run code: write <|begin_code_call|>```python\nyour code here\n```<|end_code_call|>\n"
        "- You'll receive results as: <|begin_code_call_result|> execution results <|end_code_call_result|>\n\n"
        f"- You can execute code up to {MAX_CODE_CALL_NUM} times per conversation\n"
        "- If your code has errors, fix and retry, but make sure your code is correct and safe.\n\n"
        "Check your current memory for relevant information before writing code. \n\n"
        "Key guidelines:\n"
        "- Think step by step and explain your reasoning\n"
        "- Use code for calculations whenever possible\n"
        'Current Memory:\n'
        f'{current_memory}\n\n'
        f'{box_message}'
        f'Question:\n{task_info}\n\n'
    )

def get_code_o1_instruction(task_info: str, MAX_CODE_CALL_NUM: int = 5, use_boxed=False):
    if use_boxed:
        box_message = 'You should provide your final answer in the format \\boxed{YOUR_ANSWER}.\n\n'
    else:
        box_message = ''
    return (
        "You are an AI assistant that can execute Python code to solve problems. Here's how to use this ability:\n\n"
        "- To run code: write <|begin_code_call|>```python\nyour code here\n```<|end_code_call|>\n"
        "- You'll receive results as: <|begin_code_call_result|> execution results <|end_code_call_result|>\n\n"
        f"- You can execute code up to {MAX_CODE_CALL_NUM} times per conversation\n"
        "- If your code has errors, fix and retry, but make sure your code is correct and safe.\n\n"
        "Check your current memory for relevant information before writing code. \n\n"
        "Key guidelines:\n"
        "- Think step by step and explain your reasoning\n"
        "- Use code for calculations whenever possible\n"
        f'{box_message}'
        f'Question:\n{task_info}\n\n'
    )



def get_search_o1_instruction(question, max_search_limit, use_boxed=False):
    if use_boxed:
        box_message = 'You should provide your final answer in the format \\boxed{{YOUR_ANSWER}}.\n\n'
    else:
        box_message = ''
    instruction =  (
        "You are a reasoning assistant with the ability to perform web searches to help "
        "you answer the user's question accurately. You have special tools:\n\n"
        "- To perform a search: write <|begin_search_query|> your query here <|end_search_query|>.\n"
        "Then, the system will search and analyze relevant web pages, then provide you with helpful information in the format <|begin_search_result|> ...search results... <|end_search_result|>.\n\n"
        f"You can repeat the search process multiple times if necessary. The maximum number of search attempts is limited to {max_search_limit}.\n\n"
        "Once you have all the information you need, continue your reasoning.\n\n"
        "Example:\n"
        "Question: \"Alice David is the voice of Lara Croft in a video game developed by which company?\"\n"
        "Assistant thinking steps:\n"
        "- I need to find out who voices Lara Croft in the video game.\n"
        "- Then, I need to determine which company developed that video game.\n\n"
        "Assistant:\n"
        "<|begin_search_query|>Alice David Lara Croft voice<|end_search_query|>\n\n"
        "(System returns processed information from relevant web pages)\n\n"
        "Assistant thinks: The search results indicate that Alice David is the voice of Lara Croft in a specific video game. Now, I need to find out which company developed that game.\n\n"
        "Assistant:\n"
        "<|begin_search_query|>video game developed by Alice David Lara Croft<|end_search_query|>\n\n"
        "(System returns processed information from relevant web pages)\n\n"
        "Assistant continues reasoning with the new information...\n\n"
        "Remember:\n"
        "- Use <|begin_search_query|> to request a web search and end with <|end_search_query|>.\n"
        "- When done searching, continue your reasoning.\n\n"
        'Please answer the following question step by step.'
        f'{box_message}'
        'Please ensure the truthfulness of the answer. If you find that there is no valid information in the document, please directly state that there is no valid information.'
        f'Question:\n{question}\n\n'
    )
    return instruction

def get_search_o1_instruction_withmemory(question, current_memory, max_search_limit, use_boxed=False):
    if use_boxed:
        box_message = 'You should provide your final answer in the format \\boxed{{YOUR_ANSWER}}.\n\n'
    else:
        box_message = ''
    instruction =  (
        "You are a reasoning assistant with the ability to perform web searches to help "
        "you answer the user's question accurately. You have special tools:\n\n"
        "- To perform a search: write <|begin_search_query|> your query here <|end_search_query|>.\n"
        "Then, the system will search and analyze relevant web pages, then provide you with helpful information in the format <|begin_search_result|> ...search results... <|end_search_result|>.\n\n"
        f"You can repeat the search process multiple times if necessary. The maximum number of search attempts is limited to {max_search_limit}.\n\n"
        "Once you have all the information you need, continue your reasoning.\n\n"
        "Example:\n"
        "Question: \"Alice David is the voice of Lara Croft in a video game developed by which company?\"\n"
        "Assistant thinking steps:\n"
        "- I need to find out who voices Lara Croft in the video game.\n"
        "- Then, I need to determine which company developed that video game.\n\n"
        "Assistant:\n"
        "<|begin_search_query|>Alice David Lara Croft voice<|end_search_query|>\n\n"
        "(System returns processed information from relevant web pages)\n\n"
        "Assistant thinks: The search results indicate that Alice David is the voice of Lara Croft in a specific video game. Now, I need to find out which company developed that game.\n\n"
        "Assistant:\n"
        "<|begin_search_query|>video game developed by Alice David Lara Croft<|end_search_query|>\n\n"
        "(System returns processed information from relevant web pages)\n\n"
        "Assistant continues reasoning with the new information...\n\n"
        "Remember:\n"
        "- Use <|begin_search_query|> to request a web search and end with <|end_search_query|>.\n"
        "- When done searching, continue your reasoning.\n\n"
        "Check your current memory for relevant information before writing code. \n\n"
        "Current Memory:\n"
        f"{current_memory}\n\n"
        'Please answer the following question step by step.'
        f'{box_message}'
        'Please ensure the truthfulness of the answer. If you find that there is no valid information in the document, please directly state that there is no valid information.'
        f'Question:\n{question}\n\n'
    )
    return instruction

def get_multimodal_o1_instruction(task_info: str, MAX_MM_CALL_NUM: int = 5, use_boxed=False):
    if use_boxed:
        box_message = 'You should provide your final answer in the format \\boxed{{YOUR_ANSWER}}.\n\n'
    else:
        box_message = ''
    return (
        "You are an AI assistant with multimodal understanding capabilities. You can analyze images, video and audio "
        "to answer user questions. You have access to a special multimodal tool:\n\n"
        
        "- To analyze an image/video/audio and answer questions about it, use the format:\n"
        "<|begin_multimodal_call|>\n"
        "data: [path of image/video/audio]\n"
        "question: [your specific question]\n"
        "<|end_multimodal_call|>\n"
        "The system will provide analysis results in the format: "
        "<|begin_multimodal_result|> ...analysis results... <|end_multimodal_result|>\n\n"
        
        f"You can ask multiple questions about different aspects of the image/video/audio if needed. "
        f"The maximum number of multimodal analysis calls is limited to {MAX_MM_CALL_NUM}.\n\n"
        
        "Example:\n"
        "Task: \"What are the main colors of the clothing worn by people in the image?\"\n\n"
        
        "Assistant thinking steps:\n"
        "- I need to analyze the clothing colors in the image\n"
        "- I'll ask about the clothing colors\n\n"
        
        "Assistant:\n"
        "<|begin_multimodal_call|>\n"
        "data: photo.jpg\n"
        "question: What colors are the clothes that people are wearing in this image?\n"
        "<|end_multimodal_call|>\n\n"
        
        "<|begin_multimodal_result|>\n"
        "The people in the image are wearing primarily red and blue clothing. One person has "
        "on a red sweater, while another is wearing a navy blue jacket.\n"
        "<|end_multimodal_result|>\n\n"
        
        "Assistant continues reasoning with the results...\n\n"
        
        "Remember:\n"
        "- Always explain your reasoning before and after multimodal analysis\n"

        f"{box_message}"
        f"Question:\n{task_info}\n\n"
    )

def get_multimodal_o1_instruction_withmemory(task_info: str, current_memory: str, MAX_MM_CALL_NUM: int = 5, use_boxed=False):
    if use_boxed:
        box_message = 'You should provide your final answer in the format \\boxed{{YOUR_ANSWER}}.\n\n'
    else:
        box_message = ''
    return (
        "You are an AI assistant with multimodal understanding capabilities. You can analyze images, video and audio "
        "to answer user questions. You have access to a special multimodal tool:\n\n"
        
        "- To analyze an image/video/audio and answer questions about it, use the format:\n"
        "<|begin_multimodal_call|>\n"
        "data: [path of image/video/audio]\n"
        "question: [your specific question]\n"
        "<|end_multimodal_call|>\n"
        "The system will provide analysis results in the format: "
        "<|begin_multimodal_result|> ...analysis results... <|end_multimodal_result|>\n\n"
        
        f"You can ask multiple questions about different aspects of the image/video/audio if needed. "
        f"The maximum number of multimodal analysis calls is limited to {MAX_MM_CALL_NUM}.\n\n"
        
        "You are given previous analysis results as your current memory. Please check the current "
        "memory for relevant information before making new analysis requests.\n\n"
        
        "Example:\n"
        "Task: \"What are the main colors of the clothing worn by people in the image?\"\n\n"
        
        "Assistant thinking steps:\n"
        "- I need to analyze the clothing colors in the image\n"
        "- Let me check if there's relevant information in memory first\n"
        "- I'll ask about the clothing colors\n\n"
        
        "Assistant:\n"
        "<|begin_multimodal_call|>\n"
        "data: photo.jpg\n"
        "question: What colors are the clothes that people are wearing in this image?\n"
        "<|end_multimodal_call|>\n\n"
        
        "<|begin_multimodal_result|>\n"
        "The people in the image are wearing primarily red and blue clothing. One person has "
        "on a red sweater, while another is wearing a navy blue jacket.\n"
        "<|end_multimodal_result|>\n\n"
        
        "Assistant continues reasoning with the results...\n\n"
        
        "Remember:\n"
        "- Always explain your reasoning before and after multimodal analysis\n"

        "Current Memory:\n"
        f"{current_memory}\n\n"
        f"{box_message}"
        f"Question:\n{task_info}\n\n"
    )

def get_webthinker_instruction(task_info: str, MAX_SEARCH_LIMIT: int = 10, use_boxed=False):
    if use_boxed:
        box_message =  'Provide your final answer in the format \\boxed{YOUR_ANSWER}.\n\n'
    else:
        box_message = ''
    instruction = (
        "You are a reasoning assistant with the ability to perform web searches to help "
        "you answer the user's question accurately. You have special tools:\n\n"
        "- To perform a search: write <|begin_search_query|> your query here <|end_search_query|>.\n"
        "Then, the system will search and analyze relevant web pages, then provide you with helpful information in the format <|begin_search_result|> ...search results... <|end_search_result|>.\n\n"
        f"You can repeat the search process multiple times if necessary. The maximum number of search attempts is limited to {MAX_SEARCH_LIMIT}.\n\n"
        "Once you have all the information you need, continue your reasoning.\n\n"
        "Example:\n"
        "Question: \"Alice David is the voice of Lara Croft in a video game developed by which company?\"\n"
        "Assistant thinking steps:\n"
        "- I need to find out who voices Lara Croft in the video game.\n"
        "- Then, I need to determine which company developed that video game.\n\n"
        "Assistant:\n"
        "<|begin_search_query|>Alice David Lara Croft voice<|end_search_query|>\n\n"
        "(System returns processed information from relevant web pages)\n\n"
        "Assistant thinks: The search results indicate that Alice David is the voice of Lara Croft in a specific video game. Now, I need to find out which company developed that game.\n\n"
        "Assistant:\n"
        "<|begin_search_query|>video game developed by Alice David Lara Croft<|end_search_query|>\n\n"
        "(System returns processed information from relevant web pages)\n\n"
        "Assistant continues reasoning with the new information...\n\n"
        "Remember:\n"
        "- Use <|begin_search_query|> to request a web search and end with <|end_search_query|>.\n"
        "- When done searching, continue your reasoning.\n\n"
    )
    user_prompt = (
        'Please answer the following question. You should think step by step to solve it.\n\n'
        f'{box_message}'
        f'Question:\n{task_info}\n\n'
    )
    return instruction + user_prompt


def get_webthinker_instruction_withmemory(task_info: str, current_memory: str, MAX_SEARCH_LIMIT: int = 10, use_boxed=False):
    if use_boxed:
        box_message =  'Provide your final answer in the format \\boxed{YOUR_ANSWER}.\n\n'
    else:
        box_message = ''
    instruction = (
        "You are a reasoning assistant with the ability to perform web searches to help "
        "you answer the user's question accurately. You have special tools:\n\n"
        "- To perform a search: write <|begin_search_query|> your query here <|end_search_query|>.\n"
        "Then, the system will search and analyze relevant web pages, then provide you with helpful information in the format <|begin_search_result|> ...search results... <|end_search_result|>.\n\n"
        f"You can repeat the search process multiple times if necessary. The maximum number of search attempts is limited to {MAX_SEARCH_LIMIT}.\n\n"
        "Once you have all the information you need, continue your reasoning.\n\n"
        "You are given previous exploration results as your current memory, please first check the current memory for relevant information before making search. But you need to be aware that the facts involved in memory may not be comprehensive.\n\n"
        "Example:\n"
        "Question: \"Alice David is the voice of Lara Croft in a video game developed by which company?\"\n"
        "Assistant thinking steps:\n"
        "- I need to find out who voices Lara Croft in the video game.\n"
        "- Then, I need to determine which company developed that video game.\n\n"
        "Assistant:\n"
        "<|begin_search_query|>Alice David Lara Croft voice<|end_search_query|>\n\n"
        "(System returns processed information from relevant web pages)\n\n"
        "Assistant thinks: The search results indicate that Alice David is the voice of Lara Croft in a specific video game. Now, I need to find out which company developed that game.\n\n"
        "Assistant:\n"
        "<|begin_search_query|>video game developed by Alice David Lara Croft<|end_search_query|>\n\n"
        "(System returns processed information from relevant web pages)\n\n"
        "Assistant continues reasoning with the new information...\n\n"
        "Remember:\n"
        "- Use <|begin_search_query|> to request a web search and end with <|end_search_query|>.\n"
        "- When done searching, continue your reasoning.\n\n"
    )
    user_prompt = (
        'Please answer the following question. You should think step by step to solve it.\n\n'
        f'{box_message}'
        'Current Memory:\n'
        f'{current_memory}\n\n'
        f'Question:\n{task_info}\n\n'
    )
    return instruction + user_prompt

def get_searcho1_summary_instruction(search_query: str, prev_reasoning: str, document: str):
    return f"""**Task Instruction:**

You are tasked with reading and analyzing web pages based on the following inputs: **Previous Reasoning Steps**, **Current Search Query**, and **Searched Web Pages**. Your objective is to extract relevant and helpful information for **Current Search Query** from the **Searched Web Pages** and seamlessly integrate this information into the **Previous Reasoning Steps** to continue reasoning for the original question.

**Guidelines:**

1. **Analyze the Searched Web Pages:**
- Carefully review the content of each searched web page.
- Identify factual information that is relevant to the **Current Search Query** and can aid in the reasoning process for the original question.

2. **Extract Relevant Information:**
- Select the information from the Searched Web Pages that directly contributes to advancing the **Previous Reasoning Steps**.
- Ensure that the extracted information is accurate and relevant.

3. **Output Format:**
- **If the web pages provide helpful information for current search query:** Present the information beginning with `**Final Information**` as shown below.
**Final Information**

[Helpful information]

- **If the web pages do not provide any helpful information for current search query:** Output the following text.

**Final Information**

No helpful information found.

**Inputs:**
- **Previous Reasoning Steps:**  
{prev_reasoning}

- **Current Search Query:**  
{search_query}

- **Searched Web Pages:**  
{document}

Now you should analyze each web page and find helpful information based on the current search query "{search_query}" and previous reasoning steps.
"""