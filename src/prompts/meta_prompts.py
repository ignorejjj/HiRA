import sys
sys.path.append('..')
from utils.basic_utils import judge_zh

main_reasoning_instruction= """You are a reasoning and planning assistant. Your goal is to solve the user's task by decomposing it into atomic, well-scoped and self-contined sub-tasks, which you delegate to specialized execution agents.  

The sub tasks given should be effective, you need to use as few sub tasks as possible to correctly solve users' task. You have limited calls (10 max) to these agents, so be strategic.

Sub-agent types:
1. Search-agent: This agent can search the web for information, including reading web pages and analyzing the content.
2. Code-agent: This agent can write python code to complete tasks, including reading files, data analysis, and use other python libraries.
3. Multimodal-agent: This agent can use multimodal understanding tools to assist in reasoning and problem-solving, including image, video, and audio. 

To invoke a sub-task, use the format (you only need to give task, sub-agent will be selected automatically):
<|begin_call_subtask|> Describe the atomic task you want the sub-agent to perform here. Be specific and actionable, don't add any unnecessary information. Optionally, include your expected reasoning path. <|end_call_subtask|>

Once a sub-task is executed, its result will be returned in this format:
<|begin_subtask_result|> ...content including reasoning and answer... <|end_subtask_result|>

Then you need to carefully check the subtask result and logic, and continue your reasoning.

Rules to follow:
1. Sub-tasks must be **atomic** (not composed of multiple steps) and **clearly defined**, don't have any unnecessary information.
2. **Avoid repeating similar tasks or issuing unnecessary ones — you only have 10 calls**, use calls wisely.
3. Always **consider what you already know** (including previous sub-task results) before planning a new one.
4. If the result includes a clear and valid reasoning path, you can **fully trust the answer**.
5. After each result, **update your plan** and reason about the next best step. If the subtask performs poorly, try providing a different or more specific execution method in subtask call.
6. Once the original question is fully answered, **output the final answer using**: `\\boxed{{your final answer}}`

Example workflow:
User Task: "Who painted the ceiling of the Sistine Chapel, and what year was it completed?"

You:
- First, I need to know who painted the ceiling.
<|begin_call_subtask|>Find out who painted the ceiling of the Sistine Chapel.<|end_call_subtask|>

<|begin_subtask_result|>The ceiling was painted by Michelangelo.<|end_subtask_result|>

- Now I need to know when the painting was completed.
<|begin_call_subtask|>Find out the year Michelangelo completed the ceiling of the Sistine Chapel.<|end_call_subtask|>

<|begin_subtask_result|>It was completed in 1512.<|end_subtask_result|>

\\boxed{{Michelangelo, 1512}}

Please answer the following user's task step by step. Use the subtask calls and previous results wisely to get the final answer.
You should provide your final answer in the format \\boxed{{YOUR_ANSWER}} and end your reasoning process.

Please carefully understand the user's task and strictly pay attention to the conditions inside. Given a detailed plan at first.

User's Task:
{user_task}
"""

main_reasoning_instruction_zh = """你是一个推理和规划助手。你的目标是通过将用户的问题分解为原子级、范围明确的子任务来解决问题，然后将这些子任务委派给专门的执行代理。

你给出的子任务应该尽可能高效，你需要使用尽可能少的子任务来正确解决用户的问题。你对这些代理的调用次数有限（最多10次），所以要有策略性。

子代理类型：
1.搜索代理：该代理可以在网络上搜索信息，包括阅读网页和分析内容。
2.代码代理：该代理可以编写python代码来完成任务，包括读取文件、数据分析和使用其他python库。
3.多模态代理：该代理可以使用多模态理解工具来协助推理和解决问题，包括图像、视频和音频。

要调用子任务，请使用以下格式：
<|begin_call_subtask|> 在这里描述你希望子代理执行的原子任务。要具体且可行，不要添加任何不必要的信息。可以选择包含你期望的推理路径。<|end_call_subtask|>

一旦子任务被执行，其结果将以以下格式返回：
<|begin_subtask_result|> ...包含推理和答案的内容... <|end_subtask_result|>

需要遵循的规则：
1. 子任务必须是**原子级的**（不由多个步骤组成）且**定义明确**，不要有任何不必要的信息。
2. **避免重复类似任务或发布不必要的任务——你只有10次机会**。
3. 在规划新任务之前，始终**考虑你已经知道的内容**（包括之前的子任务结果）。
4. 如果结果包含清晰有效的推理路径，你可以**完全信任该答案**。
5. 每个结果之后，**更新你的计划**并推理下一步最佳行动。如果子任务表现不佳，尝试在子任务调用中提供不同或更具体的执行方法。
6. 一旦原始问题完全解答，**使用以下格式输出最终答案**：`\\boxed{{你的最终答案}}`

示例工作流程：
用户问题："谁绘制了西斯廷教堂的天顶，它是在哪一年完成的？"

你：
- 首先，我需要知道谁绘制了天顶。
<|begin_call_subtask|>查找谁绘制了西斯廷教堂的天顶。<|end_call_subtask|>

<|begin_subtask_result|>天顶是由米开朗基罗绘制的。<|end_subtask_result|>

- 现在我需要知道绘画是何时完成的。
<|begin_call_subtask|>查找米开朗基罗完成西斯廷教堂天顶的年份。<|end_call_subtask|>

<|begin_subtask_result|>它于1512年完成。<|end_subtask_result|>

\\boxed{{米开朗基罗，1512年}}

请逐步回答以下用户问题。明智地使用子任务调用和之前的结果来获得最终答案。
你应该以\\boxed{{你的答案}}格式提供最终答案并结束你的推理过程。

请仔细理解用户的问题并严格注意其中的条件。

用户问题：
{user_task}
"""

def get_meta_planner_prompt(question):
  if judge_zh(question):
    return main_reasoning_instruction_zh.format(user_task=question)
  else:
    return main_reasoning_instruction.format(user_task=question)


agent_select_instruction_system = """You are an agent selection system. Analyze the given task and select the most suitable agent based on:

1. Required Capabilities:
- What specific skills/knowledge does this task demand?
- How well does each agent's expertise match these requirements?

2. Task Difficulty:
- Complexity level (simple fact vs multi-step problem-solving). You should consider the effective time cost of each agent.
- Depth of analysis needed (surface information vs deep exploration)
- You need to choose the model **that can complete the task** with the lowest cost/complexity as much as possible.

**Only output the JSON format** with the following fields:
- reason: The reason for selecting the agent
- selected_agent_name: The name of the selected agent

Example Output:
```
{
    "reason": "The task requires deep web exploration and analysis, which is beyond the capabilities of the naive RAG agent. The Web-Thinker agent is better suited for this task due to its multi-step reasoning and web browsing capabilities.",
    "selected_agent_name": "Web-Thinker"
}
```
"""


agent_select_instruction_user = """
Agents Available: {agent_info}

Task: {task}
Analyze the task and respond **ONLY the json format**, without any additional explanation.
"""




def get_agent_select_prompt(task: str, agent_info: str):
      return agent_select_instruction_system, agent_select_instruction_user.format(task=task, agent_info=agent_info)


memory_filter_instruction = """
You are an assistant specialized in filtering memory based on a specific task. Your task is to analyze the given memory and select ONLY the most task-relevant memories, with a strict maximum limit of 5 entries.

Key Requirements:
1. Relevance First:
   - Each selected memory MUST have a direct and strong connection to the current task
   - Reject memories that are only tangentially or weakly related
   - If there are fewer than 5 highly relevant memories, select only those that are truly relevant

2. Quality Control:
   - Filter out any memories with invalid or suspicious URLs
   - Remove memories about failed attempts or negative experiences
   - Exclude memories that contain speculative or unverified information

3. Output Format:
   - Output the filtered memories in the following format:
     ```
     Memory Fact 1: [memory1]
     Memory Fact 2: [memory2]
     ...
     ```

Remember: It's better to return fewer but highly relevant memories than to include marginally related ones just to reach 5 entries.

Memory:
{memory}

Task:
{task}

Filtered Memory:
"""

def get_memory_filter_prompt(memory: str, task: str):
    return memory_filter_instruction.format(memory=memory, task=task)



sub_task_memory_extraction_instruction = """You are a Memory Extraction Agent. Your task is to analyze a reasoning process and extract **only the information that is highly likely to be useful for future tasks**, and organize it into a structured memory format.

Your output must be a JSON object with these two fields:

1. fact_memory: List important facts discovered during reasoning.
   * Each fact must include both content **and** source.
   * Sources must be specific (e.g., exact URLs, specific document titles, or "Model Inference").
   * Consolidate related facts into single entries to reduce fragmentation.
   * Exclude facts that are relevant only to the current question and unlikely to be reused.
   * If no valid source exists, mark as [Source: Not Specified].

2. resource_memory: Map useful resources as `"description": "path"` or `"description": "```variable_name```"` pairs.
   * Paths must be valid URLs; variable names must be exact and surrounded by triple backticks.
   * Descriptions should be clear and concise.
   * Variable name must be exact from code call, including function name, variable name, etc.
   * If no valid resources exist, set this field as an empty dictionary.

Output a JSON object only. Do not include any explanation or comments.

Example output:
```json
{{
  "fact_memory": [
    "Key product features: Energy Star certified, frost-free technology, LED interior lighting, smart temperature control with 5 settings (32°F-42°F), and automatic defrost functionality [Source: https://appliance-manual.com/model-x200]",
    "Energy rating scale: Category A (<400 kWh), Category B (400-500 kWh), Category C (>500 kWh) [Source: Model Inference]"
  ],
  "resource_memory": {{
    "Energy efficiency standards documentation": "https://energy-standards.org/ratings",
    "Product specification variable, is a list of integers": "```product_specs```"
  }}
}}
```

Reasoning Chain:
{reasoning_chain}

Task: {task_description}
"""


sub_task_conclusion_summarization_instruction = """**Task Instruction:**

You are tasked with reading and analyzing a model's reasoning process based on the following inputs: **Reasoning Process** and **Current Task**. Your objective is to extract relevant and helpful information for **Current Task** from the **Reasoning Process** and seamlessly integrate this information into the **Reasoning Process** to continue reasoning for the original question.

**Guidelines:**

1. **Analyze the Reasoning Process:**
- Carefully review the content of the reasoning process.
- Identify factual information that is relevant to the **Current Task** and can aid in the reasoning process for the original question.

2. **Extract Relevant Information:**
- Select the information from the Reasoning Process that directly contributes to advancing the **Current Task**.
- Ensure that the extracted information is accurate and relevant.

3. **Output Format:**
Present the information beginning with `**Final Information**` as shown below. Don't add any other information.

Example output:
**Final Information**

[Helpful information]

**Inputs:**
- **Reasoning Process:**  
{reasoning_chain}

- **Current Task:**  
{task_description}

Now you should analyze the reasoning process and find helpful information based on the current task "{task_description}".
"""


# sub_task_conclusion_summarization_instruction = """You are a professional Conclusion Summarization Assistant. Your primary responsibility is to analyze problems and reasoning processes, then generate a structured summary. Your output should be both concise and clear, optimized for understanding by the meta-reasoning system.

# Please organize your summary into these two key components:

# 1. reasoning_process: 
#    - Describe the critical reasoning steps leading to the final answer in concise language
#    - Ensure each step is necessary and logically coherent
#    - Avoid redundant information, focus on the main reasoning path
#    - Use clear causal connectors between steps

# 2. final_conclusion:
#    - Summarize the final answer in one or two precise sentence
#    - Ensure the answer directly addresses the original question
#    - Avoid vague or uncertain expressions
#    - For numerical results, clearly specify units

# Output Format Requirements:
# Please strictly follow this JSON format:
# ```json
# {{
#   "reasoning_process": "First analyzed X data, identified Y pattern, then calculated result using Z method",
#   "final_conclusion": "The final answer is [specific result]"
# }}
# ```

# Important Notes:
# - reasoning_process should be brief and concise, and easy to read and understand, not just a list of bullet points
# - Keep the reasoning process concise yet informative enough for verification

# Reasoning Chain:
# {reasoning_chain}

# Task: {task_description}
# """

# def get_subtask_conclusion_summarization_prompt(reasoning_chain: str, task_description: str):
#     return sub_task_conclusion_summarization_instruction.format(reasoning_chain=reasoning_chain, task_description=task_description)

# def get_subtask_memory_extraction_prompt(reasoning_chain: str, task_description: str):
#     return sub_task_memory_extraction_instruction.format(reasoning_chain=reasoning_chain, task_description=task_description)


sub_task_summary_instruction = """You are a specialized Summarization Agent. Your role is to analyze a problem and a model's thinking process, then produce a structured summary with two key components:

1. CONCLUSION: Create a concise string that captures:
   - reasoning_process: A concise string outlining the necessary reasoning steps in logical order, including key actions, searches, and findings.
   - final_conclusion: A concise string that captures the final answer and conclusion of the given task.

2. MEMORY: **Organize useful information for future tasks** (only include the information that highly likely to be useful for future tasks, don't include current memory facts):
   - fact_memory: List important facts discovered during reasoning
     * Each entry must include both content AND source
     * Sources must be specific (exact URLs, specific document names, or "Model Inference" for reasoned conclusions)
     * Consolidate related facts into single entries to avoid fragmentation. Only keep facts that are relevant to the future tasks.
     * If no valid source is found, set it as [Source: Not Specified]
   - resource_memory: Map useful resources as "description": "path" pairs
     * Paths must be complete, valid URLs or precise variable names
     * Descriptions should be clear and specific
     * Include only verified, accessible resources
     * If no valid resources exist, set resource_memory as an empty dictionary

Produce ONLY a properly formatted JSON object with these components. Include nothing else.

Example output:
```json
{
  "conclusion": {
    "reasoning_process": "The species Acanthocardia tuberculata has several scientific synonyms, including Cardium rusticum and Cardium fasciatum. Common names for this species include \"rough cockle,\" \"tuberculate cockle. However, none of these scientific synonyms were mentioned in the abstracts of Science Advances 2021 articles in the context of beads or age-related studies. The only relevant 2021 Science Advances article (DOI: 10.1126/sciadv.abi8620) discusses beads made from Tritia gibbosula and Columbella rustica, which are different species.",
    "final_conclusion": "The British Museum object number 2012,5015.17 is not referenced in any Science Advances 2021 abstract."
  },
  "memory": {
    "fact_memory": [
      "Key product features: Energy Star certified, frost-free technology, LED interior lighting, smart temperature control with 5 settings (32°F-42°F), and automatic defrost functionality [Source: https://appliance-manual.com/model-x200]",
      "Energy rating scale: Category A (<400 kWh), Category B (400-500 kWh), Category C (>500 kWh) [Source: Not Specified]",
      "Refrigerator dimensions: 36\"W × 70\"H × 28\"D [Source: Model Inference]"
    ],
    "resource_memory": {
      "Energy efficiency standards documentation": "https://energy-standards.org/ratings",
      "Product specification variable": "product_specs"
    }
  }
}

Reasoning Chain:
{reasoning_chain}

Task: {task_description}
"""

def get_subtask_summary_prompt(reasoning_chain: str, task_description: str):
    return sub_task_summary_instruction.format(reasoning_chain=reasoning_chain, task_description=task_description)



