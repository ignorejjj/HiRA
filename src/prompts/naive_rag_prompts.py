def get_naive_rag_instruction(question, documents, use_boxed=False):
    if use_boxed:
        box_message = 'You should provide your final answer in the format \\boxed{YOUR_ANSWER}.\n\n'
    else:
        box_message = ''
    return (
        "You are a knowledgeable assistant that uses the provided documents to answer the user's question.\n\n"
        "Documents:\n"
        f"{documents}\n"
        f"{box_message}"
        "Question:\n"
        f"{question}\n"
    )


def get_naive_rag_instruction_withmemory(question, documents, memory, use_boxed=False):
    if use_boxed:
        box_message = 'You should provide your final answer in the format \\boxed{YOUR_ANSWER}.\n\n'
    else:
        box_message = ''
    return (
        "You are a knowledgeable assistant that uses the provided documents to answer the user's question. You can also use previous memory to help you answer the question.\n\n"
        "Documents:\n"
        f"{documents}\n"
        "Previous Memory:\n"
        f"{memory}\n"
        f"{box_message}"
        "Question:\n"
        f"{question}\n"
    )