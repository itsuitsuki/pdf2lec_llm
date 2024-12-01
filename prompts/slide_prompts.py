def get_each_page_prompt():
    EACH_PAGE_PROMPT_PATH = './prompts/each_page_prompt' # Path to the prompt file
    with open(EACH_PAGE_PROMPT_PATH, 'r') as file:
        each_page_prompt = file.read()
    return each_page_prompt

def get_detailed_prompt():
    DETAILED_PROMPT_PATH = './prompts/detailed_prompt' # Path to the prompt file
    with open(DETAILED_PROMPT_PATH, 'r') as file:
        detailed_prompt = file.read()
    return detailed_prompt

def get_brief_prompt():
    BRIEF_PROMPT_PATH = './prompts/brief_prompt' # Path to the prompt file
    with open(BRIEF_PROMPT_PATH, 'r') as file:
        brief_prompt = file.read()
    return brief_prompt
        
def get_summarizing_prompt():
    SUMMARIZING_PROMPT_PATH = './prompts/summarizing_prompt' # Path to the prompt file
    with open(SUMMARIZING_PROMPT_PATH, 'r') as file:
        summarizing_prompt = file.read()
    return summarizing_prompt

def get_each_page_prompt(complexity):
    if complexity == 1:
        return get_brief_prompt()
    if complexity == 2:
        return get_each_page_prompt()
    if complexity == 3:
        return get_detailed_prompt()
    else:
        return 1