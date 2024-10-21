def get_each_page_prompt():
    EACH_PAGE_PROMPT_PATH = './prompts/each_page_prompt' # Path to the prompt file
    with open(EACH_PAGE_PROMPT_PATH, 'r') as file:
        each_page_prompt = file.read()
    return each_page_prompt
        
def get_summarizing_prompt():
    SUMMARIZING_PROMPT_PATH = './prompts/summarizing_prompt' # Path to the prompt file
    with open(SUMMARIZING_PROMPT_PATH, 'r') as file:
        summarizing_prompt = file.read()
    return summarizing_prompt

def get_introduction_prompt():
    INTRODUCTION_PROMPT_PATH = './prompts/intro_prompt' # Path to the prompt file
    with open(INTRODUCTION_PROMPT_PATH, 'r') as file:
        introduction_prompt = file.read()
    return introduction_prompt