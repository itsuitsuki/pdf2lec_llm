def get_default_prompt():
    DEFAULT_PROMPT_PATH = './prompts/default_prompt' # Path to the prompt file
    with open(DEFAULT_PROMPT_PATH, 'r') as file:
        default_prompt = file.read()
    return default_prompt

def get_brief_prompt():
    BRIEF_PROMPT_PATH = './prompts/brief_prompt' # Path to the prompt file
    with open(BRIEF_PROMPT_PATH, 'r') as file:
        brief_prompt = file.read()
    return brief_prompt

def get_detailed_prompt():
    DETAILED_PROMPT_PATH = './prompts/detailed_prompt' # Path to the prompt file
    with open(DETAILED_PROMPT_PATH, 'r') as file:
        detailed_prompt = file.read()
    return detailed_prompt

def get_slide_parsing_prompt():
    SLIDE_PARSING_PROMPT_PATH = './prompts/slide_parsing_agent.txt' # Path to the prompt file
    with open(SLIDE_PARSING_PROMPT_PATH, 'r') as file:
        slide_parsing_prompt = file.read()
    return slide_parsing_prompt

def get_each_slide_prompt(complexity):
    if complexity == 1:
        return get_brief_prompt()
    elif complexity == 2:
        return get_default_prompt()
    elif complexity == 3:
        return get_detailed_prompt()
    else:
        raise ValueError(f"Invalid complexity value {complexity}. Must be 1, 2, or 3.")
        
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