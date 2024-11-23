def get_clarity_prompt():
    CLARITY_PROMPT_PATH = './prompts/clarity_agent' # Path to the prompt file
    with open(CLARITY_PROMPT_PATH, 'r') as file:
        clarity_prompt = file.read()
    return clarity_prompt

def get_engagement_prompt():
    ENGAGEMENT_PROMPT_PATH = './prompts/engagement_agent' # Path to the prompt file
    with open(ENGAGEMENT_PROMPT_PATH, 'r') as file:
        engagement_prompt = file.read()
    return engagement_prompt

def get_assembler_prompt():
    ASSEMBLER_PROMPT_PATH = './prompts/assembler_agent' # Path to the prompt file
    with open(ASSEMBLER_PROMPT_PATH, 'r') as file:
        assembler_prompt = file.read()
    return assembler_prompt