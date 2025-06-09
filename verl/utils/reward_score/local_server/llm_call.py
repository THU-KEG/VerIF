import re
from build_model import APIModel

api_url=""
model_name=""
api_model = APIModel(api_url, model_name)


def generate_chat(messages, max_tokens=128, temperature=0.0):
    response = api_model.generate_chat(messages, max_tokens, temperature)
    return response.strip()


def extract_chat(messages, max_tokens=128, temperature=0.0):
    response = api_model.generate_chat(messages, max_tokens, temperature)
    return response.strip()


prompt_template = """
请判断以下文本是否满足给定的约束，仅回答是或否，不要输出其他内容。


原始指令：$I$

文本：$R$

约束：$C$

原始指令描述了基本的任务信息，给定的约束介绍了应该满足的具体的一个约束。
请判断以下文本是否满足给定的这个约束（仅仅判断是否满足给定的约束），仅回答是或否，不要输出其他内容。
"""


def llm_judge(instruction, response, constraint):
    if isinstance(response, list):
        response = "\n\n".join(response)
    prompt = prompt_template.replace("$R$", response).replace("$C$", constraint).replace("$I", instruction)
    data = [
        {"role": "user", "content": prompt}
    ]
    response = generate_chat(data)

    return response[0] == "是"


def llm_extract(instruction, response, specific_prompt):
    prompt_suffix = "\n请直接输出文本中的原文信息，不要改写，不要添加任何额外的信息。"

    prompt = f"文本：{response}\n抽取要求：{specific_prompt}" + prompt_suffix
    data = [
        {"role": "user", "content": prompt}
    ]
    response = extract_chat(data, max_tokens=1024)
    return response


def extract_score(text):
    match = re.search(r'\[\[(\d+)\]\]', text)
    try:
        return int(match.group(1))
    except:
        return 0


def llm_score(instruction, response, checkers):
    prompt = f"""
    请判断给定的回复是否遵循指令中的约束，比如长度、风格、格式等约束。
    
    [指令]
    {instruction}

    [回复]
    {response}

    [约束]
    {checkers}

    请判断给定的回复是否遵循指令中的约束，比如长度、风格、格式等约束。
    请在回答的最开始用[[score]]格式输出你的分数。
    如果遵循所有的约束，请输出[[1]]，否则输出[[0]]
    """
    data = [
        {"role": "user", "content": prompt}
    ]
    response = generate_chat(data, max_tokens=4096)
    score = extract_score(response)
    return score



if __name__ == "__main__":
    result = llm_judge(
        "What is the speed of light, and how does it compare to the speed of sound in a vacuum? Please answer with a tone of excitement and wonder.The word 'light' should appear at least 3 times, and your response should contain exactly 3 sentences.",
        "Oh, the speed of light is a mind-blowing marvel of the universe, traveling at a staggering 299,792,458 meters per second (m/s)! 🌟 In comparison, the speed of sound in a vacuum is non-existent because sound needs a medium to travel, whereas light races through the void with unparalleled grace and swiftness. Imagine the thrill of light zooming across the cosmos, effortlessly outpacing any sound, and illuminating the mysteries of space with its incredible speed!",
        "Your response should contain exactly 3 sentences",
    )
    print(result)

    


