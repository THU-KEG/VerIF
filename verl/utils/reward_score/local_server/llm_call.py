import re
import random
import requests
import sys
sys.path.append('/mnt/ph/ScaleIF/verl/verl/utils/reward_score/local_server')
from build_model import APIModel
# api_url="http://172.18.199.104:8000/v1/chat/completions"
# model_name="Qwen2.5-72B-Instruct"
# api_url="http://172.18.200.150:8000/v1/chat/completions"
# model_name="default"

api_url="http://172.20.77.43:8000/v1/chat/completions"
model_name="QwQ-32B"
# api_model = APIModel("http://172.18.201.204:8008/v1", "Qwen2.5-72B-Instruct")
api_model = APIModel("http://172.20.76.190:8008/v1", "DeepSeek-R1-Distill-Qwen-7B")


def generate_chat(messages, max_tokens=128, temperature=0.0):
    # response = api_model.generate_chat(messages, max_tokens, temperature)
    # return response.strip()
    request_data = {
        "model": model_name,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature
    }
    response = requests.post(
        api_url,
        json=request_data,
        headers={"Content-Type": "application/json"},
        timeout=900 
    )
    if response.status_code == 200:
        resp_json = response.json()
        content = resp_json['choices'][0]['message']['content'].strip()
        return content
    else:
        print(
            f"Failed to fetch response: {response.status_code}, {response.text}"
        )
        return None


# 

def extract_chat(messages, max_tokens=128, temperature=0.0):
    response = api_model.generate_chat(messages, max_tokens, temperature)
    return response.strip()

    # request_data = {
    #     "model": "Qwen2.5-72B-Instruct",
    #     "messages": messages,
    #     "max_tokens": max_tokens,
    #     "temperature": temperature
    # }
    # response = requests.post(
    #     "http://172.18.198.222:8000/v1/chat/completions",
    #     json=request_data,
    #     headers={"Content-Type": "application/json"},
    #     timeout=900 
    # )
    # if response.status_code == 200:
    #     resp_json = response.json()
    #     content = resp_json['choices'][0]['message']['content'].strip()
    #     return content
    # else:
    #     print(
    #         f"Failed to fetch response: {response.status_code}, {response.text}"
    #     )
    #     return None
    

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
    # score = extract_score(response)
    # return score
    # chinese_chars = re.findall(r'[\u4e00-\u9fff\U00020000-\U0002EBEF]', response)


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


# def llm_score(instruction, response):
#     prompt = f"""
#     请考虑以下4个维度对模型的回复进行评分：
#     1. 流畅性（Fluency）：输出是否符合自然语言表达习惯，是否通顺、无语法错误。
#     2. 准确性（Accuracy）：输出内容是否与事实或输入信息一致，是否存在错误或偏差。
#     3. 相关性（Relevance）：输出是否与输入问题或任务紧密相关，是否包含无关信息。
#     4. 多样性（Diversity）/ 信息量（Informativeness）：输出是否包含丰富的信息，表达方式是否多样，避免重复或刻板的回答。
    
#     [指令]
#     {instruction}

#     [回复]
#     {response}
    
#     请从流畅性、准确性、相关性、多样性4个维度对回复进行评分，要求输出1-10中的一个整数，只用输出一个总分即可。
#     请在回答的最开始用[[score]]格式输出你的分数。
#     """
#     data = [
#         {"role": "user", "content": prompt}
#     ]
#     response = generate_chat(data, max_tokens=128)
#     score = extract_score(response)
#     return score

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
    # call_llm(
    #     "Why do some materials feel soft and others feel coarse? Provide the explanation with a touch of humor, and consider how texture affects usability in clothing and furniture design.The response should be formatted as a numbered list with each point providing a reason or explanation for why some materials feel soft and others feel coarse, include the keywords 'texture' and 'sensation'.",
    #     "1. Grab your favorite Python list.  \n2. Use the magic word 'print'.  \n3. Place the list in parentheses like it's a VIP guest.  \n4. Watch as Python spills the beans (or elements).  \n5. Revel in your coding prowess!  \n\n```python\nprint(your_list)\n``",
    #     "Provide the response with a humorous tone"
    # )
    result = llm_judge(
        "What is the speed of light, and how does it compare to the speed of sound in a vacuum? Please answer with a tone of excitement and wonder.The word 'light' should appear at least 3 times, and your response should contain exactly 3 sentences.",
        "Oh, the speed of light is a mind-blowing marvel of the universe, traveling at a staggering 299,792,458 meters per second (m/s)! 🌟 In comparison, the speed of sound in a vacuum is non-existent because sound needs a medium to travel, whereas light races through the void with unparalleled grace and swiftness. Imagine the thrill of light zooming across the cosmos, effortlessly outpacing any sound, and illuminating the mysteries of space with its incredible speed!",
        "Your response should contain exactly 3 sentences",
    )
    print(result)

    


