import pandas as pd
import ast
import re
import json
import os

from langchain_community.chat_models import ChatOpenAI
from openai import OpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)

sys_template = """Please act as an impartial judge and evaluate the implementation method and reason for the convergence of the two technologies involved in the IPC pair within the field of large language models, based on the information from the IPC, as well as the titles and abstracts of related patents or papers in the meta-path. Assign a score according to the following four levels:
RATING-1: Unreasonable implementation method - The implementation method only involves one IPC in the IPC pair or the "reason" only involves one IPC in the IPC pair. Alternatively, the technologies involved in this IPC pair cannot be converged in the field of large language models through this implementation method.
RATING-2: Reasonable implementation method, unacceptable reason - The implementation method involves the two IPCs of the IPC pair. And the technologies involved in this IPC pair can be converged in the field of large language models through this implementation method. However, the "reason" cannot sufficiently justify the convergence of technologies. For example, there are errors in the description of IPCs or there are errors in the content of patent/paper (if the "reason" cites specific patents or papers). Or, the "reason" did not mention what problem could be solved by the convergence of the two IPCs, which means that it just only separately explained the independent problems that each IPC can solve.
RATING-3: Reasonable implementation method, acceptable reason - The implementation method involves the convergence of the IPC pair. And the technologies involved in this IPC pair can be converged in the field of large language models through this implementation method. In addition, there is no errors in the information of IPCs or errors in the description of patent/paper (if the "reason" cites specific patents or papers). Moreover, the “reason” also includes an explanation of what problem could be solved by the convergence of the two IPC. However, the "reason" lacks some reliable information to enhance its persuasiveness. For example, none of patents or papers were cited in the "reason" to support the conclusion. Or, the “reason” does not involve anything about the way of technology convergence, despite not elaborating on the details.
RATING-4: Reasonable implementation method, satisfying reason - The implementation method involves the convergence of the IPC pair. And the technologies involved in this IPC pair can be converged in the field of large language models through this implementation method. In addition, there is no errors in the information of IPCs or errors in the description of patent/paper (if the "reason" cites specific patents or papers), and the "reason" has included an explanation of what problem could be solved by the convergence of the two IPC. Moreover, the "reason" has cited one or more specific patents or papers to enhance its persuasiveness, even though the patent or paper is only related to one IPC in the IPC pair. And the “reason” has involved what problems the convergence of technologies can solve and the way of technology convergence (Elaborating on the details is not necessary).
Notes：1. The patents or papers cited are generally related to only one IPC.  If such patents or papers have been referenced, or the cited patents or papers do not involve both two IPCs, it shouldn't fall under the category of information error in RATING-2 or the category of lack of reliable information in RATING-3.  
2.  Moreover, even though the cited patent or paper does not mention one of the IPCs, it cannot be concluded that there is no connection between the patent or paper and that IPC (there may be an indirect connection).If partial content of the patent or paper can utilize the technology involved in that IPC, it is considered that there is an indirect connection between the patent or paper and that IPC, and it shouldn't fall under the category of information error in RATING-2.
3. If the “reason” cites a patent or paper as an example to demonstrate that technologies can be integrated to improve that patent or paper, and although the patent or paper is only related to one IPC, another unrelated IPC can be applied to improve it, then this does not fall within the scope of erroneous information in RATING-2.
Please give your score in the form of <br>RATING</br>, for example, if the rating is 1, output <br>RATING-1</br>.  Please strictly follow the standards for scoring and do not create your own criteria. Pay special attention to the content of the Notes. Be as objective as possible.
"""

hum_template = """ipc pair: {ipc_pair}
ipc information: {ipc_info}
{metapath_desc}
The implementation method and reason: {implementation_path}"""

RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
RESET = "\033[0m"

def extract_score(text):
    pattern = r"<br>(.*?)</br>"
    matches = re.findall(pattern, text)
    score = int(matches[0][-1])
    return score

def check(file_path,path_idx):
    if not os.path.exists(file_path):
        return False
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            data = json.loads(line)
            if data["path_idx"] == path_idx:
                return True
    return False

if __name__ == '__main__':
    data = pd.read_excel(r"output\lot\rating\human_rating\sampled_data_for_ai_rating_same_with_human.xlsx")
    data = data.to_dict(orient="records")
    print("共{}条数据。".format(len(data)))

    """chat = ChatOpenAI(
        base_url="https://api.deepseek.com",
        model_name="deepseek-reasoner",
        openai_api_key="sk-ac7364efef8d48ba9bdf02fae50a0020"
    )"""
    client = OpenAI(
        # 此为默认路径，您可根据业务所在地域进行配置
        base_url="https://ark.cn-beijing.volces.com/api/v3",
        # 从环境变量中获取您的 API Key。此为默认方式，您可根据需要进行修改
        api_key="8b499071-dc00-4ead-a100-30089e342311",
    )

    result_data_path = "output/lot/rating/ai_rating_doubao.jsonl"
    for path_idx,record in enumerate(data):
        # 检查是否已经评分
        exist = check(result_data_path,path_idx)
        if exist:
            print(f"path_idx{path_idx}评分结果已存在于文件 {result_data_path} 中。")
            continue

        sys_msg = SystemMessagePromptTemplate.from_template(sys_template)
        hum_msg = HumanMessagePromptTemplate.from_template(hum_template)
        chat_prompt = ChatPromptTemplate.from_messages([sys_msg, hum_msg])

        chain = record["chain"]
        if not record["metapath_desc"].endswith("'}") and not record["metapath_desc"].endswith('"}'):
            try:
                record["metapath_desc"] += '"}'
                record_metapath_desc = ast.literal_eval(record["metapath_desc"].replace("\n",""))
                record_metapath_desc["2"] = record_metapath_desc["0"] + "\n" + record_metapath_desc["1"]
            except Exception as  e:
                record["metapath_desc"] += "'}"
                record_metapath_desc = ast.literal_eval(record["metapath_desc"].replace("\n", ""))
                record_metapath_desc["2"] = record_metapath_desc["0"] + "\n" + record_metapath_desc["1"]

        else:
            record_metapath_desc = ast.literal_eval(record["metapath_desc"])

        if record["prompt_type"] == "output_path":
            formatted_prompt = chat_prompt.format_prompt(ipc_pair=record["ipc_pair"],
                                                         ipc_info=record["ipc_info"],
                                                         metapath_desc=record_metapath_desc[str(chain)],
                                                         implementation_path=record["implementation_path"]
                                                         ).to_messages()
        if record["prompt_type"] == "output_path_without_metapath":
            formatted_prompt = chat_prompt.format_prompt(ipc_pair=record["ipc_pair"],
                                                         ipc_info=record["ipc_info"],
                                                         metapath_desc="\n",
                                                         implementation_path=record["implementation_path"]
                                                         ).to_messages()

        while True:
            try:
                print(BLUE + formatted_prompt[0].content+"\n"+formatted_prompt[1].content+ RESET)
                #response = chat(formatted_prompt)
                #score = extract_score(response.content)
                response = client.chat.completions.create(
                    # 指定您创建的方舟推理接入点 ID，此处已帮您修改为您的推理接入点 ID
                    model="doubao-seed-1-6-thinking-250715",
                    messages=[
                        {"role": "system", "content": formatted_prompt[0].content},
                        {"role": "user", "content": formatted_prompt[1].content},
                    ],
                    max_tokens=8192
                )
                score = extract_score(response.choices[0].message.content)
                break
            except Exception as e:
                print("RATING failed with error:", e)
                print("Retrying...")

        record["path_idx"] = path_idx
        record["AI_rating"] = score
        with open(result_data_path, "a", encoding="utf-8") as file:
            json.dump(record, file, ensure_ascii=False)
            file.write("\n")  # JSONL 格式要求每条记录占一行
        if score<4:
            print(RED + response.choices[0].message.model_extra["reasoning_content"]+ RESET)
        print(YELLOW +f"path_idx{path_idx}评分结果已成功添加到文件 {result_data_path} 中。type:{record['prompt_type']},score:{score}"+ RESET)
