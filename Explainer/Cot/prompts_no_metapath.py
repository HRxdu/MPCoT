from easydict import EasyDict

SYS_PROMPT_p1 = """Acting as an AI expert proficient in analysis, please provide a detailed analysis of the following information.
<information>
The connotation of IPC number {CN1} is: {CN1_info}.
The connotation of IPC number {CN2} is: {CN2_info}.
</information>
"""

SYS_PROMPT_p2 = """According to the information about the patents or papers in the metapaths, analyse step by step and present your analysis in detail (Format the results of each step as an unordered list. Your analysis should quote exact and relevant information as support):
#1. {Parallel_argument_s1}
#2. {Parallel_argument_s2}
#3. {Parallel_argument_s3}
#4. {Parallel_argument_s4}
"""

SYS_PROMPT_p3 = """Answer format: 
#1. The first step of the analysis process. 
#2. The second step of the analysis process. 
#3. The third step of the analysis process. 
#4. The fourth step of the analysis process.
"""

SYS_PROMPT_p4 = """Answer in detail for each step:(Your answer should quote exact quote as support.)
"""

prompt_dict = {
    "p1":SYS_PROMPT_p1,
    "p2":SYS_PROMPT_p2,
    "p3":SYS_PROMPT_p3,
    "p4":SYS_PROMPT_p4
}

SYS_PROMPTS = prompt_dict["p1"] + "\n" + prompt_dict["p2"] + "\n" + prompt_dict["p3"] + "\n" + prompt_dict["p4"]
ARGUE_PROMPTS = prompt_dict["p1"] + "\n" + prompt_dict["p4"]

# 1.找到CN1(CN2)相关的专利或论文的共同点
# 2.CN2(CN1)如何应用在这些共同点上
# 3.分析将CN2引入CN1能实现什么技术创新
# 4.分析将CN2引入CN1能在大模型领域实现什么技术创新
## Analyze why and how {CN1} and {CN2} might converge in the field of large language model in the future.

Parallel_argument_s1 = {
    0:"Identify components or modules related to {CN1} that could utilize the technology described in IPC classification {CN2}.",
    1:"Identify components or modules related to {CN2} that could utilize the technology described in IPC classification {CN1}.",
    2:"Identify components or modules that not only could utilize the technologies described in IPC classification {CN1}, but also could utilize the technologies described in IPC classification {CN2}."
}
Parallel_argument_s2 = {
    0: "Analyze how {CN2} used in the components or modules related to {CN1} that could utilize the technology described in IPC classification {CN2}.",
    1: "Analyze how {CN1} used in the components or modules related to {CN2} that could utilize the technology described in IPC classification {CN1}.",
    2: "Analyze how the technologies described in both IPC classifications {CN1} and {CN2} are used together in the components or modules that could utilize both these technologies."
}

Parallel_argument_s3 = {
    0:"Analyze whether introducing {CN2} technology into {CN1} technology could achieve meaningful scientific and technological innovation. If yes, please specify what problems can be solved for {CN1} if the technology of {CN2} is introduced. If not, please explain why.",
    1:"Analyze whether introducing {CN1} technology into {CN2} technology could achieve meaningful scientific and technological innovation. If yes, please specify what problems can be solved for {CN2} if the technology of {CN1} is introduced. If not, please explain why.",
    2:"Analyze whether the convergence of {CN1} technology and {CN2} technology could achieve meaningful scientific and technological innovation for the components or modules. If yes, please specify what problems can be solved for the components or modules. If not, please explain why."
}
Parallel_argument_s4 = {
    0:"Analyze and summarize the implementation path of the convergence of {CN1} and {CN2} in the field of large language model. (Emphasize the implementation of the convergence of {CN1} and {CN2} in the field of large language models.)",
    1:"Analyze and summarize the implementation path of the convergence of {CN1} and {CN2} in the field of large language model. (Emphasize the implementation of the convergence of {CN1} and {CN2} in the field of large language models.)",
    2:"Analyze and summarize the implementation path of the convergence of {CN1} and {CN2} in the field of large language model. (Emphasize the implementation of the convergence of {CN1} and {CN2} in the field of large language models.)"
}

## 最后：分析大语言模型领域技术融合的原因以及总结技术融合的实现路径

Parallel_argument = {
    "s1":Parallel_argument_s1,
    "s2":Parallel_argument_s2,
    "s3":Parallel_argument_s3,
    "s4":Parallel_argument_s4
}

ARGUE_PT = """
#{col}. {P}

---------------------------------
Let's clarify the reasoning step #{col} for the the consistency with the quesion premises or with previous reasoning steps.
Clarification of the #{col} step():
Step #{col} is true because ..."""

# PF_prompt = PT_prompt.replace('is true because', 'is false because')

ARGUE_PF = """
#{col}. {P}

---------------------------------
Let's criticize the reasoning step #{col} for the the conflictions to the quesion premises or to previous reasoning steps.
Criticisim of the #{col} step:
Step #{col} is false because ..."""


ARGUE_PJ = """
#{col}. {P}

---------------------------------
Verification of the #{col} step:
Let's check two different reviews (X and Y).
Suport the more plausible one and criticise the other one.
Review X: <review> step #{col} is TRUE because {PT} </review>
Review Y: <review> step #{col} is FALSE because {PF} </review>

Let's start by analyzing one by one:
I. What are the premises and previous steps to support the verification of step #{col}? (Your answer should quote exact quote as support.)
II. Criticise the incorrect review.
(Note the examined step doesn't have to tackle the whole problem at once.)
Finally, identify whether step #{col} is true or false.(Please summarize 'step #{col} is true' or 'step #{col} is false')

Analysis and conclusion:
"""

ARGUE_PR_NOREIVEW = """
Revision for the step #{col}:
---------------------------------
Original step #{col}: 
<Step {col}>
{P}
</Step {col}>
---------------------------------
(Hint: It is not good to directly adopt the step #{col}.)
Let's revise for a better version based on the question premises and on the reasoning steps so far.
The revision should keep the question and options intact. Only revise the reasoning process.
Revision of step #{col}:
"""

ARGUE_PR = """
Revision for the step #{col}:
---------------------------------
Original step #{col}: 
<Step {col}>
{P}
</Step {col}>
---------------------------------
(Hint: It is not good to directly adopt the step #{col} because there is a review says <review> {PF} </review>.)
Let's revise for a better version of step #{col} based on the question premises and on the reasoning steps so far.
(Note: You only need to modify the incorrect part of the original step #{col} and return the modified version of Step #{col}.)
Revision of step #{col}:
"""

ARGUE = EasyDict(
    {
        "PT": ARGUE_PT,
        "PF": ARGUE_PF,
        "PJ": ARGUE_PJ,
        "PR": ARGUE_PR,
        "PR_N": ARGUE_PR_NOREIVEW,
    }
)

# 将具体实践路径以字典形式输出
OUTPUT_PROMPTS = {
    0:"""·According to the above analysis steps, extract the specific implementation methods of the convergence of {CN1} and {CN2} in the field of large language model from the perspective of introducing {CN2} into {CN1}, and explain the reasons step by step.
    ·For the reasons behind each implementation method, you should clarify the connotations of {CN1} and {CN2}.
    ·You should provide the reasons as far as possible in accordance with the four analytical steps mentioned above.
    
    Please organize your results in the following format:
    [  
    {{"implementation_method": "implementation_method1","reason":"reason1"}},  
    {{"implementation_method": "implementation_method2","reason":"reason2"}}  
    ] """,

    1:"""·According to the above analysis steps, extract the specific implementation methods of the convergence of {CN1} and {CN2} in the field of large language model from the perspective of introducing {CN1} into {CN2}, and explain the reasons step by step.
    ·For the reasons behind each implementation method, you should clarify the connotations of {CN1} and {CN2}.
    ·You should provide the reasons as far as possible in accordance with the four analytical steps mentioned above.
    
    Please organize your results in the following format:
    [  
    {{"implementation_method": "implementation_method1","reason":"reason1"}},  
    {{"implementation_method": "implementation_method2","reason":"reason2"}}  
    ] """,

    2:"""·According to the above analysis steps, extract the specific implementation methods of the convergence of {CN1} and {CN2} in the field of large language model from the perspective of introducing the convergence of {CN1} and {CN2} into another technology, and explain the reasons step by step.
    ·For the reasons behind each implementation method, you should clarify the connotations of {CN1} and {CN2}.
    ·You should provide the reasons as far as possible in accordance with the four analytical steps mentioned above.
    
    Please organize your results in the following format:
    [  
    {{"implementation_method": "implementation_method1","reason":"reason1"}},  
    {{"implementation_method": "implementation_method2","reason":"reason2"}}  
    ] """
}

# 融合的具体方式（实现路径）
# 对于每一个实现路径的原因，应该阐明两个分类号的内涵，并且引用相关专利或论文的信息加以支持。尽可能按以上四个步骤依次进行解释。

SYS_PROMPTS_fin_Pr = """Let's learn the following meta-paths and their information:
IPCs: {ipc_desc}
Patents: {patent_desc}
Papers: {paper_desc}
Meta-paths: {metapath_desc}

Based on the given information as well as the known information, analyze why and how {CN1} and {CN2} might converge in the field of large language model in the future.
Let's think step by step.

Answer:
"""

SYS_PROMPTS_fin_noPr = """Let's learn the following meta-paths and their information:
IPCs: {ipc_desc}
Patents: {patent_desc}
Meta-paths: {metapath_desc}

Based on the given information as well as the known information, analyze why and how {CN1} and {CN2} might converge in the field of large language model in the future.
Let's think step by step.

Answer:
"""
SYS_PROMPTS_fin = {
    0: SYS_PROMPTS_fin_noPr,
    1: SYS_PROMPTS_fin_Pr
}
ANSWER_PROMPTS = "Integrate the above aspects to explain why and how {CN1} and {CN2} might converge in the field of large language model in the future: (Your answer should quote exact quote as support.)"
ANSWER_DEFAULT = "Actually, the probability of the future convergence of this pair of IPCs is very low."