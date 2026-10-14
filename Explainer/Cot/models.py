import re
import json
from loguru import logger
from langchain.prompts.chat import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import SystemMessage, HumanMessage
import networkx as nx
import matplotlib.pyplot as plt
from .prompts import (
    ARGUE,
    ANSWER_DEFAULT,
    SYS_PROMPTS,
    ANSWER_PROMPTS,
    Parallel_argument,
    ARGUE_PROMPTS,
    OUTPUT_PROMPTS,
    prompt_dict
)
from .utils import load_data, INVALID_ANS
from arguments import args
import copy

COLORMAP = {
    "maybe": "yellow",
    "yes": "green",
    "no": "red",
    "comment": "lightgreen",
    "never": "gray",
    "done": "purple",
    "edge": "gray",
}

class LogiCoTEnv:
    def __init__(
        self,
        chat,
        max_steps=25,
        output=None,
        debug=False,
        parallel_argument_num=3,
        refined_tolerance=3
    ) -> None:
        self.chat = chat
        self.max_steps = max_steps
        self.thoughts_spliter = re.compile(r"(?:\n+|^)#\d+\.\s?")
        self.sys_template = SystemMessagePromptTemplate.from_template(
            SYS_PROMPTS
        )
        self.argue_template = SystemMessagePromptTemplate.from_template(
            ARGUE_PROMPTS
        )
        self.output_templates = {}
        for i in range(args.parallel_argument_num):
            self.output_templates[i] = SystemMessagePromptTemplate.from_template(
                OUTPUT_PROMPTS[i]
            )
        self.ans_template = HumanMessagePromptTemplate.from_template(ANSWER_PROMPTS)
        #self.answer_msg = HumanMessage(content=ANSWER_PROMPTS)
        self._debug = debug
        if debug:
            logger.level("DEBUG")
            logger.add("logs/debug/file_{time}.log")
        else:
            logger.level("INFO")
            logger.add("logs/release/file_{time}.log")
        self.output = output
        self.datas = load_data(self.output, format="dict")
        self.index_node = None
        self._recovered = False
        self.parallel_argument_num = parallel_argument_num
        self.refined_tolerance = refined_tolerance
        self.verifications = {}
        """self.sys_template_fin = SystemMessagePromptTemplate.from_template(
            SYS_PROMPTS_fin[prompt_version]
        )"""
        self._remove_col = lambda x: [
            t for t in self.thoughts_spliter.split(x) if len(t) > 2
        ][-1]

    def reset(self, data):
        terminate = self._reset(data)
        if terminate:
            state = None, None, None, False
            return state, terminate

        all_thoughts = []
        while len(all_thoughts)!=args.thinking_steps:
            if len(all_thoughts)>args.thinking_steps:
                all_thoughts = []
            init_content = ""
            if len(all_thoughts)>0:
                for index,step in enumerate(all_thoughts):
                    init_content = init_content+f"#{index+1}. "+step +"\n"

            init_sys_msg = SystemMessage(content=self.sys_msg[0].content + init_content + f"#{len(all_thoughts)+1}. ")
            #init_sys_msg = SystemMessage(content=self.sys_msg[0].content)
            resp = self.chat([init_sys_msg])
            resp.content = resp.content.split("#1. ")[-1] #剔除#1. 之前冗余的部分
            thoughts = self.thoughts_spliter.split(resp.content)  # 按“#1.”分割
            thoughts = [s for s in thoughts if len(s.strip()) != 0]
            all_thoughts = all_thoughts + thoughts
        chain = init_sys_msg.content + resp.content
        logger.debug("<< G0 CHAIN 1 >>\n" + chain)
        self.default_chain = {0:SystemMessage(content=chain)}
        thoughts = all_thoughts
        assert len(thoughts) == args.thinking_steps

        root = self.get_index(1, 0)
        self.G[0].add_node(
            root,
            thought=self.sys_msg[0].content,
            color=COLORMAP["yes"],
            refined=True,
            mod_num=0
        )
        self.G_idx = 0
        self._grow_thoughts_graph(thoughts, root, G_idx=self.G_idx)
        next_node = next(iter(self.G[0].neighbors(root)))
        self.index_node = next_node
        P = self.G[0].nodes[self.index_node]["thought"].replace('"', "'")
        refined = self.G[0].nodes[self.index_node]["refined"]
        mod_num = self.G[0].nodes[self.index_node]["mod_num"]

        #state = self.sys_msg, 1, P, refined, self.G_idx,mod_num
        state = self.argue_msg, 1, P, refined, self.G_idx,mod_num
        terminate = len(thoughts) < 1
        return state, terminate

    def _reset(self, data):
        terminate = False
        idx = data["idx"]
        if idx in self.datas:
            logger.warning(
                f"Loading existing data from {self.output} with idx {idx}. Terminate early."
            )
            data = self.datas[idx]
            #self.G = nx.node_link_graph(data["G"])
            self.data = data
            terminate = True
            return terminate

        self.G = {}
        for i in range(self.parallel_argument_num):
            self.G[i] = nx.DiGraph()

        self._last_node = None
        self._recovered = False

        self.parallel_arguments = {}
        for step,pr in Parallel_argument.items():
            self.parallel_arguments[step] = {}
            for i in range(self.parallel_argument_num):
                self.parallel_arguments[step][i] = Parallel_argument[step][i].format(CN1=data["ipc_pair"][0],CN2=data["ipc_pair"][1],CN1_info=data["ipc_info"][data["CN1"]],CN2_info=data["ipc_info"][data["CN2"]])

        self.sys_msg = {}
        self.argue_msg = {}
        self.output_msg = {}
        for i in range(self.parallel_argument_num):
            self.sys_msg[i] = self.sys_template.format(CN1=data["CN1"], CN2=data["CN2"],
                                                       CN1_info=data["ipc_info"][data["CN1"]],
                                                       CN2_info=data["ipc_info"][data["CN2"]],
                                                       Supporting_information=data["metapath_desc"][i],
                                                       Parallel_argument_s1=self.parallel_arguments["s1"][i],
                                                       Parallel_argument_s2=self.parallel_arguments["s2"][i],
                                                       Parallel_argument_s3=self.parallel_arguments["s3"][i],
                                                       Parallel_argument_s4=self.parallel_arguments["s4"][i])

            self.argue_msg[i] = self.argue_template.format(CN1=data["CN1"], CN2=data["CN2"],
                                                       CN1_info=data["ipc_info"][data["CN1"]],
                                                       CN2_info=data["ipc_info"][data["CN2"]],
                                                       Supporting_information=data["metapath_desc"][i],
                                                       Parallel_argument_s1=self.parallel_arguments["s1"][i],
                                                       Parallel_argument_s2=self.parallel_arguments["s2"][i],
                                                       Parallel_argument_s3=self.parallel_arguments["s3"][i],
                                                       Parallel_argument_s4=self.parallel_arguments["s4"][i])

            self.output_msg[i] = self.output_templates[i].format(CN1=data["CN1"], CN2=data["CN2"])

        self.answer_msg = self.ans_template.format(CN1=data["ipc_pair"][0],CN2=data["ipc_pair"][1])
        self.default_chain = None
        self.data = data
        logger.info(f"Reset for edge {idx}.")
        return terminate

    def get_next_node(self):
        next_nodes = list(self.G[self.G_idx].neighbors(self.index_node))
        if len(next_nodes) == 0:
            next_node = None
        else:
            next_node = next_nodes[0]
        return next_node

    def recover(self, e):
        self._recovered = True
        self._terminate()
        self.sys_msg = self.default_chain
        logger.warning(f"Recovered from: {e}")
        return

    def step(self, action):
        terminate = False
        G = self.G[self.G_idx]
        advice, passed, mod_num = action
        row, col = self.get_index_loc(self.index_node)
        if passed: #如果通过了验证，则将节点的颜色设为yes，并更新sys_msg，获取下一个节点
            G.nodes[self.index_node]["color"] = COLORMAP["yes"]
            self._update_sys_on(self.index_node)
            next_node = self.get_next_node()
        else: #如果没有通过验证，弃用原来的节点以及后续节点，新建节点（内容为修订后的步骤），然后新建思维链
            self._color_drop_branch(G, self.index_node)
            next_node = self.get_index(row + 1, col)
            G.add_node(
                next_node,
                thought=f"#{col}. {advice}\n",
                color=COLORMAP["comment"],
                refined=True,
                mod_num=mod_num
            )
            G.add_edge(self.index_node, next_node, color=COLORMAP["edge"])

            #需要生成的步骤
            all_thoughts = []
            while len(all_thoughts)!= args.thinking_steps - col:
                if len(all_thoughts)>args.thinking_steps - col:
                    all_thoughts = []
                if len(all_thoughts)>0:
                    new_content = ""
                    for index,step in enumerate(all_thoughts):
                        new_content = new_content+f"#{col+1+index}. "+step+"\n"
                    new_sys_msg = SystemMessage(
                        content=self.sys_msg[self.G_idx].content + self.G[self.G_idx].nodes[next_node][
                            "thought"] + "\n" +new_content+ f"#{col + 2+index}. "
                    )
                else:
                    new_sys_msg = SystemMessage(
                        content=self.sys_msg[self.G_idx].content + self.G[self.G_idx].nodes[next_node]["thought"] + f"#{col+1}. "
                    )
                resp = self.chat([new_sys_msg])
                resp.content = resp.content.split(f"#{col+1}. ")[-1]  # 剔除#{col+1}.之前冗余的部分
                thoughts = self.thoughts_spliter.split(resp.content)
                thoughts = [t for t in thoughts if t.strip() != ""]
                all_thoughts =all_thoughts+thoughts
            assert len(all_thoughts) == args.thinking_steps - col

            new_content = ""
            for index, step in enumerate(all_thoughts):
                new_content = new_content + f"#{col + 1 + index}. " + step + "\n"
            chain = (
                self.sys_msg[self.G_idx].content + self.G[self.G_idx].nodes[next_node]["thought"] + new_content
            )
            logger.debug(f"<< G{self.G_idx} CHAIN {row + 1} >>\n" + chain)

            self._grow_thoughts_graph(all_thoughts, next_node,self.G_idx)

        # 三条路径都结束
        #if col > self.max_steps or ((next_node is None or (mod_num > self.refined_tolerance and col > 2)) and self.G_idx == self.parallel_argument_num-1):
        if col > self.max_steps or ((next_node is None or (mod_num > self.refined_tolerance)) and self.G_idx == self.parallel_argument_num - 1):
            # TODO: if exceeds maxstep, just return the initial
            if mod_num > self.refined_tolerance:
                self.verifications[self.G_idx] = False
            else:
                self.verifications[self.G_idx] = True
            terminate = True
            self.terminate()
            return None, terminate
        # 当前路径结束步入下一条路径
        #elif (next_node is None or (mod_num > self.refined_tolerance and col > 2)) and self.G_idx < self.parallel_argument_num-1:
        elif (next_node is None or (mod_num > self.refined_tolerance)) and self.G_idx < self.parallel_argument_num - 1:
            if mod_num > self.refined_tolerance:
                self.verifications[self.G_idx] = False
            else:
                self.verifications[self.G_idx] = True
            self.G_idx += 1
            G0_root = self.get_root(self.G[0])
            G0_nodes_ = nx.shortest_path(self.G[0], G0_root, self._last_node[0])
            G0_nodes = [x for x in G0_nodes_ if self.G[0].nodes[x]["color"] != COLORMAP["no"]]

            root = self.get_index(1, 0)
            self.G[self.G_idx].add_node(
                root,
                thought=self.sys_msg[self.G_idx].content,
                color=COLORMAP["yes"],
                refined=True,
                mod_num=0
            )
            next_node = root
            """for i in range(1,3):
                CN_node = G0_nodes[i]
                CN_thought = self.G[0].nodes[CN_node]["thought"].replace('"', "'")
                previous_index = self.get_index(1, i - 1)
                index = self.get_index(1, i)
                self.G[self.G_idx].add_node(
                    index,
                    thought=f"#{i}. {CN_thought}\n",
                    color=COLORMAP["comment"],
                    refined=True,
                    mod_num=0
                )
                self.G[self.G_idx].add_edge(previous_index, index, color=COLORMAP["edge"])
                self.index_node = index"""

            #新思维链所有的步骤
            all_thoughts = []
            """all_thoughts.append(self._remove_col(self.G[0].nodes[G0_nodes[1]]["thought"].replace('"', "'")))
            all_thoughts.append(self._remove_col(self.G[0].nodes[G0_nodes[2]]["thought"].replace('"', "'")))"""

            while len(all_thoughts)!=args.thinking_steps:
                if len(all_thoughts)>args.thinking_steps:
                    all_thoughts = []
                    """all_thoughts.append(self._remove_col(self.G[0].nodes[G0_nodes[1]]["thought"].replace('"', "'")))
                    all_thoughts.append(self._remove_col(self.G[0].nodes[G0_nodes[2]]["thought"].replace('"', "'")))"""
                init_content = ""
                for index, step in enumerate(all_thoughts):
                    init_content = init_content +f"#{index+1}. "+ step + "\n"
                init_sys_msg = SystemMessage(content=self.sys_msg[self.G_idx].content + "\n" + init_content + f"#{len(all_thoughts)+1}. ")
                resp = self.chat([init_sys_msg])
                resp.content = resp.content.split(f"#{len(all_thoughts)+1}. ")[-1]
                thoughts = self.thoughts_spliter.split(resp.content)  # 按“#1.”分割
                thoughts = [s for s in thoughts if len(s.strip()) != 0]
                all_thoughts = all_thoughts + thoughts

            chain = init_sys_msg.content + resp.content
            logger.debug(f"<< G{self.G_idx} CHAIN 1 >>\n" + chain)
            self.default_chain[self.G_idx] = SystemMessage(content=chain)

            #self._grow_thoughts_graph(all_thoughts, self.index_node, G_idx=self.G_idx)
            self._grow_thoughts_graph(all_thoughts, root, G_idx=self.G_idx)
            #next_node = self.get_next_node()
            next_node = next(iter(self.G[self.G_idx].neighbors(root)))
            terminate = next_node is None
            #self.sys_msg[self.G_idx].content += self.G[0].nodes[G0_nodes[1]]["thought"].replace('"', "'") + "\n" + \
                                                #self.G[0].nodes[G0_nodes[2]]["thought"].replace('"', "'") + "\n"

        G = self.G[self.G_idx]
        self.index_node = next_node
        P = G.nodes[self.index_node]["thought"].replace('"', "'")
        refined = G.nodes[self.index_node]["refined"]
        mod_num = G.nodes[self.index_node]["mod_num"]
        _, col = self.get_index_loc(self.index_node)
        #state = self.sys_msg, col, P, refined,self.G_idx, mod_num
        state = self.argue_msg, col, P, refined, self.G_idx, mod_num
        return state, terminate

    def get_root(self,G):
        root_nodes = [node for node, in_degree in G.in_degree() if in_degree == 0]
        return root_nodes[0]

    def terminate(self):
        self._terminate()

    def _terminate(self):
        self.G[self.G_idx].nodes[self.index_node]["color"] = COLORMAP["done"]
        self._store_graph_callback()
        self._compute_answer_callback()
        self._save_data_callback()
        self.render()
        return

    def _store_graph_callback(self):
        # 存储图
        self.data["G"] = {}
        # 存储思维链
        self.data["sys_msg"] = {}
        for i in range(self.parallel_argument_num):
            G_dict = nx.node_link_data(self.G[i])
            self.data["G"][i] = G_dict
            self.data["sys_msg"][i] = self.sys_msg[i].content
        return
    """
    最后的结果输出，待修改
    """
    def _compute_answer_callback(self):
        output_path = {}

        G_nodes_ = {}
        G_nodes = {}
        for i in range(self.parallel_argument_num):
            Gi_root = self.get_root(self.G[i])
            G_nodes_[i] = nx.shortest_path(self.G[i], Gi_root, self._last_node[i])
            G_nodes[i] = [x for x in G_nodes_[i] if self.G[i].nodes[x]["color"] != COLORMAP["no"]]

        sys_msg_v2 = copy.deepcopy(self.sys_msg)
        for i in range(self.parallel_argument_num):
            if self.verifications[i]:
                sys_msg_v2[i].content.replace(prompt_dict["p3"],"")
                out = self.chat([sys_msg_v2[i], self.output_msg[i]]).content
                try:
                    out_list = json.loads(out)
                    output_path[i] = out_list
                except Exception as e:
                    output_path[i] = [out]
            else:
                output_path[i] = []
        self.data.update(
            {
                "output_path": output_path
            }
        )

        """recall = 1 #记录是否有答案
        G_nodes_ = {}
        G_nodes = {} #三条平行论证的最终结果思维链
        ver_result = {}
        for i in range(self.parallel_argument_num):
            Gi_root = self.get_root(self.G[i])
            G_nodes_[i] = nx.shortest_path(self.G[i], Gi_root, self._last_node[i])
            G_nodes[i] = [x for x in G_nodes_[i] if self.G[i].nodes[x]["color"] != COLORMAP["no"]]
            thought = self.G[i].nodes[G_nodes[i][3]]["thought"].replace('"', "'")
            ver_result[i] = self.remove_col(thought)

        #step1,step2
        self.sys_msg_fin = SystemMessage(
            content=self.G[0].nodes[G_nodes[0][1]]["thought"] + "\n" + self.G[0].nodes[G_nodes[0][2]]["thought"] + "\n"
        )
        col = 2
        #step3
        for i in range(self.parallel_argument_num):
            if self.verifications[i]:
                col += 1
                self.sys_msg_fin = SystemMessage(
                    content=self.sys_msg_fin.content + f"#{col}. {ver_result[i]}\n"
                )
        if col != 2:
            answer_logi = self._extract_answer(chain=self.sys_msg_fin)
        else:
            answer_logi = ANSWER_DEFAULT
            recall = 0
        self.data.update(
            {
                "answer_logi": answer_logi,
                "recall": recall,
                "recovered":self._recovered
            }
        )"""
        return

    def _extract_answer(self, chain):
        answer_pred = self.chat([chain, self.answer_msg]).content
        logger.debug(chain.content)
        logger.debug(self.answer_msg.content)
        logger.debug(answer_pred)
        return answer_pred

    def _save_data_callback(self):
        if self.output is None:
            return
        with open(self.output, "a+") as fout:
            data_json = json.dumps(self.data)
            fout.write(data_json + "\n")
            logger.info(f"Saved one line to {self.output}")
        return

    def render(self):
        if not self._debug:
            return

        def draw_graph(G, ax):
            labels = nx.get_node_attributes(G, "thought")
            colors = nx.get_node_attributes(G, "color").values()
            pos = nx.spring_layout(G)  # 生成布局
            nx.draw_networkx(G, pos, labels=labels, node_color=colors, ax=ax)

        fig, axs = plt.subplots(1, self.parallel_argument_num, figsize=(15, 5))
        for i in range(self.parallel_argument_num):
            draw_graph(self.G[i], axs[i])
        plt.show()
        return

    def get_index(self, row, col):
        return f"T[{row}, {col}]"

    def get_index_loc(self, index):
        row, col = index[2:-1].split(",")
        return int(row.strip()), int(col.strip())

    def _grow_thoughts_graph(self, thoughts, index,G_idx):
        row, col = self.get_index_loc(index)
        for thought in thoughts:
            thought = thought.strip("\n")
            thought = thought.strip("<step> ")
            thought = thought.strip(" </step>")
            col += 1
            previous_index = self.get_index(row, col - 1)
            index = self.get_index(row, col)
            self.G[G_idx].add_node(
                index,
                thought=f"#{col}. {thought}",
                color=COLORMAP["maybe"],
                refined=False,
                mod_num=0
            )
            self.G[G_idx].add_edge(previous_index, index)
        if G_idx == 0:
            self._last_node = {}
        self._last_node[G_idx] = index
        # self.render()
        return

    def _update_sys_on(self, index):
        self.sys_msg[self.G_idx] = SystemMessage(
            content=self.sys_msg[self.G_idx].content + self.G[self.G_idx].nodes[index]["thought"] + "\n"
        )

        self.argue_msg[self.G_idx] = SystemMessage(
            content=self.argue_msg[self.G_idx].content + self.G[self.G_idx].nodes[index]["thought"] + "\n"
        )
        return

    def _color_drop_branch(self, G, drop_start): ## 将起始节点的颜色设置为"no"，后续节点颜色设置为"never"
        G.nodes[drop_start]["color"] = COLORMAP["no"]
        drop_nodes = nx.shortest_path(G, drop_start, self._last_node[self.G_idx])
        for dnode in drop_nodes[1:]:
            G.nodes[dnode]["color"] = COLORMAP["never"]
        return

    def remove_col(self,x):
        thought_polisher = re.compile(r"#\d+[:\.]\s")
        result = [t for t in thought_polisher.split(x) if len(t) > 2][-1]
        return result

class LogiAgent:
    def __init__(self, chat, check_refined=True, mode="argue_review") -> None:
        self.chat = chat
        self._mode = mode  # ['negation', 'argue_review', 'argue_noreview']
        self.thought_polisher = re.compile(r"#\d+[:\.]\s")
        self._remove_col = lambda x: [
            t for t in self.thought_polisher.split(x) if len(t) > 2
        ][-1]
        self.check_refined = check_refined
        self.mod_num = 0

    def policy(self, state):
        sys_msg, col, P, refined, G_idx,mod_num = state
        P = self._remove_col(P) #去掉“#1.”
        if refined and not self.check_refined:
            return P, True,mod_num
        if self._mode.startswith("argue"):
            # Role maters
            self.PT_template = HumanMessagePromptTemplate.from_template(ARGUE.PT)
            self.PF_template = HumanMessagePromptTemplate.from_template(ARGUE.PF)
            self.PJ_template = HumanMessagePromptTemplate.from_template(ARGUE.PJ)
            PT_msg = self.PT_template.format(P=P, col=col)
            msgs = [sys_msg[G_idx], PT_msg]
            # logger.debug(sys_msg.content + PT_msg.content)
            PT = self.chat(msgs).content #引导大模型推理推理步骤col为什么为真

            PF_msg = self.PF_template.format(P=P, col=col)
            msgs = [sys_msg[G_idx], PF_msg]
            # logger.debug(sys_msg.content + PF_msg.content)
            PF = self.chat(msgs).content #引导大模型推理推理步骤col为什么为假
            PJ_msg = self.PJ_template.format(P=P, PT=PT, PF=PF, col=col)
            msgs = [sys_msg[G_idx], PJ_msg]
            choice = self.chat(msgs).content #利用PT和PF引导大模型推理推理步骤col是真还是假
            logger.debug("PTF msg:\n" + sys_msg[G_idx].content + PJ_msg.content)
            logger.debug("Choice:\n" + choice)
            if " is true" in choice.lower(): #如果步骤col是真,设置修改次数为0
                advice = P
                passed = True
            elif " is false" in choice.lower(): #如果步骤col是假
                if self._mode == "argue_noreview":
                    self.PR_template = HumanMessagePromptTemplate.from_template(
                        ARGUE.PR_N
                    )
                    PFR_msg = self.PR_template.format(P=P, col=col) #不包含评论的修订模板
                else:
                    self.PR_template = HumanMessagePromptTemplate.from_template(
                        ARGUE.PR
                    )
                    PFR_msg = self.PR_template.format(P=P, PF=PF, col=col) #包含评论的修订模板
                logger.debug(sys_msg[G_idx].content + PFR_msg.content)
                PFR = self.chat([sys_msg[G_idx], PFR_msg]).content #引导大模型进行修订
                advice = self._remove_col(PFR)
                logger.info("Refinement:\n" + advice)
                passed = False
                mod_num += 1
            else:
                logger.warning(choice)
                advice = P
                passed = True

        return advice, passed, mod_num #advice:步骤col的内容，passed:是否通过,mod_num:验证后的修改的次数