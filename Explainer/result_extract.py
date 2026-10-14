import json
import pandas as pd


if __name__ == '__main__':
    # 打开 JSONL 文件
    file_path = 'output/lot/output_dev_no_metapath.jsonl'  # 替换为你的 JSONL 文件路径

    # 创建一个空列表来存储提取的数据
    data_list = []
    keys_to_extract = ["ipc_pair", "ipc_info","metapath_desc","idx","sys_msg", "output_path"]

    # 打开文件并逐行读取
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            # 将每一行解析为 JSON 对象
            json_object = json.loads(line)
            new_dict = {key: json_object[key] for key in keys_to_extract if key in json_object}

            total_length = 0
            metapath_desc = new_dict["metapath_desc"]
            for value in metapath_desc.values():
                total_length += len(value)

            if total_length>10:
                data_list.append(new_dict)
            else:
                continue

    df = pd.DataFrame(data_list)
    df.to_excel("output/lot/metapath_explaination_no_metapath.xlsx")

    """with open('output/output_list.json','w',encoding='utf-8') as f:
        json.dump(data_list,f,indent=4)"""
    # 输出提取的数据
    print(data_list)