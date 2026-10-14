import json
import pandas as pd
import ast

if __name__ == '__main__':
    data = pd.read_excel("output/lot/metapath_explaination_combine.xlsx")
    data_dicts = data.to_dict(orient="records")

    process_dicts=[]
    for record in data_dicts:
        exp_record = {}
        exp_record["idx"] = record["idx"]
        exp_record["ipc_pair"] = record["ipc_pair"]
        exp_record["ipc_info"] = record["ipc_info"]
        exp_record["metapath_desc"] = record["metapath_desc"]

        for prompt_type in ["output_path","output_path_without_metapath"]:
            output_path = ast.literal_eval(record[prompt_type]) #所有的路径
            for chain in range(3):
                chain_paths = output_path[str(chain)] #每个思维链的路径
                for path in chain_paths:
                    exp_record_new = {k: v for k, v in exp_record.items()}
                    exp_record_new.update({"prompt_type":prompt_type,"chain":chain,"implementation_path":path})
                    process_dicts.append(exp_record_new)

    process_result = pd.DataFrame(process_dicts)
    process_result.to_excel("output/lot/explaination_process_result.xlsx",index=False)

    print(data.info())