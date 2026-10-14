import pandas as pd
import scipy.stats as stats
from statsmodels.stats.inter_rater import aggregate_raters, fleiss_kappa
import kendall_w as kw
import json
import numpy as np

def read_jsonl(file_path):
    dict_list = []  # 用于存储解析后的字典
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            # 解析每一行为 JSON 对象并添加到列表中
            dict_list.append(json.loads(line))
    return pd.DataFrame(dict_list)

if __name__ == '__main__':
    group1 = read_jsonl("output/lot/rating/ai_rating.jsonl")
    group2 = read_jsonl("output/lot/rating/ai_rating_doubao.jsonl")
    group3 = read_jsonl("output/lot/rating/ai_rating_gemini.jsonl")

    match_cols = ["idx","prompt_type","chain","implementation_path"]
    group2_match_tuples = set(group2[match_cols].apply(tuple, axis=1))
    group1 = group1[group1[match_cols].apply(tuple, axis=1).isin(group2_match_tuples)]

    group1 = group1.sort_values(by=match_cols[:-1]).reset_index(drop=True)
    group2 = group2.sort_values(by=match_cols[:-1]).reset_index(drop=True)
    group3 = group3.sort_values(by=match_cols[:-1]).reset_index(drop=True)

    merging_data = pd.concat([group1,group2,group3],axis=0)
    data_with_metapath = merging_data[merging_data["prompt_type"]=="output_path"]
    data_without_metapath = merging_data[merging_data["prompt_type"] == "output_path_without_metapath"]

    print("元路径评分:", data_with_metapath["AI_rating"].mean())
    print("元路径标准差:", data_with_metapath["AI_rating"].std())
    print("无元路径评分:", data_without_metapath["AI_rating"].mean())
    print("无元路径标准差:", data_without_metapath["AI_rating"].std())

    group1.to_excel("output/lot/rating/ai_rating.xlsx",index = False)
    group2.to_excel("output/lot/rating/ai_rating_doubao.xlsx", index=False)
    group3.to_excel("output/lot/rating/ai_rating_gemini.xlsx", index=False)