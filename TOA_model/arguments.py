import dgl
import torch
import argparse
import os

static_model_path=os.path.dirname(os.path.abspath(__file__))
graph = dgl.load_graphs(f'{static_model_path}/result/LLM/thg.dgl')[0][0]

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=str, default='LLM', help='数据集')

parser.add_argument('--log_dir', type=str, default='log', help='日志保存路径')
parser.add_argument('--save_dir', type=str, default='output', help='模型输出向量保存路径')
parser.add_argument('--csv_dir', type=str, default='csv', help='模型评估结果保存路径')
parser.add_argument('--model_dir', type=str, default='model', help='模型保存路径')
parser.add_argument('--toa_model_dir', type=str, default=f'{static_model_path}', help='toa模型')

parser.add_argument('--GPU_ID', type=str, default='0', help='GPU id')

parser.add_argument('--graph_path', type=str, default=f"{static_model_path}/result/LLM/thg.dgl", help='网络保存地址(绝对路径)')
# parser.add_argument('--graph_path', type=str, default="D:/Hu/技术机会预测/result/大模型/thg1.dgl", help='网络保存地址(绝对路径)')
parser.add_argument('--graphs_path', type=str, default=f"{static_model_path}/result/LLM/target_graphs_static.dgl", help='网络snapshots保存地址(绝对路径)')
# parser.add_argument('--metapath_path', type=str, default="D:/Hu/技术机会预测/result/大模型/temporal_metapath_dict.pkl", help='metapath字典保存地址(绝对路径)')
parser.add_argument('--metapath_path', type=str, default=f"{static_model_path}/result/LLM/temporal_metapath_dict_paper1.pkl", help='metapath字典保存地址(绝对路径)')
parser.add_argument('--time_range', type=list, default=list(range(2010, 2023)), help='训练数据时间范围')
parser.add_argument('--test_timestamp', type=int, default=2023, help='测试时间步')
parser.add_argument('--min_timestamp', type=int, default=2010, help='最小timestamp')
parser.add_argument('--max_timestamp', type=int, default=2022, help='最大timestamp')
parser.add_argument('--num_metapath_types', type=int, default=3, help='metapath种类数')

parser.add_argument('--device', default=torch.device('cuda:0'), help='训练设备')
# parser.add_argument('--device', default='cpu', help='训练设备')
parser.add_argument('--num_epochs', type=int, default=100, help='训练轮次')
parser.add_argument('--explain_num_epochs', type=int, default=300, help='解释器训练轮次')
parser.add_argument('--batch_size', type=int, default=64, help='批大小')
parser.add_argument('--lr', type=float, default=2e-4, help='学习率')
parser.add_argument('--explain_lr', type=float, default=3e-1, help='解释器学习率')
parser.add_argument('--patience', type=int, default=5, help='early stop')
parser.add_argument('--explain_patience', type=int, default=10, help='explainer early stop')

parser.add_argument('--feature_dim', type=int, default=128, help='特征向量维度')
parser.add_argument('--hidden_dim', type=int, default=64, help='隐向量维度', choices=[16, 32, 64, 128, 256]) # best=64
parser.add_argument('--output_dim', type=int, default=128, help='输出向量维度', choices=[16, 32, 64, 128, 256]) # best=128
parser.add_argument('--dropout_rate', type=float, default=0.5, help='dropout概率')
parser.add_argument('--num_heads', type=int, default=12, help='多头注意力机制', choices=[4, 6, 8, 12, 16]) # best=12
parser.add_argument('--num_logits_heads', type=int, default=12, help='talking-heads', choices=[4, 6, 8, 12, 16])

parser.add_argument('--sample_size', type=float, default=30, help='抽样次数')
parser.add_argument('--metapath_size', type=float, default=1,choices=[0.005,0.01,0.05,0.1,0.5,1], help='metapath_size惩罚系数')
parser.add_argument('--feature_size', type=float, default=0.5, help='feature_size惩罚系数')

parser.add_argument('--num_ipc', type=int, default=graph.num_nodes(ntype='ipc'), help='ipc节点数')
parser.add_argument('--num_patent', type=int, default=graph.num_nodes(ntype='patent'), help='patent节点数')
parser.add_argument('--num_paper', type=int, default=graph.num_nodes(ntype='paper'), help='paper节点数')

parser.add_argument("--env", default="lot")
parser.add_argument("--seed", default="1")
parser.add_argument("--max_steps", default=30, type=int)
parser.add_argument("--temperature", default=0.0, type=float)
parser.add_argument("--max_tokens", default=4096, type=int)
parser.add_argument("--agent_mode", default="argue_review")
parser.add_argument("--prompt_version", default=0, type=int)
parser.add_argument("--use_wandb", action="store_true")
parser.add_argument("--debug", action="store_true")
parser.add_argument("--model_name", default="deepseek-ai/DeepSeek-V3")
parser.add_argument("--parallel_argument_num", default=3)
parser.add_argument("--refined_tolerance", default=3)
parser.add_argument("--thinking_steps", default=4)
parser.add_argument('--threshold', type=float, default=0.9, help='预测阈值')
parser.add_argument('--api_key', type=str, default="sk-ncqckmctyibzcmyeverjkctlgrqihjhbwfughrmxtvdlmutw", help='api key')

args = parser.parse_args(args=[])