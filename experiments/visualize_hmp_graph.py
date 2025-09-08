"""
python visualize_hmp_graph.py \
    --model_path results/saved_models/HMP-MACE_QM9_best.pt \
    --model_name HMP-MACE \
    --dataset_name QM9 \
    --dataset_root ../dataset_sample \
    --graph_idx 10
    
这将会加载第10个图，使用训练好的HMP-MACE模型预测其虚拟边，并生成一张名为 hmp_visualization_QM9_10.png 的图片。
"""

# visualize_hmp_graph.py
import argparse
import torch
import os
import json
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.data import Batch

# --- 确保可以从项目目录导入工具和模型 ---
# (假设此脚本与 run_tautobase_benchmark_parallel.py 在同一目录)
from run_tautobase_benchmark_parallel import get_model_configs, MapAtomTypes
from utils.tautobase_utils import MolecularDataset

def visualize_graph(data, edge_index_virtual, save_path="graph_visualization.png"):
    """
    使用 networkx 和 matplotlib 可视化分子图及其虚拟边。
    
    Args:
        data (torch_geometric.data.Data): 单个图的数据对象。
        edge_index_virtual (torch.Tensor): 形状为 [2, N_virtual] 的虚拟边索引。
        save_path (str): 保存图像的路径。
    """
    print(f"Visualizing graph and saving to {save_path}...")
    
    g = nx.Graph()
    pos_3d = data.pos.cpu().numpy()
    
    # 使用 x, y 坐标进行 2D 布局
    pos_2d = {i: (pos_3d[i, 0], pos_3d[i, 1]) for i in range(pos_3d.shape[0])}
    
    # 添加节点
    g.add_nodes_from(range(data.num_nodes))
    
    # 准备用于绘图的边列表
    original_edges = data.edge_index.cpu().numpy().T.tolist()
    virtual_edges = edge_index_virtual.cpu().numpy().T.tolist()
    
    plt.figure(figsize=(12, 12))
    
    # 绘制节点
    nx.draw_networkx_nodes(g, pos_2d, node_color='skyblue', node_size=500, alpha=0.9)
    
    # 绘制原始边 (黑色实线)
    nx.draw_networkx_edges(g, pos_2d, edgelist=original_edges, width=1.5, alpha=0.8, edge_color='black', style='solid')
    
    # 绘制虚拟边 (红色虚线)
    nx.draw_networkx_edges(g, pos_2d, edgelist=virtual_edges, width=2.0, alpha=0.7, edge_color='red', style='dashed')
    
    # 绘制节点标签
    nx.draw_networkx_labels(g, pos_2d, font_size=12, font_color='black')
    
    plt.title("Graph with Original (Black) and HMP Virtual (Red, Dashed) Edges")
    plt.axis('off')
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"Visualization saved successfully to {save_path}")

def main():
    parser = argparse.ArgumentParser(description='Visualize HMP Virtual Edges on a Graph')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the saved model .pt file.')
    parser.add_argument('--model_name', type=str, required=True, choices=['HMP-MACE'], help='Name of the HMP model to load.')
    parser.add_argument('--dataset_name', type=str, required=True, choices=['QM9', 'PC9', 'ANI-1E'], help='Name of the dataset the model was trained on.')
    parser.add_argument('--dataset_root', type=str, default='../dataset_sample', help='Root directory of the datasets.')
    parser.add_argument('--graph_idx', type=int, required=True, help='Index of the graph to visualize from the dataset.')
    parser.add_argument('--L', type=int, default=4, help='Number of hierarchical layers (must match the trained model).')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use for inference.')
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Using device: {device}")

    # 1. 加载原子类型映射
    atom_map_file = "QTautobase_atom_map.json"
    if not os.path.exists(atom_map_file):
        print(f"Error: Atom map file '{atom_map_file}' not found. Please run the training script first to generate it.")
        return
    with open(atom_map_file, 'r') as f:
        atom_map = {int(k): v for k, v in json.load(f).items()}
    num_atom_classes = len(atom_map)
    atom_mapping_transform = MapAtomTypes(atom_map)

    # 2. 加载数据集和指定的图
    dataset_map = {
        'QM9': 'qm9',
        'PC9': 'PC9/XYZ',
        'ANI-1E': 'ANI-1E/xyz'
    }
    dataset_path = os.path.join(args.dataset_root, dataset_map[args.dataset_name])
    print(f"Loading dataset from: {dataset_path}")
    dataset = MolecularDataset(root=dataset_path, transform=atom_mapping_transform)
    
    if args.graph_idx >= len(dataset):
        print(f"Error: graph_idx {args.graph_idx} is out of bounds for dataset with size {len(dataset)}.")
        return
        
    data = dataset[args.graph_idx]
    print(f"Loaded graph {args.graph_idx} with {data.num_nodes} nodes and {data.num_edges} edges.")

    # 3. 实例化并加载模型
    print(f"Loading model '{args.model_name}' from {args.model_path}")
    model_configs = get_model_configs(args.L, num_atom_classes)
    config = model_configs.get(args.model_name)
    if not config:
        print(f"Error: Model name '{args.model_name}' not found in configurations.")
        return
        
    model = config['class'](**config['params'])
    
    # 加载状态字典
    state_dict = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # 4. 执行推理以获取虚拟边
    print("Running model inference to generate virtual edges...")
    with torch.no_grad():
        # 将单个图转换为批次大小为1的批处理对象
        batch = Batch.from_data_list([data]).to(device)
        
        # 调用 forward 方法并请求返回虚拟边
        _, edge_index_virtual_global = model(batch, return_virtual_edges=True)

    print(f"Generated {edge_index_virtual_global.shape[1]} virtual edges.")

    # 5. 可视化
    save_path = f"hmp_visualization_{args.dataset_name}_{args.graph_idx}.png"
    visualize_graph(data, edge_index_virtual_global, save_path)

if __name__ == '__main__':
    main()