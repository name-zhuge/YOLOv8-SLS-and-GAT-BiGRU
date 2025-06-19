import os
import numpy as np
import pandas as pd
import torch
import scipy.sparse as sp

def encode_onehot(labels, concentration_labels=[0, 10, 100, 1000]):
    """Convert concentration labels into one hot encoding"""
    classes_dict = {c: np.identity(len(concentration_labels))[i, :] for i, c in enumerate(concentration_labels)}
    labels_onehot = np.array([classes_dict.get(c) for c in labels], dtype=np.int32)
    return labels_onehot

def load_data(path):
    """
    Load all CSV file data from the data folder and generate the corresponding graph feature matrix for each file.
    Parameters:
    Path: Data folder path
    return:
    Graphs_dict: The graph data dictionary corresponding to each folder
    """
    graphs_dict = {}

    csv_files = [f for f in os.listdir(path) if f.endswith('.csv')]
    csv_files.sort()

    for csv_file in csv_files:
        concentration = int(csv_file.split('-')[1].split('A')[0])

        if concentration not in graphs_dict:
            graphs_dict[concentration] = []

        csv_file_path = os.path.join(path, csv_file)
        data = pd.read_csv(csv_file_path)

        grouped = data.groupby('frame_id')

        for frame_id, group in grouped:
            features = group[['center_x', 'center_y']].values.astype(np.float32)

            num_nodes = 10

            if features.shape[0] < num_nodes:
                features = np.pad(features, ((0, num_nodes - features.shape[0]), (0, 0)), mode='constant')

            dist_matrix = np.zeros((num_nodes, num_nodes), dtype=np.float32)
            for i in range(num_nodes):
                for j in range(num_nodes):
                    dist_matrix[i, j] = np.linalg.norm(features[i] - features[j])

            adj = sp.csr_matrix(dist_matrix)

            adj = normalize(adj + sp.eye(num_nodes))

            features_tensor = torch.FloatTensor(features)
            adj_tensor = sparse_mx_to_torch_sparse_tensor(adj)

            labels = encode_onehot([concentration])[0]
            labels_tensor = torch.LongTensor(np.argmax(labels, axis=0))

            # 存储该帧的图数据
            graph_data = {
                'features': features_tensor,
                'adj': adj_tensor,
                'labels': labels_tensor
            }
            graphs_dict[concentration].append(graph_data)

    return graphs_dict

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse_coo_tensor(indices, values, shape)

def print_graph_data(graphs_dict, concentration=0):
    if concentration in graphs_dict:
        graph_data_list = graphs_dict[concentration]
        print(f"concentration {concentration} graph data：")

        # 打印前5帧的 features 和 adj（根据需要调整）
        for idx, graph_data in enumerate(graph_data_list[:5]):  # 只展示前5帧，避免数据过多
            print(f"Frame {idx+1} image data:")
            print(f"    - features：\n{graph_data['features']}")
            print(f"    - adj：\n{graph_data['adj']}")
    else:
        print(f"No graph data for {concentration} was found")
