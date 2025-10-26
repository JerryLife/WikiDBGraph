import torch
import dgl
import networkx as nx
from pyvis.network import Network
import random
import os

def build_dgl_graph_from_pt(pt_file_path):
    tensor = torch.load(pt_file_path, map_location="cpu")
    print("Loaded tensor with shape", tensor.shape)

    valid_edges = tensor[tensor[:, 4] == 1]
    print("Valid edges:", valid_edges.shape)

    src = valid_edges[:, 0].long()
    dst = valid_edges[:, 1].long()

    g = dgl.graph((src, dst))
    g.edata["weight"] = valid_edges[:, 2]
    print(f"DGLGraph built with {g.num_nodes()} nodes and {g.num_edges()} edges.")

    return g

def save_dgl_graph(g, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    dgl.save_graphs(path, [g])
    print(f"DGLGraph saved to {path}")

def load_dgl_graph(path):
    graphs, _ = dgl.load_graphs(path)
    g = graphs[0]
    print(f"Loaded DGLGraph with {g.num_nodes()} nodes and {g.num_edges()} edges from {path}.")
    return g

if __name__ == "__main__":
    # === Your real file paths ===
    pt_file = "/hpctmp/e1351271/wkdbs/out/col_matcher_bge-m3_database/weights/finetuned_bge_m3_softmax_lr1e-05/test_results_exhaustive_split/all_exhaustive_predictions.pt"
    dgl_graph_save_path = "/hpctmp/e1351271/wkdbs/out/col_matcher_bge-m3_database/weights/finetuned_bge_m3_softmax_lr1e-05/test_results_exhaustive_split/full_graph.dgl"
    sample_output_html = "/hpctmp/e1351271/wkdbs/out/col_matcher_bge-m3_database/weights/finetuned_bge_m3_softmax_lr1e-05/test_results_exhaustive_split/sampled_random_walk_10k_graph.html"

    # Build and save full DGLGraph
    g = build_dgl_graph_from_pt(pt_file)
    save_dgl_graph(g, dgl_graph_save_path)