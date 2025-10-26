import os
import torch
import pandas as pd
from tqdm import tqdm
import numpy as np
from joblib import Parallel, delayed

input_dir = "/hpctmp/e1351271/wkdbs/out/col_matcher_bge-m3_database/weights/finetuned_bge_m3_softmax_lr1e-05/test_results_exhaustive_split"
output_dir = input_dir
final_output_pt = os.path.join(output_dir, "all_exhaustive_predictions.pt")

def process_single_csv(csv_file):
    csv_path = os.path.join(input_dir, csv_file)
    pt_path = os.path.join(output_dir, csv_file.replace(".csv", ".pt"))

    if os.path.exists(pt_path):
        return pt_path

    df = pd.read_csv(csv_path)

    anchor_ids = df["anchor_id"].astype(int).values
    target_ids = df["target_id"].astype(int).values
    similarities = df["similarity"].astype(float).values
    labels = (df["label"] == 1).astype(int).values
    edges = (similarities > 0.6713).astype(int)

    data = np.stack([anchor_ids, target_ids, similarities, labels, edges], axis=1)
    tensor = torch.from_numpy(data).to(torch.float32)

    torch.save(tensor, pt_path)

    return pt_path

if __name__ == "__main__":
    csv_files = [f for f in os.listdir(input_dir) if f.endswith(".csv")]

    pt_files = Parallel(n_jobs=8)(
        delayed(process_single_csv)(csv_file) for csv_file in tqdm(csv_files, desc="Processing CSVs")
    )

    print(f"Processed {len(pt_files)} files. Now merging...")

    # 加载所有pt文件
    all_tensors = []
    for pt_path in tqdm(pt_files, desc="Loading PT files"):
        tensor = torch.load(pt_path)
        all_tensors.append(tensor)

    merged_tensor = torch.cat(all_tensors, dim=0)

    torch.save(merged_tensor, final_output_pt)
    print(f"Saved merged tensor with shape {merged_tensor.shape} to {final_output_pt}")
