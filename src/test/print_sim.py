import torch

def find_similarity(pt_file_path, anchor_id_query, target_id_query):
    tensor = torch.load(pt_file_path, map_location="cpu")
    print("Loaded tensor with shape", tensor.shape)

    cond1 = (tensor[:, 0] == anchor_id_query) & (tensor[:, 1] == target_id_query)
    cond2 = (tensor[:, 0] == target_id_query) & (tensor[:, 1] == anchor_id_query)

    matches = tensor[cond1 | cond2]

    if matches.shape[0] == 0:
        print(f"No match found for anchor_id={anchor_id_query}, target_id={target_id_query}")
        return None

    similarity = matches[0, 2].item()

    return similarity

if __name__ == "__main__":
    pt_path = "/hpctmp/e1351271/wkdbs/out/col_matcher_bge-m3_database/weights/finetuned_bge_m3_softmax_lr1e-05/test_results_exhaustive_split/all_exhaustive_predictions.pt"
    anchor_id = 12345
    target_id = 67890

    print("finding similarity")
    sim = find_similarity(pt_path, anchor_id, target_id)
    print(f"Similarity between {anchor_id} and {target_id}: {sim}")