import json
import random
import os

triplet_dir = "/hpctmp/e1351271/wkdbs/data/split_triplets"
all_db_ids = {f"{i:05d}" for i in range(100000)}
seeds = [0, 1, 2, 3, 4]

def load_triplets(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def extract_all_db_ids_from_triplets(path):
    db_ids = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            triplet = json.loads(line)
            db_ids.add(triplet["anchor"])
            db_ids.add(triplet["positive"])
            db_ids.update(triplet.get("negatives", []))
    return db_ids

def main():
    test_path = os.path.join(triplet_dir, "triplets_test.jsonl")
    test_triplets = load_triplets(test_path)

    used_ap_ids = set()
    for t in test_triplets:
        used_ap_ids.add(t["anchor"])
        used_ap_ids.add(t["positive"])

    existing_ids = set()
    for split in ["train", "val", "test"]:
        path = os.path.join(triplet_dir, f"triplets_{split}.jsonl")
        if os.path.exists(path):
            existing_ids |= extract_all_db_ids_from_triplets(path)

    neg_candidates = sorted(all_db_ids - used_ap_ids - existing_ids)
    print(f"🧮 Total negative candidates: {len(neg_candidates)}")

    for seed in seeds:
        random.seed(seed)
        neg_pool = neg_candidates.copy()
        random.shuffle(neg_pool)
        neg_index = 0

        def get_negatives(k=6):
            nonlocal neg_index
            selected = neg_pool[neg_index:neg_index + k]
            neg_index += k
            return selected

        output_path = os.path.join(triplet_dir, f"triplets_test_seed{seed}.jsonl")
        with open(output_path, "w", encoding="utf-8") as f:
            for triplet in test_triplets:
                new_triplet = {
                    "anchor": triplet["anchor"],
                    "positive": triplet["positive"],
                    "negatives": get_negatives(6)
                }
                f.write(json.dumps(new_triplet) + "\n")
        print(f"✅ Generated: {output_path}")

if __name__ == "__main__":
    main()
