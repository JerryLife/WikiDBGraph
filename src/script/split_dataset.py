import csv
import json
import random
from collections import defaultdict
import os

qid_pairs_path = "/hpctmp/e1351271/wkdbs/data/qid_pairs.csv"
neg_pool_path = "/hpctmp/e1351271/wkdbs/data/negative_candidates.csv"
triplet_output_dir = "/hpctmp/e1351271/wkdbs/data/split_triplets"

SPLIT_SEED = 2024
SEEDS_FOR_TEST = [42, 43, 44, 45, 46]


def load_qid_pairs():
    qid_groups = defaultdict(list)
    with open(qid_pairs_path, newline='', encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            qid = row["qid"]
            db1 = row["db_1"].zfill(5)
            db2 = row["db_2"].zfill(5)
            qid_groups[qid].append((db1, db2))
    return qid_groups


def get_test_split(qid_groups):
    all_qids = list(qid_groups.keys())
    random.Random(SPLIT_SEED).shuffle(all_qids)

    total_pairs = sum(len(pairs) for pairs in qid_groups.values())
    target_train = int(total_pairs * 0.7)
    target_val = int(total_pairs * 0.1)

    count_train = 0
    count_val = 0
    test_pairs = []

    for qid in all_qids:
        group = qid_groups[qid]
        group_size = len(group)
        if count_train + group_size <= target_train:
            count_train += group_size
        elif count_val + group_size <= target_val:
            count_val += group_size
        else:
            test_pairs.extend(group)

    return test_pairs


def generate_test_triplets(split_pairs, seeds):
    with open(neg_pool_path, newline='', encoding='utf-8') as f:
        next(f)  # skip header
        negative_pool_base = [line.strip() for line in f]

    for seed in seeds:
        random.seed(seed)
        negative_pool = negative_pool_base.copy()
        random.shuffle(negative_pool)
        neg_index = 0

        def get_negatives(k=6):
            nonlocal neg_index
            result = negative_pool[neg_index:neg_index + k]
            neg_index += k
            return result

        output_path = os.path.join(triplet_output_dir, f"triplets_test_seed{seed}.jsonl")
        with open(output_path, "w", encoding="utf-8") as f:
            for anchor, positive in split_pairs:
                negatives = get_negatives()
                triplet = {
                    "anchor": anchor,
                    "positive": positive,
                    "negatives": negatives
                }
                f.write(json.dumps(triplet) + "\n")
        print(f"✅ Generated {len(split_pairs)} triplets for test (seed={seed})")


def main():
    qid_groups = load_qid_pairs()
    test_pairs = get_test_split(qid_groups)
    generate_test_triplets(test_pairs, seeds=SEEDS_FOR_TEST)


if __name__ == "__main__":
    main()
