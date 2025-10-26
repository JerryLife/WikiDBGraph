import pandas as pd

def check_duplicate_db1(csv_path):
    df = pd.read_csv(csv_path, dtype=str)
    df["db_1"] = df["db_1"].str.zfill(5)

    duplicates = df["db_1"].value_counts()
    duplicates = duplicates[duplicates > 1]

    if not duplicates.empty:
        print(f"❗ Found {len(duplicates)} duplicate db_1 entries.")
        print(duplicates.head(10))  # 打印前 10 个重复项
    else:
        print("✅ No duplicate db_1 entries found.")

def fix_duplicate_db1(csv_path, save_path=None):
    df = pd.read_csv(csv_path, dtype=str)
    df["db_1"] = df["db_1"].str.zfill(5)
    df["db_2"] = df["db_2"].str.zfill(5)

    seen_db1 = set()

    fixed_rows = []
    for idx, row in df.iterrows():
        db1, db2 = row["db_1"], row["db_2"]
        if db1 in seen_db1:
            # 交换 db_1 和 db_2
            db1, db2 = db2, db1
        seen_db1.add(db1)
        fixed_rows.append({"db_1": db1, "db_2": db2, "qid": row["qid"]})

    fixed_df = pd.DataFrame(fixed_rows)

    # 保存或返回
    if save_path:
        fixed_df.to_csv(save_path, index=False)
        print(f"✅ Fixed CSV saved to: {save_path}")
    else:
        return fixed_df

import json

def check_dataset_overlap():
    train_path = "/hpctmp/e1351271/wkdbs/data/split_triplets/train.jsonl"
    val_path = "/hpctmp/e1351271/wkdbs/data/split_triplets/val.jsonl"
    test_path = "/hpctmp/e1351271/wkdbs/data/split_triplets/test.jsonl"

    def load_triplets(path):
        with open(path, "r") as f:
            return set((item["anchor"], item["positive"]) for item in map(json.loads, f))

    train_set = load_triplets(train_path)
    val_set = load_triplets(val_path)
    test_set = load_triplets(test_path)

    overlap_train_val = train_set & val_set
    overlap_train_test = train_set & test_set
    overlap_val_test = val_set & test_set

    result = {
        "train ∩ val": len(overlap_train_val),
        "train ∩ test": len(overlap_train_test),
        "val ∩ test": len(overlap_val_test),
        "total overlaps": len(overlap_train_val | overlap_train_test | overlap_val_test)
    }
    print(result)



if __name__ == "__main__":
    # check_duplicate_db1("/hpctmp/e1351271/wkdbs/data/qid_pairs_fixed.csv")
    # fix_duplicate_db1(
    #     csv_path="/hpctmp/e1351271/wkdbs/data/qid_pairs_fixed.csv",
    #     save_path="/hpctmp/e1351271/wkdbs/data/qid_pairs_fixed.csv"
    # )
    # check_duplicate_db1("/hpctmp/e1351271/wkdbs/data/qid_pairs_fixed.csv")
    check_dataset_overlap()