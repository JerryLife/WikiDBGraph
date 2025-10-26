import pandas as pd
import random
import os

def generate_negative_samples(pos_file, output_dir, seeds, num_range=(0, 99999)):
    os.makedirs(output_dir, exist_ok=True)

    pos_df = pd.read_csv(pos_file, dtype=str)
    pos_pairs = set()
    for _, row in pos_df.iterrows():
        a, b = row["db_1"].zfill(5), row["db_2"].zfill(5)
        pos_pairs.add((a, b))
        pos_pairs.add((b, a))

    db1_list = [x.zfill(5) for x in pos_df["db_1"].tolist()]
    all_ids = [str(i).zfill(5) for i in range(num_range[0], num_range[1] + 1)]

    for seed in seeds:
        random.seed(seed)
        neg_rows = []

        for db1 in db1_list:
            while True:
                db2 = random.choice(all_ids)
                if db2 != db1 and (db1, db2) not in pos_pairs and (db2, db1) not in pos_pairs:
                    neg_rows.append((db1, db2))
                    break

        neg_df = pd.DataFrame(neg_rows, columns=["db_1", "db_2"])
        out_path = os.path.join(output_dir, f"neg_seed{seed}.csv")
        neg_df.to_csv(out_path, index=False)
        print(f"✅ Saved negative samples to {out_path} (seed={seed})")

if __name__ == "__main__":
    generate_negative_samples(
        pos_file="/hpctmp/e1351271/wkdbs/data/qid_pairs.csv",
        output_dir="/hpctmp/e1351271/wkdbs/data/neg_samples",
        seeds=[1, 42, 123, 212, 2025]
    )
