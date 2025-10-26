# src/model/bge_similarity.py


import os
import csv
import time
import json
import torch
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
from utils.schema_formatter import format_schema_from_loader
from utils.load_from_uci import load_wdbc_dataset, format_schema_from_dataframe
import pandas as pd
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def generate_and_save_all_embeddings(
    loader,
    embedder,
    output_path: str,
    batch_size: int = 8,
    chunk_size: int = 1000,
    db_id_range: tuple = (0, 100000),
    sample: bool = True,
    show_wikidata_property_id: bool = False
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    os.makedirs(output_path, exist_ok=True)

    embedder.model.eval()

    all_embeddings = []
    all_ids = []
    texts = []
    ids = []

    for i in tqdm(range(*db_id_range), desc="Embedding DBs"):
        db_id = str(i).zfill(5)
        try:
            schema_text = format_schema_from_loader(loader, db_id, sample=sample, show_wikidata_property_id=show_wikidata_property_id)
            texts.append(schema_text)
            ids.append(db_id)
        except Exception as e:
            print(f"⚠️ Skipped {db_id}: {e}")
            continue

        if len(texts) == batch_size or (i == db_id_range[1] - 1):
            try:
                with torch.no_grad():
                    embs = embedder.get_embedding(texts, batch_size=batch_size).cpu()
                all_embeddings.append(embs)
                all_ids.extend(ids)
            except Exception as e:
                print(f"❌ Error processing batch at db_id {db_id}: {e}")

            texts = []
            ids = []
        
        if len(all_ids) >= chunk_size:
            current_embeddings = torch.cat(all_embeddings, dim=0)
            db_id_to_index = {db_id: idx for idx, db_id in enumerate(all_ids)}
            torch.save({
                "embeddings": current_embeddings,
                "db_id_to_index": db_id_to_index
            }, os.path.join(output_path, "all_embeddings.pt"))

            print(f"Intermediate save with {len(all_ids)} entries")

    if all_ids:
        current_embeddings = torch.cat(all_embeddings, dim=0)
        db_id_to_index = {db_id: idx for idx, db_id in enumerate(all_ids)}
        torch.save({
            "embeddings": current_embeddings,
            "db_id_to_index": db_id_to_index
        }, os.path.join(output_path, "all_embeddings.pt"))

        print(f"Final save completed: {len(all_ids)} entries → all_embeddings.pt")


class BGEEmbedder:
    def __init__(self, model_type="bge-m3", model_path=None):
        self.model_type = model_type

        if model_type == "bge-m3":
            model_name = model_path if model_path is not None else "BAAI/bge-m3"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name).to(device)
            self.model.gradient_checkpointing_enable()
        elif model_type == "bge-large-en-v1.5":
            model_name = "BAAI/bge-large-en-v1.5"
            self.model = SentenceTransformer(model_name).to(device)
            self.tokenizer = None
        else:
            raise ValueError("Unsupported model_type. Choose 'bge-m3' or 'bge-large-en-v1.5'.")

        self.model.eval()
        self.embeddings = None
        self.db_id_to_index = None

    def get_topk_similar_dbs(self, embedding, embedding_path, k=10):
        if self.embeddings is None or self.db_id_to_index is None:
            self.load_embeddings(embedding_path)

        index_to_db_id = {v: k for k, v in self.db_id_to_index.items()}

        sims = F.cosine_similarity(embedding, self.embeddings)
        topk_indices = torch.argsort(sims, descending=True)[:k]
        topk_db_ids = [
            (index_to_db_id[idx.item()], sim.item())
            for idx, sim in zip(topk_indices, sims[topk_indices])
        ]
        return topk_db_ids

    def get_embedding(self, texts, is_train=False, batch_size=4) -> torch.Tensor:
        if isinstance(texts, str):
            texts = [texts]

        if self.model_type == "bge-m3":
            all_embeddings = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                inputs = self.tokenizer(
                    batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=1024
                ).to(device)

                if is_train:
                    embedding = F.normalize(self.model(**inputs).last_hidden_state[:, 0], p=2, dim=-1)
                    all_embeddings.append(embedding)
                else:
                    with torch.no_grad():
                        embedding = F.normalize(self.model(**inputs).last_hidden_state[:, 0], p=2, dim=-1)
                        all_embeddings.append(embedding)

            return torch.cat(all_embeddings, dim=0)

        elif self.model_type == "bge-large-en-v1.5":
            with torch.no_grad():
                embedding = self.model.encode(
                    texts,
                    convert_to_tensor=True,
                    normalize_embeddings=True,
                    batch_size=batch_size,
                    device=device
                )
            return embedding
        else:
            raise ValueError("Unsupported model_type.")

    def get_embedding_from_db_id(self, db_id, loader, show_wikidata_property_id: bool = False, sample: bool = True):
        schema = [format_schema_from_loader(loader, db_id, sample=sample, show_wikidata_property_id=show_wikidata_property_id)]
        return self.get_embedding(schema)


    def database_similarity(self, loader, db_id_1: str, db_id_2: str, show_wikidata_property_id: bool = False, sample: bool = True):
        start_time = time.time()

        schema_1 = [format_schema_from_loader(loader, db_id_1, sample=sample, show_wikidata_property_id=show_wikidata_property_id)]
        schema_2 = [format_schema_from_loader(loader, db_id_2, sample=sample, show_wikidata_property_id=show_wikidata_property_id)]

        emb1 = self.get_embedding(schema_1)
        emb2 = self.get_embedding(schema_2)

        similarity = F.cosine_similarity(emb1, emb2).item()
        # print(f"[{self.model_type}] Similarity between {db_id_1} and {db_id_2}: {similarity:.4f}")

        elapsed = time.time() - start_time
        matched_pair = ("_", "_")
        return similarity, matched_pair, elapsed

    def fit(
        self,
        train_path,
        val_path,
        loader,
        save_dir,
        batch_size=32,
        epochs=10,
        lr=5e-4,
        temperature=0.5,
        max_length=1024,
    ):
        import json
        import os
        from torch.utils.data import Dataset, DataLoader
        from torch.optim import AdamW
        from tqdm import tqdm

        class TripletJSONLDataset(Dataset):
            def __init__(self, path):
                self.data = []
                with open(path, "r") as f:
                    for line in f:
                        self.data.append(json.loads(line))

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                return self.data[idx]

        def info_nce_loss(anchor_emb, pos_emb, neg_embs, temperature=0.5):
            all_embs = torch.cat([pos_emb, neg_embs], dim=0)
            sims = F.cosine_similarity(anchor_emb, all_embs, dim=1) / temperature
            # print(f"sims: {sims}")
            labels = torch.zeros(1, dtype=torch.long).to(anchor_emb.device)
            loss = F.cross_entropy(sims.unsqueeze(0), labels)
            return loss

        model = self.model
        tokenizer = self.tokenizer
        model.train()

        train_set = TripletJSONLDataset(train_path)
        val_set = TripletJSONLDataset(val_path)
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=lambda x: x)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, collate_fn=lambda x: x)

        optimizer = AdamW(model.parameters(), lr=lr)
        best_val_loss = float("inf")

        def evaluate():
            model.eval()
            total_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    for item in batch:
                        anchor = format_schema_from_loader(loader, item["anchor"])
                        pos = format_schema_from_loader(loader, item["positive"])
                        negs = [format_schema_from_loader(loader, neg) for neg in item["negatives"]]

                        texts = [anchor, pos] + negs
                        emb = self.get_embedding(texts, batch_size=batch_size)
                        loss = info_nce_loss(emb[0:1], emb[1:2], emb[2:], temperature)
                        total_loss += loss
            model.train()
            return total_loss / len(val_loader)

        for epoch in range(epochs):
            total_loss = 0.0
            progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

            for batch in progress:
                torch.cuda.reset_peak_memory_stats(device)
                optimizer.zero_grad()
                batch_loss = 0.0

                for item in batch:
                    anchor = format_schema_from_loader(loader, item["anchor"])
                    pos = format_schema_from_loader(loader, item["positive"])
                    negs = [format_schema_from_loader(loader, neg) for neg in item["negatives"]]

                    texts = [anchor, pos] + negs
                    emb = self.get_embedding(texts, is_train=True, batch_size=batch_size)
                    loss = info_nce_loss(emb[0:1], emb[1:2], emb[2:], temperature)
                    batch_loss += loss
                    del emb
                
                batch_loss.backward()
                optimizer.step()
                total_loss += batch_loss.item()
                progress.set_postfix(loss=batch_loss.item())
                peak_memory = torch.cuda.max_memory_allocated(device) / 1024**2
                print(f"[Step {epoch+1}] 💡 Peak GPU memory usage: {peak_memory:.2f} MB")
                torch.cuda.empty_cache()

            avg_train_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch+1} train loss: {avg_train_loss:.4f}")

            val_loss = evaluate()
            print(f"Epoch {epoch+1} val loss: {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                os.makedirs(os.path.join(save_dir, "best"), exist_ok=True)
                model.save_pretrained(os.path.join(save_dir, "best"))
                tokenizer.save_pretrained(os.path.join(save_dir, "best"))
                print(f"💾 Saved best model to {save_dir}/best")

            os.makedirs(os.path.join(save_dir, "last"), exist_ok=True)
            model.save_pretrained(os.path.join(save_dir, "last"))
            tokenizer.save_pretrained(os.path.join(save_dir, "last"))
            print(f"📝 Saved last model to {save_dir}/last")

    def test(self, test_path, loader, batch_size=32, save_dir="test_results"):

        os.makedirs(save_dir, exist_ok=True)

        class TripletJSONLDataset:
            def __init__(self, path):
                self.data = []
                with open(path, "r") as f:
                    for line in f:
                        self.data.append(json.loads(line))

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                return self.data[idx]

        dataset = TripletJSONLDataset(test_path)
        y_true = []
        y_scores = []
        records = []

        self.model.eval()
        with torch.no_grad():
            for item in dataset:
                anchor_id = item["anchor"]
                pos_id = item["positive"]
                neg_ids = item["negatives"]

                anchor = format_schema_from_loader(loader, anchor_id)
                pos = format_schema_from_loader(loader, pos_id)
                negs = [format_schema_from_loader(loader, nid) for nid in neg_ids]

                texts = [anchor, pos]
                emb = self.get_embedding(texts, batch_size=batch_size)
                sim = F.cosine_similarity(emb[0:1], emb[1:2]).item()
                y_true.append(1)
                y_scores.append(sim)
                records.append((anchor_id, pos_id, sim, 1))

                for nid, neg in zip(neg_ids, negs):
                    texts = [anchor, neg]
                    emb = self.get_embedding(texts, batch_size=batch_size)
                    sim = F.cosine_similarity(emb[0:1], emb[1:2]).item()
                    y_true.append(0)
                    y_scores.append(sim)
                    records.append((anchor_id, nid, sim, 0))

        # ROC + AUC
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        youden_j = tpr - fpr
        best_idx = youden_j.argmax()
        best_threshold = thresholds[best_idx]

        # Save ROC curve plot
        plt.figure()
        plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.scatter(fpr[best_idx], tpr[best_idx], color='red', label=f'Best threshold = {best_threshold:.4f}')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc='lower right')
        plt.grid(True)
        roc_path = os.path.join(save_dir, "roc_curve.png")
        plt.savefig(roc_path)
        plt.close()

        # Save result CSV
        csv_path = os.path.join(save_dir, "predictions.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["anchor_id", "target_id", "similarity", "label"])
            writer.writerows(records)

        # Save ROC points
        roc_data_path = os.path.join(save_dir, "roc_data.csv")
        with open(roc_data_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["fpr", "tpr", "threshold"])
            for i in range(len(fpr)):
                writer.writerow([fpr[i], tpr[i], thresholds[i]])

        # Save summary
        summary_path = os.path.join(save_dir, "summary.txt")
        with open(summary_path, "w") as f:
            f.write(f"AUC: {roc_auc:.4f}\n")
            f.write(f"Best Threshold: {best_threshold:.4f}\n")
            f.write(f"Best TPR: {tpr[best_idx]:.4f}, Best FPR: {fpr[best_idx]:.4f}\n")

        print(f"✅ AUC: {roc_auc:.4f}")
        print(f"📌 Best threshold (Youden’s J): {best_threshold:.4f} at FPR={fpr[best_idx]:.4f}, TPR={tpr[best_idx]:.4f}")
        print(f"💾 Results saved in: {save_dir}")

    def load_embeddings(self, embedding_path):
        saved_data = torch.load(embedding_path, map_location=device)
        db_id_to_index = saved_data["db_id_to_index"]
        embedding_matrix = saved_data["embeddings"].to(device)
        self.embeddings = embedding_matrix
        self.db_id_to_index = db_id_to_index


    def test_scalable(self, test_path, embedding_path, save_dir="test_results"):
        os.makedirs(save_dir, exist_ok=True)

        class TripletJSONLDataset:
            def __init__(self, path):
                self.data = []
                with open(path, "r") as f:
                    for line in f:
                        self.data.append(json.loads(line))

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                return self.data[idx]

        dataset = TripletJSONLDataset(test_path)
        y_true = []
        y_scores = []
        records = []

        print(f"📂 Loading precomputed embeddings from {embedding_path}")
        saved_data = torch.load(embedding_path, map_location=device)
        db_id_to_index = saved_data["db_id_to_index"]
        embedding_matrix = saved_data["embeddings"].to(device)

        with torch.no_grad():
            for item in tqdm(dataset, desc="Evaluating triplets"):
                anchor_id = item["anchor"]
                pos_id = item["positive"]
                neg_ids = item["negatives"]
                all_ids = [pos_id] + neg_ids

                if anchor_id not in db_id_to_index or any(x not in db_id_to_index for x in all_ids):
                    print(f"⚠️ Skipping triplet ({anchor_id}, {pos_id}, ...) due to missing embeddings")
                    continue

                emb_anchor = embedding_matrix[db_id_to_index[anchor_id]].unsqueeze(0)
                emb_targets = torch.stack([embedding_matrix[db_id_to_index[x]] for x in all_ids], dim=0)

                sims = F.cosine_similarity(emb_anchor, emb_targets)

                y_true.append(1)
                y_scores.append(sims[0].item())
                records.append((anchor_id, pos_id, sims[0].item(), 1))

                for nid, sim_val in zip(neg_ids, sims[1:]):
                    y_true.append(0)
                    y_scores.append(sim_val.item())
                    records.append((anchor_id, nid, sim_val.item(), 0))

        # ROC + AUC
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        youden_j = tpr - fpr
        best_idx = youden_j.argmax()
        best_threshold = thresholds[best_idx]

        # Save ROC curve plot
        plt.figure()
        plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.scatter(fpr[best_idx], tpr[best_idx], color='red', label=f'Best threshold = {best_threshold:.4f}')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc='lower right')
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, "roc_curve.png"))
        plt.close()

        # Save prediction results
        with open(os.path.join(save_dir, "predictions.csv"), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["anchor_id", "target_id", "similarity", "label"])
            writer.writerows(records)

        # Save ROC points
        with open(os.path.join(save_dir, "roc_data.csv"), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["fpr", "tpr", "threshold"])
            for i in range(len(fpr)):
                writer.writerow([fpr[i], tpr[i], thresholds[i]])

        # Save summary
        with open(os.path.join(save_dir, "summary.txt"), "w") as f:
            f.write(f"AUC: {roc_auc:.4f}\n")
            f.write(f"Best Threshold: {best_threshold:.4f}\n")
            f.write(f"Best TPR: {tpr[best_idx]:.4f}, Best FPR: {fpr[best_idx]:.4f}\n")

        print(f"✅ AUC: {roc_auc:.4f}")
        print(f"📌 Best threshold (Youden’s J): {best_threshold:.4f} at FPR={fpr[best_idx]:.4f}, TPR={tpr[best_idx]:.4f}")
        print(f"💾 Results saved in: {save_dir}")

    def test_all_possible_pairs(
        self,
        embedding_path,
        qid_pairs_path="data/qid_pairs_fixed.csv",
        save_dir="out/test",
        batch_size=256,
        sim_threshold=0.6713,
    ):
        import os
        import csv
        import torch
        import torch.nn.functional as F
        from tqdm import tqdm

        os.makedirs(save_dir, exist_ok=True)

        print(f"Loading labeled qid pairs from {qid_pairs_path}")
        qid_pairs = set()
        with open(qid_pairs_path, "r") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                a, b = row["db_1"], row["db_2"]
                qid_pairs.add(tuple(sorted((a, b))))
        print(f"Loaded {len(qid_pairs):,} labeled pairs")

        print(f"oading precomputed embeddings from {embedding_path}")
        saved_data = torch.load(embedding_path, map_location=device)
        db_id_to_index = saved_data["db_id_to_index"]
        embedding_matrix = saved_data["embeddings"].to(device)

        existing_pairs = set()
        original_records = []

        all_db_ids = sorted(db_id_to_index.keys())
        num_ids = len(all_db_ids)
        total_possible = num_ids * (num_ids - 1) // 2
        remaining_to_match = total_possible - len(existing_pairs)

        print(f"Total unique IDs: {num_ids}")
        print(f"Total unordered pairs (C(n,2)): {total_possible:,}")
        print(f"Pairs to match this run: {remaining_to_match:,}")

        all_records = []
        all_records.extend(original_records)
        total_new_written = 0

        with torch.no_grad():
            for i in tqdm(range(0, num_ids), desc="Matching all pairs"):
                anchor_id = all_db_ids[i]
                emb_anchor = embedding_matrix[db_id_to_index[anchor_id]].unsqueeze(0)

                for j_start in range(i + 1, num_ids, batch_size):
                    j_end = min(j_start + batch_size, num_ids)
                    batch_target_ids = all_db_ids[j_start:j_end]

                    pairs = [tuple(sorted((anchor_id, tid))) for tid in batch_target_ids]
                    mask_unseen = [pair not in existing_pairs for pair in pairs]

                    if not any(mask_unseen):
                        continue

                    emb_targets = torch.stack([
                        embedding_matrix[db_id_to_index[tid]]
                        for flag, tid in zip(mask_unseen, batch_target_ids) if flag
                    ])

                    if emb_targets.size(0) == 0:
                        continue

                    emb_anchor_expanded = emb_anchor.expand(emb_targets.size(0), -1)
                    sims = F.cosine_similarity(emb_anchor_expanded, emb_targets)

                    idx = 0
                    for flag, tid in zip(mask_unseen, batch_target_ids):
                        if flag:
                            sim_val = sims[idx].item()
                            edge = 1 if sim_val > sim_threshold else 0
                            pair = tuple(sorted((anchor_id, tid)))
                            label = 1 if pair in qid_pairs else 0
                            all_records.append([anchor_id, tid, sim_val, label, edge])
                            total_new_written += 1
                            idx += 1

        save_path = os.path.join(save_dir, "all_exhaustive_predictions.pt")
        torch.save(all_records, save_path)

        summary_path = os.path.join(save_dir, "summary.txt")
        with open(summary_path, "w") as f:
            f.write(f"Total available IDs: {num_ids}\n")
            f.write(f"Total unordered pairs (C(n,2)): {total_possible:,}\n")
            f.write(f"Previously matched pairs: {len(original_records):,}\n")
            f.write(f"Newly matched this run: {total_new_written:,}\n")
            f.write(f"Output .pt file: {os.path.basename(save_path)}\n")

        print(f"All done. Total new pairs written: {total_new_written:,}")
        print(f"Results saved to {save_path}")
        print(f"Summary saved to {summary_path}")



if __name__ == "__main__":
    from model.WKDataset import WKDataset

    loader = WKDataset(schema_dir="../data/schema", csv_base_dir="../data/unzip")

    # for model_type in ["bge-large-en-v1.5", "bge-m3"]:
    #     print(f"\n--- Using model: {model_type} ---")
    #     embedder = BGEEmbedder(model_type=model_type)

    #     torch.cuda.empty_cache()
    #     torch.cuda.reset_peak_memory_stats()

    #     sim1, matched_pair, elapsed = embedder.database_similarity(loader, "78145", "95960")
    #     peak_memory = torch.cuda.max_memory_allocated(device="cuda")
    #     print(f"sim1: {sim1:.4f}, matched_pair: {matched_pair}, elapsed: {elapsed:.2f} seconds")
    #     print(f"peak memory: {peak_memory / 1e9:.2f} GB")

    #     torch.cuda.empty_cache()
    #     torch.cuda.reset_peak_memory_stats()

    #     sim2, matched_pair, elapsed = embedder.database_similarity(loader, "00024", "62098")
    #     peak_memory = torch.cuda.max_memory_allocated(device="cuda")
    #     print(f"sim2: {sim2:.4f}, matched_pair: {matched_pair}, elapsed: {elapsed:.2f} seconds")
    #     print(f"peak memory: {peak_memory / 1e9:.2f} GB")
    # lr=1e-05
    embedder = BGEEmbedder(model_type="bge-m3", model_path=f"/hpctmp/e1351271/wkdbs/out/col_matcher_bge-m3_database/weights/finetuned_bge_m3_softmax_lr1e-05/best")

    # dataset_dir = "/hpctmp/e1351271/wkdbs/data/uci_datasets/breast+cancer+wisconsin+diagnostic"
    # data_path = os.path.join(dataset_dir, "wdbc.data")
    # names_path = os.path.join(dataset_dir, "wdbc.names")
    dataset_dir = "/hpctmp/e1351271/wkdbs/data/uci_datasets/twitter+geospatial+data"
    data_path = os.path.join(dataset_dir, "twitter.csv")
    emb_save_path = os.path.join(dataset_dir, "embeddings.pt")
    all_embeddings_path = "/hpctmp/e1351271/wkdbs/out/col_matcher_bge-m3_database/weights/finetuned_bge_m3_softmax_lr1e-05/embeddings/all_embeddings.pt"
    df = pd.read_csv(data_path)
    db_name = "twitter_geospatial_data"
    schema = format_schema_from_dataframe(df, db_name)
    print(schema)
    embedding = embedder.get_embedding(schema)
    torch.save(embedding, emb_save_path)
    sim_list = embedder.get_topk_similar_dbs(embedding, all_embeddings_path, 50)
    print(sim_list)
    save_path = os.path.join(dataset_dir, "sim_list_schema.txt")
    with open(save_path, "w") as f:
        count = 0
        for db_id, sim in sim_list:
            db_schema = format_schema_from_loader(loader, db_id, sample=False, only_show_column_name=True)
            print(db_schema)
            f.write("-" * 100 + "\n")
            f.write(f"sim_list[{count}]: {sim:.4f}\n")
            f.write(f"DB: {db_id}\n")
            f.write(db_schema + "\n\n")
            count += 1
    # for seed in range(5):
    #     embedder.test_scalable(
    #         test_path=f"/hpctmp/e1351271/wkdbs/data/split_triplets/triplets_test_seed{seed}.jsonl",
    #         embedding_path=f"/hpctmp/e1351271/wkdbs/out/col_matcher_bge-m3_database/weights/finetuned_bge_m3_softmax_lr{lr}/embeddings/all_embeddings.pt",
    #         save_dir=f"/hpctmp/e1351271/wkdbs/out/col_matcher_bge-m3_database/weights/finetuned_bge_m3_softmax_lr{lr}/test_results_seed{seed}"
    #     )

    # embedder.test_scalable(
    #     test_path="/hpctmp/e1351271/wkdbs/data/split_triplets/triplets_test_fullneg.jsonl",
    #     embedding_path="/hpctmp/e1351271/wkdbs/out/col_matcher_bge-m3_database/weights/finetuned_bge_m3_softmax_lr1e-05/embeddings/all_embeddings.pt",
    #     save_dir="/hpctmp/e1351271/wkdbs/out/col_matcher_bge-m3_database/weights/finetuned_bge_m3_softmax_lr1e-05/test_results_fullneg"
    # )

    # embedder.test_all_possible_pairs(
    #     embedding_path="/hpctmp/e1351271/wkdbs/out/col_matcher_bge-m3_database/weights/finetuned_bge_m3_softmax_lr1e-05/embeddings/all_embeddings.pt",
    #     pred_csv_path="/hpctmp/e1351271/wkdbs/out/col_matcher_bge-m3_database/weights/finetuned_bge_m3_softmax_lr1e-05/test_results_fullneg/predictions.csv",
    #     save_dir="/hpctmp/e1351271/wkdbs/out/col_matcher_bge-m3_database/weights/finetuned_bge_m3_softmax_lr1e-05/test_results_exhaustive_split",
    #     batch_size=4096,                
    #     max_rows_per_file=int(1e8),
    #     sim_threshold=0.6713
    # )

    # generate_and_save_all_embeddings(
    #     loader=loader,
    #     embedder=embedder,
    #     output_path=f"/hpctmp/e1351271/wkdbs/out/col_matcher_bge-m3_database/weights/finetuned_bge_m3_softmax_lr{lr}/embeddings",
    #     batch_size=8,
    # )