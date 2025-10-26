import torch
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import AutoTokenizer, AutoModel
from fuzzywuzzy import fuzz
import warnings
from model.column_encoder import ColumnEncoder
import pandas as pd
import time
import os
import torch.nn.functional as F
DEFAULT_MODELS = ["sentence-transformers/all-mpnet-base-v2", "BAAI/bge-m3"]

# from cross_encoder import CrossEncoder
# model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

def compute_max_cosine_similarity(embeddings_df1, embeddings_df2, idf_input=None, idf_target=None):
    embeddings_df1 = torch.nn.functional.normalize(embeddings_df1, p=2, dim=1)
    embeddings_df2 = torch.nn.functional.normalize(embeddings_df2, p=2, dim=1)

    similarity_matrix = torch.matmul(embeddings_df1, embeddings_df2.T)
    total_column_count = similarity_matrix.shape[0] + similarity_matrix.shape[1]

    if idf_input is not None and idf_target is not None:
        idf_input_exp = idf_input.unsqueeze(1)  # (N1, 1)
        idf_target_exp = idf_target.unsqueeze(0)  # (1, N2)
        idf_matrix = torch.minimum(idf_input_exp, idf_target_exp)  # (N1, N2)
        similarity_matrix = similarity_matrix * idf_matrix
    else:
        warnings.warn("No IDF values provided, using raw similarity matrix.")

    # Global max from entire matrix
    max_sim_val = torch.max(similarity_matrix)
    max_idx_flat = torch.argmax(similarity_matrix)
    input_idx, target_idx = divmod(max_idx_flat.item(), similarity_matrix.shape[1])

    return max_sim_val, input_idx, target_idx, total_column_count



def compute_cosine_similarity_simple(embeddings_df1, embeddings_df2, k):

    embeddings_df1 = torch.nn.functional.normalize(embeddings_df1, p=2, dim=1)
    embeddings_df2 = torch.nn.functional.normalize(embeddings_df2, p=2, dim=1)

    # Compute the cosine similarity matrix
    # embeddings_df1: (N1, D), embeddings_df2: (N2, D)
    similarity_matrix = torch.matmul(embeddings_df1, embeddings_df2.T)
    # similarity_matrix: (N1, N2)

    # Get top-k similarities and their indices for each row in similarity_matrix
    topk_similarity, topk_indices = torch.topk(similarity_matrix, k, dim=1)

    return topk_similarity, topk_indices


def compute_cosine_similarity(
    embeddings_input: torch.Tensor, embeddings_target: torch.Tensor, top_k: int
):
    """
    Compute the top K cosine similarities between input and target embeddings.

    Parameters:
    - embeddings_input (torch.Tensor): Tensor of shape (num_input, embedding_dim)
    - embeddings_target (torch.Tensor): Tensor of shape (num_target, embedding_dim)
    - top_k (int): Number of top K similarities to return

    Returns:
    - top_k_similarity (np.ndarray): Array of shape (num_input, top_k) containing similarity scores
    - top_k_indices (np.ndarray): Array of shape (num_input, top_k) containing indices of the top K most similar embeddings
    """
    # Ensure embeddings are on the same device
    device = embeddings_input.device
    embeddings_target = embeddings_target.to(device)

    # Normalize embeddings
    input_norm = torch.norm(embeddings_input, dim=1, keepdim=True)
    target_norm = torch.norm(embeddings_target, dim=1, keepdim=True)

    # Compute cosine similarity
    similarities = torch.mm(embeddings_input, embeddings_target.T) / (
        input_norm * target_norm.T
    )

    # Remove self-similarities

    min_top_k = min(top_k, similarities.shape[1])

    # Get top K scores and indices
    top_k_scores, top_k_indices = torch.topk(
        similarities, min_top_k, dim=1, largest=True, sorted=True
    )

    # Convert to numpy arrays for easier handling
    top_k_scores = top_k_scores.cpu().numpy()
    top_k_indices = top_k_indices.cpu().numpy()

    return top_k_scores, top_k_indices, similarities


class EmbeddingMatcher:
    def __init__(self, params):
        self.params = params
        self.topk = params["topk"]
        self.embedding_threshold = params["embedding_threshold"]
        self.similarity_function = params["similarity_function"]
        self.is_idf_weighted = params["is_idf_weighted"]
        get_embedding_similarity_candidates_methods = {
            "cross_encoder": self.get_embedding_similarity_candidates_by_cross_encoder,
            "cosine": self.get_embedding_similarity_candidates_by_cosine
        }   
        self.get_embedding_similarity_candidates = get_embedding_similarity_candidates_methods[self.similarity_function]
        # Dynamically set device to GPU if available, else fallback to CPU
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        self.model_name = params["embedding_model"]

        # if self.similarity_function == "cross_encoder":
        #     self.model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', device=self.device)
        
        if self.similarity_function == "cosine":
            if self.model_name in DEFAULT_MODELS:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModel.from_pretrained(self.model_name)
                self.model.eval()
                self.model.to(self.device)
                print(f"Loaded ZeroShot Model {self.model_name} on {self.device}")
            else:
                # Base model
                base_model = "BAAI/bge-m3"
                self.model = AutoModel.from_pretrained(base_model, device=self.device)
                self.tokenizer = AutoTokenizer.from_pretrained(base_model, device=self.device)

                print(f"Loaded bge-m3 Model on {self.device}")

                # path to the trained model weights
                model_path = params["embedding_model"]
                if os.path.exists(model_path):
                    print(f"Loading trained model from {model_path}")
                    # Load state dict for the SentenceTransformer model
                    state_dict = torch.load(
                        model_path, map_location=self.device, weights_only=True
                    )
                    # Assuming the state_dict contains the proper model weights and is compatible with SentenceTransformer
                    self.model.load_state_dict(state_dict)
                    self.model.eval()
                    self.model.to(self.device)
                else:
                    print(
                        f"Trained model not found at {model_path}, loading default model."
                    )

    def _get_embeddings(self, texts, batch_size=32):
        if self.model_name in DEFAULT_MODELS:
            return self._get_embeddings_zs(texts, batch_size)
        else:
            return self._get_embeddings_ft(texts, batch_size)

    def _get_embeddings_zs(self, texts, batch_size=32):
        # embeddings = []
        # for i in range(0, len(texts), batch_size):
        #     batch_texts = texts[i: i + batch_size]
        #     inputs = self.tokenizer(
        #         batch_texts,
        #         padding=True,
        #         # Move inputs to device
        #         truncation=True,
        #         return_tensors="pt",
        #     ).to(self.device)
        #     with torch.no_grad():
        #         outputs = self.model(**inputs)
        #     embeddings.append(outputs.last_hidden_state.mean(dim=1))
        # return torch.cat(embeddings)
        # embeddings = self.model.encode(
        #     texts,
        #     batch_size=batch_size,
        #     convert_to_tensor=True,
        #     device=self.device,
        #     show_progress_bar=False
        # )
        # if embeddings.shape != (len(texts), 768):
        #     raise ValueError(f"Unexpected embedding shape: {embeddings.shape}")
        # print(embeddings.shape)
        # return embeddings
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i: i + batch_size]
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=512
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)

            embeddings = outputs.last_hidden_state[:, 0]  # CLS token
            embeddings = F.normalize(embeddings, p=2, dim=-1)
            all_embeddings.append(embeddings)

        return torch.cat(all_embeddings, dim=0)

    def _get_embeddings_ft(self, texts, batch_size=32):
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i: i + batch_size]
            with torch.no_grad():
                batch_embeddings = self.model.encode(
                    batch_texts, show_progress_bar=False, device=self.device
                )
            embeddings.append(torch.tensor(batch_embeddings))
        return torch.cat(embeddings)

    def get_embedding_similarity_candidates_by_cosine(self, source_df, target_df):

        encoder = ColumnEncoder(
            self.tokenizer,
            encoding_mode=self.params["encoding_mode"],
            sampling_mode=self.params["sampling_mode"],
            n_samples=self.params["sampling_size"],
        )
        input_col_repr_dict = {
            encoder.encode(source_df, col): col for col in source_df.columns
        }
        target_col_repr_dict = {
            encoder.encode(target_df, col): col for col in target_df.columns
        }

        cleaned_input_col_repr = list(input_col_repr_dict.keys())
        cleaned_target_col_repr = list(target_col_repr_dict.keys())

        start_time = time.time()
        embeddings_input = self._get_embeddings(cleaned_input_col_repr, batch_size=64)
        end_time = time.time()
        print(f"Time taken to get source embeddings: {end_time - start_time:.6f} seconds")

        start_time = time.time()
        embeddings_target = self._get_embeddings(cleaned_target_col_repr, batch_size=64)
        end_time = time.time()
        print(f"Time taken to get target embeddings: {end_time - start_time:.6f} seconds")

        start_time = time.time()
        if self.is_idf_weighted:
            idf_df = pd.read_csv("/hpctmp/e1351271/wkdbs/data/field_idf_scores_normalized.csv")
            idf_map = dict(zip(idf_df["field"], idf_df["normalized_idf"]))

            idf_input = torch.tensor([
                idf_map.get(input_col_repr_dict[col].split("::")[-1].lower(), 1.0)
                for col in cleaned_input_col_repr
            ], device=self.device)

            idf_target = torch.tensor([
                idf_map.get(target_col_repr_dict[col].split("::")[-1].lower(), 1.0)
                for col in cleaned_target_col_repr
            ], device=self.device)
        else:
            idf_input = torch.ones(len(cleaned_input_col_repr), device=self.device)
            idf_target = torch.ones(len(cleaned_target_col_repr), device=self.device)

        end_time = time.time()
        print(f"Time taken to get idf values: {end_time - start_time:.6f} seconds")

        start_time = time.time()
        max_sim_val, input_idx, target_idx, total_column_count = compute_max_cosine_similarity(
            embeddings_input, embeddings_target, idf_input, idf_target
        )
        end_time = time.time()
        print(f"Time taken to compute max cosine similarity: {end_time - start_time:.6f} seconds")
        
        original_input_col = input_col_repr_dict[cleaned_input_col_repr[input_idx]]
        original_target_col = target_col_repr_dict[cleaned_target_col_repr[target_idx]]

        if max_sim_val.item() >= self.embedding_threshold:
            matched_pair = (original_input_col, original_target_col)
        else:
            matched_pair = None

        print(f"Total column count: {total_column_count}")
        print("-"*20)
        return max_sim_val.item(), matched_pair, total_column_count
        # candidates = {}

        # for i, cleaned_input_col in enumerate(cleaned_input_col_repr):
        #     original_input_col = input_col_repr_dict[cleaned_input_col]

        #     for j in range(top_k):
        #         cleaned_target_col = cleaned_target_col_repr[topk_indices[i, j]]
        #         original_target_col = target_col_repr_dict[cleaned_target_col]
        #         similarity = topk_similarity[i, j].item()

        #         if similarity >= self.embedding_threshold:
        #             candidates[(original_input_col,
        #                         original_target_col)] = similarity

        # return candidates
        # return topk_similarity, matched_pair, total_column_count

    def get_embedding_similarity_candidates_by_cross_encoder(self, source_df, target_df):
        encoder = ColumnEncoder(
            encoding_mode=self.params["encoding_mode"],
            sampling_mode=self.params["sampling_mode"],
            n_samples=self.params["sampling_size"],
        )

        source_repr_map = {encoder.encode(source_df, col): col for col in source_df.columns}
        target_repr_map = {encoder.encode(target_df, col): col for col in target_df.columns}

        source_reprs = list(source_repr_map.keys())
        target_reprs = list(target_repr_map.keys())

        idf_df = pd.read_csv("/hpctmp/e1351271/wkdbs/data/field_idf_scores_normalized.csv")
        idf_map = dict(zip(idf_df["field"], idf_df["normalized_idf"]))

        src_idf_cache = {
            src_repr: idf_map.get(source_repr_map[src_repr].split("::")[1].lower(), 1.0)
            for src_repr in source_reprs
        }

        tgt_idf_cache = {
            tgt_repr: idf_map.get(target_repr_map[tgt_repr].split("::")[1].lower(), 1.0)
            for tgt_repr in target_reprs
        }

        input_pairs = []
        pair_idfs = []

        start_time = time.time()
        for src_repr in source_reprs:
            for tgt_repr in target_reprs:
                input_pairs.append((src_repr, tgt_repr))
                pair_idfs.append(min(src_idf_cache[src_repr], tgt_idf_cache[tgt_repr]))
        end_time = time.time()
        print(f"Time taken to create input pairs: {end_time - start_time} seconds")

        start_time = time.time()
        scores = self.model.predict(input_pairs, batch_size=64)
        end_time = time.time()
        print(f"Time taken to predict scores: {end_time - start_time} seconds")

        scores_tensor = torch.tensor(scores, device=self.device)
        idf_tensor = torch.tensor(pair_idfs, device=self.device)

        weighted_scores = torch.where(
            scores_tensor > 0,
            scores_tensor * idf_tensor,
            scores_tensor
        )
        max_idx = torch.argmax(weighted_scores).item()

        max_score = scores_tensor[max_idx].item()
        weighted_score = weighted_scores[max_idx].item()
        src_repr, tgt_repr = input_pairs[max_idx]

        src_col = source_repr_map[src_repr]
        tgt_col = target_repr_map[tgt_repr]

        matched_pair = (src_col, tgt_col)

        total_column_count = len(source_df.columns) + len(target_df.columns)

        return weighted_score, matched_pair, total_column_count