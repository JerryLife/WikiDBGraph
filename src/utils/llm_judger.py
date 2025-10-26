import pandas as pd
import random
import csv
import os
from openai import OpenAI
from model.WKDataset import WKDataset
from utils.schema_formatter import format_schema_from_loader
import re

# GPT请求封装
def ask_gpt_similarity(client, db1_text: str, db2_text: str) -> str:
    user_prompt = f"""You will be given two database schemas.

Please evaluate their similarity based on structure and semantics, and return a score **between -10 and 10**, followed by a brief explanation of your reasoning.

Respond in this format:
<score>: <short explanation in one sentence>

Schema 1:
{db1_text}

Schema 2:
{db2_text}
"""
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant skilled in database schema understanding."},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,
    )
    return response.choices[0].message.content.strip()

def extract_score_and_reason(response_text: str):
    match = re.match(r"(-?\d+)\s*:\s*(.*)", response_text.strip())
    if match:
        score = int(match.group(1))
        reason = match.group(2)
        return score, reason
    else:
        return None, response_text.strip()
    
def get_schema_string(db_id: str, loader: WKDataset) -> str:
    ret = format_schema_from_loader(loader, db_id, show_wikidata_property_id=False, sample_size=1)
    if len(ret) > 1024:
        ret = format_schema_from_loader(loader, db_id, only_show_table_name=True)
    return ret[:1024]

def evaluate_samples(input_csv_path: str, output_csv_path: str, sample_size: int = 50):
    df = pd.read_csv(input_csv_path, dtype={"anchor_id": str, "target_id": str})
    sampled_df = df.sample(n=sample_size, random_state=42)

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    loader = WKDataset(schema_dir="../data/schema", csv_base_dir="../data/unzip")

    results = []

    for idx, row in sampled_df.iterrows():
        anchor_id = row["anchor_id"].zfill(5)
        target_id = row["target_id"].zfill(5)
        similarity = row["similarity"]

        print(f"🔍 Processing {anchor_id} vs {target_id} | sim: {similarity:.4f}")

        db1_schema = get_schema_string(anchor_id, loader)
        db2_schema = get_schema_string(target_id, loader)

        try:
            response_text = ask_gpt_similarity(client, db1_schema, db2_schema)
            score, reason = extract_score_and_reason(response_text)
        except Exception as e:
            score, reason = None, f"Error: {e}"
        print("llm_score: ", score)
        print("llm_reason: ", reason)
        results.append({
            "anchor_id": anchor_id,
            "target_id": target_id,
            "similarity": similarity,
            "llm_score": score,
            "llm_reason": reason
        })

    output_df = pd.DataFrame(results)
    output_df.to_csv(output_csv_path, index=False)
    print(f"✅ Done. Saved to {output_csv_path}")

if __name__ == "__main__":
    input_csv = "/hpctmp/e1351271/wkdbs/out/col_matcher_bge-m3_database/weights/finetuned_bge_m3_softmax_lr1e-05/test_results_fullneg/label0_similarity_exceed_0.9969.csv"
    output_csv = "/hpctmp/e1351271/wkdbs/out/llm_scores_sampled.csv"

    evaluate_samples(input_csv, output_csv, sample_size=50)
