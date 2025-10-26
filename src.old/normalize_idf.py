import pandas as pd

INPUT_PATH = "/hpctmp/e1351271/wkdbs/data/field_idf_scores.csv"
OUTPUT_PATH = "/hpctmp/e1351271/wkdbs/data/field_idf_scores_normalized.csv"

def normalize_idf(df):
    idf_min = df["idf"].min()
    idf_max = df["idf"].max()
    df["normalized_idf"] = (df["idf"] - idf_min) / (idf_max - idf_min)
    return df

def main():
    df = pd.read_csv(INPUT_PATH)
    print(f"Loaded {len(df)} fields.")
    df = normalize_idf(df)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"✅ Saved normalized IDF to: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
