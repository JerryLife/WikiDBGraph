import json
from tqdm import tqdm

def generate_fullneg_triplets(test_path, output_path, id_range=100000):
    """
    Generate a new test file where each triplet contains all possible negatives
    except its anchor and positive.

    Parameters:
    - test_path (str): Path to the original triplets_test.jsonl file
    - output_path (str): Path to save the modified triplets_test_fullneg.jsonl
    - id_range (int): Total number of possible IDs (default: 100000)
    """
    # Step 1: Create the universal ID set (00000 to 99999)
    all_possible_ids = {str(i).zfill(5) for i in range(id_range)}

    # Step 2: Load original test triplets
    with open(test_path, "r") as f:
        test_data = [json.loads(line) for line in f]

    # Step 3: Replace negatives with all IDs excluding anchor and positive
    for item in tqdm(test_data, desc="Rewriting negative samples"):
        anchor = item["anchor"]
        positive = item["positive"]
        excluded = {anchor, positive}
        item["negatives"] = sorted(list(all_possible_ids - excluded))

    # Step 4: Save the modified triplets
    with open(output_path, "w") as f:
        for item in test_data:
            f.write(json.dumps(item) + "\n")

    print(f"✅ New test file saved to: {output_path}")

def main():
    test_path = "data/split_triplets/triplets_test.jsonl"
    output_path = "data/split_triplets/triplets_test_fullneg.jsonl"
    generate_fullneg_triplets(test_path, output_path)

if __name__ == "__main__":
    main()
