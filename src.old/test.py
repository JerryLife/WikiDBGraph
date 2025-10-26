from sentence_transformers import CrossEncoder

# model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu")
model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2", device="cpu")

pairs = [
    ("person::name, Alice, Bob", "employee::full_name, Alice, Bob"),
    ("location::latitude, 12.5", "book::isbn, 978-3-16-148410-0")
]

scores = model.predict(pairs)
for i, s in enumerate(scores):
    print(f"Pair {i+1} score: {s:.4f}")


# tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")
# texts = ["Hello, world!", "This is a test.", "This is a test 2.", "This is a test 3."]
# batch_size = 2
# embeddings = []
# for i in range(0, len(texts), batch_size):
#     batch_texts = texts[i: i + batch_size]
#     inputs = tokenizer(
#         batch_texts,
#         padding=True,
#         truncation=True,
#         return_tensors="pt",
#     ).to("cpu")
#     with torch.no_grad():
#         outputs = model(**inputs)
#     embeddings.append(outputs.last_hidden_state.mean(dim=1))
# embeddings = torch.cat(embeddings)
