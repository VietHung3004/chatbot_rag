import pandas as pd
import re
from src.retriever import load_policy_vs


# =========================
# LOAD VECTOR STORE
# =========================
vs = load_policy_vs()


# =========================
# NORMALIZE TEXT
# =========================
def normalize(text: str):
    return text.strip().lower()


# =========================
# HIT CHECK (SEMANTIC TEXT MATCH)
# =========================
def is_hit(docs, expected_list):
    """
    Policy không có entity rõ ràng như product
    → dùng full-text match đơn giản, ổn định
    """

    retrieved_text = normalize(" ".join([d.page_content for d in docs]))

    for exp in expected_list:
        exp_norm = normalize(exp)

        if exp_norm and exp_norm in retrieved_text:
            return True

    return False


# =========================
# PRINT TOP-K
# =========================
def print_topk(docs, k):
    print(f"\n🔎 TOP {k} RESULTS:")

    for i, doc in enumerate(docs):
        preview = doc.page_content[:250].replace("\n", " ")
        print(f"[{i+1}] {preview}")


# =========================
# EVALUATION
# =========================
def evaluate(csv_path="policy_test.csv", ks=[1, 5, 10]):
    df = pd.read_csv(csv_path, encoding="utf-8-sig")

    total = len(df)
    results = {k: 0 for k in ks}

    for i, row in df.iterrows():
        query = str(row["query"])
        expected = str(row["expected"]).split("|")

        print("\n" + "=" * 80)
        print(f"[{i+1}/{total}] QUERY: {query}")
        print("EXPECTED:", expected)

        for k in ks:
            docs = vs.similarity_search(query, k=k)

            print_topk(docs, k)

            if is_hit(docs, expected):
                results[k] += 1

    print("\n" + "=" * 80)
    print("📊 FINAL POLICY RETRIEVAL RESULTS")

    for k in ks:
        recall = results[k] / total
        print(f"Recall@{k}: {recall:.4f}")


# =========================
# RUN
# =========================
if __name__ == "__main__":
    evaluate("policy_test.csv")