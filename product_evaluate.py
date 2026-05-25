import pandas as pd
import re
from src.retriever import load_product_vs


# =========================
# LOAD VECTOR STORE
# =========================
vs = load_product_vs()


# =========================
# EXTRACT PRODUCT NAME
# =========================
def extract_product_name(text: str):
    """
    Lấy tên sản phẩm từ document
    Format:
    [SẢN PHẨM]
    Tên: iPhone 16
    """
    match = re.search(r"Tên:\s*(.*)", text)
    if match:
        return match.group(1).strip().lower()
    return text.strip().lower()


# =========================
# NORMALIZE
# =========================
def normalize(text: str):
    return text.strip().lower()


# =========================
# HIT CHECK (ENTITY-LEVEL)
# =========================
def is_hit(docs, expected_list):
    retrieved = set()

    for doc in docs:
        name = extract_product_name(doc.page_content)
        retrieved.add(name)

    for exp in expected_list:
        if normalize(exp) in retrieved:
            return True

    return False


# =========================
# PRINT TOP-K RESULTS
# =========================
def print_topk(docs, k):
    names = [extract_product_name(d.page_content) for d in docs]
    print(f"\n🔎 TOP {k}: {names}")


# =========================
# EVALUATION
# =========================
def evaluate(csv_path="product_test.csv", ks=[1, 5, 10]):
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
    print("📊 FINAL PRODUCT RETRIEVAL RESULTS")

    for k in ks:
        recall = results[k] / total
        print(f"Recall@{k}: {recall:.4f}")


# =========================
# RUN
# =========================
if __name__ == "__main__":
    evaluate("product_test.csv")