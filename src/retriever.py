from src.vectorstore import load_product_vs, load_policy_vs
from src.config import OPENAI_API_KEY, LLM_MODEL

from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model=LLM_MODEL,
    openai_api_key=OPENAI_API_KEY,
    temperature=0
)


def classify_query(query: str):
    prompt = f"""
Bạn là hệ thống phân loại câu hỏi cho chatbot bán điện thoại.

Chỉ trả về DUY NHẤT 1 từ trong 3 lựa chọn:
- product (hỏi về điện thoại, giá, cấu hình, so sánh, pin, camera...)
- policy (bảo hành, vận chuyển, trả góp, đổi trả...)
- other (chào hỏi, nói chuyện, không liên quan)

KHÔNG giải thích.

Câu hỏi:
{query}

Trả lời:
""".strip()

    res = llm.invoke(prompt).content.strip().lower()

    if res not in ["product", "policy", "other"]:
        return "product"

    return res


# =========================
# SEARCH
# =========================
def search(query: str):

    route = classify_query(query)

    print(f"🧠 Route: {route}")

    # POLICY
    if route == "policy":
        vs = load_policy_vs()
        docs = vs.similarity_search(query, k=3)
        return docs, route

    # PRODUCT
    if route == "product":
        vs = load_product_vs()
        docs = vs.similarity_search(query, k=10)
        return docs, route

    # OTHER
    return [], route


# =========================
# TEST CLI8
# =========================
if __name__ == "__main__":

    print("TEST RETRIEVER LLM ROUTER\n")

    while True:
        query = input("Query: ")

        if query.lower() == "exit":
            break

        docs, route = search(query)

        if route == "other":
            print("\n💬 Đây là câu hỏi hội thoại (LLM sẽ trả lời trực tiếp)\n")
            continue

        print(f"\n📦 Found {len(docs)} documents\n")

        for i, doc in enumerate(docs):
            print(f"--- Result {i+1} ---")
            print(doc.page_content)
            print()