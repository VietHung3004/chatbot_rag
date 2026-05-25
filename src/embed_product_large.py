from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from src.load_data import load_products
from src.config import OPENAI_API_KEY, EMBEDDING_MODEL_LARGE, FAISS_Products_Large


def build_products_faiss_large():

    raw_docs = load_products()
    print(f"📱 Products loaded: {len(raw_docs)}")

    if len(raw_docs) == 0:
        raise ValueError("❌ No product data found!")

    # convert str → Document
    docs = [Document(page_content=d) for d in raw_docs]

    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL_LARGE,
        openai_api_key=OPENAI_API_KEY
    )

    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(FAISS_Products_Large)

    print("✅ FAISS_PRODUCTS (LARGE) saved successfully!")


if __name__ == "__main__":
    build_products_faiss_large()