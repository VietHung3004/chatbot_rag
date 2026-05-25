from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

from src.config import (
    OPENAI_API_KEY,
    EMBEDDING_MODEL,
    FAISS_Products,
    FAISS_Policies_large, EMBEDDING_MODEL_LARGE
)


# =========================
# PRODUCT VECTORSTORE (LARGE ONLY)
# =========================
def load_product_vs():

    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        openai_api_key=OPENAI_API_KEY
    )

    return FAISS.load_local(
        FAISS_Products,
        embeddings,
        allow_dangerous_deserialization=True
    )


# =========================
# POLICY VECTORSTORE (LARGE ONLY)
# =========================
def load_policy_vs():

    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL_LARGE,
        openai_api_key=OPENAI_API_KEY
    )

    return FAISS.load_local(
        FAISS_Policies_large,
        embeddings,
        allow_dangerous_deserialization=True
    )