import os
import streamlit as st

def get_openai_key():
    try:
        key = st.secrets.get("OPENAI_API_KEY")
        if key:
            return key
    except Exception:
        pass

    return os.getenv("OPENAI_API_KEY")

OPENAI_API_KEY = get_openai_key()

if not OPENAI_API_KEY:
    raise ValueError("❌ OPENAI_API_KEY NOT FOUND (Streamlit Secrets or Env)")
# ======================
# Models
# ======================
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_MODEL_LARGE = "text-embedding-3-large"
LLM_MODEL = "gpt-4o-mini"

# ======================
# Base path
# ======================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ======================
# Database
# ======================
DB_PATH = os.path.join(BASE_DIR, "DB_code", "DB_Shop.db")

# ======================
# FAISS paths
# ======================
FAISS_Products = os.path.join(BASE_DIR, "vector_store", "Faiss_Product")
FAISS_Policies = os.path.join(BASE_DIR, "vector_store", "Faiss_Policies")
FAISS_Products_Large = os.path.join(BASE_DIR, "vector_store", "Faiss_Product_Large")
FAISS_Policies_large = os.path.join(BASE_DIR, "vector_store", "Faiss_Policies_Large")