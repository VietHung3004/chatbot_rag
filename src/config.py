import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# API Key
OPENAI_API_KEY = (
    st.secrets.get("OPENAI_API_KEY")
    or os.getenv("OPENAI_API_KEY")
)

# Models
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_MODEL_LARGE = "text-embedding-3-large"
LLM_MODEL = "gpt-4o-mini"

# Lấy thư mục gốc của project
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Database
DB_PATH = os.path.join(
    BASE_DIR,
    "DB_code",
    "DB_Shop.db"
)

# FAISS paths
FAISS_Products = os.path.join(
    BASE_DIR,
    "vector_store",
    "Faiss_Product"
)

FAISS_Policies = os.path.join(
    BASE_DIR,
    "vector_store",
    "Faiss_Policies"
)

FAISS_Products_Large = os.path.join(
    BASE_DIR,
    "vector_store",
    "Faiss_Product_Large"
)

FAISS_Policies_large = os.path.join(
    BASE_DIR,
    "vector_store",
    "Faiss_Policies_Large"
)