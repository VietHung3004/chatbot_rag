from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.load_data import load_policies
from src.config import OPENAI_API_KEY, EMBEDDING_MODEL, FAISS_Policies


def build_policies_faiss():

    docs = load_policies()
    print(f"📜 Policies loaded: {len(docs)}")

    if len(docs) == 0:
        raise ValueError("❌ No policy data found!")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=80
    )

    chunks = splitter.create_documents(docs)

    print(f"✂️ Policy chunks created: {len(chunks)}")
    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        openai_api_key=OPENAI_API_KEY
    )
    vectorstore = FAISS.from_documents(chunks, embeddings)

    vectorstore.save_local(FAISS_Policies)

    print("✅ FAISS_POLICIES saved successfully!")


if __name__ == "__main__":
    build_policies_faiss()