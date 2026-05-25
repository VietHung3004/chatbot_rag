import streamlit as st


from src.chatbot import (
    rewrite_query,
    generate_answer,
    update_history,
    extract_product_with_llm,
    search,
    llm_strict
)

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Hoàng Hà Mobile AI",
    page_icon="logo.png",
    layout="wide"
)

st.image("logo.svg", width=250)
st.caption("Chatbot tư vấn điện thoại thông minh")

# =========================
# SESSION STATE
# =========================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# =========================
# SIDEBAR
# =========================
with st.sidebar:

    st.header("⚙️ Điều khiển")

    st.markdown("""
### 💡 Gợi ý:
- Điện thoại chơi game tốt
- iPhone nào đáng mua?
- Máy pin trâu dưới 5 triệu
- So sánh Samsung và iPhone
- Thủ tục trả góp
""")

    if st.button("🗑️ Xóa hội thoại"):
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.rerun()

# =========================
# HIỂN THỊ CHAT
# =========================
for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# =========================
# USER INPUT
# =========================
if query := st.chat_input("Nhập câu hỏi về điện thoại..."):

    # 👉 hiện tin nhắn user
    st.session_state.messages.append({
        "role": "user",
        "content": query
    })

    with st.chat_message("user"):
        st.markdown(query)

    # =========================
    # BOT PROCESSING
    # =========================
    with st.spinner("AI đang nhập câu trả lời..."):

        # 🔄 Rewrite query
        rewritten_query = rewrite_query(
            query,
            st.session_state.chat_history
        )

        # 🧠 xử lý coreference
        if any(x in query.lower() for x in ["nó", "cái này", "con này"]):

            product = extract_product_with_llm(
                st.session_state.chat_history
            )

            if product:

                for token in ["nó", "cái này", "con này"]:
                    rewritten_query = rewritten_query.replace(
                        token,
                        product
                    )

        # 🔍 Search
        docs, route = search(rewritten_query)

        # 🤖 Generate answer
        answer = generate_answer(
            query,
            docs,
            route,
            st.session_state.chat_history
        )

        # 💾 Update memory
        st.session_state.chat_history = update_history(
            st.session_state.chat_history,
            query,
            answer
        )

    # 👉 lưu assistant message
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })

    # 👉 hiển thị assistant
    with st.chat_message("assistant"):
        st.markdown(answer)