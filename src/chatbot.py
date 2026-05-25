from src.retriever import search
from src.config import OPENAI_API_KEY, LLM_MODEL
from langchain_openai import ChatOpenAI
import re

# =========================
# INIT LLM
# =========================
llm = ChatOpenAI(
    model=LLM_MODEL,
    api_key=OPENAI_API_KEY,
    temperature=0.3
)

llm_strict = ChatOpenAI(
    model=LLM_MODEL,
    api_key=OPENAI_API_KEY,
    temperature=0
)


# =========================
# MEMORY
# =========================
MAX_TURNS = 5


def format_history(chat_history):
    return "\n".join(chat_history)


def update_history(chat_history, query, answer):
    chat_history.append(f"User: {query}")
    chat_history.append(f"Bot: {answer}")
    if len(chat_history) > MAX_TURNS * 2:
        chat_history = chat_history[-MAX_TURNS * 2:]
    return chat_history


# =========================
# EXTRACT BUDGET
# =========================
def extract_budget_million(query, chat_history):
    """
    Trích xuất ngân sách (đơn vị: triệu VND) từ query hoặc lịch sử.
    Ưu tiên query hiện tại, fallback sang lịch sử.
    Trả về float hoặc None.
    """
    for text in [query, "\n".join(chat_history)]:
        # "10 triệu", "10tr", "10 tr"
        m = re.search(r'(\d+(?:[.,]\d+)?)\s*(?:triệu|tr)\b', text.lower())
        if m:
            return float(m.group(1).replace(',', '.'))

        # Số nguyên >= 1.000.000: 10000000
        m = re.search(r'\b(\d{7,9})\b', text)
        if m:
            return float(m.group(1)) / 1_000_000

    return None


# =========================
# FILTER DOCS BY BUDGET ±1 triệu
# =========================
def filter_docs_by_budget(docs, budget_million):
    """
    Giữ lại doc có giá nằm trong [budget-1tr, budget+1tr].
    Nếu filter ra rỗng → trả về docs gốc.
    """
    if budget_million is None:
        return docs

    lower = (budget_million - 1) * 1_000_000
    upper = (budget_million + 1) * 1_000_000

    price_pattern = re.compile(
        r'\b(\d{1,3}(?:[.,]\d{3}){1,2})\b'
        r'|'
        r'\b(\d{6,9})\b'
    )

    def parse_price(raw: str) -> float:
        clean = raw.replace('.', '').replace(',', '')
        try:
            return float(clean)
        except ValueError:
            return 0.0

    filtered = []
    for doc in docs:
        for m in price_pattern.finditer(doc.page_content):
            raw = m.group(1) or m.group(2)
            price = parse_price(raw)
            if lower <= price <= upper:
                filtered.append(doc)
                break

    return filtered if filtered else docs


# =========================
# EXTRACT PRODUCT (single)
# =========================
def extract_product_with_llm(chat_history):
    if len(chat_history) < 2:
        return None

    prompt = f"""
Xác định sản phẩm điện thoại đang được nhắc tới gần nhất trong hội thoại.

QUY TẮC:
- Chỉ trả về tên sản phẩm
- Không giải thích
- Nếu không xác định được → trả về "none"

Hội thoại:
{chr(10).join(chat_history)}

Kết quả:
"""
    try:
        res = llm_strict.invoke(prompt).content.strip()
        return None if res.lower() == "none" else res
    except:
        return None


# =========================
# EXTRACT PRODUCT GROUP
# =========================
def extract_recent_products(chat_history):
    if len(chat_history) < 2:
        return []

    prompt = f"""
Trích xuất danh sách sản phẩm điện thoại gần nhất đang được nhắc tới trong hội thoại.

QUY TẮC:
- Chỉ lấy sản phẩm đang được so sánh/gợi ý gần nhất
- Trả về dạng: product1 | product2 | product3
- Nếu chỉ có 1 sản phẩm → trả về 1 tên
- Nếu không có → trả "none"

Hội thoại:
{chr(10).join(chat_history)}

Kết quả:
"""
    try:
        res = llm_strict.invoke(prompt).content.strip()
        if res.lower() == "none":
            return []
        return [x.strip() for x in res.split("|") if x.strip()]
    except:
        return []


# =========================
# REWRITE QUERY
# =========================
def rewrite_query(query, chat_history):
    """
    Chỉ điền tên sản phẩm vào câu hỏi thiếu ngữ cảnh.
    Không paraphrase, không đổi từ.
    """
    if not chat_history or len(chat_history) < 2:
        return query

    history_text = (
        format_history(chat_history)
        if isinstance(chat_history, list)
        else chat_history
    )

    # Lấy tên sản phẩm gần nhất từ lịch sử để kiểm tra
    # nếu query đã có tên sản phẩm → bỏ qua rewrite luôn
    product = extract_product_with_llm(chat_history)
    if product and product.lower() in query.lower():
        return query

    prompt = f"""Nhiệm vụ: Nếu câu hỏi thiếu tên sản phẩm, hãy thêm tên sản phẩm gần nhất từ lịch sử vào ĐẦU câu.

Quy tắc bắt buộc:
1. Chỉ THÊM tên sản phẩm vào đầu, KHÔNG thay đổi bất kỳ từ nào khác
2. Nếu câu hỏi đã đủ ngữ cảnh hoặc không liên quan sản phẩm → trả nguyên câu gốc
3. Chỉ trả về 1 câu duy nhất, không giải thích

Ví dụ đúng (lịch sử nhắc iPhone 17 Pro Max):
- "có màu gì" → "iPhone 17 Pro Max có màu gì"
- "giá bao nhiêu" → "iPhone 17 Pro Max giá bao nhiêu"
- "pin dùng được bao lâu" → "iPhone 17 Pro Max pin dùng được bao lâu"

Ví dụ KHÔNG rewrite:
- "iPhone 17 Pro Max giá bao nhiêu" → "iPhone 17 Pro Max giá bao nhiêu" (đã có tên)
- "bạn tên gì" → "bạn tên gì" (không liên quan sản phẩm)

Lịch sử hội thoại:
{history_text}

Câu hỏi gốc: {query}

Kết quả:"""

    try:
        rewritten = llm_strict.invoke(prompt).content.strip()
        # Bảo vệ: nếu rewrite dài bất thường hoặc rỗng → giữ gốc
        if not rewritten or len(rewritten) > len(query) * 4:
            return query
        return rewritten
    except:
        return query


# =========================
# GENERATE ANSWER
# =========================
def generate_answer(query, docs, route, chat_history, budget_million=None):

    history_text = (
        format_history(chat_history)
        if isinstance(chat_history, list)
        else chat_history
    )

    # ── OTHER ──────────────────────────────────────────
    if route == "other":
        prompt = f"""Bạn là nhân viên tư vấn bán hàng của Hoàng Hà Mobile.
Trò chuyện tự nhiên, thân thiện, gợi mở nhu cầu khách hàng.

Lịch sử:
{history_text}

User: {query}
Bot:"""
        return llm.invoke(prompt).content

    # ── NO DATA ────────────────────────────────────────
    if not docs:
        return "Hiện tại mình chưa có thông tin về sản phẩm bạn hỏi. Bạn có thể mô tả thêm nhu cầu để mình tư vấn phù hợp hơn nhé!"

    # ── CONTEXT ────────────────────────────────────────
    # Tăng lên 2000 để tránh cắt mất thông tin quan trọng
    context = "\n\n---\n\n".join([doc.page_content[:2000] for doc in docs])

    budget_hint = ""
    if budget_million:
        budget_hint = f"""
[NGÂN SÁCH: {budget_million:.0f} triệu VND — chỉ gợi ý sản phẩm trong khoảng {budget_million - 1:.0f}–{budget_million + 1:.0f} triệu]
"""

    prompt = f"""Bạn là nhân viên tư vấn bán hàng của Hoàng Hà Mobile. Hãy trả lời câu hỏi của khách dựa trên DATA bên dưới.
{budget_hint}
NGUYÊN TẮC:
- Dùng thông tin trong DATA để trả lời. DATA có thể dùng từ khác nhau: "RAM/ROM/storage/bộ nhớ", "màu/color", v.v. — hãy nhận diện linh hoạt.
- Nếu DATA có thông tin liên quan → PHẢI trả lời, không được nói "chưa có thông tin".
- Nếu DATA thực sự không có → nói ngắn gọn "mình chưa có thông tin chi tiết về điều này".
- Trả lời thân thiện, đúng trọng tâm, không lan man.
- Không bịa thông số ngoài DATA.

LỊCH SỬ:
{history_text}

DATA:
{context}

Câu hỏi: {query}

Trả lời:"""

    return llm.invoke(prompt).content


# =========================
# CHAT LOOP
# =========================
def chat():

    print("🤖 Chatbot RAG - Hoàng Hà Mobile\n")

    chat_history = []

    while True:

        query = input("👤 Bạn: ").strip()

        if not query:
            continue

        if query.lower() == "exit":
            print("👋 Tạm biệt bạn!")
            break

        # ── 1. REWRITE ───────────────────────────────────
        rewritten_query = rewrite_query(query, chat_history)
        print(f"🔄 Rewrite: {rewritten_query}")

        # ── 2. COREFERENCE BACKUP ────────────────────────
        coref_words = ["nó", "cái này", "con này", "máy này", "máy đó", "con đó"]
        compare_words = ["máy nào", "con nào", "mạnh nhất", "ngon nhất", "đáng mua nhất"]

        if any(x in query.lower() for x in coref_words):
            product = extract_product_with_llm(chat_history)
            if product and product.lower() not in rewritten_query.lower():
                rewritten_query = f"{product}: {rewritten_query}"

        if any(x in query.lower() for x in compare_words):
            products = extract_recent_products(chat_history)
            if len(products) >= 2:
                product_str = ", ".join(products)
                if not any(p.lower() in rewritten_query.lower() for p in products):
                    rewritten_query = f"Trong các sản phẩm {product_str}, {query}"

        # ── 3. EXTRACT BUDGET ────────────────────────────
        budget = extract_budget_million(query, chat_history)
        if budget:
            print(f"💰 Budget: {budget:.0f} triệu → filter [{budget-1:.0f}tr – {budget+1:.0f}tr]")

        # ── 4. SEARCH ────────────────────────────────────
        docs, route = search(rewritten_query)
        print(f"🧠 Route: {route} | Docs retrieved: {len(docs)}")

        # ── 5. FILTER BY BUDGET ──────────────────────────
        if budget and route == "product":
            docs = filter_docs_by_budget(docs, budget)
            print(f"🔍 Docs after price filter: {len(docs)}")

        # ── 6. GENERATE ──────────────────────────────────
        answer = generate_answer(
            query=rewritten_query,
            docs=docs,
            route=route,
            chat_history=chat_history,
            budget_million=budget,
        )

        # ── 7. UPDATE MEMORY (lưu query gốc) ─────────────
        chat_history.append(f"User: {query}")
        chat_history.append(f"Bot: {answer}")

        if len(chat_history) > MAX_TURNS * 2:
            chat_history = chat_history[-MAX_TURNS * 2:]

        # ── OUTPUT ───────────────────────────────────────
        print(f"\n🤖 Bot: {answer}\n")


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    chat()