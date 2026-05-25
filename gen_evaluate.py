import json
import time
import re
from datetime import datetime

import pandas as pd
from langchain_openai import ChatOpenAI

from src.retriever import search
from src.chatbot import generate_answer
from src.config import OPENAI_API_KEY, LLM_MODEL


# =========================
# INIT LLM JUDGE
# Dùng model mạnh nhất có thể, KHÁC với model chatbot nếu được
# =========================
llm_judge = ChatOpenAI(
    model="gpt-4o",          # dùng model khác chatbot để tránh blind spot
    openai_api_key=OPENAI_API_KEY,
    temperature=0
)


# =========================
# STRICT JUDGE — Chain-of-Thought trước, chấm điểm sau
# =========================
JUDGE_PROMPT = """
Bạn là giám khảo NGHIÊM KHẮC đánh giá chatbot tư vấn điện thoại.
Nhiệm vụ: phát hiện MỌI lỗi, dù nhỏ. Không được bỏ qua.

==================================================
QUY TẮC BẮT BUỘC
==================================================
1. Phải viết REASONING trước — liệt kê từng lỗi cụ thể tìm thấy.
2. Nếu không tìm thấy lỗi nào, giải thích rõ tại sao answer đúng.
3. Chỉ sau khi reasoning xong mới được cho điểm.
4. Mặc định nghi ngờ answer — chỉ cho điểm cao khi CÓ BẰNG CHỨNG rõ ràng từ context.

==================================================
3 TIÊU CHÍ CHẤM ĐIỂM (0–10)
==================================================

── FAITHFULNESS ──────────────────────────────────
Mỗi thông tin trong answer có nguồn gốc từ context không?

Kiểm tra từng thông tin:
  ✓ Có trong context → OK
  ✗ Không có trong context → trừ điểm
  ✗ Sai so với context → trừ điểm nặng

Thang:
  10  : 100% thông tin có trong context
  8–9 : 1 suy diễn nhỏ không ảnh hưởng (vd: "dùng tốt" từ pin lớn)
  5–7 : Thêm 1–2 thông tin không có trong context
  2–4 : Nhiều thông tin bịa, hoặc sai 1 thông số quan trọng
  0–1 : Bịa hoàn toàn, hoặc sai nhiều thông số (pin, giá, chip, RAM...)

LƯU Ý — Lỗi thường gặp bị bỏ qua:
  - Làm tròn số sai: context "4947 mAh", answer "khoảng 5000 mAh" → trừ 1–2đ
  - Tự thêm tính năng: context không nhắc sạc nhanh mà answer nói có → trừ 3–5đ
  - Tự đánh giá chủ quan: "pin trâu", "màn hình đẹp" khi context không nói vậy → trừ 1–3đ
  - Nhầm model: context nói iPhone 17 Pro, answer nói iPhone 17 → trừ 5–8đ

── ANSWER RELEVANCE ──────────────────────────────
Answer có trả lời ĐÚNG câu hỏi không?

Kiểm tra:
  - Câu hỏi hỏi gì? (thông số / so sánh / tư vấn / giá?)
  - Answer có trả lời đúng điểm đó không?
  - Có bị lạc sang chủ đề khác không?

Thang:
  10  : Trả lời chính xác đúng câu hỏi, không thừa không thiếu
  8–9 : Đúng nhưng hơi thừa hoặc thiếu 1 chi tiết nhỏ
  5–7 : Trả lời được một phần, hoặc đúng nhưng lan man
  2–4 : Lạc đề, chủ yếu nói chuyện khác
  0–1 : Hoàn toàn không liên quan

── CONTEXT RELEVANCE ─────────────────────────────
Context được retrieve có chứa thông tin để trả lời câu hỏi không?

Thang:
  10  : Context chứa đúng và đủ thông tin cần thiết
  7–9 : Có thông tin cần, nhưng lẫn nhiều thông tin thừa
  4–6 : Chỉ một phần nhỏ liên quan
  1–3 : Hầu như không liên quan
  0   : Hoàn toàn sai chủ đề

==================================================
INPUT
==================================================

QUESTION: {question}

CONTEXT:
{context}

ANSWER:
{answer}

==================================================
OUTPUT FORMAT — BẮT BUỘC ĐÚNG ĐỊNH DẠNG NÀY
==================================================
{{
  "faithfulness_reasoning": "<liệt kê từng thông tin trong answer, đối chiếu với context, ghi rõ đúng/sai/bịa>",
  "answer_relevance_reasoning": "<câu hỏi hỏi gì, answer có trả lời đúng không, thiếu/thừa gì>",
  "context_relevance_reasoning": "<context có chứa thông tin cần không, thừa thiếu ra sao>",
  "faithfulness": <0–10>,
  "answer_relevance": <0–10>,
  "context_relevance": <0–10>
}}

Chỉ trả về JSON. Không markdown. Không giải thích thêm ngoài JSON.
"""


def _parse_judge_response(raw: str) -> dict:
    """Bóc JSON từ response, handle cả trường hợp có markdown fence."""
    raw = raw.strip()

    # strip ```json ... ``` nếu có
    if raw.startswith("```"):
        match = re.search(r"```(?:json)?\s*(.*?)```", raw, re.DOTALL)
        if match:
            raw = match.group(1).strip()

    return json.loads(raw)


def judge_rag(question: str, context: str, answer: str, max_retries: int = 3) -> dict:
    """
    Chấm điểm 3 tiêu chí với CoT reasoning.
    Trả về dict gồm scores + reasoning + error flag.
    """

    # ── edge cases ───────────────────────────────────────────────
    if not context.strip():
        return {
            "faithfulness": 0.0, "answer_relevance": 0.0, "context_relevance": 0.0,
            "faithfulness_reasoning": "N/A", "answer_relevance_reasoning": "N/A",
            "context_relevance_reasoning": "N/A", "error": "empty_context"
        }
    if not answer.strip():
        return {
            "faithfulness": 0.0, "answer_relevance": 0.0, "context_relevance": 0.0,
            "faithfulness_reasoning": "N/A", "answer_relevance_reasoning": "N/A",
            "context_relevance_reasoning": "N/A", "error": "empty_answer"
        }

    prompt = JUDGE_PROMPT.format(
        question=question,
        context=context[:3000],   # giới hạn context để judge không bị overwhelm
        answer=answer
    )

    for attempt in range(1, max_retries + 1):
        try:
            raw = llm_judge.invoke(prompt).content
            data = _parse_judge_response(raw)

            result = {
                "faithfulness":               float(data.get("faithfulness", 0)),
                "answer_relevance":           float(data.get("answer_relevance", 0)),
                "context_relevance":          float(data.get("context_relevance", 0)),
                "faithfulness_reasoning":     data.get("faithfulness_reasoning", ""),
                "answer_relevance_reasoning": data.get("answer_relevance_reasoning", ""),
                "context_relevance_reasoning":data.get("context_relevance_reasoning", ""),
                "error": None
            }

            # clamp [0, 10]
            for k in ("faithfulness", "answer_relevance", "context_relevance"):
                result[k] = max(0.0, min(10.0, result[k]))

            return result

        except Exception as e:
            print(f"   ⚠️  Judge attempt {attempt}/{max_retries} failed: {e}")
            if attempt < max_retries:
                time.sleep(2 ** attempt)

    return {
        "faithfulness": 0.0, "answer_relevance": 0.0, "context_relevance": 0.0,
        "faithfulness_reasoning": "", "answer_relevance_reasoning": "",
        "context_relevance_reasoning": "", "error": "judge_failed_all_retries"
    }


# =========================
# EVALUATION LOOP
# =========================
def evaluate(csv_path: str = "gen_test.csv", output_path: str = "eval_results.csv"):
    df = pd.read_csv(csv_path)

    if "question" not in df.columns:
        raise ValueError("CSV thiếu cột 'question'")

    has_budget = "budget_million" in df.columns
    total = len(df)
    results = []

    print(f"\n🚀 STRICT LLM JUDGE — {total} samples — {datetime.now():%Y-%m-%d %H:%M:%S}")
    print("=" * 90)

    for i, row in df.iterrows():
        question = str(row["question"]).strip()
        budget   = row["budget_million"] if has_budget else None

        print(f"\n[{i + 1}/{total}] ❓ {question}")

        # ── Retrieval ────────────────────────────────────────────
        try:
            docs, route = search(question)
            context = "\n".join([d.page_content for d in docs])
        except Exception as e:
            print(f"   ❌ RETRIEVAL ERROR: {e}")
            results.append(_error_row(question, "", "", "retrieval_error", str(e)))
            continue

        # ── Generation ───────────────────────────────────────────
        try:
            answer = str(generate_answer(
                query=question, docs=docs, route=route,
                chat_history=[], budget_million=budget
            )).strip()
        except Exception as e:
            print(f"   ❌ GENERATION ERROR: {e}")
            results.append(_error_row(question, context, "", "generation_error", str(e), str(route)))
            continue

        # ── Judge ────────────────────────────────────────────────
        score  = judge_rag(question, context, answer)
        overall = (score["faithfulness"] + score["answer_relevance"] + score["context_relevance"]) / 3

        # ── In kết quả kèm reasoning ─────────────────────────────
        print(f"   Route              : {route}")
        print(f"   Faithfulness       : {score['faithfulness']:.1f}  → {score['faithfulness_reasoning'][:120]}")
        print(f"   Answer Relevance   : {score['answer_relevance']:.1f}  → {score['answer_relevance_reasoning'][:120]}")
        print(f"   Context Relevance  : {score['context_relevance']:.1f}  → {score['context_relevance_reasoning'][:120]}")
        print(f"   Overall            : {overall:.2f}  {'✅' if score['error'] is None else '⚠️  ' + str(score['error'])}")

        results.append({
            "question":                   question,
            "answer":                     answer,
            "context":                    context[:500] + "..." if len(context) > 500 else context,
            "faithfulness":               score["faithfulness"],
            "answer_relevance":           score["answer_relevance"],
            "context_relevance":          score["context_relevance"],
            "overall":                    round(overall, 2),
            "faithfulness_reasoning":     score["faithfulness_reasoning"],
            "answer_relevance_reasoning": score["answer_relevance_reasoning"],
            "context_relevance_reasoning":score["context_relevance_reasoning"],
            "route":                      str(route),
            "error":                      score.get("error")
        })

    # ── Save ─────────────────────────────────────────────────────
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"\n💾 Results saved → {output_path}")

    # ── Summary ───────────────────────────────────────────────────
    _print_summary(results_df)
    return results_df


def _error_row(question, context, answer, err_type, err_msg, route="error"):
    return {
        "question": question, "answer": answer,
        "context": context[:500] if context else "",
        "faithfulness": 0.0, "answer_relevance": 0.0, "context_relevance": 0.0,
        "overall": 0.0,
        "faithfulness_reasoning": "", "answer_relevance_reasoning": "",
        "context_relevance_reasoning": "",
        "route": route, "error": f"{err_type}: {err_msg}"
    }


def _print_summary(df: pd.DataFrame):
    clean = df[df["error"].isna()]
    n_clean, n_err = len(clean), len(df) - len(clean)

    print("\n" + "=" * 90)
    print("🔥 FINAL RESULT SUMMARY")
    print("=" * 90)
    print(f"  Total   : {len(df)}  |  Valid: {n_clean}  |  Errors: {n_err}")

    if n_clean == 0:
        print("  ⚠️  Không có sample hợp lệ để tính điểm.")
        return

    print(f"\n  Avg Faithfulness     : {clean['faithfulness'].mean():.2f} / 10")
    print(f"  Avg Answer Relevance : {clean['answer_relevance'].mean():.2f} / 10")
    print(f"  Avg Context Relevance: {clean['context_relevance'].mean():.2f} / 10")
    print(f"  ─────────────────────────────────────────")
    print(f"  Avg Overall          : {clean['overall'].mean():.2f} / 10")

    # Distribution
    print("\n  Score distribution (Overall):")
    buckets = [("Excellent 8–10", 8, 10), ("Good 6–8", 6, 8),
               ("Fair 4–6", 4, 6),        ("Poor 0–4", 0, 4)]
    for label, lo, hi in buckets:
        count = ((clean["overall"] >= lo) & (clean["overall"] < hi)).sum()
        if hi == 10:
            count = (clean["overall"] >= lo).sum()
        pct = count / n_clean * 100
        print(f"    {label:<18}: {count:>3} ({pct:5.1f}%)  {'█' * int(pct / 5)}")

    # Worst 5
    cols = ["question", "faithfulness", "answer_relevance", "context_relevance", "overall"]
    print("\n  ⚠️  Top 5 worst samples (để debug):")
    print(clean.nsmallest(5, "overall")[cols].to_string(index=False))

    # Điểm thấp bất thường — faithful thấp nhưng relevance cao (hallucination signal)
    hallucination = clean[(clean["faithfulness"] < 5) & (clean["answer_relevance"] >= 7)]
    if len(hallucination) > 0:
        print(f"\n  🚨 Hallucination risk ({len(hallucination)} samples): answer relevance cao nhưng faithful thấp")
        print(hallucination[["question", "faithfulness", "answer_relevance"]].to_string(index=False))

    print("=" * 90)


if __name__ == "__main__":
    evaluate(csv_path="gen_test.csv", output_path="eval_results.csv")