import sqlite3
from src.config import DB_PATH


def load_products():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT id, name, brand, chip, battery, description
        FROM products
    """)
    products = cursor.fetchall()

    documents = []

    for product in products:
        product_id = product[0]

        cursor.execute("""
            SELECT color, ram, storage, price, stock
            FROM product_variants
            WHERE product_id = ?
        """, (product_id,))
        variants = cursor.fetchall()

        # ===== xử lý variants =====
        variant_by_storage = {}
        prices = []

        for color, ram, storage, price, stock in variants:
            prices.append(price)

            key = f"Bộ nhớ: {storage}GB"
            if key not in variant_by_storage:
                variant_by_storage[key] = []

            variant_by_storage[key].append(
                f"Màu sắc: {color} ({price:,} VND, tồn kho {stock})"
            )

        # ===== format variants =====
        variant_text = ""
        for storage, items in variant_by_storage.items():
            variant_text += f"- {storage}:\n"
            for item in items:
                variant_text += f"  + {item}\n"

        # ===== giá min/max =====
        if prices:
            min_price = min(prices)
            max_price = max(prices)
            price_summary = f"""
[GIÁ THAM KHẢO]
- Thấp nhất: {min_price:,} VND
- Cao nhất: {max_price:,} VND
            """.strip()
        else:
            price_summary = "[GIÁ THAM KHẢO] Không có dữ liệu"

        # ===== document =====
        text = f"""
[SẢN PHẨM]
Tên: {product[1]}
Hãng: {product[2]}

[THÔNG SỐ]
- Chip: {product[3]}
- Pin: {product[4]} mAh

[PHIÊN BẢN]
{variant_text if variant_text else "Không có phiên bản"}

{price_summary}

[MÔ TẢ]
{product[5]}
        """

        documents.append(text.strip())

    conn.close()
    return documents


def load_policies():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT title, content
        FROM policies
    """)
    rows = cursor.fetchall()

    documents = []

    for title, content in rows:
        text = f"""
[CHÍNH SÁCH]
Tiêu đề: {title}

Nội dung:
{content}
        """
        documents.append(text.strip())

    conn.close()
    return documents


def load_all():
    return load_products() + load_policies()


if __name__ == "__main__":
    docs = load_all()

    print(f"\nTổng document: {len(docs)}\n")

    for i, doc in enumerate(docs, 1):
        print("=" * 80)
        print(f"DOCUMENT {i}")
        print("=" * 80)
        print(doc)
        print("\n")