CREATE TABLE products (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    brand TEXT,
    price INTEGER,
    ram TEXT,
    storage TEXT,
    chip TEXT,
    battery TEXT,

    description TEXT,   -- mô tả ngắn để embed
    embedding TEXT,     -- vector (JSON string)

    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE policies (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT,
    content TEXT,

    embedding TEXT,

    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE chat_sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE chat_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER,

    role TEXT NOT NULL,
    message TEXT NOT NULL,

    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (session_id) REFERENCES chat_sessions(id)
);