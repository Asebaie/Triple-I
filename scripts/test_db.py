import sqlite3


def check_data():
    conn = sqlite3.connect("knowledge_base.db")
    cursor = conn.cursor()
    cursor.execute("SELECT material, process, outcome FROM facts LIMIT 5;")
    rows = cursor.fetchall()
    for row in rows:
        print(f"Материал: {row[0]} | Процесс: {row[1]} | Результат: {row[2]}\n")
    conn.close()


if __name__ == "__main__":
    check_data()
