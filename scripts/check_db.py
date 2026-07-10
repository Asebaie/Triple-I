import os
import sqlite3


def get_processed_files():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    db_path = os.path.join(project_root, "knowledge_base.db")

    if not os.path.exists(db_path):
        print(f"Файл базы данных не найден по пути: {db_path}")
        return

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT DISTINCT source_file FROM facts;")
        rows = cursor.fetchall()
        print("Обработанные файлы в базе данных:\n")
        for i, row in enumerate(rows, 1):
            print(f"{i}. {row[0]}")
    except sqlite3.OperationalError as e:
        print(f"Ошибка базы данных: {e}")
    finally:
        conn.close()


if __name__ == "__main__":
    get_processed_files()
