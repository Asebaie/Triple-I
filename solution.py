import csv
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

import rag

FAILED_ANSWER = "Не удалось сформировать ответ: сервис языковой модели не ответил. Повторите вопрос позже."
RETRY_FAILED_UNTIL = 12 * 60
QUESTION_DEADLINE = 15 * 60


def main() -> None:
    start = time.monotonic()
    data_dir = Path(rag.env("DATA_DIR", "data"))
    with open(data_dir / "questions.csv", encoding="utf-8-sig", newline="") as file:
        questions = list(csv.DictReader(file))
    index = rag.build_index(data_dir)
    print(f"Документов: {len(index.documents)}, фрагментов: {len(index.chunks)}, вопросов: {len(questions)}", flush=True)

    def work(row: dict) -> str:
        if time.monotonic() - start > QUESTION_DEADLINE:
            return FAILED_ANSWER
        try:
            return rag.answer(index, row["question"]).text or FAILED_ANSWER
        except (httpx.HTTPError, RuntimeError, KeyError, IndexError, TypeError, ValueError) as error:
            print(f"Вопрос {row['id']}: {error!r}", file=sys.stderr, flush=True)
            return FAILED_ANSWER

    with ThreadPoolExecutor(int(rag.env("MAX_WORKERS", "2"))) as pool:
        answers = list(pool.map(work, questions))
    answers = [
        work(row) if text == FAILED_ANSWER and time.monotonic() - start < RETRY_FAILED_UNTIL else text
        for row, text in zip(questions, answers)
    ]

    with open("submission.csv", "w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file, quoting=csv.QUOTE_ALL, lineterminator="\n")
        writer.writerow(["id", "answer"])
        writer.writerows((row["id"], " ".join(text.split())) for row, text in zip(questions, answers))
    print(f"submission.csv: {len(answers)} ответов", flush=True)


if __name__ == "__main__":
    main()
