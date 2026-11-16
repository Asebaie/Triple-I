from __future__ import annotations

import csv
import io
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import runtime
from config import API_BASE, FINAL_PASS, LLM_KEY, MAX_WORKERS, OUTPUT_FILE, PREBUILT_INDEX, log, remaining
from kb import KB, build_kb, decode_bytes, find_questions
from runtime import choose_level, question_cost
from index import Index, embed_questions
from answering import answer_one
from dialog import Prepared, parse_dialog
from retrieval import fallback_answer
from probe import log_env, platform_probe, run_stats, wait_for_llm


def load_questions(path: Path) -> list[tuple[str, str]]:
    raw = path.read_bytes()
    text = decode_bytes(raw)
    reader = csv.DictReader(io.StringIO(text))
    fields = [f.strip().lower() for f in (reader.fieldnames or [])]
    if "id" not in fields or "question" not in fields:
        raise ValueError("В questions.csv нужны колонки id и question")
    reader.fieldnames = fields
    rows, seen = [], set()
    for r in reader:
        qid = str(r.get("id") or "").strip()
        if not qid or qid in seen:
            continue
        seen.add(qid)
        rows.append((qid, str(r.get("question") or "").strip()))
    if not rows:
        raise ValueError("В questions.csv нет вопросов")
    return rows


WRITE_LOCK = threading.Lock()


def write_submission(questions: list[tuple[str, str]], answers: list[str]) -> None:
    with WRITE_LOCK:
        tmp = OUTPUT_FILE + ".tmp"
        with open(tmp, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f, quoting=csv.QUOTE_ALL, lineterminator="\n")
            w.writerow(["id", "answer"])
            for (qid, _), a in zip(questions, answers):
                a = (a or "").strip() or "Не удалось сформировать ответ по базе знаний."
                w.writerow([qid, a[:9500]])
        os.replace(tmp, OUTPUT_FILE)


def main() -> None:
    log_env()
    questions = load_questions(find_questions())
    n = len(questions)
    log(f"Вопросов: {n}")
    kb, index, answers = load_base(n, questions)
    write_submission(questions, answers)
    log("Записан предварительный submission.csv с запасными ответами")

    if not API_BASE or not LLM_KEY:
        log("Нет доступа к API — остаются запасные ответы")
        platform_probe(kb, questions, index, None, None)
        run_stats(n, n, 0)
        return

    preps = [Prepared(*parse_dialog(q)) for _, q in questions]
    llm_ready = wait_for_llm()
    dense_sec = build_dense(kb, index, preps)
    platform_probe(kb, questions, index, llm_ready, dense_sec)

    levels: list[int | None] = [None] * n
    work = make_worker(kb, index, questions, preps, levels)
    failed = first_pass(work, questions, answers)
    rounds = retry_rounds(work, questions, answers, failed)
    if FINAL_PASS:
        final_pass(kb, index, questions, preps, answers, levels, failed)
    runtime.FLAGS.stats["fallback"] = len(failed)
    write_submission(questions, answers)
    run_stats(n, len(failed), rounds, levels)
    log(
        f"Готово: {runtime.FLAGS.stats}, tiny={runtime.FLAGS.tiny}, chat≈{runtime.TIMING.get('chat_full', 0):.1f} c (полный) / "
        f"{runtime.TIMING.get('chat_compact', 0):.1f} c (компактный) / {runtime.TIMING.get('chat_min', 0):.1f} c (минимальный), "
        f"dense={runtime.FLAGS.dense}, compact={runtime.FLAGS.compact}, minimal={runtime.FLAGS.minimal}, rerank_off={runtime.FLAGS.rerank_off}"
    )


def load_base(n: int, questions: list[tuple[str, str]]) -> tuple[KB, Index, list[str]]:
    try:
        kb = build_kb()
        index = Index(kb)
        answers = [fallback_answer(kb, index, q) for _, q in questions]
    except Exception as exc:
        log(f"База знаний не загружена: {type(exc).__name__}: {exc}")
        kb = KB([], [])
        index = Index(kb)
        answers = ["В базе знаний не найдено документов по этому вопросу, поэтому ответить по документам нельзя."] * n
    return kb, index, answers


def build_dense(kb: KB, index: Index, preps: list) -> float | None:
    if not kb.chunks:
        return None
    t_dense = time.monotonic()
    if not (PREBUILT_INDEX and index.load_prebuilt().get("used")):
        index.build_dense(cap=60.0)
    dense_sec = round(time.monotonic() - t_dense, 1)
    if index.vecs is not None:
        embed_questions(preps)
    return dense_sec


def make_worker(kb: KB, index: Index, questions: list[tuple[str, str]], preps: list, levels: list):
    def work(i: int) -> tuple[int, str | None]:
        try:
            level = choose_level()
            if level == 0:
                runtime.FLAGS.inc("skipped")
                return i, None
            info: dict = {}
            t_q = time.monotonic()
            ans = answer_one(kb, index, questions[i][1], preps[i], level, info=info)
            levels[i] = info.get("level", level)
            runtime.TIMING.add(f"q{levels[i]}", time.monotonic() - t_q)
            runtime.FLAGS.inc("llm_ok")
            return i, ans
        except Exception as exc:
            log(f"Вопрос {questions[i][0]}: запасной ответ ({type(exc).__name__}: {exc})")
            return i, None
        finally:
            with runtime.PROG.lock:
                runtime.PROG.finished += 1

    return work


def first_pass(work, questions: list[tuple[str, str]], answers: list[str]) -> list[int]:
    n = len(questions)
    runtime.PROG.n = n
    failed: list[int] = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        for k, fut in enumerate(as_completed([pool.submit(work, i) for i in range(n)]), 1):
            i, ans = fut.result()
            if ans:
                answers[i] = ans
            else:
                failed.append(i)
            if k % 10 == 0 or k == n:
                write_submission(questions, answers)
                log(
                    f"Обработано {k}/{n}; уровень сейчас {choose_level()}, "
                    f"chat≈{runtime.TIMING.get('chat_full', 0):.1f}/{runtime.TIMING.get('chat_compact', 0):.1f} c"
                )
    return failed


def retry_rounds(work, questions: list[tuple[str, str]], answers: list[str], failed: list[int]) -> int:
    rounds = 0
    while failed and remaining() > 60 and rounds < 20:
        rounds += 1
        before = len(failed)
        log(f"Проход {rounds + 1}: повторяем {before} вопросов")
        runtime.PROG.n, runtime.PROG.finished = len(failed), 0
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            for fut in as_completed([pool.submit(work, i) for i in sorted(failed)]):
                i, ans = fut.result()
                if ans:
                    answers[i] = ans
                    failed.remove(i)
        write_submission(questions, answers)
        if len(failed) == before:
            time.sleep(min(20.0, max(0.0, remaining() - 60)))
    return rounds


def final_pass(kb, index, questions, preps, answers: list, levels: list, failed: list) -> None:
    f = runtime.FLAGS
    if f.tiny:
        return
    target = 1 if f.minimal else (2 if f.compact else 4)
    rank = {None: 0, 0: 1, 1: 2, 2: 3, 3: 4}
    cands = sorted(
        (i for i in range(len(questions)) if levels[i] is None or levels[i] < target),
        key=lambda i: (rank.get(levels[i], 9), i),
    )
    if not cands:
        return
    log(f"Финальный проход: {len(cands)} вопросов на уровень {target}, осталось {remaining():.0f} c")

    def upgrade(i: int) -> tuple[int, str | None, int | None]:
        info: dict = {}
        t_q = time.monotonic()
        try:
            ans = answer_one(kb, index, questions[i][1], preps[i], target, info=info)
        except Exception as exc:
            log(f"Вопрос {questions[i][0]}: переответ не удался ({type(exc).__name__})")
            return i, None, None
        lv = info.get("level", target)
        runtime.TIMING.add(f"q{lv}", time.monotonic() - t_q)
        return i, ans, lv

    upgraded, pos = 0, 0
    while pos < len(cands) and remaining() - 15 > question_cost(target) * 1.2:
        batch = cands[pos : pos + MAX_WORKERS]
        pos += len(batch)
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            for i, ans, lv in pool.map(upgrade, batch):
                if ans and (levels[i] is None or (lv is not None and lv > levels[i])):
                    answers[i], levels[i] = ans, lv
                    upgraded += 1
                    if i in failed:
                        failed.remove(i)
        write_submission(questions, answers)
    f.stats["final_pass_upgraded"] = f.stats.get("final_pass_upgraded", 0) + upgraded
    log(f"Финальный проход: повышено {upgraded}, осталось {remaining():.0f} c")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        log(f"Критическая ошибка: {type(exc).__name__}: {exc}")
        raise
