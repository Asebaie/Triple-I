from __future__ import annotations

import base64
import hashlib
import json
import math
import operator
import sys
import time
from array import array
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, wait
from typing import TYPE_CHECKING

import runtime
from config import API_BASE, EMB_MODEL, env, log, remaining
from textproc import expand_query, tokenize
from kb import kb_hash
from dialog import search_query
from llm import APIError, embed

if TYPE_CHECKING:
    from dialog import Prepared
    from kb import KB


class Index:
    def __init__(self, kb: KB) -> None:
        self.kb = kb
        self.k1, self.b = 1.5, 0.75
        self.postings: dict[str, list[tuple[int, int]]] = {}
        self.lens: list[int] = []
        for c in kb.chunks:
            toks = tokenize(c.search)
            self.lens.append(len(toks))
            tf: dict[str, int] = {}
            for t in toks:
                tf[t] = tf.get(t, 0) + 1
            for t, f in tf.items():
                self.postings.setdefault(t, []).append((c.idx, f))
        n = max(1, len(kb.chunks))
        self.avgdl = (sum(self.lens) / n) if self.lens else 1.0
        self.idf = {t: math.log(1 + (n - len(p) + 0.5) / (len(p) + 0.5)) for t, p in self.postings.items()}
        self.vecs: list[list[float]] | None = None
        self.prebuilt: dict | None = None

    def bm25(self, query: str) -> list[tuple[int, float]]:
        scores: dict[int, float] = {}
        for t in set(tokenize(query)):
            idf = self.idf.get(t)
            if idf is None:
                continue
            for ci, f in self.postings[t]:
                dl = self.lens[ci]
                s = idf * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
                scores[ci] = scores.get(ci, 0.0) + s
        return sorted(scores.items(), key=lambda x: -x[1])

    def build_dense(self, cap: float = 60.0) -> None:
        texts = [c.search[:1200] for c in self.kb.chunks]
        if not texts or not API_BASE:
            return
        batches = [list(range(i, min(i + 16, len(texts)))) for i in range(0, len(texts), 16)]
        vecs: list[list[float] | None] = [None] * len(texts)

        def run(batch: list[int]) -> None:
            res = embed([texts[i] for i in batch])
            for i, v in zip(batch, res):
                vecs[i] = v

        t0 = time.monotonic()
        pool = ThreadPoolExecutor(max_workers=4)
        futs = [pool.submit(run, b) for b in batches]
        done, not_done = wait(futs, timeout=max(5.0, min(cap, remaining() - 120)))
        pool.shutdown(wait=False, cancel_futures=True)
        errors = [f.exception() for f in done if f.exception() is not None]
        if not_done or errors or any(v is None for v in vecs):
            reason = f"не уложились в {cap:.0f} c" if not_done else (f"{errors[0]}" if errors else "не все векторы")
            log(f"Эмбеддинги недоступны ({reason}), работаем на BM25")
            self.vecs = None
            return
        self.vecs = vecs
        runtime.FLAGS.dense = True
        log(f"Эмбеддинги базы готовы: {len(texts)} фрагментов за {time.monotonic() - t0:.1f} c")

    def fingerprint(self) -> dict:
        texts = "\n\x00".join(c.search[:1200] for c in self.kb.chunks)
        return {
            "kb_sha256": kb_hash(self.kb),
            "texts_sha256": hashlib.sha256(texts.encode("utf-8")).hexdigest(),
            "n": len(self.kb.chunks),
        }

    def save_vectors(self, path: Path, emb_model: str) -> dict:
        assert self.vecs is not None, "нет векторов"
        dim = len(self.vecs[0])
        arr = array("f", [x for v in self.vecs for x in v])
        if arr.itemsize != 4:
            raise RuntimeError("float32 недоступен")
        data = {
            "format": 1,
            **self.fingerprint(),
            "emb_model": emb_model,
            "dim": dim,
            "dtype": "float32",
            "byteorder": "little",
            "vectors": base64.b64encode(arr.tobytes() if sys.byteorder == "little" else _swap(arr)).decode("ascii"),
        }
        path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        return {k: v for k, v in data.items() if k != "vectors"}

    def load_prebuilt(self, path: Path | None = None, sample: int = 8, min_cos: float = 0.99) -> dict:
        try:
            self.prebuilt = self._load_prebuilt(path, sample, min_cos)
        except Exception as exc:
            self.vecs = None
            self.prebuilt = {"found": True, "used": False, "reason": f"ошибка {type(exc).__name__}"}
            log(f"Готовый индекс не использован: {type(exc).__name__}")
        return self.prebuilt

    def _load_prebuilt(self, path: Path | None, sample: int, min_cos: float) -> dict:
        path = path or INDEX_VECTORS
        info: dict = {"found": path.is_file(), "used": False}
        if not info["found"] or not self.kb.chunks or not API_BASE:
            return info
        t0 = time.monotonic()
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            fp = self.fingerprint()
            for key in ("kb_sha256", "texts_sha256", "n"):
                if data.get(key) != fp[key]:
                    info["reason"] = f"не совпал {key}"
                    return info
            if str(data.get("emb_model", "")).lower().split("/")[-1] != EMB_MODEL.lower().split("/")[-1]:
                info["reason"] = "другая модель эмбеддингов"
                return info
            arr = array("f")
            arr.frombytes(base64.b64decode(data["vectors"]))
            if sys.byteorder != data.get("byteorder", "little"):
                arr.byteswap()
            dim = int(data["dim"])
            if len(arr) != dim * fp["n"]:
                info["reason"] = "размер массива"
                return info
            vecs = [list(arr[i * dim : (i + 1) * dim]) for i in range(fp["n"])]
            idx = sorted({round(k * (fp["n"] - 1) / max(1, sample - 1)) for k in range(sample)})
            try:
                fresh = embed([self.kb.chunks[i].search[:1200] for i in idx])
            except APIError as exc:
                info["reason"] = f"контроль не выполнен: {exc}"[:120]
                return info
            if len(fresh[0]) != dim:
                info["reason"] = f"размерность API {len(fresh[0])} ≠ {dim}"
                return info
            cos = [
                sum(map(operator.mul, fresh[j], vecs[i])) / (math.sqrt(sum(x * x for x in vecs[i])) or 1.0)
                for j, i in enumerate(idx)
            ]
            info["min_cos"] = round(min(cos), 5)
            if min(cos) < min_cos:
                info["reason"] = "векторы API отличаются от сохранённых"
                return info
        except (OSError, ValueError, KeyError, TypeError) as exc:
            info["reason"] = f"{type(exc).__name__}"
            return info
        self.vecs = vecs
        runtime.FLAGS.dense = True
        info.update(used=True, sec=round(time.monotonic() - t0, 2), dim=dim)
        log(f"Готовый индекс загружен: {fp['n']} векторов за {info['sec']} c (контроль: косинус ≥ {info['min_cos']})")
        return info

    def dense(self, qvec: list[float] | None) -> list[tuple[int, float]]:
        if self.vecs is None or qvec is None:
            return []
        mul = operator.mul
        sc = [(i, sum(map(mul, qvec, v))) for i, v in enumerate(self.vecs)]
        return sorted(sc, key=lambda x: -x[1])

    def hybrid(self, query: str, qvec: list[float] | None, kinds: set[str]) -> list[int]:
        return self.hybrid_multi([(query, qvec)], kinds)

    def hybrid_multi(self, pairs: list[tuple[str, list[float] | None]], kinds: set[str]) -> list[int]:
        allowed = {c.idx for c in self.kb.chunks if self.kb.docs[c.doc].kind in kinds}
        lists = []
        for query, qvec in pairs:
            lists.append([i for i, _ in self.bm25(expand_query(query)) if i in allowed][:80])
            d = [i for i, _ in self.dense(qvec) if i in allowed][:80]
            if d:
                lists.append(d)
        rrf: dict[int, float] = {}
        for lst in lists:
            for r, i in enumerate(lst):
                rrf[i] = rrf.get(i, 0.0) + 1.0 / (60 + r)
        return [i for i, _ in sorted(rrf.items(), key=lambda x: -x[1])]


def embed_questions(preps: list[Prepared], cap: float = 25.0) -> None:
    qtexts = [search_query(p)[:2000] for p in preps]
    batches = [list(range(i, min(i + 16, len(qtexts)))) for i in range(0, len(qtexts), 16)]
    out: list[list[float] | None] = [None] * len(qtexts)

    def run(batch: list[int]) -> None:
        for j, v in zip(batch, embed([qtexts[i] for i in batch])):
            out[j] = v

    pool = ThreadPoolExecutor(max_workers=4)
    futs = [pool.submit(run, b) for b in batches]
    done, not_done = wait(futs, timeout=max(3.0, min(cap, remaining() - 120)))
    pool.shutdown(wait=False, cancel_futures=True)
    if not_done or any(f.exception() for f in done) or any(v is None for v in out):
        log("Эмбеддинги вопросов не получены, поиск без них")
        return
    for p, v in zip(preps, out):
        p.qvec = v


INDEX_VECTORS = Path(env("SOLUTION_INDEX_FILE") or Path(__file__).resolve().parent / "index_vectors.json")


def _swap(arr: array) -> bytes:
    a = array(arr.typecode, arr)
    a.byteswap()
    return a.tobytes()
