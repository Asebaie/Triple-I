import argparse
import gc
import glob
import json
import math
import os
import sys
import time

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import logging as hf_logging

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import core.matching as matching_module
from core.logging_utils import Heartbeat, StageTimer, setup_logger
from core.matching import CAT_TO_IDX, EMPTY_RECORD, build_records, ce_text_pair, pair_categories
from core.metrics import macro_pr_auc, macro_pr_auc_at_prevalence

MODEL_PRESETS = {
    "tiny": "cointegrated/rubert-tiny2",
    "small": "intfloat/multilingual-e5-small",
    "base": "intfloat/multilingual-e5-base",
    "rubase": "ai-forever/ruBert-base",
}


class PairDataset(Dataset):
    def __init__(self, texts_a, texts_b, targets):
        self.texts_a = texts_a
        self.texts_b = texts_b
        self.targets = targets

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        return index


class CategoryBatchSampler:
    def __init__(self, categories, batch_size, seed, drop_last=True):
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.seed = seed
        self.epoch = 0
        self.groups = [np.flatnonzero(categories == c) for c in np.unique(categories)]
        self.length = sum(len(g) // batch_size if drop_last else -(-len(g) // batch_size) for g in self.groups)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return self.length

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        batches = []
        for group in self.groups:
            shuffled = group.copy()
            rng.shuffle(shuffled)
            for start in range(0, len(shuffled), self.batch_size):
                chunk = shuffled[start : start + self.batch_size]
                if self.drop_last and len(chunk) < self.batch_size:
                    continue
                batches.append(chunk.tolist())
        rng.shuffle(batches)
        return iter(batches)


def ranking_loss(logits, labels):
    positive = logits[labels >= 0.5]
    negative = logits[labels < 0.5]
    if positive.numel() == 0 or negative.numel() == 0:
        return logits.sum() * 0.0
    return nn.functional.softplus(-(positive.unsqueeze(1) - negative.unsqueeze(0))).mean()


class Collator:
    def __init__(self, dataset, tokenizer, max_len):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __call__(self, indices):
        batch_a = [self.dataset.texts_a[i] for i in indices]
        batch_b = [self.dataset.texts_b[i] for i in indices]
        encoded = self.tokenizer(
            batch_a,
            batch_b,
            padding=True,
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )
        encoded["labels"] = torch.tensor([self.dataset.targets[i] for i in indices], dtype=torch.float32)
        return encoded


def union_find_groups(ids1, ids2):
    codes, _ = pd.factorize(np.concatenate([ids1, ids2]))
    left = codes[: len(ids1)]
    right = codes[len(ids1) :]
    parent = np.arange(codes.max() + 1, dtype=np.int64)

    def find(x):
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    for a, b in zip(left, right):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    roots = np.array([find(x) for x in left], dtype=np.int64)
    return roots


def assign_folds(groups, n_folds, seed):
    unique_groups = np.unique(groups)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_groups)
    mapping = {g: i % n_folds for i, g in enumerate(unique_groups)}
    return np.array([mapping[g] for g in groups], dtype=np.int32)


@torch.inference_mode()
def predict(model, loader, device, amp_dtype, logger, label, total):
    model.eval()
    scores = np.empty(total, dtype=np.float32)
    heartbeat = Heartbeat(logger, total, label, every_seconds=30.0)
    offset = 0
    for batch in loader:
        batch.pop("labels", None)
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        with torch.autocast("cuda", dtype=amp_dtype, enabled=device.type == "cuda"):
            logits = model(**batch).logits.squeeze(-1)
        size = logits.shape[0]
        scores[offset : offset + size] = logits.float().cpu().numpy()
        offset += size
        heartbeat.update(size)
    heartbeat.finish()
    return scores


def build_texts(records, ids1, ids2, logger):
    texts_a, texts_b = [], []
    heartbeat = Heartbeat(logger, len(ids1), "Сборка текстов пар", every_seconds=20.0)
    for i in range(len(ids1)):
        left, right = ce_text_pair(records.get(ids1[i], EMPTY_RECORD), records.get(ids2[i], EMPTY_RECORD))
        texts_a.append(left)
        texts_b.append(right)
        heartbeat.update(1)
    heartbeat.finish()
    return texts_a, texts_b


def save_checkpoint(path, model, optimizer, scheduler, scaler, epoch, step, best_score):
    tmp_path = path + ".tmp"
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict() if scaler is not None else None,
            "epoch": epoch,
            "step": step,
            "best_score": best_score,
        },
        tmp_path,
    )
    os.replace(tmp_path, path)


def train_one_fold(args, fold, data, logger, device, amp_dtype):
    ids1, ids2, targets, categories, folds, texts_a, texts_b = data
    train_mask = folds != fold
    val_mask = ~train_mask
    train_idx = np.flatnonzero(train_mask)
    val_idx = np.flatnonzero(val_mask)
    logger.info(f"Фолд {fold}: train={len(train_idx):,} пар, val={len(val_idx):,} пар")

    hf_logging.set_verbosity_error()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=1)
    model.to(device)

    if args.freeze_embeddings:
        for param in model.base_model.embeddings.parameters():
            param.requires_grad = False
        logger.info("Эмбеддинги заморожены")

    hidden = model.config.hidden_size
    lr = args.lr if args.lr is not None else (1e-4 if hidden <= 320 else 3e-5)
    logger.info(f"Модель {args.model} | hidden={hidden} | lr={lr}")

    encoder_params, head_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        (head_params if "classifier" in name or "pooler" in name else encoder_params).append(param)
    optimizer = torch.optim.AdamW(
        [{"params": encoder_params, "lr": lr}, {"params": head_params, "lr": lr * 10}],
        weight_decay=0.01,
    )

    train_dataset = PairDataset(texts_a, texts_b, targets)
    collate = Collator(train_dataset, tokenizer, args.max_len)
    batch_sampler = None
    if args.rank_loss_weight > 0:
        batch_sampler = CategoryBatchSampler(categories[train_idx], args.batch_size, args.seed)
        train_loader = DataLoader(
            train_idx,
            batch_sampler=batch_sampler,
            collate_fn=collate,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        logger.info(f"Батчи формируются внутри категорий, вес ранжирующего лосса {args.rank_loss_weight}")
    else:
        train_loader = DataLoader(
            train_idx,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collate,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True,
        )
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = int(total_steps * 0.06)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return max(0.02, 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0))))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype == torch.float16 and device.type == "cuda")
    criterion = nn.BCEWithLogitsLoss()

    checkpoint_path = os.path.join(args.out_dir, f"ckpt_fold{fold}.pt")
    start_epoch, start_step, best_score = 0, 0, -1.0
    if args.resume and os.path.exists(checkpoint_path):
        state = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        if scaler is not None and state.get("scaler") is not None:
            scaler.load_state_dict(state["scaler"])
        start_epoch = state["epoch"]
        start_step = state["step"]
        best_score = state["best_score"]
        logger.info(f"Возобновление с эпохи {start_epoch}, шага {start_step}, best={best_score:.5f}")

    model_dir = os.path.join(args.out_dir, f"ce_fold{fold}")
    global_step = start_epoch * steps_per_epoch + start_step

    for epoch in range(start_epoch, args.epochs):
        model.train()
        if batch_sampler is not None:
            batch_sampler.set_epoch(epoch)
        heartbeat = Heartbeat(logger, steps_per_epoch, f"Фолд {fold} эпоха {epoch + 1}/{args.epochs}", 30.0)
        running_loss, seen = 0.0, 0
        for step, batch in enumerate(train_loader):
            if epoch == start_epoch and step < start_step:
                heartbeat.update(1)
                continue
            labels = batch.pop("labels").to(device, non_blocking=True)
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            with torch.autocast("cuda", dtype=amp_dtype, enabled=device.type == "cuda"):
                logits = model(**batch).logits.squeeze(-1)
                loss = criterion(logits.float(), labels)
                if args.rank_loss_weight > 0:
                    loss = loss + args.rank_loss_weight * ranking_loss(logits.float(), labels)
            optimizer.zero_grad(set_to_none=True)
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            scheduler.step()
            global_step += 1
            running_loss += loss.item()
            seen += 1
            heartbeat.update(1)
            if seen % args.log_every == 0:
                logger.info(f"Фолд {fold} э{epoch + 1} шаг {step + 1}/{steps_per_epoch} loss={running_loss / seen:.4f}")
            if args.save_every and global_step % args.save_every == 0:
                save_checkpoint(checkpoint_path, model, optimizer, scheduler, scaler, epoch, step + 1, best_score)
                logger.info(f"Чекпоинт сохранён на шаге {global_step}")
        heartbeat.finish()
        start_step = 0

        eval_idx = val_idx
        if args.eval_subsample and len(val_idx) > args.eval_subsample and epoch < args.epochs - 1:
            eval_idx = np.random.default_rng(0).choice(val_idx, args.eval_subsample, replace=False)
        eval_loader = DataLoader(
            eval_idx,
            batch_size=args.eval_batch_size,
            shuffle=False,
            collate_fn=collate,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        scores = predict(model, eval_loader, device, amp_dtype, logger, f"Валидация фолд {fold}", len(eval_idx))
        score, _ = macro_pr_auc(categories[eval_idx], targets[eval_idx], scores)
        logger.info(f"Фолд {fold} эпоха {epoch + 1}: macro PR-AUC = {score:.5f}")
        if score > best_score:
            best_score = score
            model.save_pretrained(model_dir)
            tokenizer.save_pretrained(model_dir)
            with open(os.path.join(model_dir, "training_config.json"), "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "max_len": args.max_len,
                        "base_model": args.model,
                        "attr_keys": not args.no_attr_keys,
                        "categories": [c.strip() for c in args.categories.split(",") if c.strip()],
                    },
                    handle,
                    ensure_ascii=False,
                )
            logger.info(f"Новая лучшая модель сохранена в {model_dir}")
        save_checkpoint(checkpoint_path, model, optimizer, scheduler, scaler, epoch + 1, 0, best_score)

    logger.info(f"Фолд {fold}: лучший macro PR-AUC = {best_score:.5f}. Считаем OOF на полном валидационном фолде.")
    best_model = AutoModelForSequenceClassification.from_pretrained(model_dir, num_labels=1).to(device)
    full_val_loader = DataLoader(
        val_idx,
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=collate,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    oof_scores = predict(best_model, full_val_loader, device, amp_dtype, logger, f"OOF фолд {fold}", len(val_idx))
    score, per_category = macro_pr_auc(categories[val_idx], targets[val_idx], oof_scores)
    low, spread = macro_pr_auc_at_prevalence(categories[val_idx], targets[val_idx], oof_scores)
    logger.info(f"Фолд {fold} при тестовой доле позитивов 5%: macro PR-AUC = {low:.5f} (разброс {spread:.4f})")
    for cat_idx, cat_score in sorted(per_category.items()):
        logger.info(f"    категория {cat_idx}: PR-AUC = {cat_score:.4f}")
    logger.info(f"Фолд {fold} финальный OOF macro PR-AUC = {score:.5f}")

    oof_path = os.path.join(args.out_dir, f"oof_fold{fold}.parquet")
    pd.DataFrame(
        {
            "id1": ids1[val_idx],
            "id2": ids2[val_idx],
            "target": targets[val_idx],
            "category": categories[val_idx],
            "ce_score": oof_scores,
            "fold": fold,
        }
    ).to_parquet(oof_path, index=False)
    logger.info(f"OOF сохранены: {oof_path}")

    del model, best_model, optimizer, scheduler
    gc.collect()
    torch.cuda.empty_cache()
    return score


def pretrain_on_llm(args, logger, device, amp_dtype):
    logger.info(f"Предобучение на LLM-разметке: {args.llm_parquet}")
    paths = sorted(glob.glob(args.llm_parquet)) or [args.llm_parquet]
    logger.info(f"Файлов с LLM-разметкой: {len(paths)}")
    frame = pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)
    if args.categories and "category" in frame.columns:
        wanted = [c.strip() for c in args.categories.split(",") if c.strip()]
        missing = [c for c in wanted if c not in CAT_TO_IDX]
        if missing:
            raise SystemExit(f"Неизвестные категории: {missing}")
        keep = frame["category"].isin([CAT_TO_IDX[c] for c in wanted])
        logger.info(f"Отбор LLM-пар по категориям {wanted}: {int(keep.sum()):,} из {len(frame):,}")
        frame = frame[keep].reset_index(drop=True)
    if args.llm_max_pairs and len(frame) > args.llm_max_pairs:
        frame = frame.sample(args.llm_max_pairs, random_state=42).reset_index(drop=True)
    logger.info(f"LLM пар для предобучения: {len(frame):,}")

    texts_a = frame["text1"].tolist()
    texts_b = frame["text2"].tolist()
    targets = frame["target"].to_numpy(dtype=np.float32)
    del frame
    gc.collect()

    hf_logging.set_verbosity_error()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=1).to(device)
    if args.freeze_embeddings:
        for param in model.base_model.embeddings.parameters():
            param.requires_grad = False

    hidden = model.config.hidden_size
    lr = args.lr if args.lr is not None else (1e-4 if hidden <= 320 else 3e-5)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr * 2, weight_decay=0.01)

    dataset = PairDataset(texts_a, texts_b, targets)
    collate = Collator(dataset, tokenizer, args.max_len)
    loader = DataLoader(
        np.arange(len(targets)),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    total_steps = len(loader) * args.llm_epochs
    warmup_steps = int(total_steps * 0.03)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return max(0.05, 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0))))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype == torch.float16 and device.type == "cuda")
    criterion = nn.BCEWithLogitsLoss()

    out_dir = os.path.join(args.out_dir, "ce_pretrained")
    checkpoint_path = os.path.join(args.out_dir, "ckpt_pretrain.pt")
    start_epoch, start_step = 0, 0
    if args.resume and os.path.exists(checkpoint_path):
        state = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        if state.get("scaler") is not None:
            scaler.load_state_dict(state["scaler"])
        start_epoch, start_step = state["epoch"], state["step"]
        logger.info(f"Возобновление предобучения с эпохи {start_epoch}, шага {start_step}")

    steps_per_epoch = len(loader)
    for epoch in range(start_epoch, args.llm_epochs):
        model.train()
        heartbeat = Heartbeat(logger, steps_per_epoch, f"LLM предобучение эпоха {epoch + 1}/{args.llm_epochs}", 30.0)
        running_loss, seen = 0.0, 0
        for step, batch in enumerate(loader):
            if epoch == start_epoch and step < start_step:
                heartbeat.update(1)
                continue
            labels = batch.pop("labels").to(device, non_blocking=True)
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            with torch.autocast("cuda", dtype=amp_dtype, enabled=device.type == "cuda"):
                logits = model(**batch).logits.squeeze(-1)
                loss = criterion(logits.float(), labels)
            optimizer.zero_grad(set_to_none=True)
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            scheduler.step()
            running_loss += loss.item()
            seen += 1
            heartbeat.update(1)
            if seen % args.log_every == 0:
                logger.info(f"LLM э{epoch + 1} шаг {step + 1}/{steps_per_epoch} loss={running_loss / seen:.4f}")
            if args.save_every and seen % args.save_every == 0:
                save_checkpoint(checkpoint_path, model, optimizer, scheduler, scaler, epoch, step + 1, -1.0)
                model.save_pretrained(out_dir)
                tokenizer.save_pretrained(out_dir)
                logger.info("Промежуточное сохранение предобученной модели")
        heartbeat.finish()
        start_step = 0
        model.save_pretrained(out_dir)
        tokenizer.save_pretrained(out_dir)
        save_checkpoint(checkpoint_path, model, optimizer, scheduler, scaler, epoch + 1, 0, -1.0)

    logger.info(f"Предобученная модель сохранена: {out_dir}")
    del model, optimizer, scheduler
    gc.collect()
    torch.cuda.empty_cache()
    return out_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--items", default="data/items_human.parquet")
    parser.add_argument("--matches", default="data/matches.parquet")
    parser.add_argument("--out_dir", default="artifacts")
    parser.add_argument("--log_dir", default="logs")
    parser.add_argument("--model", default="tiny")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--train_folds", type=int, default=2)
    parser.add_argument("--only_fold", type=int, default=-1)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--eval_batch_size", type=int, default=256)
    parser.add_argument("--max_len", type=int, default=192)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_every", type=int, default=200)
    parser.add_argument("--save_every", type=int, default=2000)
    parser.add_argument("--eval_subsample", type=int, default=60000)
    parser.add_argument("--max_pairs", type=int, default=0)
    parser.add_argument("--freeze_embeddings", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--llm_parquet", default="")
    parser.add_argument("--llm_epochs", type=int, default=1)
    parser.add_argument("--llm_max_pairs", type=int, default=1500000)
    parser.add_argument("--pretrain_only", action="store_true")
    parser.add_argument("--rank_loss_weight", type=float, default=0.0)
    parser.add_argument("--no_attr_keys", action="store_true")
    parser.add_argument("--categories", default="")
    args = parser.parse_args()

    args.model = MODEL_PRESETS.get(args.model, args.model)
    os.makedirs(args.out_dir, exist_ok=True)
    logger, _ = setup_logger("train_ce", args.log_dir)
    logger.info(f"Аргументы: {vars(args)}")
    if args.no_attr_keys:
        matching_module.USE_ATTR_KEYS = False
        logger.info("Ключи атрибутов исключены из текста, остаются только значения")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        major, minor = torch.cuda.get_device_capability(0)
        logger.info(f"GPU: {torch.cuda.get_device_name(0)} | compute capability {major}.{minor}")
        try:
            probe = torch.zeros(64, 64, device=device)
            (probe @ probe).sum().item()
            torch.cuda.synchronize()
        except Exception as error:
            logger.error(f"GPU не работает с этой сборкой PyTorch: {error!r}")
            logger.error(f"Поддерживаемые архитектуры: {torch.cuda.get_arch_list()}")
            logger.error("Смени ускоритель на T4 (Kaggle: Settings -> Accelerator -> GPU T4 x2)")
            raise SystemExit(1)
        amp_dtype = torch.bfloat16 if major >= 8 else torch.float16
    else:
        logger.warning("CUDA недоступна, обучение пойдёт на CPU и будет очень медленным")
        amp_dtype = torch.float32
    logger.info(f"amp_dtype={amp_dtype}")

    if args.llm_parquet:
        with StageTimer(logger, "Предобучение на LLM-разметке"):
            args.model = pretrain_on_llm(args, logger, device, amp_dtype)
        if args.pretrain_only:
            logger.info(f"Предобучение завершено, модель в {args.model}. Дообучение по фолдам пропущено")
            return

    with StageTimer(logger, "Загрузка товаров и построение записей"):
        items_df = pd.read_parquet(args.items)
        logger.info(f"Товаров: {len(items_df):,}")
        records = build_records(items_df, log_every=200000, logger=logger)
        del items_df
        gc.collect()

    with StageTimer(logger, "Загрузка пар и разбиение на фолды"):
        matches = pd.read_parquet(args.matches)
        if args.max_pairs and len(matches) > args.max_pairs:
            matches = matches.sample(args.max_pairs, random_state=args.seed).reset_index(drop=True)
        ids1 = matches["id1"].to_numpy()
        ids2 = matches["id2"].to_numpy()
        targets = matches["target"].to_numpy(dtype=np.float32)
        del matches
        gc.collect()
        groups = union_find_groups(ids1, ids2)
        folds = assign_folds(groups, max(args.folds, 2), args.seed)
        categories = pair_categories(ids1, records)
        if args.categories:
            wanted = [c.strip() for c in args.categories.split(",") if c.strip()]
            missing = [c for c in wanted if c not in CAT_TO_IDX]
            if missing:
                raise SystemExit(f"Неизвестные категории: {missing}")
            keep = np.isin(categories, [CAT_TO_IDX[c] for c in wanted])
            logger.info(f"Обучение только на категориях {wanted}: остаётся {keep.sum():,} пар из {len(keep):,}")
            ids1, ids2, targets = ids1[keep], ids2[keep], targets[keep]
            categories, folds = categories[keep], folds[keep]
        logger.info(f"Пар: {len(ids1):,} | групп: {len(np.unique(groups)):,} | фолдов: {args.folds}")

    with StageTimer(logger, "Сборка текстовых пар"):
        texts_a, texts_b = build_texts(records, ids1, ids2, logger)

    data = (ids1, ids2, targets, categories, folds, texts_a, texts_b)
    fold_list = [args.only_fold] if args.only_fold >= 0 else list(range(min(args.train_folds, args.folds)))
    scores = []
    for fold in fold_list:
        with StageTimer(logger, f"Обучение фолда {fold}"):
            scores.append(train_one_fold(args, fold, data, logger, device, amp_dtype))

    logger.info(f"Средний OOF macro PR-AUC по обученным фолдам: {np.mean(scores):.5f}")
    logger.info("Готово. Дальше запускай train_gbdt.py")


if __name__ == "__main__":
    main()
