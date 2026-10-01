import argparse
import glob
import os
import shutil
import subprocess
import sys
import zipfile

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from core.logging_utils import setup_logger

KNOWN_STALE = [
    "gbdt_sklearn.joblib",
    "src/gbdt_predict.py",
    "models/paraphrase-multilingual",
    "models/rubert-tiny2",
    "models/cross-encoder-ms-marco-MiniLM-L12-v2",
    "classifier_resnet_v5.pt",
    "classifier_resnet.pt",
    "baseline_logreg_l12.joblib",
    "src/__pycache__",
    "__pycache__",
]


def convert_to_fp16(path):
    import torch
    from safetensors.torch import load_file, save_file

    tensors = load_file(path)
    converted = {k: (v.half() if v.dtype == torch.float32 else v) for k, v in tensors.items()}
    save_file(converted, path, metadata={"format": "pt"})


def directory_size(path):
    total = 0
    for root, _, files in os.walk(path):
        for name in files:
            total += os.path.getsize(os.path.join(root, name))
    return total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", default="artifacts")
    parser.add_argument("--extra_ce", nargs="*", default=[])
    parser.add_argument("--no_stack", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--only_first_fold", action="store_true")
    parser.add_argument("--submission", default="submission")
    parser.add_argument("--zip_path", default="submission.zip")
    parser.add_argument("--log_dir", default="logs")
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    logger, _ = setup_logger("make_submission", args.log_dir)
    submission_dir = os.path.join(BASE_DIR, args.submission)
    artifacts_dir = os.path.join(BASE_DIR, args.artifacts)
    models_dir = os.path.join(submission_dir, "models")
    os.makedirs(models_dir, exist_ok=True)

    for module in ("matching.py", "lgbm_numpy.py", "attr_priority.py"):
        shutil.copy2(os.path.join(BASE_DIR, "ecup", module), os.path.join(submission_dir, "src", module))
    logger.info("Модули ecup синхронизированы в submission/src")

    init_path = os.path.join(submission_dir, "src", "__init__.py")
    if not os.path.exists(init_path):
        open(init_path, "w").close()

    fold_dirs = sorted(glob.glob(os.path.join(artifacts_dir, "ce_fold*")))
    if args.only_first_fold:
        fold_dirs = fold_dirs[:1]
    if not fold_dirs:
        raise SystemExit(f"Не найдены обученные модели ce_fold* в {artifacts_dir}")

    sources = [(d, os.path.basename(d)) for d in fold_dirs]
    for spec in args.extra_ce:
        source_dir, _, tag = spec.partition(":")
        if not tag:
            tag = os.path.basename(os.path.dirname(source_dir.rstrip("/")))
        if os.path.exists(os.path.join(source_dir, "config.json")):
            sources.append((source_dir, f"ce_{tag}"))
        else:
            for extra in sorted(glob.glob(os.path.join(source_dir, "ce_fold*"))):
                sources.append((extra, f"ce_{tag}{os.path.basename(extra)[-1]}"))
    fold_dirs = [d for d, _ in sources]

    expected = {name for _, name in sources}
    for existing in sorted(glob.glob(os.path.join(models_dir, "ce*"))):
        if os.path.basename(existing) not in expected:
            shutil.rmtree(existing)
            logger.warning(f"Удалена модель от прошлой сборки: {os.path.basename(existing)}")
    for fold_dir, target_name in sources:
        target = os.path.join(models_dir, target_name)
        if os.path.exists(target):
            shutil.rmtree(target)
        shutil.copytree(fold_dir, target, ignore=shutil.ignore_patterns("optimizer*", "*.msgpack", "*.h5", "*.bin"))
        if args.fp16:
            before = directory_size(target)
            convert_to_fp16(os.path.join(target, "model.safetensors"))
            after = directory_size(target)
            logger.info(f"Скопирована модель: {target} ({before / 1e6:.1f} -> {after / 1e6:.1f} МБ, fp16)")
        else:
            logger.info(f"Скопирована модель: {target} ({directory_size(target) / 1e6:.1f} МБ)")

    copied_gbdt = False
    wanted = ("gbdt_features_lgb.txt",) if args.no_stack else ("gbdt_lgb.txt", "gbdt_features_lgb.txt")
    if args.no_stack:
        stale_stack = os.path.join(submission_dir, "gbdt_lgb.txt")
        if os.path.exists(stale_stack):
            os.remove(stale_stack)
            logger.info("Стек отключён, gbdt_lgb.txt удалён из сабмита")
    for name in wanted:
        source = os.path.join(artifacts_dir, name)
        if os.path.exists(source):
            shutil.copy2(source, os.path.join(submission_dir, name))
            logger.info(f"Скопирован {name}")
            copied_gbdt = True
    if not copied_gbdt:
        logger.warning("GBDT не найден, сабмит будет работать на чистом кросс-энкодере")

    stale = [p for p in KNOWN_STALE if os.path.exists(os.path.join(submission_dir, p))]
    if stale:
        if args.clean:
            for path in stale:
                full = os.path.join(submission_dir, path)
                shutil.rmtree(full) if os.path.isdir(full) else os.remove(full)
                logger.info(f"Удалён устаревший артефакт: {path}")
        else:
            logger.warning(f"В сабмите остались устаревшие файлы: {stale}. Запусти с --clean чтобы удалить")

    for required in (
        "run.py",
        "metadata.json",
        "src/utils.py",
        "src/matching.py",
        "src/lgbm_numpy.py",
        "src/attr_priority.py",
    ):
        if not os.path.exists(os.path.join(submission_dir, required)):
            raise SystemExit(f"Отсутствует обязательный файл: {required}")

    zip_path = os.path.join(BASE_DIR, args.zip_path)
    if os.path.exists(zip_path):
        os.remove(zip_path)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for root, dirs, files in os.walk(submission_dir):
            dirs[:] = [d for d in dirs if d != "__pycache__"]
            for name in files:
                if name == ".DS_Store":
                    continue
                full = os.path.join(root, name)
                archive.write(full, os.path.relpath(full, submission_dir))

    size_mb = os.path.getsize(zip_path) / 1e6
    logger.info(f"Архив собран: {zip_path} ({size_mb:.1f} МБ)")
    if size_mb > 5000:
        logger.error("Архив больше лимита в 5 ГБ")
    logger.info("Содержимое архива:")
    for name in sorted(zipfile.ZipFile(zip_path).namelist())[:60]:
        logger.info(f"    {name}")


if __name__ == "__main__":
    main()
