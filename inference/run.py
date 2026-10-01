import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.utils import log, predict_pipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--items_path", type=str, default=None)
    parser.add_argument("--matches_path", type=str, default=None)
    parser.add_argument("--output_path", type=str, default="submit.csv")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--max_len", type=int, default=192)
    args, unknown = parser.parse_known_args()
    if unknown:
        log(f"Неизвестные аргументы проигнорированы: {unknown}")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    predict_pipeline(
        items_path=args.items_path,
        matches_path=args.matches_path,
        output_path=args.output_path,
        models_dir=os.path.join(base_dir, "models"),
        gbdt_dir=base_dir,
        max_len=args.max_len,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
