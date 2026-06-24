from __future__ import annotations

import argparse
import json
import sys

from .compare import compare
from .config import load_config
from .doctor import doctor
from .evaluate import evaluate, link_text
from .index import build_index
from .mining import mine_negatives
from .prepare import prepare
from .train import train


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser()
    sub = root.add_subparsers(dest="command", required=True)
    for command in (
        "doctor", "prepare", "mine-negatives", "train", "compare",
    ):
        item = sub.add_parser(command)
        item.add_argument("--config", required=True)
    item = sub.add_parser("build-index")
    item.add_argument("--config", required=True)
    item.add_argument("--variant", choices=("base", "finetuned"), required=True)
    item = sub.add_parser("evaluate")
    item.add_argument("--config", required=True)
    item.add_argument("--variant", choices=("base", "finetuned"), required=True)
    item.add_argument("--split", default="test")
    item.add_argument("--rerank", action="store_true")
    item = sub.add_parser("link")
    item.add_argument("--config", required=True)
    item.add_argument("--variant", choices=("base", "finetuned"), required=True)
    item.add_argument("--text", required=True)
    item.add_argument("--top-k", type=int, default=10)
    item.add_argument("--rerank", action="store_true")
    return root


def main() -> int:
    args = parser().parse_args()
    cfg = load_config(args.config)
    if args.command == "doctor":
        result = doctor(cfg)
    elif args.command == "prepare":  # 数据准备
        result = prepare(cfg)
    elif args.command == "build-index": # 构建索引
        result = build_index(cfg, args.variant)
    elif args.command == "evaluate":
        result = evaluate(cfg, args.variant, args.rerank, args.split)
    elif args.command == "mine-negatives": # 挖掘负例
        result = mine_negatives(cfg)
    elif args.command == "train": # 训练模型
        result = train(cfg)
    elif args.command == "compare":  
        result = compare(cfg)
    elif args.command == "link":  # 推理
        result = link_text(
            cfg, args.variant, args.text, args.top_k, args.rerank
        )
    else:
        raise AssertionError(args.command)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.command == "doctor" and not result.get("passed", False):
        return 2
    return 0



"""
脚本目录：
├── compare.py
├── config.py
├── doctor.py
├── evaluate.py
├── index.py
├── io_utils.py
├── metrics.py
├── mining.py
├── models.py
├── prepare.py
├── retrieval.py
├── text_views.py
└── train.py

训练的路径

doctor
prepare 
build-index --config "$CONFIG" --variant base
evaluate --config "$CONFIG" --variant base
evaluate --config "$CONFIG" --variant base --rerank
mine-negatives --config "$CONFIG"
train --config "$CONFIG"
build-index --config "$CONFIG" --variant finetuned
evaluate --config "$CONFIG" --variant finetuned
evaluate --config "$CONFIG" --variant finetuned --rerank
compare --config "$CONFIG"

"""

if __name__ == "__main__":
    sys.exit(main())
