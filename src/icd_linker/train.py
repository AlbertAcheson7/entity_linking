"""使用困难负例微调实体链接双编码器。

输入：
    1. train_with_negatives.jsonl：训练查询、正例 UID、困难负例 UID。
    2. validation.jsonl：用于每轮评估的查询和正例 UID。
    3. target_terms.jsonl：目标 ICD 概念 UID 到 context_text 等信息的词典。
    4. cfg：基础模型、训练超参数、随机种子和输出路径。
输出：
    1. finetuned_model_dir/：验证集 Hit@10 最优轮次的模型与 tokenizer。
    2. finetuned_model_dir/training_state.json：最佳轮次及训练配置。
    3. finetuned_model_dir 的父目录/training_history.json：逐轮 loss 与 Hit@10。

目标：
    让查询文本向量靠近正确 ICD 概念向量、远离困难负例向量，从而提高候选召回质量。
"""

from __future__ import annotations

import math
import os
import random
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from .io_utils import load_lookup, read_jsonl, write_json


def _precision(torch, choice: str):
    """把配置中的精度名称转换为 PyTorch dtype；返回 None 表示关闭 autocast。"""
    if choice == "bf16":
        return torch.bfloat16
    if choice == "fp16":
        return torch.float16
    if choice == "auto":
        return (
            torch.bfloat16
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            else torch.float16
        )
    return None


def train(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """执行对比学习训练，并返回最佳 Hit@10 与逐轮训练历史。"""
    import torch
    import torch.nn.functional as F
    from torch.optim import AdamW
    from torch.utils.data import DataLoader, Dataset
    from transformers import (
        AutoModel, AutoTokenizer, get_linear_schedule_with_warmup,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("contrastive training requires a CUDA GPU")

    # 读取训练/验证查询，以及所有目标概念。load_lookup 以 term_uid 为键建立字典。
    prepared = Path(cfg["paths"]["prepared_dir"])
    train_rows = list(read_jsonl(prepared / "train_with_negatives.jsonl"))
    validation_rows = list(read_jsonl(prepared / "validation.jsonl"))
    targets = load_lookup(prepared / "target_terms.jsonl")
    train_cfg = cfg["training"]
    model_name = cfg["models"]["embedding"]
    output_dir = Path(cfg["paths"]["finetuned_model_dir"])
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).cuda()
    if train_cfg.get("gradient_checkpointing") and hasattr(
        model, "gradient_checkpointing_enable"
    ):
        # 用额外计算换取更低显存占用，适合较长文本或较大 batch。
        model.gradient_checkpointing_enable()
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False

    class TrainingDataset(Dataset):
        """DataLoader 所需的轻量包装；每个元素就是一条 JSONL 训练记录。"""

        def __len__(self):
            return len(train_rows)

        def __getitem__(self, index):
            return train_rows[index]

    rng = random.Random(cfg["seed"])

    def collate(batch):
        """把若干查询展开为查询列表和“1 个正例 + N 个困难负例”的候选池。"""
        query_texts, candidate_texts, candidate_uids = [], [], []
        labels, all_positives = [], []
        for row in batch:
            query_texts.append(row["query_context_text"])

            # 一个源概念可能映射到多个正确目标；每次随机选一个作为当前监督标签。
            positive_uid = rng.choice(row["positive_target_uids"])

            # label 指向该查询所选正例在“整个 batch 候选池”中的位置。
            labels.append(len(candidate_texts))
            selected = [positive_uid] + row["hard_negative_uids"]
            candidate_uids.extend(selected)
            candidate_texts.extend(
                targets[uid]["context_text"] for uid in selected
            )
            all_positives.append(set(row["positive_target_uids"]))
        return {
            "query_texts": query_texts,
            "candidate_texts": candidate_texts,
            "candidate_uids": candidate_uids,
            "labels": labels,
            "all_positives": all_positives,
        }

    loader = DataLoader(
        TrainingDataset(), batch_size=train_cfg["batch_size"], shuffle=True,
        num_workers=train_cfg["num_workers"], collate_fn=collate,
    )
    optimizer = AdamW(
        model.parameters(), lr=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
    )

    # scheduler 的 step 次数按“参数更新次数”计算，而非原始 mini-batch 次数。
    update_steps = math.ceil(
        len(loader) / train_cfg["gradient_accumulation"]
    ) * train_cfg["epochs"]
    warmup_steps = int(update_steps * train_cfg["warmup_ratio"])
    scheduler = get_linear_schedule_with_warmup(
        optimizer, warmup_steps, update_steps
    )
    scaler = torch.cuda.amp.GradScaler(
        enabled=_precision(torch, train_cfg["mixed_precision"]) == torch.float16
    )
    amp_dtype = _precision(torch, train_cfg["mixed_precision"])

    def embed(texts: List[str], max_length: int):
        """编码文本并返回单位向量；单位向量点积等价于余弦相似度。"""
        tokens = tokenizer(
            texts, padding=True, truncation=True, max_length=max_length,
            return_tensors="pt",
        )
        tokens = {k: v.cuda(non_blocking=True) for k, v in tokens.items()}
        hidden = model(**tokens, return_dict=True).last_hidden_state[:, 0]
        return F.normalize(hidden, dim=1)

    best_hit = -1.0
    history = []
    accumulation = train_cfg["gradient_accumulation"]
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(train_cfg["epochs"]):
        model.train()
        total_loss = 0.0
        for step, batch in enumerate(loader, 1):
            autocast_enabled = amp_dtype is not None
            with torch.autocast(
                device_type="cuda", dtype=amp_dtype,
                enabled=autocast_enabled,
            ):
                query_vectors = embed(
                    batch["query_texts"], train_cfg["query_max_length"]
                )
                candidate_vectors = embed(
                    batch["candidate_texts"], train_cfg["target_max_length"]
                )

                # 每个查询与 batch 内全部候选做相似度比较，形成 in-batch negatives。
                # temperature 越小，softmax 分布越尖锐，对难例的惩罚通常越强。
                logits = (
                    query_vectors @ candidate_vectors.T
                ) / train_cfg["temperature"]

                # 若某查询的另一个真实目标恰好出现在候选池中，不能把它当负例惩罚。
                # 将对应 logit 设为极小值，使其不参与该查询的交叉熵竞争。
                for query_index, positives in enumerate(batch["all_positives"]):
                    selected_label = batch["labels"][query_index]
                    for candidate_index, uid in enumerate(batch["candidate_uids"]):
                        if uid in positives and candidate_index != selected_label:
                            logits[query_index, candidate_index] = -1e4
                labels = torch.tensor(batch["labels"], device="cuda")

                # 除以 accumulation，使累积多个 mini-batch 后的总梯度尺度保持一致。
                loss = F.cross_entropy(logits, labels) / accumulation
            scaler.scale(loss).backward()
            total_loss += float(loss.detach()) * accumulation
            if step % accumulation == 0 or step == len(loader):
                # 到达累积步数（或 epoch 最后一批）后才真正更新一次参数。
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

        hit10 = _validation_hit10(
            model, tokenizer, validation_rows, targets, train_cfg
        )
        epoch_result = {
            "epoch": epoch + 1,
            "loss": total_loss / max(len(loader), 1),
            "validation_hit@10": hit10,
        }
        history.append(epoch_result)
        if hit10 > best_hit:
            # 只保存验证集 Hit@10 最好的 checkpoint，而不是无条件保存最后一轮。
            best_hit = hit10
            output_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            write_json(output_dir / "training_state.json", {
                "base_model": model_name,
                "best_epoch": epoch + 1,
                "best_validation_hit@10": best_hit,
                "training": train_cfg,
                "seed": cfg["seed"],
            })
    result = {"best_validation_hit@10": best_hit, "history": history}
    write_json(output_dir.parent / "training_history.json", result)
    return result


def _validation_hit10(model, tokenizer, rows, targets, train_cfg) -> float:
    """计算验证查询的 Hit@10：前 10 个预测命中任一真实目标即记为成功。"""
    import torch
    import torch.nn.functional as F

    model.eval()
    target_uids = sorted(targets)
    target_vectors = []
    batch_size = train_cfg["batch_size"] * 2

    # 先编码全部目标概念，并转置为 [向量维度, 目标数]，方便后续矩阵乘法。
    with torch.no_grad():
        for start in range(0, len(target_uids), batch_size):
            texts = [
                targets[uid]["context_text"]
                for uid in target_uids[start:start + batch_size]
            ]
            tokens = tokenizer(
                texts, padding=True, truncation=True,
                max_length=train_cfg["target_max_length"],
                return_tensors="pt",
            )
            tokens = {k: v.cuda() for k, v in tokens.items()}
            hidden = model(**tokens, return_dict=True).last_hidden_state[:, 0]
            target_vectors.append(F.normalize(hidden, dim=1))
    target_matrix = torch.cat(target_vectors).T.contiguous()
    hits = 0

    # 再逐批编码验证查询，与全量目标做精确向量检索并取相似度最高的 10 个。
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            batch = rows[start:start + batch_size]
            tokens = tokenizer(
                [row["query_context_text"] for row in batch],
                padding=True, truncation=True,
                max_length=train_cfg["query_max_length"],
                return_tensors="pt",
            )
            tokens = {k: v.cuda() for k, v in tokens.items()}
            hidden = model(**tokens, return_dict=True).last_hidden_state[:, 0]
            query_vectors = F.normalize(hidden, dim=1)
            top_indices = (query_vectors @ target_matrix).topk(10, dim=1).indices
            for row, indices in zip(batch, top_indices):
                predicted = {target_uids[i] for i in indices.tolist()}
                hits += bool(predicted & set(row["positive_target_uids"]))

    # 验证期间切换到了 eval 模式；返回前恢复 train 模式供下一轮训练使用。
    model.train()
    return hits / max(len(rows), 1)
