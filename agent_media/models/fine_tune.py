# -*- coding: utf-8 -*-
"""
fine_tune.py - PyTorch 微调示例（可直接运行）
============================================================
场景：训练一个"短剧内容质量"二分类器（0=低质量，1=高质量），
用于给每集正文一个确定性质量分，辅助 Reviewer 做审查。

数据格式（jsonl，每行一条）：
    {"text": "灵云宗后山，陆辰缓缓睁开眼……", "label": 1}

运行：
    python -m agent_media.models.fine_tune \
        --data data/sample_train.jsonl \
        --base-model uer/roberta-base-finetuned-jd-binary-chinese  # 或任意中文 seq-classification 底座

说明：
- 这是"会写训练代码"的展示件：数据加载 / 分词 / 训练循环 / 指标 /
  save_pretrained / 推理 一条龙；
- 生产可进一步换成 Trainer API、加 EarlyStopping、SWA、量化等。
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

# ---------- 数据 ----------
def load_dataset(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def split_train_val(rows: list[dict], ratio: float = 0.9):
    n = int(len(rows) * ratio)
    return rows[:n], rows[n:]


# ---------- 训练 ----------
def train(data_path: Path, base_model: str, out_dir: Path, epochs: int = 3, lr: float = 2e-5):
    import torch
    from torch.utils.data import DataLoader, Dataset
    from torch.optim import AdamW
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        get_linear_schedule_with_warmup,
    )

    rows = load_dataset(data_path)
    train_rows, val_rows = split_train_val(rows)

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForSequenceClassification.from_pretrained(base_model, num_labels=2)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    class SimpleDataset(Dataset):
        def __init__(self, rows_):
            self.texts = [r["text"] for r in rows_]
            self.labels = [int(r["label"]) for r in rows_]

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, i):
            enc = tokenizer(
                self.texts[i], truncation=True, max_length=256, padding="max_length"
            )
            return {
                "input_ids": torch.tensor(enc["input_ids"]),
                "attention_mask": torch.tensor(enc["attention_mask"]),
                "labels": torch.tensor(self.labels[i]),
            }

    train_dl = DataLoader(SimpleDataset(train_rows), batch_size=8, shuffle=True)
    val_dl = DataLoader(SimpleDataset(val_rows), batch_size=16)

    optimizer = AdamW(model.parameters(), lr=lr)
    total_steps = math.ceil(len(train_dl) * epochs)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for batch in train_dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            total_loss += loss.item()
        acc = evaluate(model, val_dl, device)
        print(f"[epoch {epoch + 1}/{epochs}] loss={total_loss / len(train_dl):.4f} val_acc={acc:.4f}")

    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"模型已保存到 {out_dir}")


def evaluate(model, val_dl, device) -> float:
    import torch

    model.eval()
    correct = total = 0
    with torch.no_grad():
        for batch in val_dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(**batch).logits
            preds = logits.argmax(dim=-1)
            correct += (preds == batch["labels"]).sum().item()
            total += len(batch["labels"])
    return correct / max(total, 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True, help="jsonl 训练数据")
    parser.add_argument("--base-model", type=str, default="uer/roberta-base-finetuned-jd-binary-chinese")
    parser.add_argument("--out-dir", type=Path, default=Path("checkpoints/content_scorer"))
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()
    train(args.data, args.base_model, args.out_dir, args.epochs)


if __name__ == "__main__":
    main()
