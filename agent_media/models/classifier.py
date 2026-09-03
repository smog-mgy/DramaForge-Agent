# -*- coding: utf-8 -*-
"""
classifier.py - 内容质量分类器推理封装（PyTorch + transformers）
============================================================
定位：作为"模型打底 + LLM 修正"混合审查的一环。
- 先用微调过的小模型（例如"内容是否符合高质量爽文标准"的二分类）
  给出一个确定性分数；
- 再把分数与 Reviewer(LLM) 的结论合并，作为是否重写的依据。

说明：
- 模型文件由 fine_tune.py 训练得到，存放在 checkpoints/ 目录；
- 若未训练模型，本模块返回 None，流水线自动跳过模型环节，不影响主流程。
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from ..config import Settings, get_settings

LABELS = ["低质量", "高质量"]


class ContentScorer:
    """基于 Transformer 序列分类的内容质量打分器。"""

    def __init__(self, model, tokenizer, device: str = "cpu"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    @classmethod
    def from_checkpoint(
        cls, checkpoint_dir: Path | str, settings: Settings | None = None
    ) -> Optional["ContentScorer"]:
        """从微调产物加载；目录不存在或依赖缺失时返回 None（优雅降级）。"""
        checkpoint_dir = Path(checkpoint_dir)
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            model = AutoModelForSequenceClassification.from_pretrained(str(checkpoint_dir))
            tokenizer = AutoTokenizer.from_pretrained(str(checkpoint_dir))
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model.to(device).eval()
            return cls(model, tokenizer, device)
        except Exception:  # noqa: BLE001
            return None

    def score(self, text: str) -> dict:
        """返回 {label, prob}。"""
        import torch

        inputs = self.tokenizer(
            text[:512], return_tensors="pt", truncation=True, max_length=512
        ).to(self.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits
        prob = float(torch.softmax(logits, dim=-1)[0, 1].item())
        label = LABELS[1] if prob >= 0.5 else LABELS[0]
        return {"label": label, "prob": round(prob, 4)}


def load_scorer(settings: Settings | None = None) -> Optional[ContentScorer]:
    """尝试加载分类器（默认读取项目 checkpoints/content_scorer 目录）。"""
    settings = settings or get_settings()
    ckpt = settings.output_path.parent / "checkpoints" / "content_scorer"
    if not ckpt.exists():
        return None
    return ContentScorer.from_checkpoint(ckpt)
