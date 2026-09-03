# -*- coding: utf-8 -*-
"""
embeddings.py - 本地语义向量化（PyTorch 技能点）
============================================================
说明：
- 底层是 sentence-transformers（封装 HuggingFace Transformers + PyTorch）；
- 在这里显式封装"加载本地模型 -> 编码 -> 余弦相似度"，方便面试时讲解
  BERT 类模型如何把句子映射到向量空间、如何在 RAG 中做召回；
- rag.py 的 EmbeddingProvider 在本地模式下底层复用的是同一套能力。
"""
from __future__ import annotations

from functools import lru_cache

from ..config import Settings, get_settings

DEFAULT_MODEL = "BAAI/bge-small-zh-v1.5"


@lru_cache(maxsize=1)
def _encoder(model_name: str):
    """
    懒加载本地 sentence-transformers 模型（首次调用会下载权重，之后走缓存）。
    返回对象实现 encode(texts) -> numpy array。
    """
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def encode(texts: list[str], model_name: str | None = None) -> "object":
    """把文本列表编码为向量矩阵（PyTorch 本地模型）。"""
    name = model_name or DEFAULT_MODEL
    return _encoder(name).encode(texts, normalize_embeddings=True, show_progress_bar=False)


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """两个向量的余弦相似度（纯 Python 实现，可当面试手写题讲）。"""
    if len(a) != len(b):
        raise ValueError("向量维度不一致")
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5 or 1.0
    nb = sum(y * y for y in b) ** 0.5 or 1.0
    return dot / (na * nb)


def semantic_similarity(a: str, b: str) -> float:
    """两条文本的语义相似度（0~1）。"""
    [va, vb] = encode([a, b])
    return float(cosine_similarity(list(va), list(vb)))


def build_embedding_provider(settings: Settings | None = None):
    """为 RAG 提供与 config 一致的本地向量化器。"""
    settings = settings or get_settings()
    from ..rag import EmbeddingProvider

    return EmbeddingProvider(settings)
