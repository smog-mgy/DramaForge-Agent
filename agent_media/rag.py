# -*- coding: utf-8 -*-
"""
rag.py - 检索增强生成（RAG）模块
============================================================
职责：把"设定集 / 角色圣经 / 历史剧情"切片、向量化、存入向量库，
在写每一集正文前检索最相关的上下文注入提示词 —— 解决长篇连载
"剧情跑偏、设定遗忘、人物崩坏"的根因。

分层设计（工程上可讲）：
- 存储层：Chroma 向量库（langchain-chroma 集成），可换 FAISS / Milvus；
- 向量化层：优先本地 sentence-transformers 模型（PyTorch），
  其次支持云端 Embedding API，最后降级为内置词法向量（离线可跑）；
- 检索层：语义 top-k 检索 + 格式化为 prompt 上下文。

降级策略：依赖缺失时自动退化为"关键词重叠打分"的轻量检索，
保证 RAG 链路在任何环境下都能演示，且代码层清晰标注。
"""
from __future__ import annotations

import hashlib
import math
import re
from typing import Any

from .config import Settings, get_settings


# ============================================================
# 1. 向量化层
# ============================================================
class EmbeddingProvider:
    """Embedding 提供器：API -> 本地模型 -> 内置词法向量 三级降级。"""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._kind: str | None = None
        self._model_warned: bool = False

    def _model_cached(self, model_name: str) -> bool:
        """
        检查 sentence-transformers 模型是否已在本地 HuggingFace 缓存。
        未缓存时不自动触发下载（国内网络下载模型容易挂死），
        而是降级为词法向量并提示用户手动预下载。
        """
        from pathlib import Path

        try:
            from huggingface_hub.constants import HF_HUB_CACHE

            cache = Path(HF_HUB_CACHE)
        except Exception:  # noqa: BLE001
            cache = Path.home() / ".cache" / "huggingface" / "hub"
        model_dir = cache / f"models--{model_name.replace('/', '--')}"
        if not model_dir.exists():
            return False
        snapshots = model_dir / "snapshots"
        if not snapshots.exists():
            return False
        # 至少有一个 snapshot 目录且非空
        try:
            return any(s.is_dir() and any(s.iterdir()) for s in snapshots.iterdir())
        except Exception:  # noqa: BLE001
            return False

    def _load(self) -> Any:
        s = self.settings
        # 1) 云端 Embedding API
        if s.embedding_api_base_url and s.embedding_api_key:
            try:
                from langchain_openai import OpenAIEmbeddings

                self._kind = "api"
                return OpenAIEmbeddings(
                    model=s.embedding_api_model or "BAAI/bge-large-zh-v1.5",
                    api_key=s.embedding_api_key,
                    base_url=s.embedding_api_base_url,
                )
            except Exception:  # noqa: BLE001
                pass
        # 2) 本地 sentence-transformers（PyTorch 语义向量）
        #    langchain 1.x 起移到 langchain-huggingface；0.x 在 langchain-community
        try:
            from langchain_huggingface import HuggingFaceEmbeddings  # langchain >= 1.0
        except Exception:  # noqa: BLE001
            try:
                from langchain_community.embeddings import HuggingFaceEmbeddings  # langchain 0.x
            except Exception:  # noqa: BLE001
                HuggingFaceEmbeddings = None
        if HuggingFaceEmbeddings is not None:
            if self._model_cached(s.embedding_model):
                try:
                    self._kind = "local"
                    return HuggingFaceEmbeddings(
                        model_name=s.embedding_model,
                        model_kwargs={"device": "cpu"},
                        encode_kwargs={"normalize_embeddings": True},
                    )
                except Exception as e:  # noqa: BLE001
                    print(f"[rag] 本地 embedding 模型加载失败({e})，降级为词法向量。")
            else:
                if not self._model_warned:
                    self._model_warned = True
                    print(
                        f"[rag] 本地 embedding 模型「{s.embedding_model}」未缓存，"
                        f"跳过自动下载以避免挂死。如需启用语义向量，请先运行：\n"
                        f'  python -c "from sentence_transformers import SentenceTransformer; '
                        f"SentenceTransformer('{s.embedding_model}')\""
                    )
        # 3) 内置词法向量（离线降级，仅供演示）
        self._kind = "lexical"
        return None

    def embed_query(self, text: str) -> list[float]:
        provider = self._load()
        if provider is not None:
            return provider.embed_query(text)
        return _lexical_vector(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        provider = self._load()
        if provider is not None:
            return provider.embed_documents(texts)
        return [_lexical_vector(t) for t in texts]

    @property
    def kind(self) -> str:
        return self._kind or "unknown"


def _lexical_vector(text: str, dim: int = 256) -> list[float]:
    """
    内置降级向量：基于字符 n-gram 的确定性哈希向量。
    无语义能力，仅保证 RAG 离线可跑；安装 sentence-transformers 后自动被替换。
    """
    vec = [0.0] * dim
    text = re.sub(r"\s+", "", (text or "").lower())
    for i in range(len(text)):
        for n in (1, 2):
            gram = text[i : i + n]
            h = int(hashlib.md5(gram.encode("utf-8")).hexdigest(), 16)
            vec[h % dim] += 1.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


# ============================================================
# 2. 文本切片
# ============================================================
def chunk_text(text: str, chunk_size: int = 800, overlap: int = 120) -> list[str]:
    """先按 LangChain 分词器切，缺失则回退到滑动窗口切片。"""
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            separators=["\n\n", "\n", "。", "！", "？", "；"],
        )
        return [c for c in splitter.split_text(text) if c.strip()]
    except Exception:  # noqa: BLE001
        chunks, cur = [], ""
        for para in re.split(r"(?<=[。！？\n])", text):
            cur += para
            if len(cur) >= chunk_size:
                chunks.append(cur)
                cur = cur[-overlap:] if overlap else ""
        if cur.strip():
            chunks.append(cur)
        return [c for c in chunks if c.strip()]


# ============================================================
# 3. 知识库（Chroma + 降级内存库）
# ============================================================
class StoryKnowledgeBase:
    """短剧设定知识库：支持写入设定集 / 历史剧情，并按语义检索。"""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.embedder = EmbeddingProvider(self.settings)
        self._collection = None
        self._fallback_docs: list[dict[str, Any]] = []
        self._load_store()

    def _load_store(self) -> None:
        try:
            from langchain_chroma import Chroma

            self._collection = Chroma(
                collection_name="story_kb",
                embedding_function=self.embedder,
                persist_directory=str(self.settings.chroma_db_path),
            )
        except Exception:  # noqa: BLE001
            # Chroma / 底层依赖缺失 -> 内存降级
            self._collection = None

    # ---------- 写入 ----------
    def add_document(self, doc_id: str, text: str, metadata: dict | None = None) -> int:
        """写入一段长文本（自动切片）。返回切片数。"""
        chunks = chunk_text(text, self.settings.rag_chunk_size)
        if not chunks:
            return 0
        if self._collection is not None:
            ids = [f"{doc_id}#{i}" for i in range(len(chunks))]
            metas = [dict(metadata or {}, source=doc_id, chunk=i) for i in range(len(chunks))]
            self._collection.add_texts(chunks, metadatas=metas, ids=ids)
        else:
            for i, c in enumerate(chunks):
                self._fallback_docs.append(
                    {"text": c, "metadata": dict(metadata or {}, source=doc_id, chunk=i)}
                )
        return len(chunks)

    def add_bible(self, bible_text: str) -> int:
        """写入世界观设定集。"""
        return self.add_document("bible", bible_text, {"type": "bible"})

    def add_character_bible(self, characters: dict[str, str]) -> int:
        """写入角色圣经（每个角色一条，检索时按角色名命中）。"""
        total = 0
        for name, feature in characters.items():
            total += self.add_document(
                f"character_{name}",
                f"角色「{name}」的固定外观设定：{feature}",
                {"type": "character", "character": name},
            )
        return total

    def add_episode(self, episode: int, content: str, summary: str = "") -> int:
        """写入一集正文（正文 + 摘要一起入库，供后续检索）。"""
        return self.add_document(
            f"episode_{episode}",
            f"[第{episode}集摘要] {summary}\n\n{content}",
            {"type": "episode", "episode": episode},
        )

    # ---------- 检索 ----------
    def search(self, query: str, top_k: int | None = None) -> list[dict[str, Any]]:
        """语义检索，返回 [{text, metadata, score?}]。"""
        import warnings

        top_k = top_k or self.settings.rag_top_k
        if self._collection is not None:
            try:
                # 词法/降级向量的相似度分可能略超 [0,1]，langchain-chroma 会发 UserWarning，
                # 不影响检索结果，这里静默。
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    results = self._collection.similarity_search_with_relevance_scores(query, k=top_k)
                return [
                    {
                        "text": doc.page_content,
                        "metadata": doc.metadata,
                        "score": round(float(score), 4),
                    }
                    for doc, score in results
                ]
            except Exception:  # noqa: BLE001
                pass
        return self._keyword_search(query, top_k)

    def _keyword_search(self, query: str, top_k: int) -> list[dict[str, Any]]:
        """降级检索：基于字符 bigram 重叠度打分（对中文短查询更鲁棒）。"""
        def bigrams(t: str) -> set[str]:
            t = re.sub(r"\s+", "", t)
            return {t[i : i + 2] for i in range(len(t) - 1)}

        q_bigrams = bigrams(query)
        scored = []
        for doc in self._fallback_docs:
            d_bigrams = bigrams(doc["text"])
            if not q_bigrams:
                continue
            overlap = len(q_bigrams & d_bigrams)
            score = overlap / len(q_bigrams)  # 覆盖率 0~1
            scored.append((score, doc))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            {"text": d["text"], "metadata": d["metadata"], "score": round(float(s), 4)}
            for s, d in scored[:top_k]
            if s > 0
        ]

    def format_context(self, results: list[dict[str, Any]], max_chars: int = 2000) -> str:
        """把检索结果格式化为 prompt 上下文（带来源标注，便于追溯）。"""
        parts = []
        used = 0
        for r in results:
            src = r.get("metadata", {}).get("source", "unknown")
            text = r["text"]
            if used + len(text) > max_chars:
                text = text[: max_chars - used]
            parts.append(f"[来源:{src}] {text}")
            used += len(text)
            if used >= max_chars:
                break
        return "\n".join(parts)

    @property
    def backend(self) -> str:
        return "chroma" if self._collection is not None else "memory_fallback"
