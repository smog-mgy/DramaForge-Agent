# -*- coding: utf-8 -*-
"""
memory.py - 多级记忆系统
============================================================
长篇连载最大的坑是"几十集后模型忘记设定 / 人物崩坏 / 剧情跑偏"。
这里用"四级记忆"解决（可面试展开）：

1. 设定集级（Bible Memory）  ：世界观、战力、反派，全局唯一权威；
2. 角色级（Character Memory）：角色视觉/性格基因锁，供画面一致性；
3. 逐集摘要级（Episode Memory）：每集结尾摘要 -> 滚动前情提要；
4. 向量记忆级（Vector/RAG）  ：全部历史 + 设定入库，写作前检索最相关片段。

其中第 4 级即 RAG，是"超长上下文"场景下比硬塞全部历史更省 token、
更精准的方案。
"""
from __future__ import annotations

from typing import Any

from .rag import StoryKnowledgeBase


class MemoryStore:
    def __init__(self, kb: StoryKnowledgeBase):
        self.kb = kb
        # 1) 设定集级
        self.bible: str = ""
        # 2) 角色级
        self.characters: dict[str, str] = {}
        # 3) 逐集摘要级
        self.rolling_summary: str = "长篇开始，暂无前情提要。"
        self.episode_summaries: list[dict[str, Any]] = []

    # ---------- 写入 ----------
    def set_bible(self, text: str) -> None:
        """写入并索引世界观设定集。"""
        self.bible = text.strip()
        if self.bible:
            self.kb.add_bible(self.bible)

    def set_characters(self, characters: dict[str, str]) -> None:
        """写入并索引角色圣经。"""
        self.characters = characters
        if characters:
            self.kb.add_character_bible(characters)

    def remember_episode(self, episode: int, content: str, summary: str) -> None:
        """记录一集：入库向量记忆 + 更新滚动摘要 + 追加逐集摘要。"""
        self.kb.add_episode(episode, content, summary)
        self.episode_summaries.append({"episode": episode, "summary": summary})
        self.rolling_summary = summary

    # ---------- 读取 ----------
    def retrieve_context(self, query: str, top_k: int | None = None) -> str:
        """RAG 检索与 query 最相关的设定/历史片段，格式化为上下文。"""
        results = self.kb.search(query, top_k)
        return self.kb.format_context(results)

    def recent_summaries(self, n: int = 3) -> str:
        """最近 n 集摘要（供快速回看）。"""
        tail = self.episode_summaries[-n:]
        return "\n".join(f"第{e['episode']}集：{e['summary']}" for e in tail)

    def stats(self) -> dict[str, Any]:
        return {
            "bible_len": len(self.bible),
            "characters": len(self.characters),
            "episodes_remembered": len(self.episode_summaries),
            "rag_backend": self.kb.backend,
            "embedding_kind": self.kb.embedder.kind,
        }
