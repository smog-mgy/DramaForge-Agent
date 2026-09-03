# -*- coding: utf-8 -*-
"""
agents.py - 多智能体编排（Agent 架构技能点）
============================================================
把"短剧生产"拆成五个各司其职的角色（Plan-Do-Check-Act 循环）：

    Architect(设定集) -> Planner(细纲) -> Writer(正文)
                                             |---> Reviewer(审查)
                                             |        |-- 不通过 -> 带着修改意见重写（反射回路）
                                             |<-------/
                                             |---> Summarizer(摘要 -> 记忆 -> RAG 入库)

工程上可讲的点：
- 单一 Agent 职责拆分：稳定输出、便于单独调优与评估；
- 反射/自纠错回路：Writer 写完先 Review，不合格带修改意见重写，最多 N 次；
- 结构化交接：Agent 之间传 Pydantic 对象，而不是裸字符串；
- 记忆+RAG：写下一集前，从向量库检索相关设定，保证长篇一致性。
"""
from __future__ import annotations

from typing import Any

from .config import Settings, get_settings
from .llm import get_llm, invoke_structured, invoke_text
from .memory import MemoryStore
from .prompts import (
    architect_prompt,
    build_writer_input,
    planner_prompt,
    reviewer_prompt,
    summarizer_prompt,
    writer_prompt,
)
from .rag import StoryKnowledgeBase
from .schemas import EpisodeOutline, OutlineMatrix, ReviewResult, StoryEpisode, EpisodeSummary


class MediaAgentOrchestrator:
    """短剧生产多智能体编排器。"""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.llm = get_llm(self.settings)
        self.kb = StoryKnowledgeBase(self.settings)
        self.memory = MemoryStore(self.kb)
        self.logs: list[dict[str, Any]] = []

    # ---------- 日志 ----------
    def _log(self, stage: str, msg: str, **extra: Any) -> None:
        self.logs.append({"stage": stage, "msg": msg, **extra})

    # ---------- 1. Architect：点子 -> 设定集 ----------
    def build_bible(self, idea: str) -> str:
        self._log("architect", "正在推演世界观设定集...")
        bible = invoke_text(self.llm, architect_prompt(), idea=idea)
        self.memory.set_bible(bible)
        return bible

    # ---------- 2. Planner：设定集 -> 分集细纲矩阵 ----------
    def build_outline(self, total_episodes: int) -> OutlineMatrix:
        self._log("planner", f"正在拆解 {total_episodes} 集细纲...")
        prompt = planner_prompt()
        input_text = f"设定集：\n{self.memory.bible}\n请规划 1 到 {total_episodes} 集的细纲矩阵。"
        matrix = invoke_structured(self.llm, prompt, OutlineMatrix, input=input_text)
        return matrix

    # ---------- 3. Writer + Reviewer + Summarizer：单集生产 ----------
    def produce_episode(
        self,
        outline: EpisodeOutline,
        target_words: int,
        max_rewrites: int = 2,
    ) -> StoryEpisode:
        ep = outline.episode

        # 3.1 RAG：检索与"本集细纲"最相关的设定/历史片段
        query = f"第{ep}集：{outline.core_conflict} {outline.hook}"
        rag_context = self.memory.retrieve_context(query)
        self._log("rag", f"检索到上下文 {len(rag_context)} 字符，backend={self.kb.backend}")

        # 3.2 写作（含反射回路）
        content = self._write_with_review(
            ep,
            outline,
            rag_context,
            target_words,
            max_rewrites=max_rewrites,
        )

        # 3.3 摘要 -> 记忆 + 入库
        summary = invoke_text(
            self.llm,
            summarizer_prompt(),
            content=content,
        )
        self.memory.remember_episode(ep, content, summary)
        self._log("memory", f"第 {ep} 集已记忆并入库。")

        return StoryEpisode(episode=ep, content=content)

    def _write_with_review(
        self,
        ep: int,
        outline: EpisodeOutline,
        rag_context: str,
        target_words: int,
        max_rewrites: int,
    ) -> str:
        review_enabled = self.settings.enable_review
        feedback = ""
        content = ""

        for attempt in range(max_rewrites + 1):
            # 构造主笔输入：把上一轮审查意见并入提示词（反射回路）
            inputs = build_writer_input(
                episode=ep,
                outline=outline.core_conflict + "；结尾悬念：" + outline.hook,
                rolling_summary=self.memory.rolling_summary,
                rag_context=rag_context,
                target_words=target_words,
            )
            if feedback:
                inputs["input"] += f"\n\n上一轮审查未通过，请务必修正：{feedback}"
            self._log("writer", f"第 {ep} 集 第{attempt + 1}次写作...")
            # 创意写作必须用纯文本调用（invoke_text），不能用结构化输出。
            # 800 字正文强制包 JSON 会导致模型直接吐原文，function_calling/json_mode/手动提取全失败。
            content = invoke_text(self.llm, writer_prompt(), **inputs)

            if not review_enabled or attempt >= max_rewrites:
                break

            # 审查
            result: ReviewResult = invoke_structured(
                self.llm,
                reviewer_prompt(),
                ReviewResult,
                outline=f"第{ep}集 核心矛盾：{outline.core_conflict}；结尾悬念：{outline.hook}",
                context=rag_context,
                content=content,
            )
            self._log("reviewer", f"第 {ep} 集审查：passed={result.passed}")
            if result.passed:
                break
            feedback = "；".join(result.issues)
            if result.suggestions:
                feedback += "；" + "；".join(result.suggestions)

        return content

    # ---------- 4. 端到端：连载生产 ----------
    def produce_series(
        self,
        idea: str,
        total_episodes: int,
        target_words: int,
    ) -> list[StoryEpisode]:
        self.build_bible(idea)
        matrix = self.build_outline(total_episodes)

        episodes: list[StoryEpisode] = []
        for outline in sorted(matrix.episodes, key=lambda x: x.episode):
            ep = self.produce_episode(outline, target_words)
            episodes.append(ep)
        return episodes
