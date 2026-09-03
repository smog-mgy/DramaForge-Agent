# -*- coding: utf-8 -*-
"""
schemas.py - 结构化输出 Schema（Pydantic）
============================================================
价值点：
1. 用 Pydantic 定义"模型必须返回什么"，配合 LLM 的
   with_structured_output / JSON mode，把原来脆弱的
   `json.loads(裸字符串 + 暴力 replace)`` 变成类型安全的解析；
2. 面试时可讲：如何用 Schema 约束让 LLM 输出可被下游程序
   （数据库、视频渲染、检索引擎）直接消费。
"""
from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


class EpisodeOutline(BaseModel):
    """一集的细纲：核心矛盾 + 结尾悬念。"""

    episode: int = Field(..., description="集数（从 1 开始）")
    core_conflict: str = Field(..., description="本集核心矛盾 / 剧情任务")
    hook: str = Field(..., description="本集结尾悬念（钩子），用于吸引下一集")


class OutlineMatrix(BaseModel):
    """整部长篇的分集细纲矩阵。"""

    title: str = Field(..., description="作品名")
    total_episodes: int = Field(..., description="总集数")
    episodes: List[EpisodeOutline] = Field(default_factory=list)


class StoryEpisode(BaseModel):
    """一集正文。"""

    episode: int = Field(..., description="集数")
    title: Optional[str] = Field(default=None, description="本集标题")
    content: str = Field(..., description="正文内容，纯文本，禁止 Markdown 符号")

    @field_validator("content")
    @classmethod
    def strip_markdown(cls, v: str) -> str:
        """清洗常见 Markdown 符号并折叠空白，避免污染下游 TTS / 渲染。"""
        import re

        v = re.sub(r"[*#_`~>]", "", v)
        v = re.sub(r"\s+", " ", v).strip()
        return v


class EpisodeSummary(BaseModel):
    """一集的剧情摘要，用于滚动记忆。"""

    episode: int
    summary: str = Field(..., description="一句话到两句话的剧情摘要，重点写结尾走向与悬念")


class ReviewResult(BaseModel):
    """Reviewer 审查结论。"""

    passed: bool = Field(..., description="是否通过审查")
    issues: List[str] = Field(default_factory=list, description="发现的问题列表")
    suggestions: Optional[List[str]] = Field(default_factory=list, description="改进建议列表")


class StoryboardShot(BaseModel):
    """分镜镜头。"""

    id: int
    text: str = Field(..., description="画面/情节描述")
    prompt: str = Field(..., description="英文渲染提示词（供文生图/文生视频）")


class StoryboardShotList(BaseModel):
    """分镜镜头列表（with_structured_output 要求传入 Pydantic 模型而非 list[Model]）。"""

    shots: List[StoryboardShot] = Field(default_factory=list, description="分镜镜头列表")


class CharacterBible(BaseModel):
    """角色视觉基因锁：角色名 -> 固定英文外观特征。"""

    characters: dict[str, str] = Field(
        default_factory=dict,
        description='{"男主": "1boy, handsome, ...", ...}',
    )

    def lock_prompt(self, text: str) -> str:
        """把角色圣经注入一段画面提示词中（角色一致性核心逻辑）。"""
        injected = []
        for name, feature in self.characters.items():
            if name in text:
                injected.append(f"{name}: {feature}")
        if injected:
            return text + "，" + "；".join(injected)
        return text
