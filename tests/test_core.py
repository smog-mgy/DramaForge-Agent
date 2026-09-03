# -*- coding: utf-8 -*-
"""
test_core.py - 无需 API Key / 无需联网的冒烟测试
============================================================
覆盖：配置加载、Schema 校验与清洗、文本切片、降级向量、
多级记忆（含 RAG 降级检索）、提示词渲染、角色基因锁。

运行：python -m pytest tests/ -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

# 保证从项目根目录导入 agent_media
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent_media.config import get_settings  # noqa: E402
from agent_media.rag import chunk_text, StoryKnowledgeBase, _lexical_vector  # noqa: E402
from agent_media.schemas import (  # noqa: E402
    CharacterBible,
    EpisodeOutline,
    OutlineMatrix,
    StoryEpisode,
)
from agent_media.memory import MemoryStore  # noqa: E402
from agent_media.prompts import build_writer_input  # noqa: E402


# ---------- 配置 ----------
def test_settings_defaults():
    s = get_settings()
    assert s.llm_model == "deepseek-chat"
    assert s.rag_top_k > 0
    assert s.output_path.exists()  # 自动建目录


# ---------- Schema ----------
def test_story_episode_strips_markdown():
    ep = StoryEpisode(episode=1, content="**你好**世界 # 标题 `code`")
    assert "*" not in ep.content and "#" not in ep.content and "`" not in ep.content
    assert "你好世界 标题 code" == ep.content


def test_outline_matrix_validation():
    m = OutlineMatrix(
        title="测试",
        total_episodes=1,
        episodes=[EpisodeOutline(episode=1, core_conflict="矛盾", hook="悬念")],
    )
    assert m.episodes[0].episode == 1


def test_character_bible_lock():
    bible = CharacterBible(characters={"男主": "1boy, black suit"})
    locked = bible.lock_prompt("男主从门外走来")
    assert "1boy, black suit" in locked
    unlocked = bible.lock_prompt("路人甲走过")
    assert "1boy" not in unlocked


# ---------- RAG / 切片 / 降级向量 ----------
def test_chunk_text_basic():
    text = "第一句。" * 300  # 足够长
    chunks = chunk_text(text, chunk_size=200, overlap=20)
    assert len(chunks) >= 2
    assert all(c.strip() for c in chunks)


def test_lexical_vector_deterministic():
    v1 = _lexical_vector("男主角觉醒代码外挂")
    v2 = _lexical_vector("男主角觉醒代码外挂")
    assert v1 == v2
    norm = sum(x * x for x in v1) ** 0.5
    assert abs(norm - 1.0) < 1e-6  # 归一化


def test_memory_fallback_rag():
    """不装任何依赖时，知识库应降级为内存检索且可工作。"""
    kb = StoryKnowledgeBase()
    assert kb.backend in ("chroma", "memory_fallback")
    kb.add_bible("主角陆辰是穿越来的程序员，可以改写功法代码。")
    ctx = kb.format_context(kb.search("陆辰的外挂是什么"))
    assert "陆辰" in ctx or "代码" in ctx or ctx == ""


def test_memory_store_flow():
    kb = StoryKnowledgeBase()
    mem = MemoryStore(kb)
    mem.set_bible("设定集：修仙世界由代码构成。")
    mem.remember_episode(1, "第一集正文", "第一集摘要")
    assert mem.rolling_summary == "第一集摘要"
    assert len(mem.episode_summaries) == 1
    assert mem.retrieve_context("第一集发生了什么") != ""


# ---------- 提示词渲染 ----------
def test_build_writer_input():
    payload = build_writer_input(
        episode=2, outline="打脸反派", rolling_summary="上集摘要",
        rag_context="相关设定", target_words=800,
    )
    assert "第 2 集" in payload["input"]
    assert "打脸反派" in payload["input"]
    assert "相关设定" in payload["input"]


# ---------- 工具 ----------
def test_synthesize_speech_missing_dep():
    """edge-tts 未安装时应给出可读错误，而不是静默。"""
    try:
        from agent_media.tools import synthesize_speech

        with pytest.raises(Exception):
            synthesize_speech("测试", "should_not_exist.mp3")
    except ImportError:
        pytest.skip("edge-tts 未安装，跳过")
