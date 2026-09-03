# -*- coding: utf-8 -*-
"""
tools.py - 工具层（LangChain Tool / Function Calling 技能点）
============================================================
把流水线里的"非 LLM 能力"封装成可被 Agent 调用的工具：
- tts_synthesize   ：文本 -> 语音（edge-tts）
- search_story_kb  ：RAG 检索知识库
- lock_character   ：把角色圣经注入画面提示词（角色一致性）
- list_characters  ：列出当前角色圣经

工程上可讲：工具注册表 + 参数 Schema（由类型注解自动生成），
既可供 LangChain ReAct/function-calling Agent 调用，
也可被上层流水线直接复用（本工程采用后者，保证可读性）。
"""
from __future__ import annotations

import asyncio
import json

from langchain_core.tools import tool

from .rag import StoryKnowledgeBase


def synthesize_speech(
    text: str,
    output_path: str,
    voice: str = "zh-CN-YunxiNeural",
    rate: str = "+0%",
    retries: int = 3,
) -> str:
    """调用 edge-tts 生成中文语音文件（异步 API 的同步封装）。

    edge-tts 偶发网络波动会返回 "No audio was received"，这里加自动重试
    提升流水线稳定性（对白/配音失败会直接丢镜头，代价高）。
    """
    import os
    import time

    async def _run() -> None:
        import edge_tts

        communicate = edge_tts.Communicate(text=text, voice=voice, rate=rate)
        await communicate.save(output_path)

    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            asyncio.run(_run())
            if os.path.exists(output_path) and os.path.getsize(output_path) >= 100:
                return output_path
            last_err = RuntimeError(f"语音合成产出文件无效：{output_path}")
        except Exception as e:  # noqa: BLE001
            last_err = e
        if attempt < retries - 1:
            time.sleep(1.5 * (attempt + 1))  # 1.5s / 3s 退避
    raise RuntimeError(f"语音合成失败（重试 {retries} 次）：{last_err}")


def build_tools(kb: StoryKnowledgeBase, characters: dict[str, str] | None = None) -> list:
    """构造绑定知识库与角色圣经的工具集。"""
    characters = characters or {}

    @tool
    def tts_synthesize(text: str, output_path: str) -> str:
        """把一段文本合成为中文语音 mp3 文件，返回文件路径。"""
        return synthesize_speech(text, output_path)

    @tool
    def search_story_kb(query: str) -> str:
        """在短剧设定/历史剧情知识库中检索与 query 最相关的片段（RAG）。"""
        results = kb.search(query)
        return kb.format_context(results)

    @tool
    def lock_character(render_prompt: str) -> str:
        """把角色圣经中出现的角色英文特征注入画面提示词，保证角色一致性。"""
        injected = []
        for name, feature in characters.items():
            if name in render_prompt:
                injected.append(f"{name}: {feature}")
        return render_prompt + ("，" + "；".join(injected) if injected else "")

    @tool
    def list_characters() -> str:
        """列出当前已登记的角色（角色名 -> 外观特征）。"""
        return json.dumps(characters, ensure_ascii=False)

    return [tts_synthesize, search_story_kb, lock_character, list_characters]
