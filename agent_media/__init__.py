# -*- coding: utf-8 -*-
"""
agent_media - AI 短剧全栈自动化媒体引擎（重构版）

定位：一套"世界观设定 -> 分集细纲 -> 正文 -> 审查 -> 配音/成片"的
      LLM Agent 生产流水线。本包用于展示工程化的 LLM / Prompt /
      LangChain / RAG / PyTorch 能力。
"""

__version__ = "2.0.0"

from .config import get_settings, Settings  # noqa: F401
