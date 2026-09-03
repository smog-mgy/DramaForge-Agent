# -*- coding: utf-8 -*-
"""
config.py - 集中式配置管理
============================================================
价值点：把散落在代码里的魔法数字、硬编码密钥、模型名统一收敛到
配置层，通过 .env 注入。既安全（密钥不进代码库），又可移植
（换模型/换环境只改配置）。这是生产级工程的基本功。
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# 项目根目录（本文件所在目录的上一级）
PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """全局配置。所有字段均可通过环境变量或 .env 覆盖。"""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ---------- LLM（默认 DeepSeek，OpenAI 兼容协议） ----------
    deepseek_api_key: str = ""
    llm_model: str = "deepseek-chat"
    llm_base_url: str = "https://api.deepseek.com/v1"
    llm_temperature: float = 0.8
    llm_max_tokens: int = 4096

    # ---------- Embedding / RAG ----------
    embedding_model: str = "BAAI/bge-small-zh-v1.5"
    embedding_api_base_url: str = ""   # 云端 embedding，留空则走本地模型
    embedding_api_key: str = ""
    embedding_api_model: str = ""
    chroma_db_dir: str = ".chroma_db"
    rag_top_k: int = 4
    rag_chunk_size: int = 800

    # ---------- 产物目录 ----------
    output_dir: str = "output"

    # ---------- 流水线控制 ----------
    default_episodes: int = 3
    default_words_per_episode: int = 800
    enable_review: bool = True        # 是否启用 Reviewer 审查 + 重写回路

    # ---------- 视频渲染（图文成片） ----------
    comfyui_url: str = "http://127.0.0.1:8000"  # 你的 ComfyUI Desktop 跑在 8000 端口
    comfyui_workflow: str = "workflow_api.json"  # ComfyUI 导出的 API 工作流路径
    render_use_comfyui: bool = True   # False = 始终用文字画面兜底（无需显卡）
    video_width: int = 1280
    video_height: int = 720
    video_fps: int = 24

    @property
    def chroma_db_path(self) -> Path:
        p = PROJECT_ROOT / self.chroma_db_dir
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def output_path(self) -> Path:
        p = PROJECT_ROOT / self.output_dir
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def llm_ready(self) -> bool:
        """LLM 是否已配置（用于在 UI / CLI 中给出友好提示）。"""
        return bool(self.deepseek_api_key)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """单例获取配置，避免反复读 .env。"""
    return Settings()
