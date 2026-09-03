# -*- coding: utf-8 -*-
"""
pipeline.py - 端到端生产流水线（产品化入口）
============================================================
把多智能体编排 + RAG + 语音合成（+ 可选视频合成）串成一条可命令行
调用、可出产物清单（manifest）的流水线。产物统一落到 output/ 下，
避免把中间文件散落在项目根目录。
"""
from __future__ import annotations

import json
from pathlib import Path

from .agents import MediaAgentOrchestrator
from .config import Settings, get_settings
from .schemas import StoryEpisode
from .tools import synthesize_speech


class MediaPipeline:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.orchestrator = MediaAgentOrchestrator(self.settings)

    # ---------- 文本 + 语音 ----------
    def produce_series(
        self,
        idea: str,
        total_episodes: int,
        target_words: int,
        with_tts: bool = True,
    ) -> dict:
        """生产整部连载：正文 + 语音，返回 manifest。"""
        out_dir = self.settings.output_path
        manifest = {"idea": idea, "episodes": []}

        episodes: list[StoryEpisode] = self.orchestrator.produce_series(
            idea, total_episodes, target_words
        )

        for ep in episodes:
            txt_path = out_dir / f"episode_{ep.episode:02d}.txt"
            txt_path.write_text(ep.content, encoding="utf-8")

            item: dict = {"episode": ep.episode, "text": txt_path.name, "audio": None}

            if with_tts:
                mp3_path = out_dir / f"episode_{ep.episode:02d}.mp3"
                try:
                    synthesize_speech(ep.content, str(mp3_path))
                    item["audio"] = mp3_path.name
                except Exception as e:  # noqa: BLE001
                    item["audio_error"] = str(e)

            manifest["episodes"].append(item)

        # 汇总 + 记忆统计
        manifest["stats"] = self.orchestrator.memory.stats()
        manifest["logs"] = self.orchestrator.logs
        manifest_path = out_dir / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        manifest["manifest_file"] = manifest_path.name
        return manifest

    # ---------- 分镜（供后续文生图/视频） ----------
    def storyboard_episode(self, episode_text: str) -> list[dict]:
        """把一集正文拆成分镜（调用 storyboarder）。"""
        from .llm import invoke_structured
        from .prompts import storyboarder_prompt
        from .schemas import StoryboardShotList

        prompt = storyboarder_prompt()
        input_text = f"请为这段剧情拆解分镜：{episode_text}"
        result = invoke_structured(
            self.orchestrator.llm, prompt, StoryboardShotList, input=input_text
        )
        return [s.model_dump() for s in result.shots]

    # ---------- 视频渲染（图文成片） ----------
    def render_episode_video(
        self,
        episode_text: str,
        episode_num: int,
        storyboard: list[dict] | None = None,
        character_bible: dict[str, str] | None = None,
        progress_cb: callable | None = None,
    ) -> str:
        """
        把一集正文渲染成视频（分镜图片 + 语音 → 拼接成片）。

        - 自动生成分镜（如未提供）；
        - 优先 ComfyUI 生成画面，失败退回文字画面兜底；
        - 每镜头 TTS 语音 + 图片 → MoviePy 片段 → 拼接成 mp4；
        - 返回输出视频路径。
        """
        from .renderer import VideoRenderer

        # 未提供分镜时自动生成
        if storyboard is None:
            storyboard = self.storyboard_episode(episode_text)

        renderer = VideoRenderer(self.settings)
        return renderer.render_episode(
            episode_text=episode_text,
            episode_num=episode_num,
            storyboard=storyboard,
            character_bible=character_bible,
            progress_cb=progress_cb,
        )

    # ---------- 动态漫剧（漫画出图 + Ken Burns 运镜 + 配音） ----------
    def render_comic_episode(
        self,
        episode_text: str,
        episode_num: int,
        storyboard: list[dict] | None = None,
        character_bible: dict[str, str] | None = None,
        progress_cb: callable | None = None,
    ) -> str:
        """
        把一集正文渲染成动态漫剧视频（每镜头动漫出图 + 运镜 + 对白配音）。

        - 复用分镜拆解 + 角色圣经注入；
        - 走 ComfyUI 动漫模型出图（ComicRenderer 复用 VideoRenderer 的 ComfyUI 能力）；
        - Ken Burns 运镜让静态漫画图动起来；
        - 返回输出 mp4 路径。
        """
        from .comic_renderer import ComicRenderer

        if storyboard is None:
            storyboard = self.storyboard_episode(episode_text)

        renderer = ComicRenderer(self.settings)
        return renderer.render_comic_episode(
            episode_text=episode_text,
            episode_num=episode_num,
            storyboard=storyboard,
            character_bible=character_bible,
            progress_cb=progress_cb,
        )

    # ---------- 角色动态漫剧（AnimateDiff 多帧动画 + 配音） ----------
    def render_animated_comic_episode(
        self,
        episode_text: str,
        episode_num: int,
        storyboard: list[dict] | None = None,
        character_bible: dict[str, str] | None = None,
        progress_cb: callable | None = None,
    ) -> str:
        """
        把一集正文渲染成「角色动态漫剧」：每镜头 AnimateDiff 生成
        多帧动画（角色真正动起来）+ 对白配音，合成 mp4。

        比 render_comic_episode 更进一步：画面本身是动态的（AnimateDiff），
        而非静态图 + Ken Burns 运镜。
        """
        from .comic_renderer import ComicRenderer

        if storyboard is None:
            storyboard = self.storyboard_episode(episode_text)

        renderer = ComicRenderer(self.settings)
        return renderer.render_animated_comic_episode(
            episode_text=episode_text,
            episode_num=episode_num,
            storyboard=storyboard,
            character_bible=character_bible,
            progress_cb=progress_cb,
        )


def run_cli(idea: str, episodes: int, words: int, with_tts: bool) -> None:
    """命令行入口：python -m agent_media.pipeline --idea "..." --episodes 3"""
    pipeline = MediaPipeline()
    manifest = pipeline.produce_series(idea, episodes, words, with_tts)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="AI 短剧全栈流水线")
    parser.add_argument("--idea", type=str, required=True, help="核心点子")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--words", type=int, default=800)
    parser.add_argument("--no-tts", action="store_true")
    args = parser.parse_args()
    run_cli(args.idea, args.episodes, args.words, not args.no_tts)
