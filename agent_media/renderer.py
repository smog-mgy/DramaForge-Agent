# -*- coding: utf-8 -*-
"""
renderer.py - 视频渲染层（图文成片 / 音画同步）
============================================================
从 legacy/web_studio1.py 的 Tab 4 移植并优化：

原方案问题：
- 逐句生成图片+音频，一集 20-30 张图，极慢；
- ComfyUI 失败时退回黑屏，观感差；
- 用 os.system 调 edge-tts，脆弱；
- URL / workflow 路径 / 分辨率硬编码；
- MoviePy clip 不关闭，内存泄漏。

优化方案：
- 复用 storyboard_episode() 的结构化分镜（每集 5-10 个镜头，而非逐句）；
- 复用 synthesize_speech() 的 edge-tts 封装；
- ComfyUI 失败时退回「文字画面」（PIL 渐变底+居中文字），比黑屏可看；
- 全部参数走 config.py（ComfyUI URL / workflow 路径 / 分辨率 / FPS）；
- 每个 clip 用完即 close，临时文件统一清理；
- 可选进度回调（供 Streamlit UI 展示进度条）。

技能点：MoviePy 音视频合成 · edge-tts 语音 · ComfyUI API · PIL 图像处理
"""
from __future__ import annotations

import json
import os
import random
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Callable

from .config import Settings, get_settings
from .tools import synthesize_speech


class VideoRenderer:
    """图文视频渲染器：分镜图片 + 语音 → 拼接成片。"""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self._temp_files: list[str] = []

    # ----------------------------------------------------------
    # 主入口
    # ----------------------------------------------------------
    def render_episode(
        self,
        episode_text: str,
        episode_num: int,
        storyboard: list[dict[str, Any]] | None = None,
        character_bible: dict[str, str] | None = None,
        output_dir: str | Path | None = None,
        progress_cb: Callable[[int, int, str], None] | None = None,
    ) -> str:
        """
        渲染一集视频，返回输出 mp4 路径。

        Args:
            episode_text:  剧集正文（用于 TTS）
            episode_num:   集号（用于文件名）
            storyboard:    分镜列表（每项含 id/text/prompt）；为 None 则不生成图片，只用文字画面
            character_bible: 角色圣经（注入画面提示词，保证一致性）
            output_dir:    输出目录，默认 settings.output_path
            progress_cb:   进度回调 (current, total, message)
        """
        from moviepy.editor import AudioFileClip, ImageClip, concatenate_videoclips

        output_dir = Path(output_dir or self.settings.output_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 1. 准备镜头：有分镜用分镜，没分镜用单镜头（整集文字）
        shots = self._prepare_shots(episode_text, storyboard)
        total = len(shots)
        video_clips = []

        try:
            for idx, shot in enumerate(shots):
                msg = f"镜头 {idx + 1}/{total}: {shot['text'][:20]}..."
                if progress_cb:
                    progress_cb(idx, total, msg)

                # 2. 生成该镜头的语音
                audio_path = str(output_dir / f"_tmp_ep{episode_num}_s{idx}.mp3")
                try:
                    synthesize_speech(shot["text"], audio_path)
                except Exception as e:  # noqa: BLE001
                    print(f"[renderer] 镜头 {idx} 语音合成失败({e})，跳过")
                    continue
                if not os.path.exists(audio_path) or os.path.getsize(audio_path) < 100:
                    continue
                self._temp_files.append(audio_path)

                # 3. 读取音频时长
                try:
                    audio_clip = AudioFileClip(audio_path)
                    duration = max(audio_clip.duration, 1.0)  # 至少 1 秒
                except Exception:  # noqa: BLE001
                    continue

                # 4. 生成画面：ComfyUI → 文字画面兜底
                image_path = self._render_image(
                    prompt=shot.get("prompt", ""),
                    text=shot["text"],
                    episode_num=episode_num,
                    shot_idx=idx,
                    output_dir=output_dir,
                    character_bible=character_bible,
                )

                # 5. 音画合并
                try:
                    video_clip = ImageClip(image_path).set_duration(duration)
                    video_clip = video_clip.set_audio(audio_clip)
                    video_clips.append(video_clip)
                except Exception as e:  # noqa: BLE001
                    print(f"[renderer] 镜头 {idx} 音画合并失败({e})")
                    audio_clip.close()

            # 6. 拼接成片
            if not video_clips:
                raise RuntimeError("没有成功生成任何视频片段，请检查 TTS 和图片生成。")

            if progress_cb:
                progress_cb(total, total, "拼接成片中...")

            final_video = concatenate_videoclips(video_clips, method="compose")
            output_path = str(output_dir / f"Final_Episode_{episode_num}.mp4")
            final_video.write_videofile(
                output_path,
                fps=self.settings.video_fps,
                codec="libx264",
                audio_codec="aac",
                logger=None,
            )
            final_video.close()
            return output_path

        finally:
            # 7. 清理：关闭所有 clip + 删除临时文件
            for clip in video_clips:
                try:
                    clip.close()
                except Exception:  # noqa: BLE001
                    pass
            self._cleanup_temp()

    # ----------------------------------------------------------
    # 镜头准备
    # ----------------------------------------------------------
    def _prepare_shots(
        self, episode_text: str, storyboard: list[dict] | None
    ) -> list[dict[str, str]]:
        """把分镜转成统一的镜头列表；无分镜时整集作为一个镜头。"""
        if storyboard:
            shots = []
            for s in storyboard:
                shots.append({
                    "text": str(s.get("text", "")).strip(),
                    "prompt": str(s.get("prompt", "")).strip(),
                })
            return [s for s in shots if s["text"]]
        # 无分镜：整集文字作为一个镜头
        return [{"text": episode_text.strip(), "prompt": ""}]

    # ----------------------------------------------------------
    # 图片生成
    # ----------------------------------------------------------
    def _render_image(
        self,
        prompt: str,
        text: str,
        episode_num: int,
        shot_idx: int,
        output_dir: Path,
        character_bible: dict[str, str] | None = None,
    ) -> str:
        """生成一张画面：优先 ComfyUI，失败退回文字画面。"""
        # 有 ComfyUI 且有英文 prompt → 尝试生成
        if self.settings.render_use_comfyui and prompt.strip():
            # 注入角色圣经
            full_prompt = prompt
            if character_bible:
                injected = []
                for name, feature in character_bible.items():
                    if name in text:
                        injected.append(feature)
                if injected:
                    full_prompt = prompt + ", " + ", ".join(injected)

            img = self._comfyui_generate(full_prompt, episode_num, shot_idx, output_dir)
            if img:
                return img

        # 兜底：文字画面
        return self._fallback_text_image(text, episode_num, shot_idx, output_dir)

    def _comfyui_generate(
        self, prompt: str, episode_num: int, shot_idx: int, output_dir: Path
    ) -> str | None:
        """调用本地 ComfyUI API 生成图片，成功返回路径，失败返回 None。"""
        output_dir = Path(output_dir)
        workflow_path = self.settings.comfyui_workflow
        if not os.path.isabs(workflow_path):
            workflow_path = str(Path.cwd() / workflow_path)
        if not os.path.exists(workflow_path):
            print(f"[renderer] workflow 文件不存在: {workflow_path}")
            return None

        try:
            with open(workflow_path, "r", encoding="utf-8") as f:
                workflow = json.load(f)
        except Exception as e:  # noqa: BLE001
            print(f"[renderer] 读取 workflow 失败: {e}")
            return None

        # 注入提示词和随机种子（节点 ID 6=正向提示词，3=采样器，按 ComfyUI 默认工作流）
        try:
            workflow["6"]["inputs"]["text"] = prompt
            workflow["3"]["inputs"]["seed"] = random.randint(1, 10**9)
        except KeyError:
            print("[renderer] workflow 节点 ID 不匹配（期望 6/3），请检查导出的 workflow_api.json")
            return None

        url = self.settings.comfyui_url.rstrip("/")
        # 提交任务
        try:
            data = json.dumps({"prompt": workflow}).encode("utf-8")
            req = urllib.request.Request(f"{url}/prompt", data=data)
            with urllib.request.urlopen(req, timeout=30) as resp:
                prompt_id = json.loads(resp.read())["prompt_id"]
        except Exception as e:  # noqa: BLE001
            print(f"[renderer] 连接 ComfyUI 失败: {e}")
            return None

        # 轮询等待（最多 120 秒）
        for _ in range(60):
            time.sleep(2)
            try:
                with urllib.request.urlopen(f"{url}/history/{prompt_id}", timeout=10) as resp:
                    history = json.loads(resp.read())
                if prompt_id in history:
                    break
            except Exception:  # noqa: BLE001
                continue
        else:
            print("[renderer] ComfyUI 渲染超时（>120s）")
            return None

        # 下载图片
        try:
            outputs = history[prompt_id]["outputs"]
            for node_id in outputs:
                node_out = outputs[node_id]
                if "images" in node_out:
                    img = node_out["images"][0]
                    params = urllib.parse.urlencode({
                        "filename": img["filename"],
                        "subfolder": img["subfolder"],
                        "type": img["type"],
                    })
                    with urllib.request.urlopen(f"{url}/view?{params}", timeout=30) as resp:
                        img_bytes = resp.read()
                    out_path = str(output_dir / f"_tmp_ep{episode_num}_s{shot_idx}.png")
                    with open(out_path, "wb") as f:
                        f.write(img_bytes)
                    self._temp_files.append(out_path)
                    return out_path
        except Exception as e:  # noqa: BLE001
            print(f"[renderer] 下载 ComfyUI 图片失败: {e}")
        return None

    def _fallback_text_image(
        self, text: str, episode_num: int, shot_idx: int, output_dir: Path
    ) -> str:
        """
        兜底画面：渐变底色 + 居中文字（比黑屏可看）。
        使用 PIL 生成 1280x720 图片。
        """
        output_dir = Path(output_dir)
        from PIL import Image, ImageDraw, ImageFont

        W, H = self.settings.video_width, self.settings.video_height
        img = Image.new("RGB", (W, H), color=(30, 30, 50))
        draw = ImageDraw.Draw(img)

        # 渐变底色（从上到下深蓝到深紫）
        for y in range(H):
            r = int(20 + (y / H) * 30)
            g = int(20 + (y / H) * 10)
            b = int(50 + (y / H) * 40)
            draw.line([(0, y), (W, y)], fill=(r, g, b))

        # 尝试加载中文字体，失败用默认
        font = None
        for font_path in [
            "C:/Windows/Fonts/msyh.ttc",
            "C:/Windows/Fonts/simhei.ttf",
            "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        ]:
            if os.path.exists(font_path):
                try:
                    font = ImageFont.truetype(font_path, 36)
                    break
                except Exception:  # noqa: BLE001
                    continue
        if font is None:
            font = ImageFont.load_default()

        # 文字自动换行
        max_width = int(W * 0.8)
        lines = self._wrap_text(text, font, max_width, draw)
        line_height = 50
        total_height = len(lines) * line_height
        y_start = (H - total_height) // 2

        for i, line in enumerate(lines):
            bbox = draw.textbbox((0, 0), line, font=font)
            text_w = bbox[2] - bbox[0]
            x = (W - text_w) // 2
            y = y_start + i * line_height
            # 文字阴影
            draw.text((x + 2, y + 2), line, font=font, fill=(0, 0, 0))
            draw.text((x, y), line, font=font, fill=(255, 255, 255))

        # 集号水印
        small_font = ImageFont.truetype(
            "C:/Windows/Fonts/msyh.ttc", 20
        ) if os.path.exists("C:/Windows/Fonts/msyh.ttc") else font
        draw.text((20, H - 40), f"第 {episode_num} 集 · 镜头 {shot_idx + 1}", font=small_font, fill=(180, 180, 200))

        out_path = str(output_dir / f"_tmp_ep{episode_num}_s{shot_idx}_fb.png")
        img.save(out_path)
        self._temp_files.append(out_path)
        return out_path

    @staticmethod
    def _wrap_text(text: str, font, max_width: int, draw) -> list[str]:
        """简单中文换行：按字符逐个累加，超宽就换行。"""
        lines = []
        current = ""
        for ch in text:
            test = current + ch
            bbox = draw.textbbox((0, 0), test, font=font)
            if bbox[2] - bbox[0] > max_width and current:
                lines.append(current)
                current = ch
            else:
                current = test
        if current:
            lines.append(current)
        return lines if lines else [""]

    # ----------------------------------------------------------
    # 清理
    # ----------------------------------------------------------
    def _cleanup_temp(self) -> None:
        """删除所有临时文件（_tmp 前缀）。"""
        for f in self._temp_files:
            try:
                if os.path.exists(f):
                    os.remove(f)
            except Exception:  # noqa: BLE001
                pass
        self._temp_files.clear()
