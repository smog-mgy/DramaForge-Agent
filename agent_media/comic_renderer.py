# -*- coding: utf-8 -*-
"""
comic_renderer.py - 动态漫剧渲染层（漫画出图 + 运镜 + 音画合成）
============================================================
在图文成片基础上，把"静态漫画图"升级为"动态漫剧"：

- ComfyUI 动漫模型出图（复用 VideoRenderer._comfyui_generate）；
- Ken Burns 运镜：zoom in / zoom out / 左右平移，让每格画面"活"起来；
- 每镜头 TTS 对白 + 运镜画面 → MoviePy 合成 → 单集漫剧 MP4。

技能点：ComfyUI API · Ken Burns 镜头语言 · MoviePy 音视频合成 · edge-tts
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image

from .config import Settings
from .renderer import VideoRenderer
from .tools import synthesize_speech


class ComicRenderer(VideoRenderer):
    """动态漫剧渲染器：每镜头动漫出图 + Ken Burns 运镜 + 配音合成。"""

    # 运镜方向循环表（每个镜头换一种，避免全片单调）
    MOTIONS = ["zoom_in", "pan_right", "zoom_out", "pan_left"]

    def render_comic_episode(
        self,
        episode_text: str,
        episode_num: int,
        storyboard: list[dict[str, Any]] | None = None,
        character_bible: dict[str, str] | None = None,
        output_dir: str | Path | None = None,
        progress_cb: Callable[[int, int, str], None] | None = None,
    ) -> str:
        """
        渲染一集动态漫剧，返回 mp4 路径。

        Args:
            episode_text:  剧集正文（用于对白 TTS）
            episode_num:   集号
            storyboard:    分镜列表（每项含 text/prompt）；None 则整集一个镜头
            character_bible: 角色圣经（注入画面提示词）
            output_dir:    输出目录
            progress_cb:   进度回调 (current, total, message)
        """
        from moviepy.editor import AudioFileClip, CompositeVideoClip, concatenate_videoclips

        output_dir = Path(output_dir or self.settings.output_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        shots = self._prepare_shots(episode_text, storyboard)
        total = len(shots)
        video_clips = []

        try:
            for idx, shot in enumerate(shots):
                msg = f"镜头 {idx + 1}/{total}：出图 + 运镜 + 配音"
                if progress_cb:
                    progress_cb(idx, total, msg)

                # 1. ComfyUI 动漫出图（失败退回文字画面）
                image_path = self._render_image(
                    prompt=shot.get("prompt", ""),
                    text=shot["text"],
                    episode_num=episode_num,
                    shot_idx=idx,
                    output_dir=output_dir,
                    character_bible=character_bible,
                )

                # 2. TTS 对白
                audio_path = str(output_dir / f"_tmp_ep{episode_num}_c{idx}.mp3")
                try:
                    synthesize_speech(shot["text"], audio_path)
                except Exception as e:  # noqa: BLE001
                    print(f"[comic] 镜头 {idx} 对白合成失败({e})，跳过")
                    continue
                if not os.path.exists(audio_path) or os.path.getsize(audio_path) < 100:
                    continue
                self._temp_files.append(audio_path)

                # 3. 读音频时长
                try:
                    audio_clip = AudioFileClip(audio_path)
                    duration = max(audio_clip.duration, 1.0)
                except Exception:  # noqa: BLE001
                    continue

                # 4. Ken Burns 运镜（静态图 -> 动态画面）
                motion = self.MOTIONS[idx % len(self.MOTIONS)]
                visual = self._ken_burns_clip(
                    image_path, duration, motion,
                    size=(self.settings.video_width, self.settings.video_height),
                )

                # 5. 音画合成
                try:
                    final_slice = visual.set_audio(audio_clip)
                    video_clips.append(final_slice)
                except Exception as e:  # noqa: BLE001
                    print(f"[comic] 镜头 {idx} 合成失败({e})")
                    audio_clip.close()
                    continue

            if not video_clips:
                raise RuntimeError("没有成功生成任何漫剧片段，请检查出图和 TTS。")

            if progress_cb:
                progress_cb(total, total, "拼接成片中...")

            final_video = concatenate_videoclips(video_clips, method="compose")
            output_path = str(output_dir / f"Comic_Episode_{episode_num}.mp4")
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
            for clip in video_clips:
                try:
                    clip.close()
                except Exception:  # noqa: BLE001
                    pass
            self._cleanup_temp()

    # ----------------------------------------------------------
    # 角色动态漫剧（AnimateDiff 多帧动画 + 配音）
    # ----------------------------------------------------------
    def render_animated_comic_episode(
        self,
        episode_text: str,
        episode_num: int,
        storyboard: list[dict[str, Any]] | None = None,
        character_bible: dict[str, str] | None = None,
        output_dir: str | Path | None = None,
        progress_cb: Callable[[int, int, str], None] | None = None,
        frames: int = 4,
        fps: int = 6,
    ) -> str:
        """
        渲染一集「角色动态漫剧」：每镜头用 AnimateDiff 生成多帧动画，
        角色真正动起来（转身/发丝/衣摆飘动），再配音合成 mp4。

        与 render_comic_episode（静态图+Ken Burns）的区别：
        本方法画面本身是动态的（AnimateDiff 生成视频帧序列）。

        注意：frames 默认 4（RTX 4060 8GB 显存被系统占用后只剩约 4GB，
        AnimateDiff 的 temporal attention 中间激活随帧数平方增长，
        batch>=6 会超出显存导致灰屏）。释放显存后可调大到 8-16 帧。
        """
        from moviepy.editor import AudioFileClip, ImageSequenceClip, concatenate_videoclips

        output_dir = Path(output_dir or self.settings.output_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        shots = self._prepare_shots(episode_text, storyboard)
        total = len(shots)
        video_clips = []

        try:
            for idx, shot in enumerate(shots):
                msg = f"镜头 {idx + 1}/{total}：AnimateDiff 动态生成 + 配音"
                if progress_cb:
                    progress_cb(idx, total, msg)

                # 1. 合成 AnimateDiff 正向提示词（含角色圣经注入）
                prompt = shot.get("prompt", "")
                if character_bible:
                    injected = []
                    for name, feature in character_bible.items():
                        if name in shot.get("text", ""):
                            injected.append(feature)
                    if injected:
                        prompt = (prompt + ", " + ", ".join(injected)).strip()

                # 2. AnimateDiff 生成动态帧（返回帧图片目录）
                frame_paths = self._animatediff_generate(
                    prompt=prompt,
                    episode_num=episode_num,
                    shot_idx=idx,
                    output_dir=output_dir,
                    frames=frames,
                )
                if not frame_paths:
                    print(f"[comic] 镜头 {idx} AnimateDiff 生成失败，跳过")
                    continue

                # 3. TTS 对白
                audio_path = str(output_dir / f"_tmp_ep{episode_num}_ad{idx}.mp3")
                try:
                    synthesize_speech(shot["text"], audio_path)
                except Exception as e:  # noqa: BLE001
                    print(f"[comic] 镜头 {idx} 对白合成失败({e})，跳过")
                    continue
                if not os.path.exists(audio_path) or os.path.getsize(audio_path) < 100:
                    continue
                self._temp_files.append(audio_path)

                # 4. 读取音频时长
                try:
                    audio_clip = AudioFileClip(audio_path)
                    duration = max(audio_clip.duration, 1.0)
                except Exception:  # noqa: BLE001
                    continue

                # 5. 帧序列 -> 视频 clip（画面循环到覆盖对白时长）
                try:
                    # 放大到视频规格（AnimateDiff 输出 512x512 -> 1280x720）
                    target = (self.settings.video_width, self.settings.video_height)
                    resized = [self._resize_frame(p, target) for p in frame_paths]
                    visual = ImageSequenceClip(resized, fps=fps)
                    if visual.duration < duration:
                        visual = visual.loop(duration=duration)
                    final_slice = visual.set_audio(audio_clip)
                    video_clips.append(final_slice)
                except Exception as e:  # noqa: BLE001
                    print(f"[comic] 镜头 {idx} 帧合成失败({e})")
                    audio_clip.close()
                    continue

            if not video_clips:
                raise RuntimeError("没有成功生成任何动态漫剧片段，请检查 AnimateDiff 和 TTS。")

            if progress_cb:
                progress_cb(total, total, "拼接成片中...")

            final_video = concatenate_videoclips(video_clips, method="compose")
            output_path = str(output_dir / f"AnimatedComic_Episode_{episode_num}.mp4")
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
            for clip in video_clips:
                try:
                    clip.close()
                except Exception:  # noqa: BLE001
                    pass
            self._cleanup_temp()

    def _animatediff_generate(
        self,
        prompt: str,
        episode_num: int,
        shot_idx: int,
        output_dir: Path,
        frames: int = 16,
        width: int = 512,
        height: int = 512,
    ) -> list[str]:
        """
        调用 ComfyUI AnimateDiff 生成多帧动态动画，返回帧图片路径列表。

        使用内置节点：LoaderWithContext + StandardUniformContextOptions + KSampler。
        """
        import json as _json
        import time as _time
        import urllib.request as _urlreq
        import urllib.parse as _urlparse

        output_dir = Path(output_dir)
        base_url = self.settings.comfyui_url.rstrip("/")

        # 从 workflow_api.json 读取动漫模型名（保证与现有出图一致）
        ckpt_name = "动漫majicmixRealistic_v6_nigi3d_v1.0.safetensors"
        wf_path = self.settings.comfyui_workflow
        if not os.path.isabs(wf_path):
            wf_path = str(Path.cwd() / wf_path)
        try:
            if os.path.exists(wf_path):
                with open(wf_path, "r", encoding="utf-8") as f:
                    wf = _json.load(f)
                ckpt_name = wf.get("4", {}).get("inputs", {}).get("ckpt_name", ckpt_name)
        except Exception:  # noqa: BLE001
            pass

        # AnimateDiff 工作流（16 帧动画）
        workflow = {
            "4": {"inputs": {"ckpt_name": ckpt_name}, "class_type": "CheckpointLoaderSimple"},
            "6": {"inputs": {"text": prompt, "clip": ["4", 1]}, "class_type": "CLIPTextEncode"},
            "7": {"inputs": {"text": "text, watermark, blurry, low quality, deformed", "clip": ["4", 1]}, "class_type": "CLIPTextEncode"},
            "5": {"inputs": {"width": width, "height": height, "batch_size": frames}, "class_type": "EmptyLatentImage"},
            "10": {"inputs": {"context_length": frames, "context_stride": 1, "context_overlap": 4, "fuse_method": "pyramid", "use_on_equal_length": False, "start_percent": 0.0, "guarantee_steps": 1}, "class_type": "ADE_StandardUniformContextOptions"},
            "12": {"inputs": {"model": ["4", 0], "model_name": "mm_sd_v15_v2.safetensors", "beta_schedule": "autoselect", "context_options": ["10", 0]}, "class_type": "ADE_AnimateDiffLoaderWithContext"},
            "3": {"inputs": {"seed": 12345 + shot_idx, "steps": 20, "cfg": 7.0, "sampler_name": "euler", "scheduler": "normal", "denoise": 1.0, "model": ["12", 0], "positive": ["6", 0], "negative": ["7", 0], "latent_image": ["5", 0]}, "class_type": "KSampler"},
            "8": {"inputs": {"samples": ["3", 0], "vae": ["4", 2]}, "class_type": "VAEDecode"},
            "13": {"inputs": {"images": ["8", 0], "filename_prefix": f"AD_ep{episode_num}_s{shot_idx}", "fps": 12.0, "lossless": False, "quality": 90, "method": "default"}, "class_type": "SaveAnimatedWEBP"},
        }

        # 提交任务
        try:
            data = _json.dumps({"prompt": workflow}).encode("utf-8")
            req = _urlreq.Request(f"{base_url}/prompt", data=data, headers={"Content-Type": "application/json"})
            with _urlreq.urlopen(req, timeout=30) as resp:
                prompt_id = _json.loads(resp.read())["prompt_id"]
        except Exception as e:  # noqa: BLE001
            print(f"[comic] AnimateDiff 提交失败: {e}")
            return []

        # 轮询等待（最多 5 分钟）
        for _ in range(100):
            _time.sleep(3)
            try:
                with _urlreq.urlopen(f"{base_url}/history/{prompt_id}", timeout=10) as resp:
                    history = _json.loads(resp.read())
                if prompt_id in history:
                    break
            except Exception:  # noqa: BLE001
                continue
        else:
            print("[comic] AnimateDiff 渲染超时")
            return []

        # 收集帧：下载 webp 动画后拆成逐帧 PNG
        frame_paths: list[str] = []
        try:
            outputs = history[prompt_id]["outputs"]
            for node_id in outputs:
                node_out = outputs[node_id]
                if "images" in node_out:
                    for img in node_out["images"]:
                        params = _urlparse.urlencode({
                            "filename": img["filename"],
                            "subfolder": img["subfolder"],
                            "type": img["type"],
                        })
                        with _urlreq.urlopen(f"{base_url}/view?{params}", timeout=30) as resp:
                            anim_bytes = resp.read()
                        # 拆帧
                        from PIL import Image as _PILImage
                        import io as _io

                        try:
                            anim = _PILImage.open(_io.BytesIO(anim_bytes))
                            n = getattr(anim, "n_frames", 1)
                            for fi in range(n):
                                anim.seek(fi)
                                out_path = str(output_dir / f"_ad_ep{episode_num}_s{shot_idx}_f{fi:03d}.png")
                                anim.save(out_path, format="PNG")
                                frame_paths.append(out_path)
                                self._temp_files.append(out_path)
                        except Exception as e:  # noqa: BLE001
                            print(f"[comic] 拆帧失败: {e}")
        except Exception as e:  # noqa: BLE001
            print(f"[comic] 下载动画帧失败: {e}")
            return []

        # 按帧序号排序
        frame_paths.sort(key=lambda p: int(p.rsplit("_f", 1)[1].split(".")[0]))
        return frame_paths

    def _resize_frame(self, path: str, target: tuple[int, int]) -> np.ndarray:
        """把单帧图片 cover 裁剪 + 放大到目标尺寸，返回 RGB numpy 数组。"""
        img = Image.open(path).convert("RGB")
        base = self._cover_resize(img, target[0], target[1])
        return np.asarray(base)

    # ----------------------------------------------------------
    # Ken Burns 运镜
    # ----------------------------------------------------------
    def _ken_burns_clip(
        self,
        image_path: str,
        duration: float,
        motion: str = "zoom_in",
        size: tuple[int, int] = (1280, 720),
        fps: int = 24,
    ):
        """
        把静态图变成带运镜的动态 clip。

        原理：图片先放大 1.25 倍，然后一个大小=输出尺寸的"窗口"
        在放大图上随时间移动/缩放，形成推拉/平移的镜头语言。
        纯 numpy + VideoClip 实现，跨 MoviePy 版本稳定。
        """
        from moviepy.editor import VideoClip

        W, H = size
        # 读取并统一尺寸
        img = Image.open(image_path).convert("RGB")
        # cover 裁剪到目标比例，再放大 1.25 倍作为运镜底图
        base = self._cover_resize(img, W, H)
        zf = 1.25
        big = base.resize((int(W * zf), int(H * zf)), Image.LANCZOS)
        big_arr = np.asarray(big)
        big_w, big_h = big_arr.shape[1], big_arr.shape[0]

        def make_frame(t: float) -> np.ndarray:
            p = t / duration if duration > 0 else 0.0
            p = max(0.0, min(1.0, p))

            if motion == "zoom_in":
                # 窗口从全尺寸逐渐缩小（视觉放大）
                win_scale = 1.0 - 0.15 * p
                win_w = int(W * win_scale)
                win_h = int(H * win_scale)
                cx, cy = big_w // 2, big_h // 2
                x1 = max(0, cx - win_w // 2)
                y1 = max(0, cy - win_h // 2)
                crop = big_arr[y1:y1 + win_h, x1:x1 + win_w]
            elif motion == "zoom_out":
                win_scale = 0.85 + 0.15 * p
                win_w = int(W * win_scale)
                win_h = int(H * win_scale)
                cx, cy = big_w // 2, big_h // 2
                x1 = max(0, cx - win_w // 2)
                y1 = max(0, cy - win_h // 2)
                crop = big_arr[y1:y1 + win_h, x1:x1 + win_w]
            elif motion == "pan_right":
                max_off = big_w - W
                x1 = int(max_off * p)
                y1 = (big_h - H) // 2
                crop = big_arr[y1:y1 + H, x1:x1 + W]
            elif motion == "pan_left":
                max_off = big_w - W
                x1 = int(max_off * (1 - p))
                y1 = (big_h - H) // 2
                crop = big_arr[y1:y1 + H, x1:x1 + W]
            else:  # 静态
                x1 = (big_w - W) // 2
                y1 = (big_h - H) // 2
                crop = big_arr[y1:y1 + H, x1:x1 + W]

            # 统一回输出尺寸
            frame = Image.fromarray(crop).resize((W, H), Image.LANCZOS)
            return np.asarray(frame)

        return VideoClip(make_frame, duration=duration)

    @staticmethod
    def _cover_resize(img: Image.Image, W: int, H: int) -> Image.Image:
        """按 cover 模式裁剪到目标比例（不拉伸变形）。"""
        src_w, src_h = img.size
        target_ratio = W / H
        src_ratio = src_w / src_h
        if src_ratio > target_ratio:
            # 原图更宽，裁左右
            new_w = int(src_h * target_ratio)
            x = (src_w - new_w) // 2
            img = img.crop((x, 0, x + new_w, src_h))
        elif src_ratio < target_ratio:
            # 原图更高，裁上下
            new_h = int(src_w / target_ratio)
            y = (src_h - new_h) // 2
            img = img.crop((0, y, src_w, y + new_h))
        return img.resize((W, H), Image.LANCZOS)
