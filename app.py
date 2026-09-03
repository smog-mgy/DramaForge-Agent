# -*- coding: utf-8 -*-
"""
app.py - AI 短剧全栈自动化工厂（新版 Web UI）
============================================================
基于 agent_media 包的 Streamlit 界面：
- Tab 1 单集灵感测试
- Tab 2 长篇连载流水线（多智能体 + RAG + 审查回路 + 语音）
- Tab 3 分镜与角色一致性
运行：streamlit run app.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import streamlit as st

from agent_media.config import get_settings
from agent_media.pipeline import MediaPipeline
from agent_media.prompts import build_writer_input

st.set_page_config(page_title="AI 短剧全栈工厂", page_icon="🎬", layout="wide")
settings = get_settings()

# ---------- 侧边栏 ----------
with st.sidebar:
    st.header("⚙️ 控制台")
    api_key = st.text_input(
        "DeepSeek API Key（也可写进 .env）",
        value=settings.deepseek_api_key,
        type="password",
    )
    if api_key:
        settings.deepseek_api_key = api_key
    st.caption("RAG 后端: 由依赖自动选择 (Chroma / 内存降级)")
    st.caption("Embedding: 本地模型 / API / 词法降级")

st.title("🎬 AI 短剧全栈自动化工厂")
st.caption("多智能体编排 · RAG 设定检索 · Reviewer 审查回路 · PyTorch 本地向量化 · edge-tts 语音")

if not settings.deepseek_api_key:
    st.warning("⚠️ 未配置 API Key：请在左侧输入，或在项目根目录创建 .env（参考 .env.example）。")

tab1, tab2, tab3, tab4 = st.tabs(["✍️ 单集灵感", "📦 长篇连载流水线", "🎨 分镜与角色一致性", "🎞️ 视频渲染车间"])

# ============================================================
# Tab 1：单集灵感测试
# ============================================================
with tab1:
    st.subheader("💡 单集灵感测试")
    idea = st.text_area("输入点子", height=100, key="tab1_idea")
    words = st.slider("目标字数", 300, 2000, 800, 100, key="tab1_words")

    if st.button("🚀 生成单集", use_container_width=True):
        if not settings.deepseek_api_key:
            st.error("请先配置 API Key")
        elif not idea.strip():
            st.warning("点子不能为空")
        else:
            try:
                from agent_media.agents import MediaAgentOrchestrator

                orch = MediaAgentOrchestrator(settings)
                orch.build_bible("这是一个都市爽文短剧：" + idea)
                with st.spinner("主笔创作中..."):
                    # 无细纲时直接按固定任务写一集
                    episode = orch._write_with_review(
                        1,
                        type("O", (), {"core_conflict": "主角觉醒外挂", "hook": "打脸反派"}),  # noqa: E731
                        orch.memory.retrieve_context(idea),
                        words,
                        max_rewrites=1,
                    )
                st.markdown(episode)
            except Exception as e:  # noqa: BLE001
                st.error(f"生成失败：{e}")

# ============================================================
# Tab 2：长篇连载流水线（核心）
# ============================================================
with tab2:
    st.subheader("📦 长篇连载流水线（多智能体 + RAG + 审查回路）")
    mega_idea = st.text_area(
        "核心世界观大纲",
        placeholder="例如：男主穿越到人人用飞剑修仙的世界，发现功法本质是代码，可以用笔记本改写报错来打脸退婚圣女……",
        height=120,
        key="tab2_idea",
    )
    c1, c2 = st.columns(2)
    n_eps = c1.slider("集数", 1, 20, 3, 1, key="tab2_eps")
    ep_words = c2.slider("单集字数", 300, 2000, 800, 100, key="tab2_words")
    with_tts = st.checkbox("同时生成语音 (edge-tts)", value=True, key="tab2_tts")

    if st.button("⚙️ 启动流水线", type="primary", use_container_width=True):
        if not settings.deepseek_api_key:
            st.error("请先配置 API Key")
        elif not mega_idea.strip():
            st.warning("大纲不能为空")
        else:
            try:
                pipeline = MediaPipeline(settings)
                with st.spinner(f"正在生产 {n_eps} 集（含设定集推演 / 细纲 / 正文 / 审查 / 记忆入库）..."):
                    manifest = pipeline.produce_series(
                        mega_idea.strip(), n_eps, ep_words, with_tts=with_tts
                    )
                st.success("✅ 流水线完成！")
                # ---- 保存到 session_state，供 Tab 3 分镜一键填入 ----
                ep_data = []
                for item in manifest["episodes"]:
                    p = settings.output_path / item["text"]
                    content = p.read_text(encoding="utf-8") if p.exists() else ""
                    ep_data.append({
                        "episode": item["episode"],
                        "content": content,
                        "audio": item.get("audio", ""),
                    })
                st.session_state.generated_episodes = ep_data
                # 角色圣经：如果流水线里有设定则一并传递
                chars = getattr(pipeline.orchestrator.memory, "characters", {})
                if chars:
                    st.session_state.character_bible = json.dumps(chars, ensure_ascii=False, indent=2)
                st.info(f"📦 已生成 {len(ep_data)} 集 → 切换到「分镜与角色一致性」Tab 可直接选用")
                st.json(manifest["stats"])
                for item in manifest["episodes"]:
                    st.markdown(f"**第 {item['episode']} 集** — 正文: `{item['text']}`"
                                + (f" / 语音: `{item['audio']}`" if item.get("audio") else ""))
                out = settings.output_path
                if manifest["episodes"]:
                    # 打包下载
                    import io, zipfile

                    buf = io.BytesIO()
                    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
                        for ep in manifest["episodes"]:
                            p = out / ep["text"]
                            if p.exists():
                                zf.writestr(ep["text"], p.read_text(encoding="utf-8"))
                            if ep.get("audio"):
                                ap = out / ep["audio"]
                                if ap.exists():
                                    zf.writestr(ep["audio"], ap.read_bytes())
                    st.download_button(
                        "🎁 下载全部资产包", data=buf.getvalue(),
                        file_name=f"serial_{n_eps}eps.zip", mime="application/zip",
                    )
            except Exception as e:  # noqa: BLE001
                st.error(f"流水线异常：{e}")

# ============================================================
# Tab 3：分镜 + 角色一致性（RAG + 基因锁）
# ============================================================
with tab3:
    st.subheader("🎨 分镜拆解 + 角色视觉基因锁")

    # ---- 如果 Tab 2 流水线已生成剧集，提供一键填入 ----
    if "generated_episodes" in st.session_state and st.session_state.generated_episodes:
        eps = st.session_state.generated_episodes
        col_sel, col_btn = st.columns([3, 1])
        sel_idx = col_sel.selectbox(
            "从流水线结果选择剧集",
            range(len(eps)),
            format_func=lambda i: f"第 {eps[i]['episode']} 集（{len(eps[i]['content'])} 字）",
            key="tab3_ep_sel",
        )
        if col_btn.button("📥 填入剧情", use_container_width=True):
            st.session_state.tab3_story = eps[sel_idx]["content"]
            st.rerun()
        st.caption("选择剧集后点「填入剧情」，文本会自动填充到下方；也可手动编辑。")
    else:
        st.caption("💡 先在「长篇连载流水线」Tab 生成剧集，回来即可一键选用；也可直接手动粘贴剧情。")

    story_text = st.text_area("输入一段剧情（用于拆解分镜）", height=150, key="tab3_story")

    # 角色圣经：优先用流水线传递的，否则用默认示例
    default_chars = st.session_state.get(
        "character_bible",
        '{"男主": "1boy, handsome, black suit", "女主": "1girl, long white hair"}',
    )
    char_json = st.text_area(
        "角色圣经（JSON）",
        value=default_chars,
        height=90,
        key="tab3_chars",
    )
    if st.button("🪄 拆解分镜", use_container_width=True):
        if not settings.deepseek_api_key or not story_text.strip():
            st.error("需要 API Key 和剧情文本")
        else:
            try:
                import json

                bible = json.loads(char_json)
                pipeline = MediaPipeline(settings)
                # 注入角色圣经到知识库
                pipeline.orchestrator.memory.set_characters(bible)
                with st.spinner("分镜师拆解中..."):
                    shots = pipeline.storyboard_episode(story_text)
                for s in shots:
                    with st.expander(f"镜头 {s['id']}：{s['text']}"):
                        st.code(s["prompt"], language="text")
            except Exception as e:  # noqa: BLE001
                st.error(f"拆解失败：{e}")

# ============================================================
# Tab 4：视频渲染车间（图文成片）
# ============================================================
with tab4:
    st.subheader("🎞️ 视频渲染车间（分镜图片 + 语音 → 拼接成片）")
    st.caption("复用 Tab 2 生成的剧集正文，自动分镜 + TTS + 画面生成，拼接成 mp4。优先 ComfyUI 生成画面，无显卡时自动退回文字画面。")

    # 剧集选择
    if "generated_episodes" in st.session_state and st.session_state.generated_episodes:
        eps = st.session_state.generated_episodes
        col_ep, col_info = st.columns([2, 1])
        ep_idx = col_ep.selectbox(
            "选择要渲染的剧集",
            range(len(eps)),
            format_func=lambda i: f"第 {eps[i]['episode']} 集（{len(eps[i]['content'])} 字）",
            key="tab4_ep_sel",
        )
        selected_ep = eps[ep_idx]
        col_info.info(f"已选：第 {selected_ep['episode']} 集\n字数：{len(selected_ep['content'])}")
        episode_text = selected_ep["content"]
        episode_num = selected_ep["episode"]
    else:
        st.warning("⚠️ 尚未生成剧集。请先到「长篇连载流水线」Tab 生成剧集，剧本会自动输送到这里。")
        # 允许手动粘贴
        episode_text = st.text_area("或手动粘贴剧集正文", height=120, key="tab4_manual_text")
        episode_num = st.number_input("集号", min_value=1, value=1, key="tab4_manual_num")

    # 角色圣经（复用 Tab 3 的配置）
    default_chars_tab4 = st.session_state.get(
        "character_bible",
        '{"男主": "1boy, handsome, black suit", "女主": "1girl, long white hair"}',
    )
    char_json_tab4 = st.text_area(
        "角色圣经（JSON，注入画面提示词保证一致性）",
        value=default_chars_tab4,
        height=80,
        key="tab4_chars",
    )

    # 渲染选项
    col_opt0, col_opt1, col_opt2 = st.columns([1, 1, 1])
    render_mode = col_opt0.selectbox(
        "渲染模式",
        ["🎞️ 图文成片", "🎨 动态漫剧（Ken Burns 运镜）", "🤖 角色动态漫剧（AnimateDiff 动画）"],
        key="tab4_mode",
    )
    use_comfyui = col_opt1.checkbox("启用 ComfyUI 生成画面（需本地运行 ComfyUI）", value=settings.render_use_comfyui, key="tab4_comfyui")
    comfyui_url = col_opt2.text_input("ComfyUI 地址", value=settings.comfyui_url, key="tab4_comfyui_url")

    # 启动渲染
    if st.button("🎬 启动视频渲染", type="primary", use_container_width=True):
        if not episode_text.strip():
            st.error("剧集正文不能为空")
        else:
            try:
                import json as _json

                bible = _json.loads(char_json_tab4)
                # 临时覆盖设置
                settings.render_use_comfyui = use_comfyui
                settings.comfyui_url = comfyui_url

                pipeline = MediaPipeline(settings)
                pipeline.orchestrator.memory.set_characters(bible)

                # 先生成分镜
                with st.spinner("分镜师拆解镜头中..."):
                    shots = pipeline.storyboard_episode(episode_text)
                st.success(f"已拆解 {len(shots)} 个镜头")
                for s in shots:
                    with st.expander(f"镜头 {s['id']}：{s['text'][:30]}", expanded=False):
                        st.code(s["prompt"], language="text")

                # 渲染视频
                progress_bar = st.progress(0)
                status_text = st.empty()

                def _progress(cur, total, msg):
                    progress_bar.progress(cur / max(total, 1))
                    status_text.text(msg)

                mode_label = "角色动态漫剧（AnimateDiff 动画 + 配音）" if "AnimateDiff" in render_mode else ("动态漫剧（出图 + 运镜 + 配音）" if "漫剧" in render_mode else "图文成片（TTS + 画面 + 拼接）")
                with st.spinner(f"{mode_label}渲染中..."):
                    if "AnimateDiff" in render_mode:
                        video_path = pipeline.render_animated_comic_episode(
                            episode_text=episode_text,
                            episode_num=episode_num,
                            storyboard=shots,
                            character_bible=bible,
                            progress_cb=_progress,
                        )
                    elif "漫剧" in render_mode:
                        video_path = pipeline.render_comic_episode(
                            episode_text=episode_text,
                            episode_num=episode_num,
                            storyboard=shots,
                            character_bible=bible,
                            progress_cb=_progress,
                        )
                    else:
                        video_path = pipeline.render_episode_video(
                            episode_text=episode_text,
                            episode_num=episode_num,
                            storyboard=shots,
                            character_bible=bible,
                            progress_cb=_progress,
                        )

                progress_bar.progress(1.0)
                status_text.text("✅ 渲染完成！")
                st.success(f"🎬 视频已生成：{video_path}")

                # 展示和下载
                if os.path.exists(video_path):
                    st.video(video_path)
                    with open(video_path, "rb") as f:
                        st.download_button(
                            "💾 下载视频",
                            data=f.read(),
                            file_name=os.path.basename(video_path),
                            mime="video/mp4",
                        )
            except Exception as e:  # noqa: BLE001
                st.error(f"渲染失败：{e}")
