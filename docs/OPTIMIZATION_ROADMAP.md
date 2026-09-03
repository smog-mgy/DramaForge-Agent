# 优化路线图：从「能跑」到「能讲、能投」

> 面向 **AI Agent 实习/校招** 的项目优化记录。
> 目标：不是把项目改得更花哨，而是让每一个面试官要求的能力点
> （LLM / Prompt / LangChain / RAG / PyTorch / Agent 设计）都有
> **明确可指认的代码落点 + 可讲的工程理由**。

---

## 0. 结论先行

**现状**：旧版项目（`phase1_mvp.py` / `web_studio*.py`）是一个**可运行但不可展示**的 demo：

- ✅ 已具备：多角色 Agent 雏形（agno 框架）、提示词有一定约束、能跑通 TTS/ComfyUI/MoviePy 全链路
- ❌ 缺失：LangChain、RAG、PyTorch 三项硬技能完全空白
- ⚠️ 问题：prompt 散落、JSON 解析脆弱、无结构化输出、无配置管理、**API Key 硬编码**、无 README/requirements/git、单文件无架构

**判断**：以"投 AI Agent 实习"为标准，旧版停留在"脚本 demo"，达不到简历上可写的水平。本次重构把缺失的技能点补齐，并把已有能力工程化。

---

## 1. 差距清单（旧 vs 新）

| 维度 | 旧版 | 新版 | 招聘价值 |
| --- | --- | --- | --- |
| LLM 接入 | agno + DeepSeek，写死在文件里 | LangChain + 配置化，模型即插即用 | 讲得出"抽象层" |
| Prompt | 每条写死在函数里、重复 | 集中式角色化模板 + few-shot + 结构化约束 | 讲得出"工程化" |
| LangChain | ❌ 无 | chains / tools / with_structured_output / memory | 硬技能补齐 |
| RAG | ❌ 无（只靠滚动摘要） | 切片→向量化→Chroma→检索→上下文注入 | 硬技能补齐 |
| PyTorch | ❌ 无 | 本地语义向量 + 微调示例 + 分类器 | 硬技能补齐 |
| 多智能体 | 有角色但靠手写拼装 | Architect/Planner/Writer/Reviewer/Summarizer + 审查回路 | 讲得出"编排与自纠错" |
| 输出可靠性 | `json.loads(裸字符串.replace)` 易崩 | Pydantic Schema + with_structured_output | 讲得出"类型安全" |
| 配置/安全 | 硬编码 Key | .env + pydantic-settings | 讲得出"生产习惯" |
| 可复现 | 无依赖清单 | requirements.txt + 冒烟测试 + 文档 | 讲得出"工程规范" |

---

## 2. 落地内容（本仓库已完成）

| # | 交付物 | 说明 |
| --- | --- | --- |
| 1 | `agent_media/` 核心包 | 8 个模块 + PyTorch 子包，见 README 结构 |
| 2 | `app.py` | 基于新包的 Streamlit UI（3 个 Tab） |
| 3 | `tests/test_core.py` | 无需 API Key 的冒烟测试 |
| 4 | `requirements.txt` / `.env.example` / `.gitignore` | 工程规范 |
| 5 | `README.md` | 招聘视角项目叙事（架构图 + 技能落点表） |

## 3. 分阶段建议

- **P0（本次已做）**：补齐技能点代码 + 文档 + 测试，让项目"能讲"。
- **P1（建议尽快）**：本机 `conda create -n media-ai python=3.11` 装依赖，用自己 Key 跑通一遍 `pipeline` 和 `app.py`，把产物和日志截图存进仓库 `output/demo/`。**面试官爱看真实跑出来的东西。**
- **P2（加分）**：按 README Roadmap 接入 LangGraph / FAISS / Agentic RAG / 训练质量模型。
- **P3（投递前）**：写 2~3 页 **Project Write-up**（痛点→方案→结果→量化指标），配架构图与运行截图，GitHub 置顶 README。

## 4. 面试可能的追问与准备

| 追问 | 建议回答要点 |
| --- | --- |
| 为什么用 RAG 而不是把全部历史塞进 prompt？ | token 成本随集数线性增长、上下文窗口有限、检索更精准（可在面试时算：40 集×1000 字 ≈ 8 万 token，远超上下文窗口） |
| 审查回路如何防止模型"自己审自己"通过？ | Reviewer 是独立 Prompt/独立调用；后续可用不同模型或接确定性规则（如字符数、Markdown 符号检测）做硬校验 |
| 降级策略设计？ | RAG 依赖缺失时自动退化为内存检索 + 词法向量，保证演示链路不断 —— 体现"容错优先"的工程观 |
| PyTorch 这块具体做过什么？ | 本地 embedding（讲 BERT 如何把句子映射成向量 + 余弦相似度）、一份可运行的微调脚本（训练循环/指标/保存加载） |

## 5. 已知边界（诚实声明）

- 本次交付代码经**语法校验**通过；端到端跑通需要你的 API Key + Python 3.10+ 环境，我这边无法代替验证。
- 旧版视频渲染（ComfyUI/MoviePy）链路保留在旧脚本中，新版未重写该部分；建议作为 P2 并入。
- `fine_tune.py` 与 `classifier.py` 需要先准备训练数据并跑一遍训练，属增量能力。
