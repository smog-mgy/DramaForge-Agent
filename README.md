# 🎬 DramaForge Agent — AI 短剧全栈自动化媒体引擎

> 项目原名「Agent 架构全栈自动化媒体引擎」，现整理为 `dramaforge-agent`。
> 一套 **LLM Agent 驱动的短剧生产流水线**：从一句话世界观 → 长篇设定集 → 分集细纲 → 正文 → 审查自纠错 → RAG 记忆 → 语音/视频成片。
> 重点展示 **LLM、Prompt Engineering、LangChain、RAG、PyTorch、多智能体编排** 六项硬技能。

---

## 1. 它解决什么问题

做"几十集 AI 短剧"最大的三个技术痛点，也是本项目的核心命题：

| 痛点 | 工程解法（本仓库落地） |
| --- | --- |
| 剧情跑偏、设定遗忘 | **RAG 记忆库**：设定集/角色圣经/历史剧情入库，每集写作前检索最相关上下文注入提示词 |
| 长篇人物崩坏、不一致 | **多级记忆 + 角色基因锁**：设定集级/角色级/逐集摘要级/向量记忆级四级记忆 |
| 模型输出不可控、质量不稳 | **结构化输出 + Reviewer 审查回路**：Pydantic Schema 约束输出，不合格自动带修改意见重写 |

---

## 2. 系统架构

![系统架构图](docs/architecture.png)

> 上图由 `docs/architecture.html` 渲染（可直接浏览器打开 / 截图复用）。
> 简化版流程：

```
                        ┌─────────────────────────────────────────────┐
  用户点子 idea ───────▶│             MediaPipeline (入口)             │
                        └─────────────────────┬───────────────────────┘
                                              │
        ┌─────────────────────────────────────▼───────────────────────────┐
        │                    多智能体编排 MediaAgentOrchestrator            │
        │                                                                  │
        │   Architect(设定集) → Planner(细纲矩阵) → Writer(正文)            │
        │                                          │  ▲                    │
        │                                          ▼  │ (不通过→带意见重写) │
        │                                       Reviewer(审查)             │
        │                                          │                       │
        │                                          ▼                       │
        │                                     Summarizer(摘要)             │
        └─────────────────────────────────────┬───────────────────────────┘
                                              │
        ┌─────────────────────────────────────▼───────────────────────────┐
        │                 MemoryStore（四级记忆）                          │
        │   Bible ─ 角色圣经 ─ 逐集摘要 ─ 向量记忆(RAG)                    │
        └─────────────────────────────────────┬───────────────────────────┘
                                              ▼
        ┌──────────────────────────────────────────────────────────────────┐
        │   工具层 tools.py : edge-tts 语音 / RAG 检索 / 角色一致性注入     │
        │   产出 : output/episode_NN.txt · .mp3 · manifest.json            │
        └──────────────────────────────────────────────────────────────────┘
```

**技术栈**：Python 3.10+ · LangChain(LangChain-Core/OpenAI/Community/Chroma) · DeepSeek(OpenAI 兼容) · ChromaDB · sentence-transformers(PyTorch) · Streamlit · edge-tts · Pydantic v2

---

## 3. 代码里的落点

| 技能 | 作用 | 代码位置 |
| --- | --- | --- |
| **LLM** | 模型即插即用（DeepSeek 走 OpenAI 兼容协议，换模型只改配置）；温度/超参统一管理 | `agent_media/config.py` · `agent_media/llm.py` |
| **Prompt Engineering** | 角色化 System Prompt 集中管理；Few-shot 约束 JSON 格式；RAG 上下文注入（Grounded Generation）；结构化输出约束 | `agent_media/prompts.py` |
| **LangChain** | ChatPromptTemplate / FewShotChatMessagePromptTemplate / Runnable 链 / with_structured_output / @tool 工具注册 | `agent_media/llm.py` · `prompts.py` · `tools.py` |
| **RAG** | 切片(chunking) → 向量化(本地模型/API/降级) → Chroma 向量库 → top-k 语义检索 → 上下文注入；一致性评测 | `agent_media/rag.py` · `memory.py` |
| **PyTorch** | 本地语义向量（sentence-transformers 后端）；完整微调示例（数据加载/训练循环/指标/保存加载） | `agent_media/models/embeddings.py` · `fine_tune.py` · `classifier.py` |
| **Agent 架构** | 多角色拆分(Architect/Planner/Writer/Reviewer/Summarizer)；Plan-Do-Check-Act 反射回路；结构化交接 | `agent_media/agents.py` · `pipeline.py` |


---

## 4. 快速开始

### 4.1 环境（重要）
当前你本机 Anaconda 是 **Python 3.8.5**，而 LangChain 新栈 / Chroma / 新版本 torch 都需要 **Python ≥ 3.10**。建议新建环境：

```bash
conda create -n media-ai python=3.11 -y
conda activate media-ai
pip install -r requirements.txt
```

> 若暂不安装 `sentence-transformers / torch / chromadb`，程序会自动降级到「内存检索 + 词法向量」，RAG 链路依然可演示（功能受限）。

### 4.2 配置密钥
```bash
copy .env.example .env        # Windows
# 编辑 .env，填入 DEEPSEEK_API_KEY
```

### 4.3 运行
```bash
# 命令行流水线
python -m agent_media.pipeline --idea "男主穿越到修仙世界，发现功法是代码，可以改写报错打脸退婚圣女" --episodes 3 --words 800

# Web UI
streamlit run app.py

# 跑测试（无需 API Key）
python -m pytest tests/ -v
```

产物统一输出到 `output/`，含每集 txt / mp3 与 `manifest.json`（含记忆统计与流水线日志，方便复盘）。

---

## 5. 项目结构

```
dramaforge-agent/
├── agent_media/                  # 核心包（11 个模块）
│   ├── config.py                 # pydantic-settings 集中配置（API Key / 模型 / RAG / 路径）
│   ├── schemas.py                # Pydantic 结构化输出（设定集 / 细纲 / 正文 / 审查 / 角色）
│   ├── prompts.py                # 全部 Prompt 模板（角色化 System + few-shot + RAG 注入）
│   ├── llm.py                    # LangChain ChatOpenAI 封装 DeepSeek + with_structured_output
│   ├── rag.py                    # Chroma 向量库 + 三级降级（API/本地模型/词法向量）+ 关键词检索
│   ├── memory.py                 # 四级记忆（Bible / 角色圣经 / 逐集摘要 / 向量 RAG）
│   ├── tools.py                  # LangChain @tool 工具集（RAG 检索 / 角色查询 / 字数统计）
│   ├── agents.py                 # 五角色多智能体编排 + Reviewer 反射回路
│   ├── pipeline.py               # 端到端流水线 + CLI 入口
│   └── models/                   # PyTorch 本地模型（展示深度学习能力）
│       ├── embeddings.py         #   sentence-transformers 本地语义向量 + 余弦相似度
│       ├── classifier.py         #   内容质量分类器推理封装（transformers + torch）
│       └── fine_tune.py          #   PyTorch 微调示例（LoRA / Trainer，可直接运行）
├── tests/
│   └── test_core.py              # 10 个无 Key 冒烟测试（pytest，离线可跑）
├── docs/
│   ├── architecture.html         # 系统架构图（HTML/SVG，可浏览器打开）
│   ├── architecture.png          # 架构图渲染成品
│   └── OPTIMIZATION_ROADMAP.md  # 优化路线图 + 面试问答（HR/技术面视角）
├── app.py                        # Streamlit Web UI（4 Tab：单集 / 连载 / 分镜 / 渲染）
├── requirements.txt              # 依赖清单（兼容 langchain 0.3 与 1.x）
├── .env.example                  # 密钥配置模板（复制为 .env 填入 Key）
└── .gitignore
```

---
