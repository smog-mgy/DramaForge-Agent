# -*- coding: utf-8 -*-
"""
prompts.py - 提示词工程层（Prompt Engineering 技能点）
============================================================
设计要点：
1. 所有提示词集中管理、带版本注释 —— 可读、可复用、可 A/B；
2. 每个角色 = 明确的 Role / Instructions / 输出约束；
3. 使用 Few-shot 示例约束输出格式（尤其 JSON 结构）；
4. 把 RAG 检索到的"设定集/角色圣经/历史剧情"作为上下文
   （Grounded Generation），用结构化的 Prompt 模板拼装，
   而不是把几千字上下文塞进一条 f-string 里。
"""
from __future__ import annotations

from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

# ============================================================
# 角色系统提示词（角色 + 职责 + 输出纪律）
# ============================================================

SYSTEM_PROMPTS: dict[str, str] = {
    # 世界观架构师：把一句话点子扩写成不矛盾的设定集
    "architect": """你是资深世界观架构师。根据用户给出的粗略点子，推演出一套结构庞大、前后不矛盾的长篇设定集。
要求：
- 必须包含：主角（名字/性格/核心外挂机制）、主要反派势力、核心爽点走势、战力成长路径；
- 全局设定必须自洽，为后续几十集连载提供"可检索的设定锚点"；
- 直接输出设定集正文，不要输出标题以外的任何解释。
""",
    # 剧本总监：把设定集切成带钩子的分集细纲矩阵
    "planner": """你是顶级剧本总监。你的任务是把一部长篇切分为"每一集都有独立矛盾 + 结尾悬念钩子"的分集细纲。
要求：
- 每集细纲必须交代：本集核心矛盾、本集要推进的剧情、结尾悬念；
- 钩子必须足够强，能推动观众追下一集；
- 输出严格遵循给定 JSON 结构，字段名与示例完全一致。
""",
    # 主笔：基于细纲 + 前情 + 检索到的设定上下文，写正文
    "writer": """你是高产的连载短剧主笔。你必须在严格遵守设定集的前提下写出单集正文。
写作纪律：
- 严格围绕本集细纲推进，绝不快进、绝不崩人设、绝不违背设定集；
- 充分展开对白、神态、冲突细节，把情绪张力拉到极致；
- 单集字数要尽量贴近目标字数；
- 直接输出正文，严禁任何 Markdown 符号（* # _ ` 等），严禁任何前言后语。
""",
    # 审查官：检查一致性、爽点、语法，返回结构化结论
    "reviewer": """你是严格的短剧内容审查官。你会拿到一集正文、本集细纲、相关设定上下文。
逐项审查：
1) 是否违背设定集 / 与前情矛盾（一致性）；
2) 是否偏离本集细纲任务；
3) 是否出现 Markdown 符号或脏文本；
4) 节奏是否拖沓、悬念是否够强。
以 JSON 输出结论：passed / issues / suggestions。绝不输出 JSON 之外的文字。
""",
    # 摘要器：把一集正文压缩成下一集可用的前情提要
    "summarizer": """你是剧情摘要助手。用一两句话高度概括一段剧情的"最新走向与结尾悬念"，
作为下一集的滚动记忆。只输出摘要本身，不要任何前缀。
""",
    # 分镜师：把剧情拆成可供文生图/文生视频使用的镜头提示词
    "storyboarder": """你是资深分镜导演。把一段剧情拆解为若干核心镜头（5-10个）。
每个镜头输出：镜头 id、画面/情节描述（中文，简洁）、英文渲染提示词（纯英文逗号分隔关键词）。
严格纪律：
- 绝对不要回显、复述或引用输入的剧情原文；
- 渲染提示词必须是纯英文关键词，禁止中文、禁止 Markdown；
- 只输出 JSON 对象，格式：{{"shots": [{{"id": 1, "text": "画面描述", "prompt": "english, keywords, here"}}]}}；
- 不要输出 JSON 之外的任何文字、解释或代码块标记。
""",
}

# ============================================================
# Few-shot 示例（约束输出格式）
# ============================================================

# 分集细纲的示例
PLANNER_EXAMPLES = [
    {
        "input": "设定集：主角陆辰穿越修真界，发现功法本质是代码，可改写报错修正天道。第1集：穿越+觉醒外挂，被圣女退婚打脸。",
        "output": '{"title": "代码修真", "total_episodes": 3, "episodes": [{"episode": 1, "core_conflict": "陆辰穿越觉醒代码外挂，当众被圣女退婚羞辱", "hook": "圣女功法忽然全屏报错，全场哗然"}]}',
    }
]

# 正文（给一段"严禁 Markdown"的正面示例）
WRITER_EXAMPLES = [
    {
        "input": "细纲：第1集觉醒外挂。前情：无。",
        "output": "灵云宗后山，陆辰缓缓睁开眼。脑子里多了一行不断跳动的红色报错——【经脉堵塞：Line 42 SyntaxError】。他还没弄清这行字是什么，山门外便传来一道刺耳的通报：圣女沈清霜，退婚。",
    }
]

# 分镜拆解的示例（关键：给模型一个具体的"输入长文本 → 输出结构化分镜"参照）
STORYBOARDER_EXAMPLES = [
    {
        "input": "剧情：灵云宗后山，陆辰缓缓睁开眼。脑子里多了一行红色报错。山门外传来通报：圣女沈清霜，退婚。陆辰站起身，嘴角勾起一抹冷笑。",
        "output": '{"shots": [{"id": 1, "text": "陆辰在后山睁开眼，脑中浮现红色报错", "prompt": "a young man opening eyes on a mountain, red glowing text floating in front of him, mystical atmosphere, cinematic lighting"}, {"id": 2, "text": "山门外传来圣女退婚的通报", "prompt": "ancient chinese mountain gate, a messenger shouting, distant mountains, dramatic sky"}, {"id": 3, "text": "陆辰站起身，嘴角勾起冷笑", "prompt": "a young man standing up with a cold smirk, close-up shot, dramatic shadows, determined eyes"}]}',
    }
]


def _few_shot_prompt(role_key: str, examples: list[dict], input_key: str) -> ChatPromptTemplate:
    """构造角色专用的 few-shot 提示模板。"""
    system = SystemMessagePromptTemplate.from_template(SYSTEM_PROMPTS[role_key])
    example_prompt = ChatPromptTemplate.from_messages(
        [
            ("human", "{" + input_key + "}"),
            ("ai", "{output}"),
        ]
    )
    few_shot = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=examples,
    )
    return ChatPromptTemplate.from_messages([system, few_shot, ("human", "{" + input_key + "}")])


# ============================================================
# 对外暴露的 Prompt 构造器
# ============================================================

def planner_prompt() -> ChatPromptTemplate:
    """分集细纲 Prompt（含 few-shot 约束 JSON 结构）。"""
    return _few_shot_prompt("planner", PLANNER_EXAMPLES, "input")


def writer_prompt() -> ChatPromptTemplate:
    """单集正文 Prompt（含 few-shot 正面示例）。"""
    return _few_shot_prompt("writer", WRITER_EXAMPLES, "input")


def reviewer_prompt() -> ChatPromptTemplate:
    """审查 Prompt：输入结构化字段，输出 JSON 结论。"""
    return ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(SYSTEM_PROMPTS["reviewer"]),
            HumanMessagePromptTemplate.from_template(
                "本集细纲：\n{outline}\n\n"
                "相关设定/前情上下文：\n{context}\n\n"
                "本集正文：\n{content}"
            ),
        ]
    )


def summarizer_prompt() -> ChatPromptTemplate:
    """摘要 Prompt。"""
    return ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(SYSTEM_PROMPTS["summarizer"]),
            HumanMessagePromptTemplate.from_template("{content}"),
        ]
    )


def architect_prompt() -> ChatPromptTemplate:
    """设定集生成 Prompt。"""
    return ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(SYSTEM_PROMPTS["architect"]),
            HumanMessagePromptTemplate.from_template("核心点子：{idea}"),
        ]
    )


def storyboarder_prompt() -> ChatPromptTemplate:
    """分镜拆解 Prompt（含 few-shot JSON 约束）。"""
    return _few_shot_prompt("storyboarder", STORYBOARDER_EXAMPLES, "input")


# ============================================================
# Grounded Generation：把 RAG 结果注入正文写作
# ============================================================

def build_writer_input(
    episode: int,
    outline: str,
    rolling_summary: str,
    rag_context: str,
    target_words: int,
) -> dict[str, str]:
    """
    组装主笔的输入上下文（这是"检索增强生成"的关键落点）：
    - outline:        本集细纲（任务锚点）
    - rolling_summary: 上一集滚动记忆（连贯性）
    - rag_context:     RAG 从向量库检索到的相关设定/历史片段（一致性）
    - target_words:    目标字数
    """
    return {
        "input": (
            f"当前集数：第 {episode} 集。\n"
            f"本集细纲：{outline}\n"
            f"前情提要：{rolling_summary}\n"
            f"相关设定上下文（务必遵守）：\n{rag_context}\n"
            f"请写出约 {target_words} 字的正文。"
        )
    }
