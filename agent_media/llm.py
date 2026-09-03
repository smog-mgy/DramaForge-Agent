# -*- coding: utf-8 -*-
"""
llm.py - LangChain 模型封装层（LangChain 技能点）
============================================================
说明：
- DeepSeek 提供 OpenAI 兼容接口，因此直接复用 langchain-openai 的
  ChatOpenAI，仅改 base_url 与 model —— 这是"模型即插即用"的典型做法；
- 提供 with_structured_output 的辅助函数，把"模型返回 Pydantic 结构"
  这一能力暴露给上层，替代原来脆弱的字符串 JSON 解析；
- 若将来换模型（GPT / Qwen / GLM），只需改 config，无需动业务代码。
"""
from __future__ import annotations

from typing import Any, Type, TypeVar

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from .config import Settings, get_settings

T = TypeVar("T")


def _extract_and_parse_json(raw_text: str, schema: Type[T]) -> T | None:
    """
    从 LLM 原始输出中提取 JSON 并用 Pydantic 解析。
    处理：markdown 代码块、前后多余文字、嵌套大括号。
    成功返回 schema 实例，失败返回 None。
    """
    import json
    import re

    text = raw_text.strip()
    # 去掉 markdown 代码块标记
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    # 策略 1：直接尝试整段解析
    try:
        data = json.loads(text)
        return schema.model_validate(data)
    except Exception:  # noqa: BLE001
        pass

    # 策略 2：找到第一个 { 并匹配对应的 }（处理嵌套）
    start = text.find("{")
    if start >= 0:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start : i + 1]
                    try:
                        data = json.loads(candidate)
                        return schema.model_validate(data)
                    except Exception:  # noqa: BLE001
                        break
    # 策略 3：找第一个 [ 并匹配对应的 ]（数组型 schema）
    start = text.find("[")
    if start >= 0:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "[":
                depth += 1
            elif text[i] == "]":
                depth -= 1
                if depth == 0:
                    candidate = text[start : i + 1]
                    try:
                        data = json.loads(candidate)
                        return schema.model_validate(data)
                    except Exception:  # noqa: BLE001
                        break
    return None


def get_llm(settings: Settings | None = None) -> BaseChatModel:
    """构造 DeepSeek 聊天模型（OpenAI 兼容协议）。"""
    settings = settings or get_settings()
    if not settings.deepseek_api_key:
        raise RuntimeError(
            "未配置 DEEPSEEK_API_KEY，请在项目根目录创建 .env（参考 .env.example）。"
        )
    return ChatOpenAI(
        model=settings.llm_model,
        api_key=settings.deepseek_api_key,
        base_url=settings.llm_base_url,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
        timeout=120,
    )


def invoke_structured(
    llm: BaseChatModel,
    prompt: ChatPromptTemplate,
    schema: Type[T],
    **kwargs: Any,
) -> T:
    """
    以结构化方式调用 LLM，返回类型安全的 Pydantic 对象。

    双模式容错：
    1. 优先 function_calling（把 schema 注册为 tool，强制模型调用，精度最高）；
    2. 若返回 None（模型未发起 tool call）或抛异常，自动降级到 json_mode
       （response_format=json_object + Pydantic 解析）；
    3. 两者均失败才抛出带上下文的 RuntimeError。

    注：DeepSeek 不支持 OpenAI 专有的 json_schema 模式（会 400），故不使用。

    用法示例：
        outline = invoke_structured(llm, planner_prompt(), OutlineMatrix, input=raw)
    """
    # prompt.invoke() 返回 ChatPromptValue；langchain 1.x 的 with_structured_output
    # 链要求传入纯消息列表，否则会报 "Unexpected message type: 'messages'" 强制转换错误。
    prompt_value = prompt.invoke(kwargs)
    messages = prompt_value.to_messages() if hasattr(prompt_value, "to_messages") else prompt_value

    # ---- 模式 1：function_calling（精度最高，但创意写作时模型可能不调 tool 返回 None） ----
    try:
        structured_llm = llm.with_structured_output(schema, method="function_calling")
        result = structured_llm.invoke(messages)
        if result is not None:
            return result  # type: ignore[return-value]
        print(f"[llm] function_calling 返回 None（schema={schema.__name__}），降级 json_mode")
    except Exception as e:  # noqa: BLE001
        print(f"[llm] function_calling 异常（schema={schema.__name__}）：{e}，降级 json_mode")

    # ---- 模式 2：json_mode 降级 ----
    # DeepSeek 要求 prompt 中必须出现 "json" 字样才能用 response_format=json_object，
    # 这里显式追加一条含 JSON 的系统指令，确保合规。
    from langchain_core.messages import SystemMessage

    messages_json = list(messages) + [
        SystemMessage(
            content="请严格以合法 JSON 对象格式回复，不要包含任何额外文字、解释或 markdown 代码块标记。"
        )
    ]
    try:
        structured_llm = llm.with_structured_output(schema, method="json_mode")
        result = structured_llm.invoke(messages_json)
        if result is not None:
            return result  # type: ignore[return-value]
    except Exception as e:  # noqa: BLE001
        print(f"[llm] json_mode 异常（schema={schema.__name__}）：{e}，尝试手动提取")

    # ---- 模式 3：手动兜底（直接调 LLM 拿原始文本，正则提取 JSON，Pydantic 解析） ----
    # 当 function_calling 和 json_mode 都失败时（常见于超长输入导致模型回显原文），
    # 退回到最原始的方式：拿原始输出，从中抠出 JSON 再解析。
    try:
        raw_resp = llm.invoke(messages_json)
        raw_text = getattr(raw_resp, "content", str(raw_resp))
        parsed = _extract_and_parse_json(raw_text, schema)
        if parsed is not None:
            print(f"[llm] 手动提取 JSON 成功（schema={schema.__name__}）")
            return parsed
    except Exception as e:  # noqa: BLE001
        print(f"[llm] 手动提取也失败（schema={schema.__name__}）：{e}")

    raise RuntimeError(
        f"结构化输出解析失败（schema={schema.__name__}，function_calling / json_mode / 手动提取均失败）。\n"
        f"提示：可降低 temperature、缩短输入长度、或检查 prompt 中的输出格式约束。"
    )


def invoke_text(llm: BaseChatModel, prompt: ChatPromptTemplate, **kwargs: Any) -> str:
    """普通文本调用。"""
    chain = prompt | llm
    resp = chain.invoke(kwargs)
    content = getattr(resp, "content", resp)
    return str(content).strip()
