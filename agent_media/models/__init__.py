# -*- coding: utf-8 -*-
"""
models - PyTorch / 深度学习组件（展示 PyTorch 能力）
============================================================
- embeddings.py：基于 sentence-transformers（PyTorch 后端）的本地语义向量
  + 余弦相似度工具，是 RAG 检索的向量化底座；
- classifier.py：内容质量 / 一致性评分分类器（transformers + torch）推理封装，
  可选接入 Reviewer 审查环节做"模型打底 + LLM 修正"的混合审查；
- fine_tune.py：一份可直接运行的 PyTorch 微调示例（含数据格式、训练循环、
  指标、保存加载），用于面试时展示"会写训练代码"。
"""
