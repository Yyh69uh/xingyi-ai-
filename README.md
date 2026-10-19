---
title: Xingyi AI Tour Guide
emoji: 🏔️
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# 兴义文旅 AI 导游

基于 FastAPI + 智谱 GLM 的兴义文旅智能问答应用（RAG + Function Calling）。

- 前端页面与后端 API 同端口托管（7860）
- 164 条本地知识库（景点/美食/酒店/路线/交通/活动/民俗/人文/游客评价）
- 大模型自主决策检索知识库，回答附来源与核实时间
- 含内容安全红线、五类容错兜底、AI 生成标识等合规设计
