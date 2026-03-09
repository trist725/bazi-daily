### 中文

- `main.go`：项目主程序，包含配置、模型调用、清理逻辑、报告输出
- `reports/`：程序运行后生成的 Markdown 报告目录

查看已安装模型：

List installed models:

- `main.go`: main application entry, including config, model calls, cleanup logic, and report generation
- `reports/`: output directory containing generated Markdown reports

# Bazi Daily

一个基于 **Go + Ollama + Gemini** 的八字日课多模型对比工具。  
A **Go + Ollama + Gemini** powered multi-model comparison tool for daily Bazi briefings.

本项目会根据当天日期自动构造日课输入，串行调用多个本地模型生成结果，并支持使用云端 Gemini 或本地模型作为裁判模型，对多模型输出进行总结和对比，最终自动生成 Markdown 报告。

This project automatically builds a daily prompt based on the current date, runs multiple local models sequentially, and supports using Gemini or a local model as a judge model to summarize and compare outputs. Markdown reports are generated automatically at the end.

---

## 项目简介 / Overview

### 中文

`Bazi Daily` 是一个面向个人日课生成与多模型对比的命令行工具。

它的主要目标是：

- 根据当前日期生成日课输入
- 自动发现本地 Ollama 模型
- 串行调用多个本地模型，降低显存冲突风险
- 对多个模型输出进行汇总对比
- 可选接入 Gemini 作为云端裁判模型
- 生成易读的 Markdown 报告

这个项目适合：

- 想测试不同本地模型输出差异的人
- 显存有限、不能并发运行多个模型的人
- 希望保留日报输出和模型对比记录的人

### English

`Bazi Daily` is a command-line tool for generating daily Bazi briefings and comparing results across multiple models.

Its main goals are:

- Build a daily prompt based on the current date
- Automatically discover local Ollama models
- Run local models sequentially to reduce GPU memory conflicts
- Compare and summarize outputs from multiple models
- Optionally use Gemini as a cloud judge model
- Generate readable Markdown reports

This project is suitable for:

- People who want to compare outputs from different local models
- Users with limited GPU memory who cannot run multiple models concurrently
- Anyone who wants to keep daily records and model comparison reports

---

## 核心功能 / Core Features

### 中文

- **日课生成**：根据当前日期生成固定格式输入
- **本地模型自动发现**：自动从 Ollama 读取本地已安装模型
- **本地模型筛选**：根据过滤关键字和排除关键字选择参与比较的模型
- **串行执行**：严格按顺序调用本地模型，避免同时占用显存
- **自动清理残留模型**：执行前检查并清理仍在运行的本地模型
- **自动释放模型**：每个本地模型调用结束后尝试卸载并等待释放
- **失败重试**：本地模型调用失败时自动重试
- **云端模型支持**：支持 Gemini 云端模型
- **裁判模型总结**：支持 Gemini 或本地模型作为裁判输出总结
- **Markdown 报告输出**：自动生成 summary、judge 和单模型报告
- **耗时统计**：输出调用耗时、释放耗时、总耗时

### English

- **Daily prompt generation** based on the current date
- **Automatic local model discovery** from Ollama
- **Local model filtering** using include and exclude keywords
- **Sequential execution** to avoid concurrent GPU memory usage
- **Automatic cleanup** of leftover running local models
- **Automatic unload** after each local model call
- **Retry mechanism** for failed local model calls
- **Cloud model support** via Gemini
- **Judge model summary** using Gemini or a local model
- **Markdown report generation** for summary, judge, and per-model outputs
- **Timing metrics** for calls, release, and total execution

---

