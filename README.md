# Bazi Daily | 八字日课多模型对比工具

A Go-powered tool that generates daily Bazi briefings by comparing multiple LLMs (Ollama/Gemini) and producing a unified judge report.
基于 Go 的八字日课工具，支持串行对比多个本地 (Ollama) 与云端 (Gemini) 模型，并自动生成深度汇总报告。

---

## 🚀 Quick Start / 快速开始

### 1. Requirements / 环境要求
- **Go 1.20+**
- **Ollama** (Running locally / 本地运行中)
- **Gemini API Key** (Optional / 可选)

### 2. Installation / 安装
```bash
git clone https://github.com/your-repository/bazi-daily.git
cd bazi-daily
go mod tidy
```

### 3. Configuration / 配置说明
项目通过文件系统进行配置，核心文件如下：

| 路径 | 说明 | Description |
| :--- | :--- | :--- |
| `secrets/gemini_api_key.txt` | 放置 Gemini API Key | Place your Gemini API Key here |
| `prompts/system_prompt.txt` | 自定义系统提示词 | Custom system prompt |
| `prompts/judge_prompt.txt`  | 自定义裁判提示词 | Custom judge prompt |

*Note: The program will use built-in prompts if files are missing.*
*注意：若文件不存在，程序将使用内置的默认提示词。*

### 4. Run / 运行
```bash
go run main.go
```
运行结束后，程序会自动：
1. 生成 `reports/` 目录并按时间戳存放报告。
2. 自动在浏览器打开 `final.html` 可视化结论。

---

## 🛠 Features / 核心特性

- **Sequential Execution**: Runs local models one by one to save VRAM.
  **串行执行**：逐个调用本地模型，避免显存爆炸。
- **Auto Cleanup**: Automatically unloads models after use.
  **自动清理**：调用结束后自动释放模型，保持显存清爽。
- **Smart Filtering**: Filters Ollama models by keywords (e.g., skips "embedding" models).
  **智能筛选**：自动识别并过滤 Ollama 中的非对话类模型。
- **Judge Mechanism**: Uses a high-capability model (like Gemini) to summarize all outputs.
  **裁判机制**：由高能力模型对所有结果进行横向对比与整合输出。

---

## 📂 Structure / 目录结构
- `main.go`: Application logic / 主逻辑
- `prompts/`: Prompt templates / 提示词模版
- `secrets/`: API keys and sensitive data / 密钥
- `reports/`: Generated Markdown & HTML reports / 运行报告
