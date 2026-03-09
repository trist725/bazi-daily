
---

## Timing Metrics / 耗时统计

### 中文

程序会输出以下时间信息：

- 启动前清理耗时
- 每个模型调用耗时
- 每个模型释放耗时
- 每个模型总耗时
- 裁判模型耗时
- 整轮任务总耗时

### English

The program prints timing information including:

- Startup cleanup duration
- Per-model call duration
- Per-model release duration
- Per-model total duration
- Judge model duration
- Entire task duration

---

## Typical Workflow / 典型工作流程

### 中文

1. 启动程序
2. 自动清理残留本地模型
3. 获取本地已安装 Ollama 模型
4. 根据过滤规则选择可参与对比的模型
5. 串行调用本地模型
6. 每个模型完成后释放显存
7. 可选调用 Gemini 或本地模型作为裁判
8. 生成 Markdown 报告

### English

1. Start the program
2. Clean up leftover local models
3. Load installed Ollama models
4. Apply model filtering rules
5. Run local models sequentially
6. Release resources after each local model
7. Optionally call Gemini or a local judge model
8. Generate Markdown reports

---

## Troubleshooting / 常见问题

### 1. 本地模型调用超时
### 1. Local model request timeout

**中文：**
- 检查 Ollama 是否启动
- 检查模型是否已安装
- 尝试减少模型数量
- 使用更轻量的模型
- 检查是否有残留模型未释放

**English:**
- Check whether Ollama is running
- Check whether the model is installed
- Reduce the number of models
- Use smaller models
- Check for leftover running models

---

### 2. 某个模型一直无法释放
### 2. A model cannot be unloaded

**中文：**
- 查看当前运行模型：
  ```bash
  curl http://localhost:11434/api/ps
  ```
- 确认是否有其他程序也在使用同一个 Ollama 服务
- 必要时重启 Ollama

**English:**
- Check current running models:
  ```bash
  curl http://localhost:11434/api/ps
  ```
- Make sure no other app is using the same Ollama instance
- Restart Ollama if necessary

---

### 3. Gemini 返回 404
### 3. Gemini returns 404

**中文：**
- 检查模型名是否正确
- 确认当前 API 版本支持该模型
- 优先使用已验证可用的模型名，例如 `gemini-2.5-pro`

**English:**
- Check whether the model name is correct
- Verify that the model is available for the current API version
- Prefer a verified model name such as `gemini-2.5-pro`

---

### 4. Gemini 返回 429
### 4. Gemini returns 429

**中文：**
- 说明当前 API 配额不足
- 检查账户计划和 billing
- 可以先禁用 Gemini，改用本地裁判模型

**English:**
- This means your current API quota is exceeded
- Check your plan and billing settings
- You can temporarily disable Gemini and use a local judge model instead

---

## Recommended Git Ignore / 推荐忽略文件

建议在 `.gitignore` 中至少加入：

```
results/
```

Recommended `.gitignore` entries:
