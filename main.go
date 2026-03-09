package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/6tail/lunar-go/calendar"
)

type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type OllamaChatRequest struct {
	Model     string    `json:"model"`
	Messages  []Message `json:"messages"`
	Stream    bool      `json:"stream"`
	KeepAlive any       `json:"keep_alive,omitempty"`
}

type OllamaChatResponse struct {
	Message Message `json:"message"`
	Error   string  `json:"error,omitempty"`
}

type OllamaTagsResponse struct {
	Models []OllamaModel `json:"models"`
}

type OllamaModel struct {
	Name string `json:"name"`
}

type ModelResult struct {
	Model   string
	Content string
	Err     error
}

type JudgeResult struct {
	Model   string
	Content string
	Err     error
	Enabled bool
}

type Config struct {
	BaseURL      string
	SystemPrompt string
	JudgePrompt  string

	JudgeEnabled bool
	JudgeModel   string

	ModelFilter []string
	ModelSkip   []string
	ModelLimit  int
}

var appConfig = Config{
	BaseURL:      "http://localhost:11434",
	SystemPrompt: "你现在是我的私人能量管理系统。请严格按照我的原局（庚午、癸未、辛卯、戊戌）与今日干支进行推演，输出：核心引动、能量体感预测、今日策略（宜/忌）。",
	JudgePrompt:  "你是一个严谨的结果评审助手。请基于多个本地模型对同一问题的输出结果，进行横向比较，并输出：1）整体结论；2）每个模型的优缺点；3）哪个模型最完整；4）哪个模型最稳定；5）推荐最终采用哪个模型及理由。请使用简洁清晰的中文。",
	JudgeEnabled: true,
	JudgeModel:   "qwen3.5:9b",
	ModelFilter:  []string{"qwen", "llama", "gemma", "glm"},
	ModelSkip:    []string{"embed", "rerank", "llava", "vision", "32b", "72b"},
	ModelLimit:   10,
}

func main() {
	now := time.Now()

	models, err := resolveModels(appConfig.BaseURL, appConfig)
	if err != nil {
		fmt.Println("获取模型列表失败:", err)
		return
	}
	if len(models) == 0 {
		fmt.Println("过滤后没有可用模型，请检查本地 Ollama 模型或调整代码中的配置。")
		return
	}

	fmt.Println("本次将串行调用以下模型：")
	for i, model := range models {
		fmt.Printf("%d. %s\n", i+1, model)
	}

	promptContent := buildPrompt(now)
	results := compareModelsSequentially(appConfig, models, promptContent)

	judgeResult := JudgeResult{Enabled: appConfig.JudgeEnabled}
	if appConfig.JudgeEnabled {
		judgeModel := resolveJudgeModel(models, appConfig)
		fmt.Printf("正在调用裁判模型: %s\n", judgeModel)

		judgeContent, judgeErr := judgeModelResults(appConfig, judgeModel, promptContent, results)
		judgeResult = JudgeResult{
			Model:   judgeModel,
			Content: judgeContent,
			Err:     judgeErr,
			Enabled: true,
		}

		if releaseErr := unloadModel(appConfig.BaseURL, judgeModel); releaseErr != nil {
			fmt.Printf("裁判模型卸载失败: %v\n", releaseErr)
		}

		if judgeErr != nil {
			fmt.Println("裁判模型调用失败:", judgeErr)
		} else {
			fmt.Println("裁判模型调用完成")
		}
	}

	reportDir, err := saveComparisonReports(now, promptContent, results, judgeResult)
	if err != nil {
		fmt.Println("保存报告失败:", err)
	} else {
		fmt.Println("报告已保存到:", reportDir)
	}

	fmt.Println("========== 多模型结果对比 ==========")
	for _, result := range results {
		fmt.Printf("\n----- 模型: %s -----\n", result.Model)
		if result.Err != nil {
			fmt.Println("调用失败:", result.Err)
			continue
		}
		fmt.Println(result.Content)
	}

	if judgeResult.Enabled {
		fmt.Println("\n========== 裁判模型总结 ==========")
		fmt.Printf("裁判模型: %s\n\n", judgeResult.Model)
		if judgeResult.Err != nil {
			fmt.Println("调用失败:", judgeResult.Err)
		} else {
			fmt.Println(judgeResult.Content)
		}
		fmt.Println("===================================")
	}

	fmt.Println("\n===================================")
}

func buildPrompt(t time.Time) string {
	d := calendar.NewLunarFromDate(t)

	yearGanzhi := d.GetYearInGanZhi()
	monthGanzhi := d.GetMonthInGanZhi()
	dayGanzhi := d.GetDayInGanZhi()

	currentDate := t.Format("2006年01月02日")
	return fmt.Sprintf("%s，%s年%s月%s日", currentDate, yearGanzhi, monthGanzhi, dayGanzhi)
}

func compareModelsSequentially(cfg Config, models []string, userPrompt string) []ModelResult {
	results := make([]ModelResult, 0, len(models))

	for i, modelName := range models {
		fmt.Printf("[%d/%d] 正在调用模型: %s\n", i+1, len(models), modelName)

		content, err := chatWithOllama(cfg.BaseURL, modelName, cfg.SystemPrompt, userPrompt)
		results = append(results, ModelResult{
			Model:   modelName,
			Content: content,
			Err:     err,
		})

		if releaseErr := unloadModel(cfg.BaseURL, modelName); releaseErr != nil {
			fmt.Printf("[%d/%d] 模型 %s 卸载失败: %v\n", i+1, len(models), modelName, releaseErr)
		} else {
			fmt.Printf("[%d/%d] 模型 %s 已请求卸载\n", i+1, len(models), modelName)
		}

		if err != nil {
			fmt.Printf("[%d/%d] 模型 %s 调用失败: %v\n", i+1, len(models), modelName, err)
			continue
		}

		fmt.Printf("[%d/%d] 模型 %s 调用完成\n", i+1, len(models), modelName)
		time.Sleep(2 * time.Second)
	}

	return results
}

func chatWithOllama(baseURL, modelName, systemPrompt, userPrompt string) (string, error) {
	url := strings.TrimRight(baseURL, "/") + "/api/chat"

	reqBody := OllamaChatRequest{
		Model: modelName,
		Messages: []Message{
			{
				Role:    "system",
				Content: systemPrompt,
			},
			{
				Role:    "user",
				Content: userPrompt,
			},
		},
		Stream:    false,
		KeepAlive: "0s",
	}

	jsonData, err := json.Marshal(reqBody)
	if err != nil {
		return "", fmt.Errorf("JSON 编码失败: %w", err)
	}

	req, err := http.NewRequest(http.MethodPost, url, bytes.NewBuffer(jsonData))
	if err != nil {
		return "", fmt.Errorf("创建请求失败: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	client := &http.Client{
		Timeout: 300 * time.Second,
	}

	resp, err := client.Do(req)
	if err != nil {
		return "", fmt.Errorf("请求本地模型失败，请检查 Ollama 是否启动: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", fmt.Errorf("读取响应失败: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("模型服务返回异常状态 %d: %s", resp.StatusCode, string(body))
	}

	var chatResp OllamaChatResponse
	if err := json.Unmarshal(body, &chatResp); err != nil {
		return "", fmt.Errorf("解析响应失败: %w，原始响应: %s", err, string(body))
	}

	if strings.TrimSpace(chatResp.Error) != "" {
		return "", fmt.Errorf("模型服务错误: %s", chatResp.Error)
	}

	content := strings.TrimSpace(chatResp.Message.Content)
	if content == "" {
		return "", fmt.Errorf("模型返回内容为空，原始响应: %s", string(body))
	}

	return content, nil
}

func unloadModel(baseURL, modelName string) error {
	url := strings.TrimRight(baseURL, "/") + "/api/chat"

	reqBody := OllamaChatRequest{
		Model:     modelName,
		Messages:  []Message{},
		Stream:    false,
		KeepAlive: 0,
	}

	jsonData, err := json.Marshal(reqBody)
	if err != nil {
		return fmt.Errorf("卸载请求编码失败: %w", err)
	}

	req, err := http.NewRequest(http.MethodPost, url, bytes.NewBuffer(jsonData))
	if err != nil {
		return fmt.Errorf("创建卸载请求失败: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	client := &http.Client{
		Timeout: 60 * time.Second,
	}

	resp, err := client.Do(req)
	if err != nil {
		return fmt.Errorf("发送卸载请求失败: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return fmt.Errorf("读取卸载响应失败: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("卸载失败，状态码 %d: %s", resp.StatusCode, string(body))
	}

	return nil
}

func judgeModelResults(cfg Config, judgeModel, originalPrompt string, results []ModelResult) (string, error) {
	judgeInput := buildJudgeInput(originalPrompt, results)
	return chatWithOllama(cfg.BaseURL, judgeModel, cfg.JudgePrompt, judgeInput)
}

func buildJudgeInput(originalPrompt string, results []ModelResult) string {
	var builder strings.Builder

	builder.WriteString("以下是同一个问题的多模型输出结果，请你做对比评审。\n\n")
	builder.WriteString("【原始问题】\n")
	builder.WriteString(originalPrompt)
	builder.WriteString("\n\n")

	for _, result := range results {
		builder.WriteString("【模型】")
		builder.WriteString(result.Model)
		builder.WriteString("\n")

		if result.Err != nil {
			builder.WriteString("【状态】失败\n")
			builder.WriteString("【错误】")
			builder.WriteString(result.Err.Error())
			builder.WriteString("\n\n")
			continue
		}

		builder.WriteString("【状态】成功\n")
		builder.WriteString("【输出】\n")
		builder.WriteString(result.Content)
		builder.WriteString("\n\n")
	}

	builder.WriteString("请基于以上结果给出最终评审结论。")
	return builder.String()
}

func resolveJudgeModel(models []string, cfg Config) string {
	if strings.TrimSpace(cfg.JudgeModel) != "" {
		return cfg.JudgeModel
	}
	if len(models) > 0 {
		return models[0]
	}
	return "qwen2.5:7b"
}

func resolveModels(baseURL string, cfg Config) ([]string, error) {
	models, err := fetchInstalledModels(baseURL)
	if err != nil {
		return nil, err
	}
	return applyModelSelectionRules(models, cfg), nil
}

func fetchInstalledModels(baseURL string) ([]string, error) {
	url := strings.TrimRight(baseURL, "/") + "/api/tags"

	client := &http.Client{
		Timeout: 30 * time.Second,
	}

	resp, err := client.Get(url)
	if err != nil {
		return nil, fmt.Errorf("请求 Ollama 模型列表失败: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("读取模型列表失败: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("获取模型列表失败，状态码 %d: %s", resp.StatusCode, string(body))
	}

	var tagsResp OllamaTagsResponse
	if err := json.Unmarshal(body, &tagsResp); err != nil {
		return nil, fmt.Errorf("解析模型列表失败: %w，原始响应: %s", err, string(body))
	}

	modelSet := make(map[string]struct{})
	for _, model := range tagsResp.Models {
		name := strings.TrimSpace(model.Name)
		if name == "" {
			continue
		}
		modelSet[name] = struct{}{}
	}

	models := make([]string, 0, len(modelSet))
	for name := range modelSet {
		models = append(models, name)
	}
	sort.Strings(models)

	return models, nil
}

func applyModelSelectionRules(models []string, cfg Config) []string {
	selected := make([]string, 0, len(models))

	for _, model := range uniqueStrings(models) {
		if !isLikelyChatModel(model) {
			continue
		}
		if len(cfg.ModelFilter) > 0 && !containsAnyKeyword(model, cfg.ModelFilter) {
			continue
		}
		if len(cfg.ModelSkip) > 0 && containsAnyKeyword(model, cfg.ModelSkip) {
			continue
		}
		selected = append(selected, model)
	}

	sort.Strings(selected)

	if cfg.ModelLimit > 0 && len(selected) > cfg.ModelLimit {
		selected = selected[:cfg.ModelLimit]
	}

	return selected
}

func isLikelyChatModel(modelName string) bool {
	name := strings.ToLower(strings.TrimSpace(modelName))
	if name == "" {
		return false
	}

	blockedKeywords := []string{
		"embed",
		"embedding",
		"bge",
		"rerank",
		"reranker",
		"vision",
		"vl",
		"llava",
		"minicpm-v",
		"moondream",
		"clip",
		"whisper",
		"asr",
		"tts",
		"stable-diffusion",
		"sdxl",
	}

	for _, keyword := range blockedKeywords {
		if strings.Contains(name, keyword) {
			return false
		}
	}

	return true
}

func containsAnyKeyword(modelName string, keywords []string) bool {
	name := strings.ToLower(strings.TrimSpace(modelName))
	for _, keyword := range keywords {
		k := strings.ToLower(strings.TrimSpace(keyword))
		if k != "" && strings.Contains(name, k) {
			return true
		}
	}
	return false
}

func uniqueStrings(values []string) []string {
	seen := make(map[string]struct{}, len(values))
	result := make([]string, 0, len(values))

	for _, value := range values {
		v := strings.TrimSpace(value)
		if v == "" {
			continue
		}
		if _, ok := seen[v]; ok {
			continue
		}
		seen[v] = struct{}{}
		result = append(result, v)
	}

	return result
}

func saveComparisonReports(t time.Time, prompt string, results []ModelResult, judgeResult JudgeResult) (string, error) {
	reportDir := filepath.Join("reports", t.Format("2006-01-02_15-04-05"))
	if err := os.MkdirAll(reportDir, 0755); err != nil {
		return "", fmt.Errorf("创建报告目录失败: %w", err)
	}

	for _, result := range results {
		filename := sanitizeFileName(result.Model) + ".txt"
		path := filepath.Join(reportDir, filename)

		var content string
		if result.Err != nil {
			content = fmt.Sprintf(
				"模型：%s\n生成时间：%s\n状态：失败\n错误：%v\n",
				result.Model,
				t.Format("2006-01-02 15:04:05"),
				result.Err,
			)
		} else {
			content = fmt.Sprintf(
				"模型：%s\n生成时间：%s\n状态：成功\n\n%s\n",
				result.Model,
				t.Format("2006-01-02 15:04:05"),
				result.Content,
			)
		}

		if err := os.WriteFile(path, []byte(content), 0644); err != nil {
			return "", fmt.Errorf("保存模型报告失败(%s): %w", result.Model, err)
		}
	}

	summary := buildSummaryReport(t, prompt, results, judgeResult)
	summaryPath := filepath.Join(reportDir, "summary.txt")
	if err := os.WriteFile(summaryPath, []byte(summary), 0644); err != nil {
		return "", fmt.Errorf("保存汇总报告失败: %w", err)
	}

	if judgeResult.Enabled {
		judgePath := filepath.Join(reportDir, "judge.txt")
		judgeContent := buildJudgeReport(t, judgeResult)
		if err := os.WriteFile(judgePath, []byte(judgeContent), 0644); err != nil {
			return "", fmt.Errorf("保存裁判报告失败: %w", err)
		}
	}

	return reportDir, nil
}

func buildSummaryReport(t time.Time, prompt string, results []ModelResult, judgeResult JudgeResult) string {
	var builder strings.Builder

	builder.WriteString("多模型对比汇总报告\n")
	builder.WriteString("====================\n")
	builder.WriteString(fmt.Sprintf("生成时间：%s\n", t.Format("2006-01-02 15:04:05")))
	builder.WriteString(fmt.Sprintf("请求内容：%s\n\n", prompt))

	for _, result := range results {
		builder.WriteString(fmt.Sprintf("----- 模型：%s -----\n", result.Model))
		if result.Err != nil {
			builder.WriteString(fmt.Sprintf("状态：失败\n错误：%v\n\n", result.Err))
			continue
		}
		builder.WriteString("状态：成功\n")
		builder.WriteString(result.Content)
		builder.WriteString("\n\n")
	}

	if judgeResult.Enabled {
		builder.WriteString("裁判模型总结\n")
		builder.WriteString("====================\n")
		builder.WriteString(fmt.Sprintf("裁判模型：%s\n", judgeResult.Model))
		if judgeResult.Err != nil {
			builder.WriteString(fmt.Sprintf("状态：失败\n错误：%v\n", judgeResult.Err))
		} else {
			builder.WriteString("状态：成功\n")
			builder.WriteString(judgeResult.Content)
			builder.WriteString("\n")
		}
	}

	return builder.String()
}

func buildJudgeReport(t time.Time, judgeResult JudgeResult) string {
	if judgeResult.Err != nil {
		return fmt.Sprintf(
			"裁判模型：%s\n生成时间：%s\n状态：失败\n错误：%v\n",
			judgeResult.Model,
			t.Format("2006-01-02 15:04:05"),
			judgeResult.Err,
		)
	}

	return fmt.Sprintf(
		"裁判模型：%s\n生成时间：%s\n状态：成功\n\n%s\n",
		judgeResult.Model,
		t.Format("2006-01-02 15:04:05"),
		judgeResult.Content,
	)
}

func sanitizeFileName(name string) string {
	replacer := strings.NewReplacer(
		"\\", "_",
		"/", "_",
		":", "_",
		"*", "_",
		"?", "_",
		"\"", "_",
		"<", "_",
		">", "_",
		"|", "_",
		" ", "_",
	)
	return replacer.Replace(name)
}
