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
	"strconv"
	"strings"
	"time"

	"github.com/6tail/lunar-go/calendar"
)

type ChatRequest struct {
	Model       string    `json:"model"`
	Messages    []Message `json:"messages"`
	Temperature float64   `json:"temperature,omitempty"`
	Stream      bool      `json:"stream,omitempty"`
}

type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type ChatResponse struct {
	Choices []struct {
		Message Message `json:"message"`
	} `json:"choices"`
	Error *struct {
		Message string `json:"message"`
	} `json:"error,omitempty"`
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

const (
	defaultBaseURL = "http://localhost:11434"
	defaultModel   = "qwen3.5:9b"
	systemPrompt   = "你现在是我的私人能量管理系统。请严格按照我的原局（庚午、癸未、辛卯、戊戌）与今日干支进行推演，输出：核心引动、能量体感预测、今日策略（宜/忌）。"
	judgePrompt    = "你是一个严谨的结果评审助手。请基于多个本地模型对同一问题的输出结果，进行横向比较，并输出：1）整体结论；2）每个模型的优缺点；3）哪个模型最完整；4）哪个模型最稳定；5）推荐最终采用哪个模型及理由。请使用简洁清晰的中文。"
)

func main() {
	now := time.Now()
	baseURL := getEnv("LLM_BASE_URL", defaultBaseURL)

	models, err := resolveModels(baseURL)
	if err != nil {
		fmt.Println("获取模型列表失败:", err)
		return
	}
	if len(models) == 0 {
		fmt.Println("过滤后没有可用的聊天模型。你可以检查本地模型列表，或调整 LLM_MODEL_FILTER / LLM_MODEL_SKIP / LLM_MODEL_LIMIT。")
		return
	}

	fmt.Println("本次将串行调用以下模型：")
	for i, model := range models {
		fmt.Printf("%d. %s\n", i+1, model)
	}

	promptContent := buildPrompt(now)
	results := compareModelsSequentially(baseURL, models, systemPrompt, promptContent)

	judgeResult := JudgeResult{
		Enabled: getEnvBool("LLM_JUDGE_ENABLED", true),
	}
	if judgeResult.Enabled {
		judgeModel := resolveJudgeModel(models)
		fmt.Printf("正在调用裁判模型: %s\n", judgeModel)

		judgeContent, judgeErr := judgeModelResults(baseURL, judgeModel, promptContent, results)
		judgeResult = JudgeResult{
			Model:   judgeModel,
			Content: judgeContent,
			Err:     judgeErr,
			Enabled: true,
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

func compareModelsSequentially(baseURL string, models []string, systemPrompt, userPrompt string) []ModelResult {
	results := make([]ModelResult, 0, len(models))

	for i, modelName := range models {
		fmt.Printf("[%d/%d] 正在调用模型: %s\n", i+1, len(models), modelName)

		content, err := chatWithLocalModel(baseURL, modelName, systemPrompt, userPrompt)
		results = append(results, ModelResult{
			Model:   modelName,
			Content: content,
			Err:     err,
		})

		if err != nil {
			fmt.Printf("[%d/%d] 模型 %s 调用失败: %v\n", i+1, len(models), modelName, err)
			continue
		}

		fmt.Printf("[%d/%d] 模型 %s 调用完成\n", i+1, len(models), modelName)
	}

	return results
}

func chatWithLocalModel(baseURL, modelName, systemPrompt, userPrompt string) (string, error) {
	url := strings.TrimRight(baseURL, "/") + "/v1/chat/completions"

	reqBody := ChatRequest{
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
		Temperature: 0.7,
		Stream:      false,
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
		return "", fmt.Errorf("请求本地模型失败，请检查本地服务是否启动: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", fmt.Errorf("读取响应失败: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("模型服务返回异常状态 %d: %s", resp.StatusCode, string(body))
	}

	var chatResp ChatResponse
	if err := json.Unmarshal(body, &chatResp); err != nil {
		return "", fmt.Errorf("解析响应失败: %w，原始响应: %s", err, string(body))
	}

	if chatResp.Error != nil {
		return "", fmt.Errorf("模型服务错误: %s", chatResp.Error.Message)
	}

	if len(chatResp.Choices) == 0 {
		return "", fmt.Errorf("未获取到模型回复，原始响应: %s", string(body))
	}

	content := strings.TrimSpace(chatResp.Choices[0].Message.Content)
	if content == "" {
		return "", fmt.Errorf("模型返回内容为空")
	}

	return content, nil
}

func judgeModelResults(baseURL, judgeModel, originalPrompt string, results []ModelResult) (string, error) {
	judgeInput := buildJudgeInput(originalPrompt, results)
	return chatWithLocalModel(baseURL, judgeModel, judgePrompt, judgeInput)
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

func resolveJudgeModel(models []string) string {
	judgeModel := strings.TrimSpace(os.Getenv("LLM_JUDGE_MODEL"))
	if judgeModel != "" {
		return judgeModel
	}
	if len(models) > 0 {
		return models[0]
	}
	return defaultModel
}

func resolveModels(baseURL string) ([]string, error) {
	explicitModels := parseCSVEnv("LLM_MODELS")
	if len(explicitModels) > 0 {
		return applyModelSelectionRules(explicitModels), nil
	}

	models, err := fetchInstalledModels(baseURL)
	if err != nil {
		fallback := strings.TrimSpace(os.Getenv("LLM_MODEL"))
		if fallback != "" {
			return applyModelSelectionRules([]string{fallback}), nil
		}
		return nil, err
	}

	if len(models) == 0 {
		fallback := strings.TrimSpace(os.Getenv("LLM_MODEL"))
		if fallback != "" {
			return applyModelSelectionRules([]string{fallback}), nil
		}
		return applyModelSelectionRules([]string{defaultModel}), nil
	}

	return applyModelSelectionRules(models), nil
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

func applyModelSelectionRules(models []string) []string {
	unique := uniqueStrings(models)
	filterKeywords := parseCSVEnv("LLM_MODEL_FILTER")
	skipKeywords := parseCSVEnv("LLM_MODEL_SKIP")
	limit := getEnvInt("LLM_MODEL_LIMIT", 0)

	selected := make([]string, 0, len(unique))
	for _, model := range unique {
		if !isLikelyChatModel(model) {
			continue
		}
		if len(filterKeywords) > 0 && !containsAnyKeyword(model, filterKeywords) {
			continue
		}
		if len(skipKeywords) > 0 && containsAnyKeyword(model, skipKeywords) {
			continue
		}
		selected = append(selected, model)
	}

	sort.Strings(selected)

	if limit > 0 && len(selected) > limit {
		selected = selected[:limit]
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
		if keyword == "" {
			continue
		}
		if strings.Contains(name, strings.ToLower(strings.TrimSpace(keyword))) {
			return true
		}
	}
	return false
}

func parseCSVEnv(key string) []string {
	raw := strings.TrimSpace(os.Getenv(key))
	if raw == "" {
		return nil
	}

	parts := strings.Split(raw, ",")
	result := make([]string, 0, len(parts))
	for _, part := range parts {
		value := strings.TrimSpace(part)
		if value != "" {
			result = append(result, value)
		}
	}
	return result
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

func getEnvInt(key string, fallback int) int {
	raw := strings.TrimSpace(os.Getenv(key))
	if raw == "" {
		return fallback
	}

	value, err := strconv.Atoi(raw)
	if err != nil || value < 0 {
		return fallback
	}
	return value
}

func getEnvBool(key string, fallback bool) bool {
	raw := strings.TrimSpace(strings.ToLower(os.Getenv(key)))
	if raw == "" {
		return fallback
	}

	switch raw {
	case "1", "true", "yes", "y", "on":
		return true
	case "0", "false", "no", "n", "off":
		return false
	default:
		return fallback
	}
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

func getEnv(key, fallback string) string {
	value := strings.TrimSpace(os.Getenv(key))
	if value == "" {
		return fallback
	}
	return value
}
