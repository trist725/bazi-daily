package main

import (
	"bytes"
	"encoding/json"
	"errors"
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

type OllamaGenerateRequest struct {
	Model     string `json:"model"`
	Prompt    string `json:"prompt"`
	Stream    bool   `json:"stream"`
	KeepAlive any    `json:"keep_alive,omitempty"`
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

type OllamaRunningModelsResponse struct {
	Models []OllamaRunningModel `json:"models"`
}

type OllamaRunningModel struct {
	Name string `json:"name"`
}

type GeminiGenerateRequest struct {
	Contents []GeminiContent `json:"contents"`
}

type GeminiContent struct {
	Parts []GeminiPart `json:"parts"`
	Role  string       `json:"role,omitempty"`
}

type GeminiPart struct {
	Text string `json:"text"`
}

type GeminiGenerateResponse struct {
	Candidates []struct {
		Content GeminiContent `json:"content"`
	} `json:"candidates"`
	Error *struct {
		Message string `json:"message"`
	} `json:"error,omitempty"`
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

type CloudModelConfig struct {
	Enabled  bool
	Name     string
	Provider string
	APIKey   string
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

	LocalCallTimeout       time.Duration
	LocalUnloadTimeout     time.Duration
	LocalPreflightTimeout  time.Duration
	LocalRetryCount        int
	LocalSwitchDelay       time.Duration
	CloudCallTimeout       time.Duration
	CloudSwitchDelay       time.Duration
	RunningPollInterval    time.Duration
	StartupCleanupEnabled  bool
	ContinueOnCleanupError bool

	CloudModels []CloudModelConfig
}

var appConfig = Config{
	BaseURL:      "http://localhost:11434",
	SystemPrompt: "你现在是我的私人能量管理系统。请严格按照我的原局（庚午、癸未、辛卯、戊戌）与今日干支进行推演，输出：核心引动、能量体感预测、今日策略（宜/忌）。",
	JudgePrompt:  "你是一个严谨的结果评审助手。请基于多个模型对同一问题的输出结果，进行横向比较，并输出：1）整体结论；2）每个模型的优缺点；3）哪个模型最完整；4）哪个模型最稳定；5）推荐最终采用哪个模型及理由。请使用简洁清晰的中文。",

	JudgeEnabled: true,
	JudgeModel:   "qwen3.5:9b",

	ModelFilter: []string{"qwen", "llama", "gemma", "glm"},
	ModelSkip:   []string{"embed", "embedding", "bge", "rerank", "llava", "vision", "vl", "32b", "72b"},
	ModelLimit:  10,

	LocalCallTimeout:       360 * time.Second,
	LocalUnloadTimeout:     25 * time.Second,
	LocalPreflightTimeout:  20 * time.Second,
	LocalRetryCount:        2,
	LocalSwitchDelay:       3 * time.Second,
	CloudCallTimeout:       120 * time.Second,
	CloudSwitchDelay:       1 * time.Second,
	RunningPollInterval:    700 * time.Millisecond,
	StartupCleanupEnabled:  true,
	ContinueOnCleanupError: false,

	CloudModels: []CloudModelConfig{
		{
			Enabled:  false,
			Name:     "gemini-3.1-pro",
			Provider: "gemini",
			APIKey:   "<YOUR_GEMINI_API_KEY>",
		},
	},
}

func main() {
	now := time.Now()

	if appConfig.StartupCleanupEnabled {
		if err := ensureNoModelsRunning(appConfig.BaseURL, appConfig.LocalPreflightTimeout, appConfig.RunningPollInterval); err != nil {
			fmt.Println("启动前清理运行中模型失败:", err)
			if !appConfig.ContinueOnCleanupError {
				return
			}
		}
	}

	localModels, err := resolveModels(appConfig.BaseURL, appConfig)
	if err != nil {
		fmt.Println("获取本地模型列表失败:", err)
		return
	}

	cloudModels := enabledCloudModels(appConfig.CloudModels)
	if len(localModels) == 0 && len(cloudModels) == 0 {
		fmt.Println("没有可用模型，请检查本地 Ollama 模型或云端配置。")
		return
	}

	allModels := append([]string{}, localModels...)
	allModels = append(allModels, resolveCloudModelNames(appConfig)...)

	fmt.Println("本次将串行调用以下模型：")
	for i, model := range allModels {
		fmt.Printf("%d. %s\n", i+1, model)
	}

	promptContent := buildPrompt(now)
	results := compareAllModelsSequentially(appConfig, localModels, promptContent)

	judgeResult := JudgeResult{Enabled: appConfig.JudgeEnabled}
	if appConfig.JudgeEnabled {
		judgeModel := resolveJudgeModel(localModels, appConfig)
		fmt.Printf("正在调用裁判模型: %s\n", judgeModel)

		judgeContent, judgeErr := judgeModelResults(appConfig, judgeModel, promptContent, results)
		judgeResult = JudgeResult{
			Model:   judgeModel,
			Content: judgeContent,
			Err:     judgeErr,
			Enabled: true,
		}

		if isLocalModel(judgeModel, localModels) {
			if releaseErr := unloadAndWaitAllClear(
				appConfig.BaseURL,
				judgeModel,
				appConfig.LocalUnloadTimeout,
				appConfig.RunningPollInterval,
			); releaseErr != nil {
				fmt.Printf("裁判模型卸载或等待释放失败: %v\n", releaseErr)
			}
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
		return
	}

	fmt.Println("报告已保存到:", reportDir)

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

func compareAllModelsSequentially(cfg Config, localModels []string, userPrompt string) []ModelResult {
	results := make([]ModelResult, 0, len(localModels)+len(cfg.CloudModels))
	total := len(localModels) + len(enabledCloudModels(cfg.CloudModels))
	index := 0

	for _, modelName := range localModels {
		index++

		fmt.Printf("[%d/%d] 准备调用本地模型: %s\n", index, total, modelName)

		if err := ensureNoModelsRunning(cfg.BaseURL, cfg.LocalPreflightTimeout, cfg.RunningPollInterval); err != nil {
			err = fmt.Errorf("调用前清理运行中模型失败: %w", err)
			fmt.Printf("[%d/%d] %v\n", index, total, err)

			results = append(results, ModelResult{
				Model: modelName,
				Err:   err,
			})

			if !cfg.ContinueOnCleanupError {
				break
			}
			continue
		}

		content, err := chatWithOllamaWithRetry(
			cfg.BaseURL,
			modelName,
			cfg.SystemPrompt,
			userPrompt,
			cfg.LocalRetryCount,
			cfg.LocalCallTimeout,
			cfg.LocalUnloadTimeout,
			cfg.RunningPollInterval,
		)

		results = append(results, ModelResult{
			Model:   modelName,
			Content: content,
			Err:     err,
		})

		if err != nil {
			fmt.Printf("[%d/%d] 模型 %s 调用失败: %v\n", index, total, modelName, err)
		} else {
			fmt.Printf("[%d/%d] 模型 %s 调用完成\n", index, total, modelName)
		}

		if releaseErr := unloadAndWaitAllClear(
			cfg.BaseURL,
			modelName,
			cfg.LocalUnloadTimeout,
			cfg.RunningPollInterval,
		); releaseErr != nil {
			fmt.Printf("[%d/%d] 模型 %s 卸载或等待释放失败: %v\n", index, total, modelName, releaseErr)
		} else {
			fmt.Printf("[%d/%d] 模型 %s 已确认释放\n", index, total, modelName)
		}

		time.Sleep(cfg.LocalSwitchDelay)
	}

	for _, cloud := range cfg.CloudModels {
		if !cloud.Enabled {
			continue
		}

		index++
		fmt.Printf("[%d/%d] 正在调用云端模型: %s\n", index, total, cloud.Name)

		content, err := chatWithCloudModel(cloud, cfg.SystemPrompt, userPrompt, cfg.CloudCallTimeout)
		results = append(results, ModelResult{
			Model:   cloud.Name,
			Content: content,
			Err:     err,
		})

		if err != nil {
			fmt.Printf("[%d/%d] 云端模型 %s 调用失败: %v\n", index, total, cloud.Name, err)
		} else {
			fmt.Printf("[%d/%d] 云端模型 %s 调用完成\n", index, total, cloud.Name)
		}

		time.Sleep(cfg.CloudSwitchDelay)
	}

	return results
}

func chatWithOllamaWithRetry(
	baseURL, modelName, systemPrompt, userPrompt string,
	maxAttempts int,
	callTimeout time.Duration,
	unloadTimeout time.Duration,
	pollInterval time.Duration,
) (string, error) {
	if maxAttempts < 1 {
		maxAttempts = 1
	}

	var lastErr error

	for attempt := 1; attempt <= maxAttempts; attempt++ {
		content, err := chatWithOllama(baseURL, modelName, systemPrompt, userPrompt, callTimeout)
		if err == nil {
			return content, nil
		}

		lastErr = err
		fmt.Printf("模型 %s 第 %d/%d 次调用失败: %v\n", modelName, attempt, maxAttempts, err)

		if attempt < maxAttempts {
			_ = unloadAndWaitAllClear(baseURL, modelName, unloadTimeout, pollInterval)
			time.Sleep(4 * time.Second)
		}
	}

	return "", fmt.Errorf("模型 %s 多次调用仍失败: %w", modelName, lastErr)
}

func chatWithOllama(baseURL, modelName, systemPrompt, userPrompt string, timeout time.Duration) (string, error) {
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
		Stream: false,
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

	client := &http.Client{Timeout: timeout}

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
		return "", errors.New(strings.TrimSpace(chatResp.Error))
	}

	content := strings.TrimSpace(chatResp.Message.Content)
	if content == "" {
		return "", fmt.Errorf("模型返回内容为空，原始响应: %s", string(body))
	}

	return content, nil
}

func unloadAndWaitAllClear(baseURL, modelName string, timeout time.Duration, pollInterval time.Duration) error {
	if err := unloadModel(baseURL, modelName); err != nil {
		return err
	}
	if err := waitUntilModelUnloaded(baseURL, modelName, timeout, pollInterval); err != nil {
		return err
	}
	if err := waitUntilNoModelsRunning(baseURL, timeout, pollInterval); err != nil {
		return err
	}
	return nil
}

func ensureNoModelsRunning(baseURL string, timeout time.Duration, pollInterval time.Duration) error {
	const maxCleanupRounds = 3

	var lastModels []string

	for round := 1; round <= maxCleanupRounds; round++ {
		models, err := getRunningModels(baseURL)
		if err != nil {
			return err
		}
		if len(models) == 0 {
			return nil
		}

		lastModels = models
		fmt.Printf("检测到已有运行中模型，准备清理(第 %d/%d 轮): %s\n", round, maxCleanupRounds, strings.Join(models, ", "))

		for _, model := range models {
			if err := unloadModel(baseURL, model); err != nil {
				fmt.Printf("通过 /api/chat 卸载模型失败(%s): %v\n", model, err)
			}
			if err := unloadModelViaGenerate(baseURL, model); err != nil {
				fmt.Printf("通过 /api/generate 卸载模型失败(%s): %v\n", model, err)
			}
		}

		if err := waitUntilNoModelsRunning(baseURL, timeout, pollInterval); err == nil {
			return nil
		}

		time.Sleep(2 * time.Second)
	}

	return fmt.Errorf("多轮清理后仍有模型未释放: %s", strings.Join(lastModels, ", "))
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

	client := &http.Client{Timeout: 60 * time.Second}

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

func unloadModelViaGenerate(baseURL, modelName string) error {
	url := strings.TrimRight(baseURL, "/") + "/api/generate"

	reqBody := OllamaGenerateRequest{
		Model:     modelName,
		Prompt:    "",
		Stream:    false,
		KeepAlive: 0,
	}

	jsonData, err := json.Marshal(reqBody)
	if err != nil {
		return fmt.Errorf("generate 卸载请求编码失败: %w", err)
	}

	req, err := http.NewRequest(http.MethodPost, url, bytes.NewBuffer(jsonData))
	if err != nil {
		return fmt.Errorf("创建 generate 卸载请求失败: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	client := &http.Client{Timeout: 60 * time.Second}

	resp, err := client.Do(req)
	if err != nil {
		return fmt.Errorf("发送 generate 卸载请求失败: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return fmt.Errorf("读取 generate 卸载响应失败: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("generate 卸载失败，状态码 %d: %s", resp.StatusCode, string(body))
	}

	return nil
}

func waitUntilModelUnloaded(baseURL, modelName string, timeout time.Duration, pollInterval time.Duration) error {
	deadline := time.Now().Add(timeout)

	for time.Now().Before(deadline) {
		running, err := isModelRunning(baseURL, modelName)
		if err != nil {
			return err
		}
		if !running {
			return nil
		}
		time.Sleep(pollInterval)
	}

	return fmt.Errorf("等待模型卸载超时: %s", modelName)
}

func waitUntilNoModelsRunning(baseURL string, timeout time.Duration, pollInterval time.Duration) error {
	deadline := time.Now().Add(timeout)

	for time.Now().Before(deadline) {
		models, err := getRunningModels(baseURL)
		if err != nil {
			return err
		}
		if len(models) == 0 {
			return nil
		}
		time.Sleep(pollInterval)
	}

	models, err := getRunningModels(baseURL)
	if err != nil {
		return fmt.Errorf("等待运行中模型清空超时，且查询失败: %w", err)
	}

	return fmt.Errorf("等待运行中模型清空超时，当前仍在运行: %s", strings.Join(models, ", "))
}

func isModelRunning(baseURL, modelName string) (bool, error) {
	models, err := getRunningModels(baseURL)
	if err != nil {
		return false, err
	}

	target := strings.TrimSpace(modelName)
	for _, model := range models {
		if model == target {
			return true, nil
		}
	}

	return false, nil
}

func getRunningModels(baseURL string) ([]string, error) {
	url := strings.TrimRight(baseURL, "/") + "/api/ps"

	client := &http.Client{Timeout: 30 * time.Second}

	resp, err := client.Get(url)
	if err != nil {
		return nil, fmt.Errorf("查询运行中模型失败: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("读取运行中模型响应失败: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("查询运行中模型失败，状态码 %d: %s", resp.StatusCode, string(body))
	}

	var runningResp OllamaRunningModelsResponse
	if err := json.Unmarshal(body, &runningResp); err != nil {
		return nil, fmt.Errorf("解析运行中模型响应失败: %w，原始响应: %s", err, string(body))
	}

	models := make([]string, 0, len(runningResp.Models))
	for _, model := range runningResp.Models {
		name := strings.TrimSpace(model.Name)
		if name != "" {
			models = append(models, name)
		}
	}

	sort.Strings(models)
	return models, nil
}

func chatWithCloudModel(cloud CloudModelConfig, systemPrompt, userPrompt string, timeout time.Duration) (string, error) {
	switch strings.ToLower(strings.TrimSpace(cloud.Provider)) {
	case "gemini":
		return chatWithGemini(cloud, systemPrompt, userPrompt, timeout)
	default:
		return "", fmt.Errorf("不支持的云端提供商: %s", cloud.Provider)
	}
}

func chatWithGemini(cloud CloudModelConfig, systemPrompt, userPrompt string, timeout time.Duration) (string, error) {
	apiKey := strings.TrimSpace(cloud.APIKey)
	if apiKey == "" || strings.Contains(apiKey, "<") {
		return "", fmt.Errorf("Gemini API Key 未配置，请替换代码中的占位符")
	}

	url := fmt.Sprintf(
		"https://generativelanguage.googleapis.com/v1beta/models/%s:generateContent?key=%s",
		cloud.Name,
		apiKey,
	)

	reqBody := GeminiGenerateRequest{
		Contents: []GeminiContent{
			{
				Role: "user",
				Parts: []GeminiPart{
					{
						Text: systemPrompt + "\n\n" + userPrompt,
					},
				},
			},
		},
	}

	jsonData, err := json.Marshal(reqBody)
	if err != nil {
		return "", fmt.Errorf("Gemini 请求编码失败: %w", err)
	}

	req, err := http.NewRequest(http.MethodPost, url, bytes.NewBuffer(jsonData))
	if err != nil {
		return "", fmt.Errorf("Gemini 创建请求失败: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	client := &http.Client{Timeout: timeout}

	resp, err := client.Do(req)
	if err != nil {
		return "", fmt.Errorf("Gemini 请求失败: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", fmt.Errorf("Gemini 读取响应失败: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("Gemini 返回异常状态 %d: %s", resp.StatusCode, string(body))
	}

	var geminiResp GeminiGenerateResponse
	if err := json.Unmarshal(body, &geminiResp); err != nil {
		return "", fmt.Errorf("Gemini 解析响应失败: %w，原始响应: %s", err, string(body))
	}

	if geminiResp.Error != nil {
		return "", fmt.Errorf("Gemini 服务错误: %s", geminiResp.Error.Message)
	}

	if len(geminiResp.Candidates) == 0 || len(geminiResp.Candidates[0].Content.Parts) == 0 {
		return "", fmt.Errorf("Gemini 未返回有效内容，原始响应: %s", string(body))
	}

	content := strings.TrimSpace(geminiResp.Candidates[0].Content.Parts[0].Text)
	if content == "" {
		return "", fmt.Errorf("Gemini 返回内容为空")
	}

	return content, nil
}

func judgeModelResults(cfg Config, judgeModel, originalPrompt string, results []ModelResult) (string, error) {
	judgeInput := buildJudgeInput(originalPrompt, results)

	if isCloudModelName(judgeModel, cfg.CloudModels) {
		cloud, ok := findCloudModelConfig(judgeModel, cfg.CloudModels)
		if !ok {
			return "", fmt.Errorf("未找到裁判云端模型配置: %s", judgeModel)
		}
		return chatWithCloudModel(cloud, cfg.JudgePrompt, judgeInput, cfg.CloudCallTimeout)
	}

	return chatWithOllama(cfg.BaseURL, judgeModel, cfg.JudgePrompt, judgeInput, cfg.LocalCallTimeout)
}

func buildJudgeInput(originalPrompt string, results []ModelResult) string {
	var builder strings.Builder

	writeString(&builder, "以下是同一个问题的多模型输出结果，请你做对比评审。\n\n")
	writeString(&builder, "【原始问题】\n")
	writeString(&builder, originalPrompt)
	writeString(&builder, "\n\n")

	for _, result := range results {
		writeString(&builder, "【模型】")
		writeString(&builder, result.Model)
		writeString(&builder, "\n")

		if result.Err != nil {
			writeString(&builder, "【状态】失败\n")
			writeString(&builder, "【错误】")
			writeString(&builder, result.Err.Error())
			writeString(&builder, "\n\n")
			continue
		}

		writeString(&builder, "【状态】成功\n")
		writeString(&builder, "【输出】\n")
		writeString(&builder, result.Content)
		writeString(&builder, "\n\n")
	}

	writeString(&builder, "请基于以上结果给出最终评审结论。")
	return builder.String()
}

func resolveJudgeModel(localModels []string, cfg Config) string {
	if strings.TrimSpace(cfg.JudgeModel) != "" {
		return cfg.JudgeModel
	}
	if len(localModels) > 0 {
		return localModels[0]
	}
	for _, cloud := range cfg.CloudModels {
		if cloud.Enabled && strings.TrimSpace(cloud.Name) != "" {
			return cloud.Name
		}
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

	client := &http.Client{Timeout: 30 * time.Second}

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
		if name != "" {
			modelSet[name] = struct{}{}
		}
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

func resolveCloudModelNames(cfg Config) []string {
	names := make([]string, 0, len(cfg.CloudModels))
	for _, cloud := range cfg.CloudModels {
		if cloud.Enabled && strings.TrimSpace(cloud.Name) != "" {
			names = append(names, cloud.Name)
		}
	}
	return names
}

func enabledCloudModels(models []CloudModelConfig) []CloudModelConfig {
	result := make([]CloudModelConfig, 0, len(models))
	for _, model := range models {
		if model.Enabled {
			result = append(result, model)
		}
	}
	return result
}

func isLocalModel(model string, localModels []string) bool {
	for _, local := range localModels {
		if local == model {
			return true
		}
	}
	return false
}

func isCloudModelName(model string, cloudModels []CloudModelConfig) bool {
	for _, cloud := range cloudModels {
		if cloud.Enabled && cloud.Name == model {
			return true
		}
	}
	return false
}

func findCloudModelConfig(model string, cloudModels []CloudModelConfig) (CloudModelConfig, bool) {
	for _, cloud := range cloudModels {
		if cloud.Enabled && cloud.Name == model {
			return cloud, true
		}
	}
	return CloudModelConfig{}, false
}

func saveComparisonReports(t time.Time, prompt string, results []ModelResult, judgeResult JudgeResult) (string, error) {
	reportDir := filepath.Join("reports", t.Format("2006-01-02_15-04-05"))
	if err := os.MkdirAll(reportDir, 0755); err != nil {
		return "", fmt.Errorf("创建报告目录失败: %w", err)
	}

	for _, result := range results {
		filename := sanitizeFileName(result.Model) + ".md"
		path := filepath.Join(reportDir, filename)

		var content string
		if result.Err != nil {
			content = fmt.Sprintf(
				"# 模型报告\n\n- 模型：`%s`\n- 生成时间：`%s`\n- 状态：失败\n- 错误：`%v`\n",
				result.Model,
				t.Format("2006-01-02 15:04:05"),
				result.Err,
			)
		} else {
			content = fmt.Sprintf(
				"# 模型报告\n\n- 模型：`%s`\n- 生成时间：`%s`\n- 状态：成功\n\n## 输出内容\n\n```text\n%s\n```\n",
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
	summaryPath := filepath.Join(reportDir, "summary.md")
	if err := os.WriteFile(summaryPath, []byte(summary), 0644); err != nil {
		return "", fmt.Errorf("保存汇总报告失败: %w", err)
	}

	if judgeResult.Enabled {
		judgePath := filepath.Join(reportDir, "judge.md")
		judgeContent := buildJudgeReport(t, judgeResult)
		if err := os.WriteFile(judgePath, []byte(judgeContent), 0644); err != nil {
			return "", fmt.Errorf("保存裁判报告失败: %w", err)
		}
	}

	return reportDir, nil
}

func buildSummaryReport(t time.Time, prompt string, results []ModelResult, judgeResult JudgeResult) string {
	var builder strings.Builder

	writeString(&builder, "# 多模型对比汇总报告\n\n")
	writeString(&builder, fmt.Sprintf("- 生成时间：`%s`\n", t.Format("2006-01-02 15:04:05")))
	writeString(&builder, fmt.Sprintf("- 请求内容：`%s`\n\n", prompt))

	writeString(&builder, "## 参与对比的模型结果\n\n")
	for _, result := range results {
		writeString(&builder, fmt.Sprintf("## 模型：`%s`\n\n", result.Model))
		if result.Err != nil {
			writeString(&builder, "- 状态：失败\n")
			writeString(&builder, fmt.Sprintf("- 错误：`%v`\n\n", result.Err))
			continue
		}

		writeString(&builder, "- 状态：成功\n\n")
		writeString(&builder, "### 输出内容\n\n")
		writeString(&builder, "```text\n")
		writeString(&builder, result.Content)
		if !strings.HasSuffix(result.Content, "\n") {
			writeString(&builder, "\n")
		}
		writeString(&builder, "```\n\n")
	}

	if judgeResult.Enabled {
		writeString(&builder, "## 裁判模型总结\n\n")
		writeString(&builder, fmt.Sprintf("- 裁判模型：`%s`\n", judgeResult.Model))
		if judgeResult.Err != nil {
			writeString(&builder, "- 状态：失败\n")
			writeString(&builder, fmt.Sprintf("- 错误：`%v`\n\n", judgeResult.Err))
		} else {
			writeString(&builder, "- 状态：成功\n\n")
			writeString(&builder, "### 裁判结论\n\n")
			writeString(&builder, "```text\n")
			writeString(&builder, judgeResult.Content)
			if !strings.HasSuffix(judgeResult.Content, "\n") {
				writeString(&builder, "\n")
			}
			writeString(&builder, "```\n")
		}
	}

	return builder.String()
}

func buildJudgeReport(t time.Time, judgeResult JudgeResult) string {
	var builder strings.Builder

	writeString(&builder, "# 裁判模型报告\n\n")
	writeString(&builder, fmt.Sprintf("- 生成时间：`%s`\n", t.Format("2006-01-02 15:04:05")))
	writeString(&builder, fmt.Sprintf("- 裁判模型：`%s`\n\n", judgeResult.Model))

	if judgeResult.Err != nil {
		writeString(&builder, "- 状态：失败\n")
		writeString(&builder, fmt.Sprintf("- 错误：`%v`\n", judgeResult.Err))
		return builder.String()
	}

	writeString(&builder, "- 状态：成功\n\n")
	writeString(&builder, "## 裁判结论\n\n")
	writeString(&builder, "```text\n")
	writeString(&builder, judgeResult.Content)
	if !strings.HasSuffix(judgeResult.Content, "\n") {
		writeString(&builder, "\n")
	}
	writeString(&builder, "```\n")

	return builder.String()
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

func writeString(builder *strings.Builder, s string) {
	_, _ = builder.WriteString(s)
}
