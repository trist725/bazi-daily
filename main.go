package main

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"sort"
	"strconv"
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

type OllamaGenerateRequest struct {
	Model     string `json:"model"`
	Prompt    string `json:"prompt"`
	Stream    bool   `json:"stream"`
	KeepAlive any    `json:"keep_alive,omitempty"`
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
	Model         string
	Content       string
	Err           error
	Provider      string
	CallDuration  time.Duration
	TotalDuration time.Duration
}

type JudgeResult struct {
	Model           string
	Content         string
	Err             error
	Enabled         bool
	Provider        string
	CallDuration    time.Duration
	ReleaseDuration time.Duration
	TotalDuration   time.Duration
}

type CloudModelConfig struct {
	Enabled    bool
	Name       string
	Provider   string
	APIKey     string
	APIKeyFile string
}

type Config struct {
	BaseURL          string
	SystemPrompt     string
	JudgePrompt      string
	SystemPromptFile string
	JudgePromptFile  string

	JudgeEnabled  bool
	JudgeModel    string
	JudgeProvider string

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

	LocalReleaseRetryCount int
	LocalReleaseRetryDelay time.Duration

	CloudModels []CloudModelConfig
}

var appConfig = Config{
	BaseURL: "http://localhost:11434",

	SystemPrompt: "你现在是我的私人能量管理系统。请严格按照我的原局（庚午、癸未、辛卯、戊戌）与今日干支进行推演，输出：核心引动、能量体感预测、今日策略（宜/忌）。",
	JudgePrompt:  "你是一个严谨的最终结论整合助手。请先横向比较，再生成一份可直接采用的最终答案，并输出今日运势评分。",

	SystemPromptFile: "prompts/system_prompt.txt",
	JudgePromptFile:  "prompts/judge_prompt.txt",

	JudgeEnabled:  true,
	JudgeModel:    "gemini-flash-latest",
	JudgeProvider: "gemini",

	ModelFilter: []string{"qwen", "llama", "gemma", "glm"},
	ModelSkip:   []string{"embed", "embedding", "bge", "rerank", "reranker", "llava", "vision", "vl", "32b", "72b"},
	ModelLimit:  10,

	LocalCallTimeout:       10 * time.Minute,
	LocalUnloadTimeout:     25 * time.Second,
	LocalPreflightTimeout:  20 * time.Second,
	LocalRetryCount:        2,
	LocalSwitchDelay:       3 * time.Second,
	CloudCallTimeout:       120 * time.Second,
	CloudSwitchDelay:       1 * time.Second,
	RunningPollInterval:    700 * time.Millisecond,
	StartupCleanupEnabled:  false,
	ContinueOnCleanupError: false,

	LocalReleaseRetryCount: 3,
	LocalReleaseRetryDelay: 2 * time.Second,

	CloudModels: []CloudModelConfig{
		{
			Enabled:    true,
			Name:       "gemini-flash-latest",
			Provider:   "gemini",
			APIKey:     "",
			APIKeyFile: "secrets/gemini_api_key.txt",
		},
	},
}

func main() {
	if err := loadRuntimeResources(&appConfig); err != nil {
		fmt.Println("加载配置资源失败:", err)
		return
	}

	startedAt := time.Now()
	now := startedAt
	promptContent := buildPrompt(now)

	reportDir, err := createReportDir(now)
	if err != nil {
		fmt.Println("创建报告目录失败:", err)
		return
	}
	if err := saveRunMeta(reportDir, now, promptContent); err != nil {
		fmt.Println("保存运行信息失败:", err)
		return
	}

	defer func() {
		cleanupStart := time.Now()
		if err := finalCleanupLocalModels(appConfig.BaseURL, appConfig.LocalUnloadTimeout, appConfig.RunningPollInterval); err != nil {
			fmt.Printf("程序结束前兜底清理失败: %v\n", err)
		} else {
			fmt.Printf("程序结束前兜底清理完成，耗时: %s\n", time.Since(cleanupStart).Round(time.Millisecond))
		}
	}()

	localModels, err := resolveModels(appConfig.BaseURL, appConfig)
	if err != nil {
		fmt.Println("获取本地模型列表失败:", err)
		_ = saveSummaryReport(reportDir, now, promptContent, nil, JudgeResult{Enabled: false}, time.Since(startedAt))
		return
	}

	cloudModels := enabledCloudModels(appConfig.CloudModels)
	if len(localModels) == 0 && len(cloudModels) == 0 {
		fmt.Println("没有可用模型，请检查本地 Ollama 模型或云端配置。")
		_ = saveSummaryReport(reportDir, now, promptContent, nil, JudgeResult{Enabled: false}, time.Since(startedAt))
		return
	}

	allModels := append([]string{}, resolveCloudModelNames(appConfig)...)
	allModels = append(allModels, localModels...)

	fmt.Println("本次将串行调用以下模型：")
	for i, model := range allModels {
		fmt.Printf("%d. %s\n", i+1, model)
	}

	results := make([]ModelResult, 0, len(localModels)+len(cloudModels))

	cloudResults, cloudErr := runCloudModelsFirst(appConfig, promptContent, reportDir, now, startedAt)
	results = append(results, cloudResults...)
	if cloudErr != nil {
		_ = saveSummaryReport(reportDir, now, promptContent, results, JudgeResult{Enabled: false}, time.Since(startedAt))
		return
	}

	localResults := compareLocalModelsSequentially(appConfig, localModels, promptContent, reportDir, now)
	results = append(results, localResults...)

	judgeResult := JudgeResult{Enabled: appConfig.JudgeEnabled}
	if appConfig.JudgeEnabled && hasSuccessfulResult(results) {
		judgeTotalStart := time.Now()
		judgeModel := resolveJudgeModel(localModels, appConfig)
		fmt.Printf("正在调用裁判模型: %s\n", judgeModel)

		judgeCallStart := time.Now()
		judgeContent, judgeErr := judgeModelResults(appConfig, localModels, judgeModel, promptContent, results)
		judgeCallCost := time.Since(judgeCallStart).Round(time.Millisecond)

		judgeResult = JudgeResult{
			Model:        judgeModel,
			Content:      judgeContent,
			Err:          judgeErr,
			Enabled:      true,
			Provider:     judgeResultProvider(appConfig, judgeModel, localModels),
			CallDuration: judgeCallCost,
		}

		if judgeResult.Provider == "ollama" && isLocalModel(judgeModel, localModels) {
			releaseStart := time.Now()
			if releaseErr := releaseLocalModelWithRetry(
				appConfig.BaseURL,
				judgeModel,
				appConfig.LocalUnloadTimeout,
				appConfig.RunningPollInterval,
				appConfig.LocalReleaseRetryCount,
				appConfig.LocalReleaseRetryDelay,
			); releaseErr != nil {
				fmt.Printf("裁判模型卸载或等待释放失败: %v\n", releaseErr)
			} else {
				judgeResult.ReleaseDuration = time.Since(releaseStart).Round(time.Millisecond)
				fmt.Printf("裁判模型释放耗时: %s\n", judgeResult.ReleaseDuration)
			}
		}

		judgeResult.TotalDuration = time.Since(judgeTotalStart).Round(time.Millisecond)
		fmt.Printf("裁判模型调用耗时: %s\n", judgeResult.CallDuration)
		fmt.Printf("裁判模型总耗时: %s\n", judgeResult.TotalDuration)

		if judgeErr != nil {
			fmt.Println("裁判模型调用失败:", judgeErr)
		} else {
			fmt.Println("裁判模型调用完成")
		}

		if err := saveJudgeReport(reportDir, now, judgeResult); err != nil {
			fmt.Println("保存裁判报告失败:", err)
		}
	} else if appConfig.JudgeEnabled {
		fmt.Println("没有任何模型成功返回，跳过裁判模型。")
	}

	if err := saveSummaryReport(reportDir, now, promptContent, results, judgeResult, time.Since(startedAt)); err != nil {
		fmt.Println("保存汇总报告失败:", err)
		return
	}

	finalPath, err := saveFinalConclusionHTML(reportDir, now, promptContent, results, judgeResult, time.Since(startedAt))
	if err != nil {
		fmt.Println("保存最终结论文件失败:", err)
	} else {
		fmt.Println("最终结论文件已生成:", finalPath)
		if err := openInDefaultBrowser(finalPath); err != nil {
			fmt.Println("自动打开最终结论文件失败:", err)
		} else {
			fmt.Println("已使用默认浏览器打开最终结论文件")
		}
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

	if judgeResult.Enabled && hasSuccessfulResult(results) {
		fmt.Println("\n========== 裁判模型总结 ==========")
		fmt.Printf("裁判模型: %s\n\n", judgeResult.Model)
		if judgeResult.Err != nil {
			fmt.Println("调用失败:", judgeResult.Err)
		} else {
			fmt.Println(judgeResult.Content)
		}
		fmt.Println("===================================")
	}

	fmt.Printf("\n整轮任务总耗时: %s\n", time.Since(startedAt).Round(time.Millisecond))
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

func loadRuntimeResources(cfg *Config) error {
	systemPrompt, err := loadPromptWithFallback(cfg.SystemPromptFile, cfg.SystemPrompt)
	if err != nil {
		return fmt.Errorf("加载 system prompt 失败: %w", err)
	}
	judgePrompt, err := loadPromptWithFallback(cfg.JudgePromptFile, cfg.JudgePrompt)
	if err != nil {
		return fmt.Errorf("加载 judge prompt 失败: %w", err)
	}
	cfg.SystemPrompt = systemPrompt
	cfg.JudgePrompt = judgePrompt

	for i := range cfg.CloudModels {
		key, err := loadAPIKeyWithFallback(cfg.CloudModels[i].APIKeyFile, cfg.CloudModels[i].APIKey)
		if err != nil {
			return fmt.Errorf("加载云端模型 API Key 失败(%s): %w", cfg.CloudModels[i].Name, err)
		}
		cfg.CloudModels[i].APIKey = key
	}

	return nil
}

func loadPromptWithFallback(filePath string, fallback string) (string, error) {
	path := strings.TrimSpace(filePath)
	if path == "" {
		if strings.TrimSpace(fallback) == "" {
			return "", fmt.Errorf("prompt 文件路径为空且默认 prompt 为空")
		}
		return strings.TrimSpace(fallback), nil
	}

	data, err := os.ReadFile(path)
	if err != nil {
		if strings.TrimSpace(fallback) == "" {
			return "", fmt.Errorf("读取文件失败: %w", err)
		}
		fmt.Printf("提示：读取 prompt 文件失败，改用内置默认内容 (%s): %v\n", path, err)
		return strings.TrimSpace(fallback), nil
	}

	content := strings.TrimSpace(string(data))
	if content == "" {
		if strings.TrimSpace(fallback) == "" {
			return "", fmt.Errorf("prompt 文件内容为空: %s", path)
		}
		fmt.Printf("提示：prompt 文件为空，改用内置默认内容 (%s)\n", path)
		return strings.TrimSpace(fallback), nil
	}

	fmt.Printf("已加载 Prompt 文件: %s\n", path)
	return content, nil
}

func loadAPIKeyWithFallback(filePath string, fallback string) (string, error) {
	path := strings.TrimSpace(filePath)
	if path != "" {
		data, err := os.ReadFile(path)
		if err == nil {
			key := strings.TrimSpace(string(data))
			if key != "" && !strings.Contains(key, "<") && key != "YOUR_GEMINI_API_KEY" {
				fmt.Printf("已加载 API Key 文件: %s\n", path)
				return key, nil
			}
		} else {
			fmt.Printf("提示：读取 API Key 文件失败 (%s): %v\n", path, err)
		}
	}

	key := strings.TrimSpace(fallback)
	if key == "" || strings.Contains(key, "<") || key == "YOUR_GEMINI_API_KEY" {
		return "", fmt.Errorf("未读取到有效 API Key")
	}
	return key, nil
}

func runCloudModelsFirst(
	cfg Config,
	userPrompt string,
	reportDir string,
	reportTime time.Time,
	startedAt time.Time,
) ([]ModelResult, error) {
	cloudModels := enabledCloudModels(cfg.CloudModels)
	results := make([]ModelResult, 0, len(cloudModels))

	for i, cloud := range cloudModels {
		fmt.Printf("[云端优先 %d/%d] 正在调用云端模型: %s\n", i+1, len(cloudModels), cloud.Name)

		modelTotalStart := time.Now()
		callStart := time.Now()
		content, callErr := chatWithCloudModel(cloud, cfg.SystemPrompt, userPrompt, cfg.CloudCallTimeout)
		callCost := time.Since(callStart).Round(time.Millisecond)

		result := ModelResult{
			Model:         cloud.Name,
			Content:       content,
			Err:           callErr,
			Provider:      cloud.Provider,
			CallDuration:  callCost,
			TotalDuration: time.Since(modelTotalStart).Round(time.Millisecond),
		}
		results = append(results, result)

		if saveErr := saveSingleModelReport(reportDir, reportTime, result); saveErr != nil {
			fmt.Printf("保存模型报告失败(%s): %v\n", cloud.Name, saveErr)
		}

		fmt.Printf("[云端优先 %d/%d] 云端模型 %s 调用耗时: %s\n", i+1, len(cloudModels), cloud.Name, callCost)

		if callErr != nil {
			fmt.Printf("[云端优先 %d/%d] 云端模型 %s 调用失败，程序终止: %v\n", i+1, len(cloudModels), cloud.Name, callErr)
			_ = saveSummaryReport(reportDir, reportTime, userPrompt, results, JudgeResult{Enabled: false}, time.Since(startedAt))
			return results, callErr
		}

		fmt.Printf("[云端优先 %d/%d] 云端模型 %s 调用完成\n", i+1, len(cloudModels), cloud.Name)
		fmt.Printf("[云端优先 %d/%d] 云端模型 %s 总耗时: %s\n", i+1, len(cloudModels), cloud.Name, result.TotalDuration)

		time.Sleep(cfg.CloudSwitchDelay)
	}

	return results, nil
}

func compareLocalModelsSequentially(cfg Config, localModels []string, userPrompt string, reportDir string, reportTime time.Time) []ModelResult {
	results := make([]ModelResult, 0, len(localModels))
	total := len(localModels)
	index := 0

	for _, modelName := range localModels {
		index++
		modelTotalStart := time.Now()

		fmt.Printf("[%d/%d] 准备调用本地模型: %s\n", index, total, modelName)

		preflightStart := time.Now()
		if err := ensureNoModelsRunning(cfg.BaseURL, cfg.LocalPreflightTimeout, cfg.RunningPollInterval); err != nil {
			err = fmt.Errorf("调用前清理运行中模型失败: %w", err)
			fmt.Printf("[%d/%d] %v\n", index, total, err)
			fmt.Printf("[%d/%d] 调用前清理耗时: %s\n", index, total, time.Since(preflightStart).Round(time.Millisecond))

			result := ModelResult{
				Model:         modelName,
				Err:           err,
				Provider:      "ollama",
				TotalDuration: time.Since(modelTotalStart).Round(time.Millisecond),
			}
			results = append(results, result)

			if saveErr := saveSingleModelReport(reportDir, reportTime, result); saveErr != nil {
				fmt.Printf("[%d/%d] 保存模型报告失败(%s): %v\n", index, total, modelName, saveErr)
			}

			if !cfg.ContinueOnCleanupError {
				break
			}
			continue
		}
		fmt.Printf("[%d/%d] 调用前清理耗时: %s\n", index, total, time.Since(preflightStart).Round(time.Millisecond))

		content, callCost, err := chatWithOllamaWithRetry(
			cfg.BaseURL,
			modelName,
			cfg.SystemPrompt,
			userPrompt,
			cfg.LocalRetryCount,
			cfg.LocalCallTimeout,
			cfg.LocalUnloadTimeout,
			cfg.RunningPollInterval,
		)

		result := ModelResult{
			Model:         modelName,
			Content:       content,
			Err:           err,
			Provider:      "ollama",
			CallDuration:  callCost,
			TotalDuration: time.Since(modelTotalStart).Round(time.Millisecond),
		}

		fmt.Printf("[%d/%d] 模型 %s 调用耗时: %s\n", index, total, modelName, callCost)

		if err != nil {
			fmt.Printf("[%d/%d] 模型 %s 调用失败: %v\n", index, total, modelName, err)
		} else {
			fmt.Printf("[%d/%d] 模型 %s 调用完成\n", index, total, modelName)
		}

		releaseStart := time.Now()
		if releaseErr := releaseLocalModelWithRetry(
			cfg.BaseURL,
			modelName,
			cfg.LocalUnloadTimeout,
			cfg.RunningPollInterval,
			cfg.LocalReleaseRetryCount,
			cfg.LocalReleaseRetryDelay,
		); releaseErr != nil {
			fmt.Printf("[%d/%d] 模型 %s 卸载或等待释放失败: %v\n", index, total, modelName, releaseErr)
		} else {
			fmt.Printf("[%d/%d] 模型 %s 释放耗时: %s\n", index, total, modelName, time.Since(releaseStart).Round(time.Millisecond))
			fmt.Printf("[%d/%d] 模型 %s 已确认释放\n", index, total, modelName)
		}

		result.TotalDuration = time.Since(modelTotalStart).Round(time.Millisecond)
		fmt.Printf("[%d/%d] 模型 %s 总耗时: %s\n", index, total, modelName, result.TotalDuration)

		results = append(results, result)

		if saveErr := saveSingleModelReport(reportDir, reportTime, result); saveErr != nil {
			fmt.Printf("[%d/%d] 保存模型报告失败(%s): %v\n", index, total, modelName, saveErr)
		}

		time.Sleep(cfg.LocalSwitchDelay)
	}

	cleanupStart := time.Now()
	if err := finalCleanupLocalModels(cfg.BaseURL, cfg.LocalUnloadTimeout, cfg.RunningPollInterval); err != nil {
		fmt.Printf("本地模型批量结束后的兜底清理失败: %v\n", err)
	} else {
		fmt.Printf("本地模型批量结束后的兜底清理完成，耗时: %s\n", time.Since(cleanupStart).Round(time.Millisecond))
	}

	return results
}

func releaseLocalModelWithRetry(
	baseURL, modelName string,
	timeout time.Duration,
	pollInterval time.Duration,
	retryCount int,
	retryDelay time.Duration,
) error {
	if retryCount < 1 {
		retryCount = 1
	}

	var lastErr error
	for attempt := 1; attempt <= retryCount; attempt++ {
		err := unloadAndWaitAllClear(baseURL, modelName, timeout, pollInterval)
		if err == nil {
			if attempt > 1 {
				fmt.Printf("模型 %s 第 %d/%d 次释放成功\n", modelName, attempt, retryCount)
			}
			return nil
		}

		lastErr = err
		fmt.Printf("模型 %s 第 %d/%d 次释放失败: %v\n", modelName, attempt, retryCount, err)

		if attempt < retryCount {
			time.Sleep(retryDelay)
		}
	}

	return fmt.Errorf("模型 %s 多次释放仍失败: %w", modelName, lastErr)
}

func finalCleanupLocalModels(baseURL string, timeout time.Duration, pollInterval time.Duration) error {
	models, err := getRunningModels(baseURL)
	if err != nil {
		return err
	}
	if len(models) == 0 {
		return nil
	}

	fmt.Printf("开始执行兜底清理，当前运行中模型: %s\n", strings.Join(models, ", "))

	for _, model := range models {
		if err := releaseLocalModelWithRetry(baseURL, model, timeout, pollInterval, 3, 2*time.Second); err != nil {
			fmt.Printf("兜底释放模型失败(%s): %v\n", model, err)
		}
	}

	if err := waitUntilNoModelsRunning(baseURL, timeout, pollInterval); err != nil {
		return err
	}

	return nil
}

func chatWithOllamaWithRetry(
	baseURL, modelName, systemPrompt, userPrompt string,
	maxAttempts int,
	callTimeout time.Duration,
	unloadTimeout time.Duration,
	pollInterval time.Duration,
) (string, time.Duration, error) {
	if maxAttempts < 1 {
		maxAttempts = 1
	}

	var lastErr error
	var totalCallDuration time.Duration

	for attempt := 1; attempt <= maxAttempts; attempt++ {
		attemptStart := time.Now()
		content, err := chatWithOllama(baseURL, modelName, systemPrompt, userPrompt, callTimeout)
		attemptCost := time.Since(attemptStart).Round(time.Millisecond)
		totalCallDuration += attemptCost

		if err == nil {
			fmt.Printf("模型 %s 第 %d/%d 次调用成功，耗时: %s\n", modelName, attempt, maxAttempts, attemptCost)
			return content, totalCallDuration.Round(time.Millisecond), nil
		}

		lastErr = err
		fmt.Printf("模型 %s 第 %d/%d 次调用失败，耗时: %s，错误: %v\n", modelName, attempt, maxAttempts, attemptCost, err)

		if attempt < maxAttempts {
			_ = unloadAndWaitAllClear(baseURL, modelName, unloadTimeout, pollInterval)
			time.Sleep(4 * time.Second)
		}
	}

	return "", totalCallDuration.Round(time.Millisecond), fmt.Errorf("模型 %s 多次调用仍失败: %w", modelName, lastErr)
}

func chatWithOllama(baseURL, modelName, systemPrompt, userPrompt string, timeout time.Duration) (string, error) {
	url := strings.TrimRight(baseURL, "/") + "/api/chat"

	reqBody := OllamaChatRequest{
		Model: modelName,
		Messages: []Message{
			{Role: "system", Content: systemPrompt},
			{Role: "user", Content: userPrompt},
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
	if err := unloadModelViaGenerate(baseURL, modelName); err != nil {
		fmt.Printf("备用卸载(generate)失败(%s): %v\n", modelName, err)
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
	if apiKey == "" || strings.Contains(apiKey, "<") || apiKey == "YOUR_GEMINI_API_KEY" {
		return "", fmt.Errorf("Gemini API Key 未配置，请检查 API Key 文件")
	}

	url := fmt.Sprintf(
		"https://generativelanguage.googleapis.com/v1beta/models/%s:generateContent?key=%s",
		strings.TrimSpace(cloud.Name),
		apiKey,
	)

	reqBody := GeminiGenerateRequest{
		Contents: []GeminiContent{
			{
				Role: "user",
				Parts: []GeminiPart{
					{Text: systemPrompt + "\n\n" + userPrompt},
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

func judgeModelResults(cfg Config, localModels []string, judgeModel, originalPrompt string, results []ModelResult) (string, error) {
	judgeInput := buildJudgeInput(originalPrompt, results)

	if isLocalModel(judgeModel, localModels) {
		return chatWithOllama(cfg.BaseURL, judgeModel, cfg.JudgePrompt, judgeInput, cfg.LocalCallTimeout)
	}

	cloud, ok := findCloudModelConfig(judgeModel, cfg.CloudModels)
	if ok {
		return chatWithCloudModel(cloud, cfg.JudgePrompt, judgeInput, cfg.CloudCallTimeout)
	}

	if strings.EqualFold(strings.TrimSpace(cfg.JudgeProvider), "gemini") {
		for _, cloudModel := range cfg.CloudModels {
			if cloudModel.Enabled && strings.EqualFold(cloudModel.Provider, "gemini") {
				return chatWithCloudModel(cloudModel, cfg.JudgePrompt, judgeInput, cfg.CloudCallTimeout)
			}
		}
		return "", fmt.Errorf("未找到可用的 Gemini 裁判模型配置: %s", judgeModel)
	}

	return chatWithOllama(cfg.BaseURL, judgeModel, cfg.JudgePrompt, judgeInput, cfg.LocalCallTimeout)
}

func buildJudgeInput(originalPrompt string, results []ModelResult) string {
	var builder strings.Builder

	writeString(&builder, "以下是同一个问题的多模型输出结果，请你先完成横向评审，再给出一份可以直接采用的最终结论。\n\n")
	writeString(&builder, "【任务要求】\n")
	writeString(&builder, "你不能只做模型优劣点评，必须在评审后输出“最终结论”。最终结论必须是整合后的可直接使用版本，而不是简单说哪个模型更好。\n")
	writeString(&builder, "你必须给出“今日运势评分”，满分 10 分，格式必须严格写成：X/10，例如 7.5/10 或 8/10。\n")
	writeString(&builder, "你还必须说明该评分的简短理由。\n\n")
	writeString(&builder, "【原始问题】\n")
	writeString(&builder, originalPrompt)
	writeString(&builder, "\n\n")

	successCount := 0
	failCount := 0
	for _, result := range results {
		if result.Err != nil {
			failCount++
		} else {
			successCount++
		}
	}

	writeString(&builder, "【结果统计】\n")
	writeString(&builder, fmt.Sprintf("成功模型数：%d\n", successCount))
	writeString(&builder, fmt.Sprintf("失败模型数：%d\n\n", failCount))

	for _, result := range results {
		writeString(&builder, "【模型】")
		writeString(&builder, result.Model)
		writeString(&builder, "\n")
		writeString(&builder, "【提供方】")
		writeString(&builder, result.Provider)
		writeString(&builder, "\n")
		writeString(&builder, "【调用耗时】")
		writeString(&builder, result.CallDuration.Round(time.Millisecond).String())
		writeString(&builder, "\n")
		writeString(&builder, "【总耗时】")
		writeString(&builder, result.TotalDuration.Round(time.Millisecond).String())
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

	writeString(&builder, "请严格按以下标题输出，不要遗漏：\n")
	writeString(&builder, "一、今日运势评分\n")
	writeString(&builder, "二、最终结论\n")
	writeString(&builder, "三、模型对比\n")
	writeString(&builder, "四、最佳模型\n")
	writeString(&builder, "五、采用建议\n")
	writeString(&builder, "六、置信度\n")

	return builder.String()
}

func resolveJudgeModel(localModels []string, cfg Config) string {
	if strings.TrimSpace(cfg.JudgeModel) != "" {
		return cfg.JudgeModel
	}
	if strings.EqualFold(strings.TrimSpace(cfg.JudgeProvider), "gemini") {
		for _, cloud := range cfg.CloudModels {
			if cloud.Enabled && strings.EqualFold(cloud.Provider, "gemini") && strings.TrimSpace(cloud.Name) != "" {
				return cloud.Name
			}
		}
	}
	if len(localModels) > 0 {
		return localModels[0]
	}
	return "gemini-flash-latest"
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

func findCloudModelConfig(model string, cloudModels []CloudModelConfig) (CloudModelConfig, bool) {
	for _, cloud := range cloudModels {
		if cloud.Enabled && cloud.Name == model {
			return cloud, true
		}
	}
	return CloudModelConfig{}, false
}

func createReportDir(t time.Time) (string, error) {
	reportDir := filepath.Join("reports", t.Format("2006-01-02_15-04-05"))
	if err := os.MkdirAll(reportDir, 0755); err != nil {
		return "", fmt.Errorf("创建报告目录失败: %w", err)
	}
	return reportDir, nil
}

func saveRunMeta(reportDir string, t time.Time, prompt string) error {
	content := fmt.Sprintf(
		"# 运行信息\n\n- 启动时间：`%s`\n- 请求内容：`%s`\n",
		t.Format("2006-01-02 15:04:05"),
		prompt,
	)
	path := filepath.Join(reportDir, "run.md")
	return os.WriteFile(path, []byte(content), 0644)
}

func saveSingleModelReport(reportDir string, t time.Time, result ModelResult) error {
	filename := sanitizeFileName(result.Model) + ".md"
	path := filepath.Join(reportDir, filename)

	var content string
	if result.Err != nil {
		content = fmt.Sprintf(
			"# 模型报告\n\n- 模型：`%s`\n- 提供方：`%s`\n- 生成时间：`%s`\n- 调用耗时：`%s`\n- 总耗时：`%s`\n- 状态：失败\n- 错误：`%v`\n",
			result.Model,
			result.Provider,
			t.Format("2006-01-02 15:04:05"),
			result.CallDuration.Round(time.Millisecond),
			result.TotalDuration.Round(time.Millisecond),
			result.Err,
		)
	} else {
		content = fmt.Sprintf(
			"# 模型报告\n\n- 模型：`%s`\n- 提供方：`%s`\n- 生成时间：`%s`\n- 调用耗时：`%s`\n- 总耗时：`%s`\n- 状态：成功\n\n## 输出内容\n\n```text\n%s\n```\n",
			result.Model,
			result.Provider,
			t.Format("2006-01-02 15:04:05"),
			result.CallDuration.Round(time.Millisecond),
			result.TotalDuration.Round(time.Millisecond),
			result.Content,
		)
	}

	return os.WriteFile(path, []byte(content), 0644)
}

func saveJudgeReport(reportDir string, t time.Time, judgeResult JudgeResult) error {
	judgePath := filepath.Join(reportDir, "judge.md")
	judgeContent := buildJudgeReport(t, judgeResult)
	return os.WriteFile(judgePath, []byte(judgeContent), 0644)
}

func saveSummaryReport(
	reportDir string,
	t time.Time,
	prompt string,
	results []ModelResult,
	judgeResult JudgeResult,
	totalDuration time.Duration,
) error {
	summary := buildSummaryReport(t, prompt, results, judgeResult, totalDuration)
	summaryPath := filepath.Join(reportDir, "summary.md")
	return os.WriteFile(summaryPath, []byte(summary), 0644)
}

func hasSuccessfulResult(results []ModelResult) bool {
	for _, result := range results {
		if result.Err == nil && strings.TrimSpace(result.Content) != "" {
			return true
		}
	}
	return false
}

func buildSummaryReport(
	t time.Time,
	prompt string,
	results []ModelResult,
	judgeResult JudgeResult,
	totalDuration time.Duration,
) string {
	var builder strings.Builder

	writeString(&builder, "# 多模型对比汇总报告\n\n")
	writeString(&builder, "## 基本信息\n\n")
	writeString(&builder, fmt.Sprintf("- 生成时间：`%s`\n", t.Format("2006-01-02 15:04:05")))
	writeString(&builder, fmt.Sprintf("- 请求内容：`%s`\n", prompt))
	writeString(&builder, fmt.Sprintf("- 整轮任务总耗时：`%s`\n\n", totalDuration.Round(time.Millisecond)))

	if judgeResult.Enabled && judgeResult.Err == nil && strings.TrimSpace(judgeResult.Content) != "" {
		writeString(&builder, "## 最终结论（裁判整合）\n\n")
		writeString(&builder, "```text\n")
		writeString(&builder, judgeResult.Content)
		if !strings.HasSuffix(judgeResult.Content, "\n") {
			writeString(&builder, "\n")
		}
		writeString(&builder, "```\n\n")
	}

	writeString(&builder, "## 参与对比的模型结果\n\n")
	if len(results) == 0 {
		writeString(&builder, "- 暂无模型结果\n\n")
	}

	for _, result := range results {
		writeString(&builder, fmt.Sprintf("## 模型：`%s`\n\n", result.Model))
		writeString(&builder, fmt.Sprintf("- 提供方：`%s`\n", result.Provider))
		writeString(&builder, fmt.Sprintf("- 调用耗时：`%s`\n", result.CallDuration.Round(time.Millisecond)))
		writeString(&builder, fmt.Sprintf("- 总耗时：`%s`\n", result.TotalDuration.Round(time.Millisecond)))

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
		writeString(&builder, fmt.Sprintf("- 提供方：`%s`\n", judgeResult.Provider))
		writeString(&builder, fmt.Sprintf("- 调用耗时：`%s`\n", judgeResult.CallDuration.Round(time.Millisecond)))
		writeString(&builder, fmt.Sprintf("- 总耗时：`%s`\n", judgeResult.TotalDuration.Round(time.Millisecond)))

		if judgeResult.Err != nil {
			writeString(&builder, "- 状态：失败\n")
			writeString(&builder, fmt.Sprintf("- 错误：`%v`\n\n", judgeResult.Err))
		} else if strings.TrimSpace(judgeResult.Model) == "" {
			writeString(&builder, "- 状态：未执行\n\n")
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
	writeString(&builder, "## 基本信息\n\n")
	writeString(&builder, fmt.Sprintf("- 生成时间：`%s`\n", t.Format("2006-01-02 15:04:05")))
	writeString(&builder, fmt.Sprintf("- 裁判模型：`%s`\n", judgeResult.Model))
	writeString(&builder, fmt.Sprintf("- 提供方：`%s`\n", judgeResult.Provider))
	writeString(&builder, fmt.Sprintf("- 调用耗时：`%s`\n", judgeResult.CallDuration.Round(time.Millisecond)))
	writeString(&builder, fmt.Sprintf("- 总耗时：`%s`\n\n", judgeResult.TotalDuration.Round(time.Millisecond)))

	if judgeResult.Err != nil {
		writeString(&builder, "## 状态\n\n")
		writeString(&builder, "- 状态：失败\n")
		writeString(&builder, fmt.Sprintf("- 错误：`%v`\n", judgeResult.Err))
		return builder.String()
	}

	writeString(&builder, "## 状态\n\n")
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

func saveFinalConclusionHTML(
	reportDir string,
	t time.Time,
	prompt string,
	results []ModelResult,
	judgeResult JudgeResult,
	totalDuration time.Duration,
) (string, error) {
	path := filepath.Join(reportDir, "final.html")

	successModels := make([]string, 0)
	for _, result := range results {
		if result.Err == nil && strings.TrimSpace(result.Content) != "" {
			successModels = append(successModels, result.Model)
		}
	}

	rawFinalContent := ""
	if judgeResult.Enabled && judgeResult.Err == nil && strings.TrimSpace(judgeResult.Content) != "" {
		rawFinalContent = judgeResult.Content
	} else {
		firstSuccess := firstSuccessfulResult(results)
		if firstSuccess != nil {
			rawFinalContent = "裁判模型未成功生成最终结论，以下为首个成功模型的参考输出：\n\n" + firstSuccess.Content
		} else {
			rawFinalContent = "本次未生成可用的最终结论，请查看 summary.md 获取详细信息。"
		}
	}

	scoreText, scoreReason := extractFortuneScore(judgeResult)
	scoreClass := fortuneScoreClass(scoreText)
	summaryText := extractFinalConclusionSummary(rawFinalContent)
	finalContent := buildFinalContentWithoutScore(rawFinalContent)

	modelsHTML := "<li>无</li>"
	if len(successModels) > 0 {
		var items strings.Builder
		for _, model := range successModels {
			items.WriteString("<li>")
			items.WriteString(htmlEscape(model))
			items.WriteString("</li>")
		}
		modelsHTML = items.String()
	}

	scoreCardHTML := fmt.Sprintf(`
	<div class="card score-card %s">
		<div class="score-label">今日运势评分</div>
		<div class="score-value">%s</div>
		<div class="score-reason">%s</div>
	</div>`,
		scoreClass,
		htmlEscape(scoreText),
		htmlEscape(scoreReason),
	)

	summaryCardHTML := ""
	if strings.TrimSpace(summaryText) != "" {
		summaryCardHTML = fmt.Sprintf(`
	<div class="card summary-card">
		<h2>结论摘要</h2>
		<div class="summary-text">%s</div>
	</div>`, htmlEscape(summaryText))
	}

	html := fmt.Sprintf(`<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>最终结论</title>
<style>
body{font-family:"Microsoft YaHei","PingFang SC",Arial,sans-serif;background:#f5f7fb;color:#1f2937;margin:0;padding:24px;}
.container{max-width:980px;margin:0 auto;}
.card{background:#fff;border-radius:16px;padding:24px;box-shadow:0 8px 30px rgba(0,0,0,.08);margin-bottom:20px;}
h1{margin:0 0 16px 0;font-size:32px;}
h2{margin:0 0 14px 0;font-size:22px;color:#111827;}
.meta{line-height:1.9;font-size:15px;color:#4b5563;}
.highlight{background:linear-gradient(135deg,#fff7ed,#fffbeb);border:1px solid #fdba74;}
.summary-card{background:linear-gradient(135deg,#faf5ff,#eef2ff);border:1px solid #c4b5fd;}
.summary-text{white-space:pre-wrap;word-break:break-word;line-height:1.9;font-size:16px;color:#312e81;font-weight:600;}
.score-card{text-align:center;}
.score-good{background:linear-gradient(135deg,#ecfdf5,#dcfce7);border:1px solid #86efac;}
.score-good .score-label{color:#15803d;}
.score-good .score-value{color:#166534;}
.score-mid{background:linear-gradient(135deg,#fffbeb,#fef3c7);border:1px solid #fcd34d;}
.score-mid .score-label{color:#b45309;}
.score-mid .score-value{color:#92400e;}
.score-low{background:linear-gradient(135deg,#fef2f2,#fee2e2);border:1px solid #fca5a5;}
.score-low .score-label{color:#b91c1c;}
.score-low .score-value{color:#991b1b;}
.score-unknown{background:linear-gradient(135deg,#eff6ff,#e0e7ff);border:1px solid #93c5fd;}
.score-unknown .score-label{color:#1d4ed8;}
.score-unknown .score-value{color:#1e3a8a;}
.score-label{font-size:16px;margin-bottom:10px;font-weight:700;}
.score-value{font-size:48px;line-height:1.2;font-weight:800;margin-bottom:10px;}
.score-reason{font-size:14px;line-height:1.8;color:#475569;}
pre{white-space:pre-wrap;word-break:break-word;background:#0f172a;color:#e5e7eb;padding:18px;border-radius:12px;line-height:1.75;font-size:15px;overflow:auto;}
ul{margin:0;padding-left:22px;line-height:1.9;}
.note{color:#6b7280;font-size:14px;line-height:1.8;}
.badge{display:inline-block;padding:6px 12px;border-radius:999px;background:#dbeafe;color:#1d4ed8;font-size:13px;margin-bottom:12px;}
</style>
</head>
<body>
<div class="container">
	<div class="card">
		<div class="badge">自动生成</div>
		<h1>最终结论</h1>
		<div class="meta">
			<div><strong>生成时间：</strong>%s</div>
			<div><strong>问题：</strong>%s</div>
			<div><strong>总耗时：</strong>%s</div>
		</div>
	</div>

	%s

	%s

	<div class="card">
		<h2>成功返回的模型</h2>
		<ul>%s</ul>
	</div>

	<div class="card highlight">
		<h2>最终采用结论</h2>
		<pre>%s</pre>
	</div>

	<div class="card">
		<h2>查看说明</h2>
		<div class="note">
			<div>• 如需查看每个模型的原始输出，请打开同目录下的各模型报告文件。</div>
			<div>• 如需查看完整横向比较，请打开 <strong>summary.md</strong> 与 <strong>judge.md</strong>。</div>
		</div>
	</div>
</div>
</body>
</html>`,
		t.Format("2006-01-02 15:04:05"),
		htmlEscape(prompt),
		totalDuration.Round(time.Millisecond),
		scoreCardHTML,
		summaryCardHTML,
		modelsHTML,
		htmlEscape(finalContent),
	)

	if err := os.WriteFile(path, []byte(html), 0644); err != nil {
		return "", err
	}
	return path, nil
}

func firstSuccessfulResult(results []ModelResult) *ModelResult {
	for i := range results {
		if results[i].Err == nil && strings.TrimSpace(results[i].Content) != "" {
			return &results[i]
		}
	}
	return nil
}

func extractFortuneScore(judgeResult JudgeResult) (string, string) {
	if judgeResult.Err != nil || strings.TrimSpace(judgeResult.Content) == "" {
		return "暂无评分", "裁判模型未成功返回，暂时无法提取今日运势评分。"
	}

	lines := strings.Split(judgeResult.Content, "\n")
	score := ""
	reason := ""

	for i, rawLine := range lines {
		line := strings.TrimSpace(rawLine)
		compact := strings.ReplaceAll(line, " ", "")
		if strings.Contains(compact, "今日运势评分") {
			if idx := strings.Index(compact, "："); idx >= 0 && idx < len(compact)-1 {
				score = strings.TrimSpace(compact[idx+len("："):])
			} else if idx := strings.Index(compact, ":"); idx >= 0 && idx < len(compact)-1 {
				score = strings.TrimSpace(compact[idx+1:])
			}

			if score == "" && i+1 < len(lines) {
				nextLine := strings.TrimSpace(lines[i+1])
				if strings.Contains(nextLine, "/10") {
					score = nextLine
				}
			}

			for j := i + 1; j < len(lines) && j <= i+3; j++ {
				nextLine := strings.TrimSpace(lines[j])
				if nextLine == "" {
					continue
				}
				if isSectionHeading(nextLine, 2) {
					break
				}
				if reason == "" && !strings.Contains(nextLine, "/10") {
					reason = nextLine
					break
				}
			}
			break
		}
	}

	if score == "" {
		for _, rawLine := range lines {
			line := strings.TrimSpace(rawLine)
			if strings.Contains(line, "/10") {
				score = line
				break
			}
		}
	}

	if score == "" {
		score = "未识别"
	}
	if reason == "" {
		reason = "已生成最终结论，但未识别到明确的评分理由。"
	}
	return score, reason
}

func buildFinalContentWithoutScore(content string) string {
	lines := strings.Split(content, "\n")
	result := make([]string, 0, len(lines))

	skipping := false
	for _, rawLine := range lines {
		line := strings.TrimSpace(rawLine)

		if isSectionHeading(line, 1) && strings.Contains(line, "今日运势评分") {
			skipping = true
			continue
		}

		if skipping {
			if isAnyMainSectionHeading(line) {
				skipping = false
				result = append(result, rawLine)
			}
			continue
		}

		result = append(result, rawLine)
	}

	cleaned := strings.TrimSpace(strings.Join(result, "\n"))
	if cleaned == "" {
		return content
	}
	return cleaned
}

func extractFinalConclusionSummary(content string) string {
	lines := strings.Split(content, "\n")
	collecting := false
	collected := make([]string, 0)

	for _, rawLine := range lines {
		line := strings.TrimSpace(rawLine)

		if !collecting && isSectionHeading(line, 2) && strings.Contains(line, "最终结论") {
			collecting = true
			continue
		}

		if collecting {
			if line == "" {
				if len(collected) > 0 {
					collected = append(collected, "")
				}
				continue
			}
			if isAnyMainSectionHeading(line) {
				break
			}
			collected = append(collected, line)
		}
	}

	summary := strings.TrimSpace(strings.Join(collected, "\n"))
	if summary != "" {
		return summary
	}

	fallback := buildFinalContentWithoutScore(content)
	runes := []rune(strings.TrimSpace(fallback))
	if len(runes) > 180 {
		return string(runes[:180]) + "..."
	}
	return string(runes)
}

func isAnyMainSectionHeading(line string) bool {
	for i := 1; i <= 6; i++ {
		if isSectionHeading(line, i) {
			return true
		}
	}
	return false
}

func isSectionHeading(line string, num int) bool {
	line = strings.TrimSpace(line)
	cnNums := []string{"一", "二", "三", "四", "五", "六", "七", "八", "九", "十"}
	if num >= 1 && num <= len(cnNums) && strings.HasPrefix(line, cnNums[num-1]+"、") {
		return true
	}
	if strings.HasPrefix(line, fmt.Sprintf("%d.", num)) {
		return true
	}
	if strings.HasPrefix(line, fmt.Sprintf("%d、", num)) {
		return true
	}
	return false
}

func fortuneScoreClass(scoreText string) string {
	value, ok := parseFortuneScore(scoreText)
	if !ok {
		return "score-unknown"
	}
	if value >= 8.0 {
		return "score-good"
	}
	if value >= 6.0 {
		return "score-mid"
	}
	return "score-low"
}

func parseFortuneScore(scoreText string) (float64, bool) {
	s := strings.TrimSpace(scoreText)
	if s == "" {
		return 0, false
	}

	if idx := strings.Index(s, "/10"); idx > 0 {
		s = strings.TrimSpace(s[:idx])
	}

	replacer := strings.NewReplacer(
		"：", "",
		":", "",
		"分", "",
		"今日运势评分", "",
		"一、", "",
		"1.", "",
		"1、", "",
		"=", "",
	)
	s = strings.TrimSpace(replacer.Replace(s))
	if s == "" {
		return 0, false
	}

	v, err := strconv.ParseFloat(s, 64)
	if err != nil {
		return 0, false
	}
	return v, true
}

func openInDefaultBrowser(path string) error {
	absPath, err := filepath.Abs(path)
	if err != nil {
		return err
	}

	target := "file:///" + filepath.ToSlash(absPath)
	fmt.Println("准备打开文件:", target)

	switch runtime.GOOS {
	case "windows":
		return exec.Command("cmd", "/c", "start", "", target).Run()
	case "darwin":
		return exec.Command("open", target).Run()
	default:
		return exec.Command("xdg-open", target).Run()
	}
}

func htmlEscape(s string) string {
	replacer := strings.NewReplacer(
		"&", "&amp;",
		"<", "&lt;",
		">", "&gt;",
		"\"", "&quot;",
		"'", "&#39;",
	)
	return replacer.Replace(s)
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

func judgeResultProvider(cfg Config, judgeModel string, localModels []string) string {
	if isLocalModel(judgeModel, localModels) {
		return "ollama"
	}
	if cloud, ok := findCloudModelConfig(judgeModel, cfg.CloudModels); ok {
		return cloud.Provider
	}
	if strings.TrimSpace(cfg.JudgeProvider) != "" {
		return cfg.JudgeProvider
	}
	return "unknown"
}
