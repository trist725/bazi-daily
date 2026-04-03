package main

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"sort"
	"strings"
	"time"
)

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
	_ = unloadModelViaGenerate(baseURL, modelName)
	_ = waitUntilModelUnloaded(baseURL, modelName, timeout, pollInterval)
	return waitUntilNoModelsRunning(baseURL, timeout, pollInterval)
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
			_ = unloadModel(baseURL, model)
			_ = unloadModelViaGenerate(baseURL, model)
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
	jsonData, _ := json.Marshal(reqBody)
	req, _ := http.NewRequest(http.MethodPost, url, bytes.NewBuffer(jsonData))
	req.Header.Set("Content-Type", "application/json")
	client := &http.Client{Timeout: 60 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
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
	jsonData, _ := json.Marshal(reqBody)
	req, _ := http.NewRequest(http.MethodPost, url, bytes.NewBuffer(jsonData))
	req.Header.Set("Content-Type", "application/json")
	client := &http.Client{Timeout: 60 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	return nil
}

func waitUntilModelUnloaded(baseURL, modelName string, timeout time.Duration, pollInterval time.Duration) error {
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		running, err := isModelRunning(baseURL, modelName)
		if err == nil && !running {
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
		if err == nil && len(models) == 0 {
			return nil
		}
		time.Sleep(pollInterval)
	}
	models, _ := getRunningModels(baseURL)
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
		return nil, err
	}
	defer resp.Body.Close()
	var runningResp OllamaRunningModelsResponse
	if err := json.NewDecoder(resp.Body).Decode(&runningResp); err != nil {
		return nil, err
	}
	models := make([]string, 0, len(runningResp.Models))
	for _, model := range runningResp.Models {
		if name := strings.TrimSpace(model.Name); name != "" {
			models = append(models, name)
		}
	}
	sort.Strings(models)
	return models, nil
}

func releaseLocalModelWithRetry(baseURL, modelName string, timeout, pollInterval time.Duration, retryCount int, retryDelay time.Duration) error {
	var lastErr error
	for attempt := 1; attempt <= retryCount; attempt++ {
		err := unloadAndWaitAllClear(baseURL, modelName, timeout, pollInterval)
		if err == nil {
			return nil
		}
		lastErr = err
		if attempt < retryCount {
			time.Sleep(retryDelay)
		}
	}
	return lastErr
}

func finalCleanupLocalModels(baseURL string, timeout, pollInterval time.Duration) error {
	models, _ := getRunningModels(baseURL)
	if len(models) == 0 {
		return nil
	}
	fmt.Printf("开始执行兜底清理，当前运行中模型: %s\n", strings.Join(models, ", "))
	for _, model := range models {
		_ = releaseLocalModelWithRetry(baseURL, model, timeout, pollInterval, 3, 2*time.Second)
	}
	return waitUntilNoModelsRunning(baseURL, timeout, pollInterval)
}
