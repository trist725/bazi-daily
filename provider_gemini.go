package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

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
	
	// 如果 Key 为空或看起来像占位符，才报错
	if apiKey == "" || apiKey == "YOUR_GEMINI_API_KEY" {
		return "", fmt.Errorf("Gemini API Key 未正确读取（内容为空或为默认占位符）")
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
		return "", fmt.Errorf("Gemini 网络请求失败: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", fmt.Errorf("Gemini 读取响应失败: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("Gemini API 返回状态码 %d: %s", resp.StatusCode, string(body))
	}

	var geminiResp GeminiGenerateResponse
	if err := json.Unmarshal(body, &geminiResp); err != nil {
		return "", fmt.Errorf("Gemini 解析响应失败: %w，原始响应: %s", err, string(body))
	}

	if geminiResp.Error != nil {
		return "", fmt.Errorf("Gemini 服务错误: %s", geminiResp.Error.Message)
	}

	if len(geminiResp.Candidates) == 0 || len(geminiResp.Candidates[0].Content.Parts) == 0 {
		return "", fmt.Errorf("Gemini 未返回有效内容")
	}

	content := strings.TrimSpace(geminiResp.Candidates[0].Content.Parts[0].Text)
	return content, nil
}
