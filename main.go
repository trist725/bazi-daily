package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
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

const (
	defaultBaseURL = "http://localhost:11434/v1"
	defaultModel   = "qwen3.5:27b"
	systemPrompt   = "你现在是我的私人能量管理系统。请严格按照我的原局（庚午、癸未、辛卯、戊戌）与今日干支进行推演，输出：核心引动、能量体感预测、今日策略（宜/忌）。"
)

func main() {
	now := time.Now()

	baseURL := getEnv("LLM_BASE_URL", defaultBaseURL)
	modelName := getEnv("LLM_MODEL", defaultModel)

	promptContent := buildPrompt(now)
	respText, err := chatWithLocalModel(baseURL, modelName, systemPrompt, promptContent)
	if err != nil {
		fmt.Println("调用本地模型失败:", err)
		return
	}

	fmt.Println("========== 今日日课简报 ==========")
	fmt.Println(respText)
	fmt.Println("===================================")
}

func buildPrompt(t time.Time) string {
	d := calendar.NewLunarFromDate(t)

	yearGanzhi := d.GetYearInGanZhi()
	monthGanzhi := d.GetMonthInGanZhi()
	dayGanzhi := d.GetDayInGanZhi()

	currentDate := t.Format("2006年01月02日")
	return fmt.Sprintf("%s，%s年%s月%s日", currentDate, yearGanzhi, monthGanzhi, dayGanzhi)
}

func chatWithLocalModel(baseURL, modelName, systemPrompt, userPrompt string) (string, error) {
	url := strings.TrimRight(baseURL, "/") + "/chat/completions"

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

func getEnv(key, fallback string) string {
	value := strings.TrimSpace(os.Getenv(key))
	if value == "" {
		return fallback
	}
	return value
}
