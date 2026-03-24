package main

import "time"

// LLM 消息结构
type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// Ollama 相关协议
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

// Gemini 相关协议
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

// 内部处理结果结构
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
