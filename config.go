package main

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"
)

type CloudModelConfig struct {
	Enabled    bool   `json:"enabled"`
	Name       string `json:"name"`
	Provider   string `json:"provider"`
	APIKey     string `json:"api_key"`
	APIKeyFile string `json:"api_key_file"`
}

type Config struct {
	BaseURL          string `json:"base_url"`
	LMStudioURL      string `json:"lmstudio_url"`
	SystemPrompt     string `json:"system_prompt"`
	JudgePrompt      string `json:"judge_prompt"`
	SystemPromptFile string `json:"system_prompt_file"`
	JudgePromptFile  string `json:"judge_prompt_file"`

	OllamaEnabled   bool   `json:"ollama_enabled"`
	LMStudioEnabled bool   `json:"lmstudio_enabled"`
	JudgeEnabled    bool   `json:"judge_enabled"`
	JudgeModel      string `json:"judge_model"`
	JudgeProvider   string `json:"judge_provider"`

	ModelFilter []string `json:"model_filter"`
	ModelSkip   []string `json:"model_skip"`
	ModelLimit  int      `json:"model_limit"`

	LocalCallTimeout       time.Duration `json:"local_call_timeout"`
	LocalUnloadTimeout     time.Duration `json:"local_unload_timeout"`
	LocalPreflightTimeout  time.Duration `json:"local_preflight_timeout"`
	LocalRetryCount        int           `json:"local_retry_count"`
	LocalSwitchDelay       time.Duration `json:"local_switch_delay"`
	CloudCallTimeout       time.Duration `json:"cloud_call_timeout"`
	CloudSwitchDelay       time.Duration `json:"cloud_switch_delay"`
	RunningPollInterval    time.Duration `json:"running_poll_interval"`
	StartupCleanupEnabled  bool          `json:"startup_cleanup_enabled"`
	ContinueOnCleanupError bool          `json:"continue_on_cleanup_error"`

	LocalReleaseRetryCount int           `json:"local_release_retry_count"`
	LocalReleaseRetryDelay time.Duration `json:"local_release_retry_delay"`

	CloudModels []CloudModelConfig `json:"cloud_models"`
}

var appConfig = Config{
	BaseURL:     "http://127.0.0.1:11434",
	LMStudioURL: "http://127.0.0.1:1234/v1",

	SystemPrompt: "你现在是我的私人能量管理系统。请严格按照我的原局进行推演。",
	JudgePrompt:  "你是一个严谨的最终结论整合助手。",

	SystemPromptFile: "prompts/system_prompt.txt",
	JudgePromptFile:  "prompts/judge_prompt.txt",

	OllamaEnabled:   false,
	LMStudioEnabled: true,
	JudgeEnabled:    true,
	JudgeModel:      "gemini-cli",
	JudgeProvider:   "local-cli",

	ModelFilter: []string{"qwen", "gemma", "deepseek"},
	ModelSkip:   []string{"embed", "embedding", "bge", "rerank", "reranker", "vision", "vl", "llava", "coder", "9b"},
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
	ContinueOnCleanupError: true,

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

func loadRuntimeResources(cfg *Config) error {
	systemPrompt, _ := loadPromptWithFallback(cfg.SystemPromptFile, cfg.SystemPrompt)
	judgePrompt, _ := loadPromptWithFallback(cfg.JudgePromptFile, cfg.JudgePrompt)
	cfg.SystemPrompt = systemPrompt
	cfg.JudgePrompt = judgePrompt

	for i := range cfg.CloudModels {
		key, err := loadAPIKeyWithFallback(cfg.CloudModels[i].APIKeyFile, cfg.CloudModels[i].APIKey)
		if err != nil {
			fmt.Printf("警告：API Key 加载失败 (%s): %v\n", cfg.CloudModels[i].Name, err)
		}
		cfg.CloudModels[i].APIKey = key
	}
	return nil
}

func loadPromptWithFallback(filePath string, fallback string) (string, error) {
	data, err := os.ReadFile(filePath)
	if err != nil {
		abs, _ := filepath.Abs(filePath)
		fmt.Printf("提示：未找到 Prompt 文件，使用内置默认值。路径: %s\n", abs)
		return strings.TrimSpace(fallback), nil
	}
	fmt.Printf("已加载 Prompt 文件: %s\n", filePath)
	return strings.TrimSpace(string(data)), nil
}

func loadAPIKeyWithFallback(filePath string, fallback string) (string, error) {
	data, err := os.ReadFile(filePath)
	if err != nil {
		abs, _ := filepath.Abs(filePath)
		return "", fmt.Errorf("无法读取 API Key 文件，绝对路径: %s, 错误: %v", abs, err)
	}

	key := strings.TrimSpace(string(data))
	if key == "" || key == "YOUR_GEMINI_API_KEY" {
		return "", fmt.Errorf("API Key 文件内容为空或仍为占位符")
	}

	// 脱敏打印
	displayKey := "****"
	if len(key) > 6 {
		displayKey = key[:6] + "..."
	}
	fmt.Printf("已成功加载 API Key: %s (前缀: %s, 长度: %d)\n", filePath, displayKey, len(key))
	return key, nil
}
