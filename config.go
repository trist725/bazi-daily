package main

import (
	"fmt"
	"os"
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
	SystemPrompt     string `json:"system_prompt"`
	JudgePrompt      string `json:"judge_prompt"`
	SystemPromptFile string `json:"system_prompt_file"`
	JudgePromptFile  string `json:"judge_prompt_file"`

	JudgeEnabled  bool   `json:"judge_enabled"`
	JudgeModel    string `json:"judge_model"`
	JudgeProvider string `json:"judge_provider"`

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
	BaseURL: "http://localhost:11434",

	SystemPrompt: "你现在是我的私人能量管理系统。请严格按照我的原局（庚午、癸未、辛卯、戊戌）与今日干支进行推演，输出：核心引动、能量体感预测、今日策略（宜/忌）。",
	JudgePrompt:  "你是一个严谨的最终结论整合助手。请先横向比较，再生成一份可直接采用的最终答案，并输出今日运势评分。",

	SystemPromptFile: "prompts/system_prompt.txt",
	JudgePromptFile:  "prompts/judge_prompt.txt",

	JudgeEnabled:  true,
	JudgeModel:    "gemini-flash-latest",
	JudgeProvider: "gemini",

	ModelFilter: []string{"qwen", "gemma", "glm", "deepseek"},
	ModelSkip:   []string{"embed", "embedding", "bge", "rerank", "reranker", "llava", "vision", "vl", "32b", "72b", "coder", "9b", "27b"},
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
