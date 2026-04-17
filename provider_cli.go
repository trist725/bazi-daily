package main

import (
	"bytes"
	"context"
	"fmt"
	"os/exec"
	"strings"
	"time"
)

// chatWithLocalCLI 调用本地命令行工具（如 gemini-cli）
func chatWithLocalCLI(modelName, systemPrompt, userPrompt string, timeout time.Duration) (string, error) {
	// 构造输入：将 System Prompt 和 User Prompt 组合
	combinedInput := fmt.Sprintf("%s\n\n%s", systemPrompt, userPrompt)

	// 使用 context 处理超时
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()

	// 构造命令
	args := []string{"ask"}
	if modelName != "" && modelName != "gemini-cli" && modelName != "local-cli" {
		args = append(args, "--model", modelName)
	}

	cmd := exec.CommandContext(ctx, "gemini", args...)
	
	cmd.Stdin = strings.NewReader(combinedInput)
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr

	fmt.Printf("[本地 CLI] 正在启动进程执行审计...\n")
	err := cmd.Run()
	if err != nil {
		return "", fmt.Errorf("本地 CLI 执行失败: %v, 错误输出: %s", err, stderr.String())
	}

	return strings.TrimSpace(stdout.String()), nil
}
