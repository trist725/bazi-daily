package main

import (
	"fmt"
	"os"
	"os/signal"
	"path/filepath"
	"strings"
	"sync"
	"syscall"
	"time"

	"github.com/6tail/lunar-go/calendar"
)

func main() {
	// 1. 设置全局时区为北京时间
	beijingLoc := time.FixedZone("CST", 8*3600)
	time.Local = beijingLoc

	if err := loadRuntimeResources(&appConfig); err != nil {
		fmt.Printf("加载配置资源失败: %v\n", err)
		return
	}

	startedAt := time.Now()
	// 解析日期参数
	targetTime := time.Now()
	if len(os.Args) > 1 {
		parsedTime, err := time.ParseInLocation("2006-01-02", os.Args[1], beijingLoc)
		if err == nil {
			targetTime = parsedTime
			fmt.Printf(">>> 目标日期设定为: %s\n", targetTime.Format("2006-01-02"))
		} else {
			fmt.Printf("警告：日期格式错误 (需为 YYYY-MM-DD)，将使用当前时间。\n")
		}
	}

	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, os.Interrupt, syscall.SIGTERM)
	go func() {
		<-sigChan
		fmt.Println("\n接收到中断信号，正在执行清理...")
		_ = finalCleanupLocalModels(appConfig.BaseURL, appConfig.LocalUnloadTimeout, appConfig.RunningPollInterval)
		os.Exit(0)
	}()

	promptContent := buildPrompt(targetTime)

	// 如果目标日期报告已生成，直接打开并退出
	baseReportDir := filepath.Join("reports", targetTime.Format("2006-01-02"))
	existingFinalPath := filepath.Join(baseReportDir, "final.html")
	if _, err := os.Stat(existingFinalPath); err == nil {
		fmt.Printf("目标日期报告已存在: %s，直接打开。\n", existingFinalPath)
		_ = openInDefaultBrowser(existingFinalPath)
		return
	}

	reportDir, err := createReportDir(targetTime)
	if err != nil {
		fmt.Printf("创建报告目录失败: %v\n", err)
		return
	}
	_ = saveRunMeta(reportDir, targetTime, promptContent)

	localModels, err := resolveModels(appConfig.BaseURL, appConfig)
	if err != nil {
		fmt.Printf("警告：获取本地模型列表失败（请检查 Ollama 是否启动）: %v\n", err)
	}
	cloudModels := enabledCloudModels(appConfig.CloudModels)

	fmt.Printf("本次任务：云端模型 %d 个，本地模型 %d 个\n", len(cloudModels), len(localModels))

	var results []ModelResult
	var mu sync.Mutex

	// 1. 并发执行云端模型
	if len(cloudModels) > 0 {
		fmt.Println(">>> 正在并发调用云端模型...")
		var wg sync.WaitGroup
		for _, cloud := range cloudModels {
			wg.Add(1)
			go func(c CloudModelConfig) {
				defer wg.Done()

				// 检测是否已有成功报告
				if res, ok := findExistingModelResultToday(targetTime, c.Name); ok {
					fmt.Printf("[云端] %s 发现已有成功报告，跳过调用。\n", c.Name)
					mu.Lock()
					results = append(results, *res)
					mu.Unlock()
					_ = saveSingleModelReport(reportDir, targetTime, *res)
					return
				}

				start := time.Now()
				content, err := chatWithCloudModel(c, appConfig.SystemPrompt, promptContent, appConfig.CloudCallTimeout)
				res := ModelResult{
					Model:         c.Name,
					Content:       content,
					Err:           err,
					Provider:      c.Provider,
					CallDuration:  time.Since(start),
					TotalDuration: time.Since(start),
				}
				mu.Lock()
				results = append(results, res)
				mu.Unlock()
				_ = saveSingleModelReport(reportDir, targetTime, res)

				if err != nil {
					fmt.Printf("[云端] %s 调用失败: %v\n", c.Name, err)
				} else {
					fmt.Printf("[云端] %s 调用成功，耗时: %s\n", c.Name, res.CallDuration.Round(time.Millisecond))
				}
			}(cloud)
		}
		wg.Wait()
		fmt.Println("<<< 云端模型阶段结束。")
	}

	// 2. 串行执行本地模型
	if len(localModels) > 0 {
		fmt.Println(">>> 正在串行调用本地模型...")
		for i, mName := range localModels {
			fmt.Printf("[%d/%d] 本地模型: %s\n", i+1, len(localModels), mName)

			// 检测是否已有成功报告
			if res, ok := findExistingModelResultToday(targetTime, mName); ok {
				fmt.Printf("[%d/%d] 模型 %s 发现已有成功报告，跳过调用。\n", i+1, len(localModels), mName)
				mu.Lock()
				results = append(results, *res)
				mu.Unlock()
				_ = saveSingleModelReport(reportDir, targetTime, *res)
				continue
			}

			_ = ensureNoModelsRunning(appConfig.BaseURL, appConfig.LocalPreflightTimeout, appConfig.RunningPollInterval)

			start := time.Now()
			content, cost, err := chatWithOllamaWithRetry(
				appConfig.BaseURL, mName, appConfig.SystemPrompt, promptContent,
				appConfig.LocalRetryCount, appConfig.LocalCallTimeout, appConfig.LocalUnloadTimeout, appConfig.RunningPollInterval,
			)
			res := ModelResult{
				Model:         mName,
				Content:       content,
				Err:           err,
				Provider:      "ollama",
				CallDuration:  cost,
				TotalDuration: time.Since(start),
			}

			mu.Lock()
			results = append(results, res)
			mu.Unlock()

			_ = saveSingleModelReport(reportDir, targetTime, res)

			if err != nil {
				fmt.Printf("[%d/%d] 模型 %s 调用失败: %v\n", i+1, len(localModels), mName, err)
			} else {
				fmt.Printf("[%d/%d] 模型 %s 调用完成，耗时: %s\n", i+1, len(localModels), mName, cost.Round(time.Millisecond))
			}

			_ = releaseLocalModelWithRetry(appConfig.BaseURL, mName, appConfig.LocalUnloadTimeout, appConfig.RunningPollInterval, appConfig.LocalReleaseRetryCount, appConfig.LocalReleaseRetryDelay)

			if i < len(localModels)-1 {
				time.Sleep(appConfig.LocalSwitchDelay)
			}
		}
		fmt.Println("<<< 本地模型阶段结束。")
	}

	// 3. 调用本地 Gemini CLI 参与推演
	fmt.Println(">>> 正在调用本地 Gemini CLI 参与推演...")

	var cliRes ModelResult
	if res, ok := findExistingModelResultToday(targetTime, "Local-Gemini-CLI"); ok {
		fmt.Printf("[CLI] 发现已有成功报告，跳过调用。\n")
		cliRes = *res
	} else {
		cliStart := time.Now()
		cliContent, cliErr := chatWithLocalCLI(appConfig.SystemPrompt, promptContent, appConfig.CloudCallTimeout)
		cliRes = ModelResult{
			Model:         "Local-Gemini-CLI",
			Content:       cliContent,
			Err:           cliErr,
			Provider:      "local-cli",
			CallDuration:  time.Since(cliStart),
			TotalDuration: time.Since(cliStart),
		}
	}

	mu.Lock()
	results = append(results, cliRes)
	mu.Unlock()
	_ = saveSingleModelReport(reportDir, targetTime, cliRes)
	if cliRes.Err != nil {
		fmt.Printf("[CLI] 调用失败: %v\n", cliRes.Err)
	} else if cliRes.Provider != "existing-report" {
		fmt.Printf("[CLI] 调用成功，耗时: %s\n", cliRes.CallDuration.Round(time.Millisecond))
	}

	// 4. 裁判整合
	var judgeResult JudgeResult
	successCount := 0
	for _, r := range results {
		if r.Err == nil && r.Content != "" {
			successCount++
		}
	}

	if appConfig.JudgeEnabled && successCount > 0 {
		fmt.Println(">>> 正在执行裁判整合...")
		judgeModel := resolveJudgeModel(localModels, appConfig)
		start := time.Now()
		content, err := judgeModelResults(appConfig, localModels, judgeModel, promptContent, results)
		judgeResult = JudgeResult{
			Model:        judgeModel,
			Content:      content,
			Err:          err,
			Enabled:      true,
			CallDuration: time.Since(start),
		}
		_ = saveJudgeReport(reportDir, targetTime, judgeResult)

		if err != nil {
			fmt.Printf("<<< 裁判整合失败: %v\n", err)
		} else {
			fmt.Printf("<<< 裁判整合成功，耗时: %s\n", judgeResult.CallDuration.Round(time.Millisecond))
		}
	} else if appConfig.JudgeEnabled {
		fmt.Println(">>> 跳过裁判整合：没有任何模型成功返回结果。")
	}

	totalDuration := time.Since(startedAt)
	_ = saveSummaryReport(reportDir, targetTime, promptContent, results, judgeResult, totalDuration)
	finalPath, _ := saveFinalConclusionHTML(reportDir, targetTime, promptContent, results, judgeResult, totalDuration)

	fmt.Printf("\n任务结束！总耗时: %s\n", totalDuration.Round(time.Millisecond))

	if finalPath != "" {
		fmt.Printf("最终结论报告: %s\n", finalPath)
		_ = openInDefaultBrowser(finalPath)
	}

	_ = finalCleanupLocalModels(appConfig.BaseURL, appConfig.LocalUnloadTimeout, appConfig.RunningPollInterval)
}

func buildPrompt(t time.Time) string {
	d := calendar.NewLunarFromDate(t)
	return fmt.Sprintf("%s，%s年%s月%s日", t.Format("2006年01月02日"), d.GetYearInGanZhi(), d.GetMonthInGanZhi(), d.GetDayInGanZhi())
}

func enabledCloudModels(models []CloudModelConfig) []CloudModelConfig {
	var res []CloudModelConfig
	for _, m := range models {
		if m.Enabled {
			res = append(res, m)
		}
	}
	return res
}

func resolveJudgeModel(localModels []string, cfg Config) string {
	if cfg.JudgeModel != "" {
		return cfg.JudgeModel
	}
	return "gemini-flash-latest"
}

func judgeModelResults(cfg Config, localModels []string, judgeModel, originalPrompt string, results []ModelResult) (string, error) {
	var sb strings.Builder
	sb.WriteString("以下是同一个问题的多模型输出结果，请你整合出一份最终结论。\n\n")
	for _, r := range results {
		if r.Err == nil && r.Content != "" {
			sb.WriteString(fmt.Sprintf("--- 模型 %s ---\n%s\n\n", r.Model, r.Content))
		}
	}
	sb.WriteString("请根据以上内容生成最终结论。")

	input := sb.String()

	// 1. 尝试使用首选 Provider (本地 CLI)
	if strings.EqualFold(cfg.JudgeProvider, "local-cli") {
		fmt.Printf("[裁判] 尝试使用本地 CLI (%s)...\n", judgeModel)
		content, err := chatWithLocalCLI(cfg.JudgePrompt, input, cfg.CloudCallTimeout)
		if err == nil {
			return content, nil
		}
		fmt.Printf("[裁判] 本地 CLI 失败: %v。将尝试 Fallback 到云端模型...\n", err)
	}

	// 2. Fallback 或 直接调用云端模型
	for _, cloud := range cfg.CloudModels {
		if cloud.Enabled && (cloud.Name == judgeModel || judgeModel == "gemini-flash-latest" || judgeModel == "gemini-cli") {
			fmt.Printf("[裁判] 正在使用云端模型 Fallback: %s...\n", cloud.Name)
			return chatWithCloudModel(cloud, cfg.JudgePrompt, input, cfg.CloudCallTimeout)
		}
	}

	// 3. 兜底尝试第一个开启的云端模型
	for _, cloud := range cfg.CloudModels {
		if cloud.Enabled {
			fmt.Printf("[裁判] 尝试使用第一个可用的云端模型: %s...\n", cloud.Name)
			return chatWithCloudModel(cloud, cfg.JudgePrompt, input, cfg.CloudCallTimeout)
		}
	}

	// 4. 降级使用本地 Ollama
	return chatWithOllama(cfg.BaseURL, judgeModel, cfg.JudgePrompt, input, cfg.LocalCallTimeout)
}

func findCloudModelConfig(name string, clouds []CloudModelConfig) (CloudModelConfig, bool) {
	for _, c := range clouds {
		if c.Name == name {
			return c, true
		}
	}
	return CloudModelConfig{}, false
}
