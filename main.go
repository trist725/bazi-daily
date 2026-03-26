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

	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, os.Interrupt, syscall.SIGTERM)
	go func() {
		<-sigChan
		fmt.Println("\n接收到中断信号，正在执行清理...")
		_ = finalCleanupLocalModels(appConfig.BaseURL, appConfig.LocalUnloadTimeout, appConfig.RunningPollInterval)
		os.Exit(0)
	}()

	startedAt := time.Now()
	promptContent := buildPrompt(startedAt)

	// 如果今日报告已生成，直接打开并退出
	baseReportDir := filepath.Join("reports", startedAt.Format("2006-01-02"))
	existingFinalPath := filepath.Join(baseReportDir, "final.html")
	if _, err := os.Stat(existingFinalPath); err == nil {
		fmt.Printf("今日报告已存在: %s，直接打开。\n", existingFinalPath)
		_ = openInDefaultBrowser(existingFinalPath)
		return
	}

	reportDir, err := createReportDir(startedAt)
	if err != nil {
		fmt.Printf("创建报告目录失败: %v\n", err)
		return
	}
	_ = saveRunMeta(reportDir, startedAt, promptContent)

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
				if res, ok := findExistingModelResultToday(c.Name); ok {
					fmt.Printf("[云端] %s 发现今日已有成功报告，跳过调用。\n", c.Name)
					mu.Lock()
					results = append(results, *res)
					mu.Unlock()
					_ = saveSingleModelReport(reportDir, startedAt, *res)
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
				_ = saveSingleModelReport(reportDir, startedAt, res)

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
			if res, ok := findExistingModelResultToday(mName); ok {
				fmt.Printf("[%d/%d] 模型 %s 发现今日已有成功报告，跳过调用。\n", i+1, len(localModels), mName)
				mu.Lock()
				results = append(results, *res)
				mu.Unlock()
				_ = saveSingleModelReport(reportDir, startedAt, *res)
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

			_ = saveSingleModelReport(reportDir, startedAt, res)

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
	if res, ok := findExistingModelResultToday("Local-Gemini-CLI"); ok {
		fmt.Printf("[CLI] 发现今日已有成功报告，跳过调用。\n")
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
	_ = saveSingleModelReport(reportDir, startedAt, cliRes)
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
		_ = saveJudgeReport(reportDir, startedAt, judgeResult)

		if err != nil {
			fmt.Printf("<<< 裁判整合失败: %v\n", err)
		} else {
			fmt.Printf("<<< 裁判整合成功，耗时: %s\n", judgeResult.CallDuration.Round(time.Millisecond))
		}
	} else if appConfig.JudgeEnabled {
		fmt.Println(">>> 跳过裁判整合：没有任何模型成功返回结果。")
	}

	totalDuration := time.Since(startedAt)
	_ = saveSummaryReport(reportDir, startedAt, promptContent, results, judgeResult, totalDuration)
	finalPath, _ := saveFinalConclusionHTML(reportDir, startedAt, promptContent, results, judgeResult, totalDuration)

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

	// 1. 如果指定了使用本地 CLI
	if strings.EqualFold(cfg.JudgeProvider, "local-cli") {
		return chatWithLocalCLI(cfg.JudgePrompt, input, cfg.CloudCallTimeout)
	}

	// 2. 如果指定了云端模型
	if cloud, ok := findCloudModelConfig(judgeModel, cfg.CloudModels); ok {
		return chatWithCloudModel(cloud, cfg.JudgePrompt, input, cfg.CloudCallTimeout)
	}

	// 3. 否则降级使用本地 Ollama
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
