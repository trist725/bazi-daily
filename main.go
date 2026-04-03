package main

import (
	"fmt"
	"os"
	"os/exec"
	"os/signal"
	"runtime"
	"strings"
	"sync"
	"syscall"
	"time"

	"github.com/6tail/lunar-go/calendar"
)

func main() {
	// 0. 在 Windows 环境下强制设置控制台编码为 UTF-8
	if runtime.GOOS == "windows" {
		_ = exec.Command("chcp", "65001").Run()
	}

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

	reportDir, err := createReportDir(targetTime)
	if err != nil {
		fmt.Printf("创建报告目录失败: %v\n", err)
		return
	}
	_ = saveRunMeta(reportDir, targetTime, promptContent)

	var localModels []string
	if appConfig.OllamaEnabled {
		localModels, err = resolveModels(appConfig.BaseURL, appConfig)
		if err != nil {
			fmt.Printf("警告：获取 Ollama 模型列表失败: %v\n", err)
		}
	}

	var lmStudioModels []string
	if appConfig.LMStudioEnabled {
		rawLMModels, _ := resolveLMStudioModels(appConfig.LMStudioURL)
		lmStudioModels = applyModelSelectionRules(rawLMModels, appConfig)
		if len(lmStudioModels) > 0 {
			fmt.Printf("发现 LM Studio 模型: %v\n", lmStudioModels)
		}
	}

	cloudModels := enabledCloudModels(appConfig.CloudModels)

	fmt.Printf("本次任务：云端 %d 个，Ollama %d 个，LM Studio %d 个\n", len(cloudModels), len(localModels), len(lmStudioModels))

	var results []ModelResult
	var mu sync.Mutex
	anyNewModelRun := false

	// 1. 并发执行云端模型
	if len(cloudModels) > 0 {
		fmt.Println(">>> 正在并发调用云端模型...")
		var wg sync.WaitGroup
		for _, cloud := range cloudModels {
			wg.Add(1)
			go func(c CloudModelConfig) {
				defer wg.Done()

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
				if err == nil {
					anyNewModelRun = true
				}
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

	// 2. 串行执行本地 Ollama 模型
	if len(localModels) > 0 {
		fmt.Println(">>> 正在串行调用 Ollama 模型...")
		for i, mName := range localModels {
			fmt.Printf("[%d/%d] Ollama 模型: %s\n", i+1, len(localModels), mName)

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
			if err == nil {
				anyNewModelRun = true
			}
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
		fmt.Println("<<< Ollama 模型阶段结束。")
	}

	// 2.1 串行执行本地 LM Studio 模型
	if len(lmStudioModels) > 0 {
		fmt.Println(">>> 正在串行调用 LM Studio 模型...")
		for i, mName := range lmStudioModels {
			fmt.Printf("[%d/%d] LM Studio 模型: %s\n", i+1, len(lmStudioModels), mName)

			if res, ok := findExistingModelResultToday(targetTime, mName); ok {
				fmt.Printf("[%d/%d] 模型 %s 已有成功报告，跳过。\n", i+1, len(lmStudioModels), mName)
				mu.Lock()
				results = append(results, *res)
				mu.Unlock()
				_ = saveSingleModelReport(reportDir, targetTime, *res)
				continue
			}

			// 显式加载模型
			fmt.Printf("   正在加载模型 %s...\n", mName)
			if loadErr := loadLMStudioModel(appConfig.LMStudioURL, mName); loadErr != nil {
				fmt.Printf("   加载失败: %v\n", loadErr)
				// 尝试直接调用，也许已经加载了
			}

			start := time.Now()
			content, err := chatWithLMStudio(appConfig.LMStudioURL, mName, appConfig.SystemPrompt, promptContent, appConfig.LocalCallTimeout)
			res := ModelResult{
				Model:         mName,
				Content:       content,
				Err:           err,
				Provider:      "lmstudio",
				CallDuration:  time.Since(start),
				TotalDuration: time.Since(start),
			}

			mu.Lock()
			results = append(results, res)
			if err == nil {
				anyNewModelRun = true
			}
			mu.Unlock()

			_ = saveSingleModelReport(reportDir, targetTime, res)

			if err != nil {
				fmt.Printf("[%d/%d] LM Studio 模型 %s 失败: %v\n", i+1, len(lmStudioModels), mName, err)
			} else {
				fmt.Printf("[%d/%d] LM Studio 模型 %s 成功，耗时: %s\n", i+1, len(lmStudioModels), mName, res.CallDuration.Round(time.Millisecond))
			}

			// 显式卸载模型以释放显存
			fmt.Printf("   正在卸载模型 %s...\n", mName)
			_ = unloadLMStudioModel(appConfig.LMStudioURL, mName)

			if i < len(lmStudioModels)-1 {
				time.Sleep(appConfig.LocalSwitchDelay)
			}
		}
		fmt.Println("<<< LM Studio 模型阶段结束。")
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
		if cliErr == nil {
			anyNewModelRun = true
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
		if !anyNewModelRun {
			if res, ok := findExistingJudgeResultToday(targetTime); ok {
				fmt.Println(">>> 发现已有成功裁判报告且无新模型运行，跳过整合。")
				judgeResult = *res
				goto assemble
			}
		}

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

assemble:
	totalDuration := time.Since(startedAt)
	_ = saveSummaryReport(reportDir, targetTime, promptContent, results, judgeResult, totalDuration)
	finalPath, _ := saveFinalConclusionHTML(reportDir, targetTime, promptContent, results, judgeResult, totalDuration)

	fmt.Printf("\n任务结束！总耗时: %s\n", totalDuration.Round(time.Millisecond))

	if finalPath != "" {
		fmt.Printf("最终结论报告: %s\n", finalPath)
		_ = openInDefaultBrowser(finalPath)
	}
}

func buildPrompt(t time.Time) string {
	d := calendar.NewLunarFromDate(t)
	gz := d.GetEightChar()
	return fmt.Sprintf("今日公历：%s，农历：%s年%s月%s日，八字：%s %s %s %s",
		t.Format("2006-01-02"),
		d.GetYearInChinese(), d.GetMonthInChinese(), d.GetDayInChinese(),
		gz.GetYear(), gz.GetMonth(), gz.GetDay(), gz.GetTime(),
	)
}

func enabledCloudModels(models []CloudModelConfig) []CloudModelConfig {
	var enabled []CloudModelConfig
	for _, m := range models {
		if m.Enabled {
			enabled = append(enabled, m)
		}
	}
	return enabled
}

func resolveJudgeModel(localModels []string, cfg Config) string {
	if cfg.JudgeModel != "" {
		return cfg.JudgeModel
	}
	return "gemini-flash-latest"
}

func judgeModelResults(cfg Config, localModels []string, judgeModel, originalPrompt string, results []ModelResult) (string, error) {
	var sb strings.Builder
	sb.WriteString("以下是多个 AI 模型对今日能量管理的推演结论，请你作为首席架构师进行整合：\n\n")

	for _, r := range results {
		if r.Err == nil {
			sb.WriteString(fmt.Sprintf("### 模型 %s 的结论：\n%s\n\n", r.Model, r.Content))
		}
	}

	input := sb.String()

	if strings.EqualFold(cfg.JudgeProvider, "local-cli") {
		fmt.Printf("[裁判] 尝试使用本地 CLI (%s)...\n", judgeModel)
		content, err := chatWithLocalCLI(cfg.JudgePrompt, input, cfg.CloudCallTimeout)
		if err == nil {
			return content, nil
		}
		fmt.Printf("[裁判] 本地 CLI 失败: %v。将尝试 Fallback 到云端模型...\n", err)
	}

	for _, cloud := range cfg.CloudModels {
		if cloud.Enabled && (cloud.Name == judgeModel || judgeModel == "gemini-flash-latest" || judgeModel == "gemini-cli") {
			fmt.Printf("[裁判] 正在使用云端模型 Fallback: %s...\n", cloud.Name)
			return chatWithCloudModel(cloud, cfg.JudgePrompt, input, cfg.CloudCallTimeout)
		}
	}

	for _, cloud := range cfg.CloudModels {
		if cloud.Enabled {
			fmt.Printf("[裁判] 尝试使用第一个可用云端模型: %s...\n", cloud.Name)
			return chatWithCloudModel(cloud, cfg.JudgePrompt, input, cfg.CloudCallTimeout)
		}
	}

	return chatWithOllama(cfg.BaseURL, judgeModel, cfg.JudgePrompt, input, cfg.LocalCallTimeout)
}
