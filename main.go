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
		parsedDate, err := time.ParseInLocation("2006-01-02", os.Args[1], beijingLoc)
		if err == nil {
			// 如果指定了日期，则将目标日期的年月日设为指定值，保留当前时分秒
			now := time.Now().In(beijingLoc)
			targetTime = time.Date(parsedDate.Year(), parsedDate.Month(), parsedDate.Day(),
				now.Hour(), now.Minute(), now.Second(), now.Nanosecond(), beijingLoc)
			fmt.Printf(">>> 目标日期设定为: %s (保留当前时刻)\n", targetTime.Format("2006-01-02 15:04:05"))
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
	totalTasks := len(cloudModels) + len(localModels) + len(lmStudioModels)
	if appConfig.LocalCLIEnabled {
		totalTasks++
	}

	fmt.Printf(">>> 任务就绪：共 %d 个计算节点 (云端 %d, Ollama %d, LM Studio %d)\n",
		totalTasks, len(cloudModels), len(localModels), len(lmStudioModels))

	var results []ModelResult
	var mu sync.Mutex
	anyNewModelRun := false
	taskCounter := 0

	// 1. 并发执行云端模型
	if len(cloudModels) > 0 {
		fmt.Println("\n[PHASE 1] 正在并发调度云端节点...")
		var wg sync.WaitGroup
		for _, cloud := range cloudModels {
			wg.Add(1)
			go func(c CloudModelConfig) {
				defer wg.Done()

				mu.Lock()
				taskCounter++
				currentTask := taskCounter
				mu.Unlock()

				prefix := fmt.Sprintf("[%d/%d] [Cloud] %s", currentTask, totalTasks, c.Name)

				if res, ok := findExistingModelResultToday(targetTime, c.Name); ok {
					fmt.Printf("%s -> SKIP (发现已有报告)\n", prefix)
					mu.Lock()
					results = append(results, *res)
					mu.Unlock()
					_ = saveSingleModelReport(reportDir, targetTime, *res)
					return
				}

				fmt.Printf("%s -> START\n", prefix)
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
					fmt.Printf("%s -> FAULT: %v\n", prefix, err)
				} else {
					fmt.Printf("%s -> SUCCESS (%s)\n", prefix, res.CallDuration.Round(time.Millisecond))
				}
			}(cloud)
		}
		wg.Wait()
	}

	// 2. 串行执行本地 Ollama 模型
	if len(localModels) > 0 {
		fmt.Println("\n[PHASE 2] 正在串行调度 Ollama 节点...")
		for _, mName := range localModels {
			taskCounter++
			prefix := fmt.Sprintf("[%d/%d] [Ollama] %s", taskCounter, totalTasks, mName)

			if res, ok := findExistingModelResultToday(targetTime, mName); ok {
				fmt.Printf("%s -> SKIP (发现已有报告)\n", prefix)
				mu.Lock()
				results = append(results, *res)
				mu.Unlock()
				_ = saveSingleModelReport(reportDir, targetTime, *res)
				continue
			}

			fmt.Printf("%s -> START\n", prefix)
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
				fmt.Printf("%s -> FAULT: %v\n", prefix, err)
			} else {
				fmt.Printf("%s -> SUCCESS (%s)\n", prefix, cost.Round(time.Millisecond))
			}

			_ = releaseLocalModelWithRetry(appConfig.BaseURL, mName, appConfig.LocalUnloadTimeout, appConfig.RunningPollInterval, appConfig.LocalReleaseRetryCount, appConfig.LocalReleaseRetryDelay)
		}
	}

	// 2.1 串行执行本地 LM Studio 模型
	if len(lmStudioModels) > 0 {
		fmt.Println("\n[PHASE 3] 正在串行调度 LM Studio 节点...")
		for _, mName := range lmStudioModels {
			taskCounter++
			prefix := fmt.Sprintf("[%d/%d] [LMStudio] %s", taskCounter, totalTasks, mName)

			if res, ok := findExistingModelResultToday(targetTime, mName); ok {
				fmt.Printf("%s -> SKIP (发现已有报告)\n", prefix)
				mu.Lock()
				results = append(results, *res)
				mu.Unlock()
				_ = saveSingleModelReport(reportDir, targetTime, *res)
				continue
			}

			fmt.Printf("%s -> START (Loading...)\n", prefix)
			if loadErr := loadLMStudioModel(appConfig.LMStudioURL, mName); loadErr != nil {
				fmt.Printf("   警告: 加载失败: %v (尝试直接调用)\n", loadErr)
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
				fmt.Printf("%s -> FAULT: %v\n", prefix, err)
			} else {
				fmt.Printf("%s -> SUCCESS (%s)\n", prefix, res.CallDuration.Round(time.Millisecond))
			}

			_ = unloadLMStudioModel(appConfig.LMStudioURL, mName)
		}
	}

	// 3. 调用本地 Gemini CLI 参与推演
	if appConfig.LocalCLIEnabled {
		taskCounter++
		fmt.Println("\n[PHASE 4] 正在调度本地 Gemini CLI 节点...")

		cliModelName := "gemini-3.1-pro-preview"
		displayName := fmt.Sprintf("Gemini-3.1-Pro (CLI)")
		prefix := fmt.Sprintf("[%d/%d] [LocalCLI] %s", taskCounter, totalTasks, displayName)

		var cliRes ModelResult
		if res, ok := findExistingModelResultToday(targetTime, displayName); ok {
			fmt.Printf("%s -> SKIP (发现已有报告)\n", prefix)
			cliRes = *res
		} else {
			fmt.Printf("%s -> START\n", prefix)
			cliStart := time.Now()
			cliContent, cliErr := chatWithLocalCLI(cliModelName, appConfig.SystemPrompt, promptContent, appConfig.CloudCallTimeout)
			cliRes = ModelResult{
				Model:         displayName,
				Content:       cliContent,
				Err:           cliErr,
				Provider:      "local-cli",
				CallDuration:  time.Since(cliStart),
				TotalDuration: time.Since(cliStart),
			}
			if cliErr == nil {
				anyNewModelRun = true
				fmt.Printf("%s -> SUCCESS (%s)\n", prefix, cliRes.CallDuration.Round(time.Millisecond))
			} else {
				fmt.Printf("%s -> FAULT: %v\n", prefix, cliErr)
			}
		}

		mu.Lock()
		results = append(results, cliRes)
		mu.Unlock()
		_ = saveSingleModelReport(reportDir, targetTime, cliRes)
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
				fmt.Println("\n[AUDIT] 发现已有成功裁判报告且无新模型运行，跳过整合。")
				judgeResult = *res
				goto assemble
			}
		}

		fmt.Println("\n[AUDIT] 正在执行系统能效审计与结论整合...")
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

		if err != nil {
			fmt.Printf("[AUDIT] -> FAULT: %v\n", err)
		} else {
			fmt.Printf("[AUDIT] -> SUCCESS (%s) 使用模型: %s\n", judgeResult.CallDuration.Round(time.Millisecond), judgeModel)
		}
	} else if appConfig.JudgeEnabled {
		fmt.Println("\n[AUDIT] 跳过审计：没有任何成功的推演节点。")
	}

assemble:
	totalDuration := time.Since(startedAt)

	// 构造结构化干支数据用于 HTML 看板
	d := calendar.NewSolarFromDate(targetTime).GetLunar()
	gz := d.GetEightChar()
	bazi := BaziInfo{
		SolarDate:   targetTime.Format("2006-01-02 15:04:05"),
		LunarDate:   fmt.Sprintf("%s年%s月%s", d.GetYearInChinese(), d.GetMonthInChinese(), d.GetDayInChinese()),
		UserGanzhi:  appConfig.DayMaster,
		TodayGanzhi: []string{gz.GetYear(), gz.GetMonth(), gz.GetDay(), gz.GetTime()},
		TodayMaster: gz.GetDayGan(),
		SolarTerm:   d.GetJieQi(),
		Nayins:      []string{gz.GetYearNaYin(), gz.GetMonthNaYin(), gz.GetDayNaYin(), gz.GetTimeNaYin()},
		Xunkongs:    []string{gz.GetYearXunKong(), gz.GetMonthXunKong(), gz.GetDayXunKong(), gz.GetTimeXunKong()},
	}
	if bazi.SolarTerm == "" {
		bazi.SolarTerm = "无"
	}

	finalPath, _ := saveFinalConclusionHTML(reportDir, targetTime, promptContent, results, judgeResult, totalDuration, bazi)

	fmt.Printf("\n任务结束！总耗时: %s\n", totalDuration.Round(time.Millisecond))

	if finalPath != "" {
		fmt.Printf("最终结论报告: %s\n", finalPath)
		_ = openInDefaultBrowser(finalPath)
	}
}

func buildPrompt(t time.Time) string {
	d := calendar.NewSolarFromDate(t).GetLunar()
	gz := d.GetEightChar()

	// 获取一些额外的细节来丰富输入
	solarTerm := d.GetJieQi()
	if solarTerm == "" {
		solarTerm = "无"
	}

	return fmt.Sprintf("今日公历：%s，农历：%s年%s月%s日，用户日元：%s，今日干支：%s %s %s %s，当日日主：%s，节气：%s，纳音：[%s %s %s %s]，旬空：[%s %s %s %s]",
		t.Format("2006-01-02 15:04:05"),
		d.GetYearInChinese(), d.GetMonthInChinese(), d.GetDayInChinese(),
		appConfig.DayMaster,
		gz.GetYear(), gz.GetMonth(), gz.GetDay(), gz.GetTime(),
		gz.GetDayGan(),
		solarTerm,
		gz.GetYearNaYin(), gz.GetMonthNaYin(), gz.GetDayNaYin(), gz.GetTimeNaYin(),
		gz.GetYearXunKong(), gz.GetMonthXunKong(), gz.GetDayXunKong(), gz.GetTimeXunKong(),
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
	return "gemini-3.1-pro-preview"
}

func judgeModelResults(cfg Config, localModels []string, judgeModel, originalPrompt string, results []ModelResult) (string, error) {
	var sb strings.Builder
	sb.WriteString("【当前生产环境：增量扫描报告】\n")
	sb.WriteString(originalPrompt + "\n\n")
	sb.WriteString("【待审计节点 Dump】\n")
	sb.WriteString("以下是多个计算节点提交的推演结论。请注意：若多个节点输出相似的“车轱辘话”（如通用健康建议、模糊的心理安慰），请直接丢弃并标记为“脏数据”。你必须从中提取出对今日干支交互最深刻、最具体的“内核级”见解：\n\n")

	for _, r := range results {
		if r.Err == nil {
			sb.WriteString(fmt.Sprintf("### 模型 %s 的结论：\n%s\n\n", r.Model, r.Content))
		}
	}

	input := sb.String()

	if strings.EqualFold(cfg.JudgeProvider, "local-cli") {
		fmt.Printf("[裁判] 尝试使用本地 CLI (%s)...\n", judgeModel)
		content, err := chatWithLocalCLI(judgeModel, cfg.JudgePrompt, input, cfg.CloudCallTimeout)
		if err == nil {
			return content, nil
		}
		fmt.Printf("[裁判] 本地 CLI 失败: %v。将尝试 Fallback 到云端模型...\n", err)
	}

	for _, cloud := range cfg.CloudModels {
		if cloud.Enabled && (cloud.Name == judgeModel || strings.HasPrefix(judgeModel, "gemini-3.1")) {
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
