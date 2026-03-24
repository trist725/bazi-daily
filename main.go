package main

import (
	"fmt"
	"os"
	"os/signal"
	"strings"
	"sync"
	"syscall"
	"time"

	"github.com/6tail/lunar-go/calendar"
)

func main() {
	// 1. 加载资源
	if err := loadRuntimeResources(&appConfig); err != nil {
		fmt.Printf("加载配置资源失败: %v\n", err)
		return
	}

	// 2. 信号处理，确保退出时清理本地模型
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, os.Interrupt, syscall.SIGTERM)
	go func() {
		<-sigChan
		fmt.Println("\n接收到中断信号，正在执行清理...")
		finalCleanupLocalModels(appConfig.BaseURL, appConfig.LocalUnloadTimeout, appConfig.RunningPollInterval)
		os.Exit(0)
	}()

	startedAt := time.Now()
	promptContent := buildPrompt(startedAt)

	// 3. 准备报告目录
	reportDir, err := createReportDir(startedAt)
	if err != nil {
		fmt.Printf("创建报告目录失败: %v\n", err)
		return
	}
	saveRunMeta(reportDir, startedAt, promptContent)

	// 4. 解析模型列表
	localModels, _ := resolveModels(appConfig.BaseURL, appConfig)
	cloudModels := enabledCloudModels(appConfig.CloudModels)

	if len(localModels) == 0 && len(cloudModels) == 0 {
		fmt.Println("没有可用模型，请检查配置。")
		return
	}

	var results []ModelResult
	var mu sync.Mutex

	// 5. 并发执行云端模型 (优化：并发执行)
	fmt.Println("正在并发调用云端模型...")
	var wg sync.WaitGroup
	for _, cloud := range cloudModels {
		wg.Add(1)
		go func(c CloudModelConfig) {
			defer wg.Done()
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
			saveSingleModelReport(reportDir, startedAt, res)
			fmt.Printf("云端模型 %s 调用完成\n", c.Name)
		}(cloud)
	}
	wg.Wait()

	// 6. 串行执行本地模型 (为了显存安全)
	fmt.Println("正在串行调用本地模型...")
	for _, mName := range localModels {
		ensureNoModelsRunning(appConfig.BaseURL, appConfig.LocalPreflightTimeout, appConfig.RunningPollInterval)
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
		results = append(results, res)
		saveSingleModelReport(reportDir, startedAt, res)
		releaseLocalModelWithRetry(appConfig.BaseURL, mName, appConfig.LocalUnloadTimeout, appConfig.RunningPollInterval, appConfig.LocalReleaseRetryCount, appConfig.LocalReleaseRetryDelay)
		time.Sleep(appConfig.LocalSwitchDelay)
	}

	// 7. 裁判整合
	var judgeResult JudgeResult
	if appConfig.JudgeEnabled && hasSuccessfulResult(results) {
		fmt.Println("正在执行裁判整合...")
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
		saveJudgeReport(reportDir, startedAt, judgeResult)
	}

	// 8. 生成汇总
	totalDuration := time.Since(startedAt)
	saveSummaryReport(reportDir, startedAt, promptContent, results, judgeResult, totalDuration)
	finalPath, _ := saveFinalConclusionHTML(reportDir, startedAt, promptContent, results, judgeResult, totalDuration)

	fmt.Printf("\n任务完成！报告保存在: %s\n", reportDir)
	if finalPath != "" {
		openInDefaultBrowser(finalPath)
	}

	// 兜底清理
	finalCleanupLocalModels(appConfig.BaseURL, appConfig.LocalUnloadTimeout, appConfig.RunningPollInterval)
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

func hasSuccessfulResult(results []ModelResult) bool {
	for _, r := range results {
		if r.Err == nil {
			return true
		}
	}
	return false
}

func resolveJudgeModel(localModels []string, cfg Config) string {
	if cfg.JudgeModel != "" {
		return cfg.JudgeModel
	}
	return "gemini-flash-latest"
}

func judgeModelResults(cfg Config, localModels []string, judgeModel, originalPrompt string, results []ModelResult) (string, error) {
	var sb strings.Builder
	sb.WriteString("以下是同一个问题的多模型输出结果，请你先完成横向评审，再给出一份可以直接采用的最终结论。\n\n")
	sb.WriteString("【任务要求】\n")
	sb.WriteString("你必须在评审后输出“最终结论”。最终结论必须是整合后的可直接使用版本。\n")
	sb.WriteString("你必须给出“今日运势评分”，格式为 X/10。\n\n")
	sb.WriteString("【原始问题】\n")
	sb.WriteString(originalPrompt)
	sb.WriteString("\n\n")

	for _, r := range results {
		if r.Err == nil {
			sb.WriteString(fmt.Sprintf("--- 模型 %s ---\n%s\n\n", r.Model, r.Content))
		}
	}

	sb.WriteString("请严格按以下标题输出：\n一、今日运势评分\n二、最终结论\n三、模型对比\n")

	input := sb.String()

	// 优先使用云端模型作为裁判，如果没有则用本地
	if cloud, ok := findCloudModelConfig(judgeModel, cfg.CloudModels); ok {
		return chatWithCloudModel(cloud, cfg.JudgePrompt, input, cfg.CloudCallTimeout)
	}

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
