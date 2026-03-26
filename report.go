package main

import (
	"fmt"
	"html/template"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"time"
)

const htmlTemplate = `<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>最终结论</title>
<style>
body{font-family:"Microsoft YaHei","PingFang SC",Arial,sans-serif;background:#f5f7fb;color:#1f2937;margin:0;padding:24px;}
.container{max-width:980px;margin:0 auto;}
.card{background:#fff;border-radius:16px;padding:24px;box-shadow:0 8px 30px rgba(0,0,0,.08);margin-bottom:20px;}
h1{margin:0 0 16px 0;font-size:32px;}
h2{margin:0 0 14px 0;font-size:22px;color:#111827;}
.meta{line-height:1.9;font-size:15px;color:#4b5563;}
.highlight{background:linear-gradient(135deg,#fff7ed,#fffbeb);border:1px solid #fdba74;}
.summary-card{background:linear-gradient(135deg,#faf5ff,#eef2ff);border:1px solid #c4b5fd;}
.summary-text{white-space:pre-wrap;word-break:break-word;line-height:1.9;font-size:16px;color:#312e81;font-weight:600;}
.score-card{text-align:center;}
.score-good{background:linear-gradient(135deg,#ecfdf5,#dcfce7);border:1px solid #86efac;}
.score-good .score-label{color:#15803d;}
.score-good .score-value{color:#166534;}
.score-mid{background:linear-gradient(135deg,#fffbeb,#fef3c7);border:1px solid #fcd34d;}
.score-mid .score-label{color:#b45309;}
.score-mid .score-value{color:#92400e;}
.score-low{background:linear-gradient(135deg,#fef2f2,#fee2e2);border:1px solid #fca5a5;}
.score-low .score-label{color:#b91c1c;}
.score-low .score-value{color:#991b1b;}
.score-unknown{background:linear-gradient(135deg,#eff6ff,#e0e7ff);border:1px solid #93c5fd;}
.score-unknown .score-label{color:#1d4ed8;}
.score-unknown .score-value{color:#1e3a8a;}
.score-label{font-size:16px;margin-bottom:10px;font-weight:700;}
.score-value{font-size:48px;line-height:1.2;font-weight:800;margin-bottom:10px;}
.score-reason{font-size:14px;line-height:1.8;color:#475569;}
pre{white-space:pre-wrap;word-break:break-word;background:#0f172a;color:#e5e7eb;padding:18px;border-radius:12px;line-height:1.75;font-size:15px;overflow:auto;}
ul{margin:0;padding-left:22px;line-height:1.9;}
.note{color:#6b7280;font-size:14px;line-height:1.8;}
.badge{display:inline-block;padding:6px 12px;border-radius:999px;background:#dbeafe;color:#1d4ed8;font-size:13px;margin-bottom:12px;}
</style>
</head>
<body>
<div class="container">
	<div class="card">
		<div class="badge">自动生成</div>
		<h1>最终结论</h1>
		<div class="meta">
			<div><strong>生成时间：</strong>{{.Time}}</div>
			<div><strong>问题：</strong>{{.Prompt}}</div>
			<div><strong>总耗时：</strong>{{.Duration}}</div>
		</div>
	</div>

	<div class="card score-card {{.ScoreClass}}">
		<div class="score-label">今日运势评分</div>
		<div class="score-value">{{.Score}}</div>
		<div class="score-reason">{{.ScoreReason}}</div>
	</div>

	{{if .Summary}}
	<div class="card summary-card">
		<h2>结论摘要</h2>
		<div class="summary-text">{{.Summary}}</div>
	</div>
	{{end}}

	<div class="card">
		<h2>成功返回的模型</h2>
		<ul>
		{{range .SuccessModels}}
			<li>{{.}}</li>
		{{else}}
			<li>无</li>
		{{end}}
		</ul>
	</div>

	<div class="card highlight">
		<h2>最终采用结论</h2>
		<pre>{{.FinalContent}}</pre>
	</div>

	<div class="card">
		<h2>查看说明</h2>
		<div class="note">
			<div>• 如需查看每个模型的原始输出，请打开同目录下的各模型报告文件。</div>
			<div>• 如需查看完整横向比较，请打开 <strong>summary.md</strong> 与 <strong>judge.md</strong>。</div>
		</div>
	</div>
</div>
</body>
</html>`

type HTMLData struct {
	Time          string
	Prompt        string
	Duration      string
	Score         string
	ScoreReason   string
	ScoreClass    string
	Summary       string
	SuccessModels []string
	FinalContent  string
}

func createReportDir(t time.Time) (string, error) {
	reportDir := filepath.Join("reports", t.Format("2006-01-02"))
	if err := os.MkdirAll(reportDir, 0755); err != nil {
		return "", err
	}
	return reportDir, nil
}

func saveRunMeta(reportDir string, t time.Time, prompt string) error {
	content := fmt.Sprintf("# 运行信息\n\n- 启动时间：`%s`\n- 请求内容：`%s`\n", t.Format("2006-01-02 15:04:05"), prompt)
	return os.WriteFile(filepath.Join(reportDir, "run.md"), []byte(content), 0644)
}

func findExistingModelResultToday(modelName string) (*ModelResult, bool) {
	todayDir := filepath.Join("reports", time.Now().Format("2006-01-02"))
	sanitizedName := sanitizeFileName(modelName)
	path := filepath.Join(todayDir, sanitizedName+".md")

	data, err := os.ReadFile(path)
	if err != nil {
		return nil, false
	}

	content := string(data)
	// 检查报告是否成功
	if strings.Contains(content, "状态：失败") {
		return nil, false
	}

	res := &ModelResult{
		Model:    modelName,
		Provider: "existing-report", // 标记为已有报告
	}

	// 提取耗时
	if idx := strings.Index(content, "- 耗时：`") ; idx != -1 {
		start := idx + len("- 耗时：`")
		if end := strings.Index(content[start:], "`"); end != -1 {
			dStr := content[start : start+end]
			d, _ := time.ParseDuration(dStr)
			res.CallDuration = d
			res.TotalDuration = d
		}
	}

	// 提取正文
	if idx := strings.Index(content, "## 输出内容\n\n"); idx != -1 {
		res.Content = strings.TrimSpace(content[idx+len("## 输出内容\n\n"):])
	}

	if res.Content != "" {
		return res, true
	}

	return nil, false
}

func saveSingleModelReport(reportDir string, t time.Time, result ModelResult) error {
	filename := sanitizeFileName(result.Model) + ".md"
	var content string
	if result.Err != nil {
		content = fmt.Sprintf("# 模型报告\n\n- 模型：`%s`\n- 状态：失败\n- 错误：`%v`\n", result.Model, result.Err)
	} else {
		content = fmt.Sprintf("# 模型报告\n\n- 模型：`%s`\n- 耗时：`%s`\n\n## 输出内容\n\n%s\n", result.Model, result.CallDuration, result.Content)
	}
	return os.WriteFile(filepath.Join(reportDir, filename), []byte(content), 0644)
}

func saveJudgeReport(reportDir string, t time.Time, judgeResult JudgeResult) error {
	content := fmt.Sprintf("# 裁判报告\n\n- 模型：`%s`\n- 耗时：`%s`\n\n## 结论\n\n%s\n", judgeResult.Model, judgeResult.CallDuration, judgeResult.Content)
	return os.WriteFile(filepath.Join(reportDir, "judge.md"), []byte(content), 0644)
}

func saveSummaryReport(reportDir string, t time.Time, prompt string, results []ModelResult, judgeResult JudgeResult, totalDuration time.Duration) error {
	var sb strings.Builder
	sb.WriteString("# 多模型汇总报告\n\n")
	sb.WriteString(fmt.Sprintf("- 时间：%s\n- 总耗时：%s\n\n", t.Format("2006-01-02 15:04:05"), totalDuration))

	for _, r := range results {
		sb.WriteString(fmt.Sprintf("## 模型：%s\n", r.Model))
		if r.Err != nil {
			sb.WriteString(fmt.Sprintf("- 错误：%v\n\n", r.Err))
		} else {
			sb.WriteString(fmt.Sprintf("- 耗时：%s\n\n%s\n\n", r.CallDuration, r.Content))
		}
	}
	return os.WriteFile(filepath.Join(reportDir, "summary.md"), []byte(sb.String()), 0644)
}

func saveFinalConclusionHTML(reportDir string, t time.Time, prompt string, results []ModelResult, judgeResult JudgeResult, totalDuration time.Duration) (string, error) {
	tmpl, err := template.New("report").Parse(htmlTemplate)
	if err != nil {
		return "", err
	}

	successModels := []string{}
	for _, r := range results {
		if r.Err == nil {
			durationStr := r.CallDuration.Round(time.Millisecond).String()
			successModels = append(successModels, fmt.Sprintf("%s (%s)", r.Model, durationStr))
		}
	}

	score, reason := extractFortuneScore(judgeResult)
	data := HTMLData{
		Time:          t.Format("2006-01-02 15:04:05"),
		Prompt:        prompt,
		Duration:      totalDuration.Round(time.Millisecond).String(),
		Score:         score,
		ScoreReason:   reason,
		ScoreClass:    fortuneScoreClass(score),
		Summary:       extractFinalConclusionSummary(judgeResult.Content),
		SuccessModels: successModels,
		FinalContent:  buildFinalContentWithoutScore(judgeResult.Content),
	}

	path := filepath.Join(reportDir, "final.html")
	f, err := os.Create(path)
	if err != nil {
		return "", err
	}
	defer f.Close()

	if err := tmpl.Execute(f, data); err != nil {
		return "", err
	}
	return path, nil
}

// 辅助提取函数
func extractFortuneScore(jr JudgeResult) (string, string) {
	if jr.Err != nil || jr.Content == "" {
		return "暂无评分", "无法提取评分"
	}
	lines := strings.Split(jr.Content, "\n")
	score := "未识别"
	reason := "未找到格式化的点评"

	scoreRegex := regexp.MustCompile(`([0-9.]+)\s*/\s*10`)
	reasonRegex := regexp.MustCompile(`(?i)(?:气场点评|点评)[：:]\s*(.*)`)

	for i, line := range lines {
		if scoreRegex.MatchString(line) {
			matches := scoreRegex.FindStringSubmatch(line)
			if len(matches) > 1 {
				score = matches[1] + " / 10"
			} else {
				score = strings.TrimSpace(line)
			}

			// 尝试寻找点评（通常在评分后面几行）
			for j := i; j < len(lines) && j < i+5; j++ {
				l := strings.TrimSpace(lines[j])
				if reasonRegex.MatchString(l) {
					rm := reasonRegex.FindStringSubmatch(l)
					reason = strings.Trim(rm[1], "*_ >")
					break
				}
			}
			return score, reason
		}
	}
	return score, reason
}

func fortuneScoreClass(score string) string {
	valStr := strings.Split(score, "/")[0]
	valStr = strings.TrimSpace(valStr)
	val, err := strconv.ParseFloat(valStr, 64)
	if err != nil {
		if strings.Contains(score, "8") || strings.Contains(score, "9") || strings.Contains(score, "10") {
			return "score-good"
		}
		return "score-mid"
	}

	if val >= 8.0 {
		return "score-good"
	} else if val <= 4.0 {
		return "score-low"
	}
	return "score-mid"
}

func extractFinalConclusionSummary(content string) string {
	if len(content) > 200 {
		return content[:200] + "..."
	}
	return content
}

func buildFinalContentWithoutScore(content string) string {
	lines := strings.Split(content, "\n")
	var result []string
	skip := false
	
	scoreRegex := regexp.MustCompile(`[0-9.]+\s*/\s*10`)

	for _, line := range lines {
		trimmed := strings.TrimSpace(line)
		// 如果发现评分标记行，开始跳过后续关联行
		if scoreRegex.MatchString(line) || strings.Contains(trimmed, "审计评分") || strings.Contains(trimmed, "最终评分") {
			skip = true
			continue
		}
		
		// 如果在跳过状态，且发现新的一级或二级标题，或者看起来像正文的行，停止跳过
		if skip {
			if strings.HasPrefix(trimmed, "#") || (len(trimmed) > 0 && !strings.HasPrefix(trimmed, ">") && !strings.HasPrefix(trimmed, "-") && !strings.HasPrefix(trimmed, "*")) {
				skip = false
			} else {
				continue
			}
		}
		
		result = append(result, line)
	}
	return strings.TrimSpace(strings.Join(result, "\n"))
}
