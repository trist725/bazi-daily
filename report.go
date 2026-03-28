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
<title>能量管理最终结�?/title>
<script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
<style>
:root {
    --primary: #0f172a;
    --accent: #3b82f6;
    --bg: #f1f5f9;
    --card-bg: #ffffff;
    --text-main: #334155;
    --text-muted: #64748b;
    --border: #e2e8f0;
    --shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1);
}

body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, "PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", sans-serif;
    background: var(--bg);
    color: var(--text-main);
    margin: 0;
    padding: 24px 16px;
    line-height: 1.6;
    -webkit-font-smoothing: antialiased;
}

.container {
    max-width: 900px;
    margin: 0 auto;
}

.card {
    background: var(--card-bg);
    border-radius: 16px;
    padding: 32px;
    box-shadow: var(--shadow);
    margin-bottom: 24px;
    border: 1px solid var(--border);
}

.header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 24px;
}

h1 {
    margin: 0;
    font-size: 26px;
    color: var(--primary);
    font-weight: 800;
    letter-spacing: -0.025em;
}

.badge {
    display: inline-flex;
    align-items: center;
    padding: 4px 12px;
    border-radius: 9999px;
    background: #eff6ff;
    color: #1d4ed8;
    font-size: 12px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

.meta {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 16px;
    font-size: 14px;
    color: var(--text-muted);
    border-top: 1px solid var(--border);
    padding-top: 20px;
}

.meta-item strong {
    color: var(--primary);
    font-weight: 600;
}

/* 评分卡片优化 */
.score-card {
    text-align: center;
    position: relative;
    overflow: hidden;
    transition: transform 0.2s;
}

.score-good { background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%); border-color: #86efac; color: #166534; }
.score-mid { background: linear-gradient(135deg, #fffbeb 0%, #fef3c7 100%); border-color: #fcd34d; color: #92400e; }
.score-low { background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%); border-color: #fca5a5; color: #991b1b; }

.score-label { font-size: 13px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.15em; opacity: 0.8; margin-bottom: 12px; }
.score-value { font-size: 72px; font-weight: 950; line-height: 1; margin: 8px 0; font-variant-numeric: tabular-nums; }
.score-reason {
    font-size: 16px;
    max-width: 600px;
    margin: 16px auto 0;
    font-weight: 600;
    padding: 12px 20px;
    background: rgba(255,255,255,0.4);
    border-radius: 12px;
    backdrop-filter: blur(4px);
}

/* 结论内容 Markdown 样式 */
.content-area {
    position: relative;
}

.content-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 24px;
    border-bottom: 2px solid var(--border);
    padding-bottom: 12px;
}

h2 { font-size: 20px; margin: 0; color: var(--primary); font-weight: 700; }

.copy-btn {
    padding: 8px 16px;
    background: #ffffff;
    border: 1px solid var(--border);
    border-radius: 8px;
    font-size: 13px;
    font-weight: 600;
    color: var(--text-main);
    cursor: pointer;
    transition: all 0.2s;
    box-shadow: 0 1px 2px rgba(0,0,0,0.05);
}
.copy-btn:hover { border-color: var(--accent); color: var(--accent); background: #f8fafc; }
.copy-btn:active { transform: translateY(1px); }

.markdown-body {
    font-size: 16px;
    color: #334155;
    line-height: 1.75;
}

/* Markdown 元素美化 */
.markdown-body h1, .markdown-body h2, .markdown-body h3 { 
    color: var(--primary); 
    margin-top: 1.5em; 
    margin-bottom: 0.75em; 
    font-weight: 700;
}
.markdown-body h2 { font-size: 1.25rem; border-bottom: 1px solid var(--border); padding-bottom: 0.3em; }
.markdown-body h3 { font-size: 1.1rem; }
.markdown-body p { margin-bottom: 1.25em; }
.markdown-body strong { color: #0f172a; font-weight: 800; }
.markdown-body ul, .markdown-body ol { padding-left: 1.5em; margin-bottom: 1.25em; }
.markdown-body li { margin-bottom: 0.5em; }
.markdown-body blockquote {
    margin: 1.5em 0;
    padding: 0.5em 1.5em;
    color: #475569;
    border-left: 4px solid var(--accent);
    background: #f8fafc;
    border-radius: 0 8px 8px 0;
}
.markdown-body table {
    width: 100%;
    border-collapse: collapse;
    margin: 1.5em 0;
    font-size: 0.95em;
    border-radius: 8px;
    overflow: hidden;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}
.markdown-body th {
    background-color: #f8fafc;
    color: #475569;
    font-weight: 700;
    text-align: left;
    padding: 12px 16px;
    border-bottom: 2px solid var(--border);
}
.markdown-body td {
    padding: 12px 16px;
    border-bottom: 1px solid var(--border);
    background-color: #ffffff;
}
.markdown-body tr:last-child td { border-bottom: none; }
.markdown-body tr:nth-child(even) td { background-color: #fcfdfe; }
.markdown-body hr { height: 1px; background-color: var(--border); border: none; margin: 2em 0; }
.markdown-body code {
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
    font-size: 0.9em;
    background: #f1f5f9;
    padding: 0.2em 0.4em;
    border-radius: 4px;
    color: #e11d48;
}

.summary-card {
    background: linear-gradient(to right, #f8fafc, #eff6ff); 
    border-left: 5px solid #6366f1;
}
.summary-title { color: #4338ca; margin-bottom: 10px; font-size: 13px; font-weight: 800; text-transform: uppercase; letter-spacing: 0.1em; display: flex; align-items: center; }
.summary-title::before { content: ""; display: inline-block; width: 8px; height: 8px; background: #6366f1; border-radius: 50%; margin-right: 8px; box-shadow: 0 0 8px rgba(99,102,241,0.5); }
.summary-text { font-size: 15px; color: #312e81; font-weight: 500; line-height: 1.7; }

.model-list {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-top: 12px;
}
.model-badge {
    background: #f8fafc;
    padding: 6px 12px;
    border-radius: 8px;
    font-size: 12px;
    font-weight: 500;
    color: #64748b;
    border: 1px solid var(--border);
}

.footer {
    text-align: center;
    font-size: 13px;
    color: var(--text-muted);
    margin-top: 48px;
    padding-bottom: 48px;
}

.rationale-card {
    border-left: 4px solid #10b981;
    background: #f0fdf4;
}
.comparison-card {
    border-left: 4px solid #f59e0b;
    background: #fffbeb;
}
.section-title {
    font-size: 14px;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 12px;
    display: flex;
    align-items: center;
}
.rationale-title { color: #059669; }
.comparison-title { color: #d97706; }

@media (max-width: 640px) {
    body { padding: 16px 12px; }
    .card { padding: 24px 20px; }
    h1 { font-size: 22px; }
    .score-value { font-size: 56px; }
    .header { flex-direction: column; align-items: flex-start; gap: 12px; }
}
</style>
</head>
<body>
<div class="container">
    <div class="card">
        <div class="header">
            <h1>能量管理报告</h1>
            <div class="badge">AI 推演 v2.1</div>
        </div>
        <div class="meta">
            <div class="meta-item"><strong>目标日期�?/strong>{{.Prompt}}</div>
            <div class="meta-item"><strong>生成时间�?/strong>{{.Time}}</div>
            <div class="meta-item"><strong>推演耗时�?/strong>{{.Duration}}</div>
        </div>
    </div>

    <div class="card score-card {{.ScoreClass}}">
        <div class="score-label">今日运势评分</div>
        <div class="score-value">{{.Score}}</div>
        {{if .ScoreReason}}
        <div class="score-reason">�?{{.ScoreReason}} �?/div>
        {{end}}
    </div>

    {{if .Summary}}
    <div class="card summary-card">
        <div class="summary-title">核心结论摘要</div>
        <div class="summary-text">{{.Summary}}</div>
    </div>
    {{end}}

    <div class="card content-area">
        <div class="content-header">
            <h2>最终推演结�?/h2>
            <button class="copy-btn" onclick="copyContent()">一键复制结�?/button>
        </div>
        <div id="final-content" class="markdown-body"></div>
    </div>

    {{if .Rationale}}
    <div class="card rationale-card">
        <div class="section-title rationale-title">💡 决策依据 / 采用逻辑</div>
        <div id="rationale-content" class="markdown-body" style="font-size: 14px; color: #065f46;"></div>
    </div>
    {{end}}

    {{if .Comparison}}
    <div class="card comparison-card">
        <div class="section-title comparison-title">📊 各模型对比点�?/div>
        <div id="comparison-content" class="markdown-body" style="font-size: 14px; color: #92400e;"></div>
    </div>
    {{end}}

    <div class="card">
        <h2 style="font-size: 15px; margin-bottom: 12px; color: var(--text-muted);">参与计算�?AI 模型</h2>
        <div class="model-list">
            {{range .SuccessModels}}
            <span class="model-badge">{{.}}</span>
            {{else}}
            <span class="model-badge">�?/span>
            {{end}}
        </div>
    </div>

    <div class="footer">
        <div>&bull; 数据基于多模型共识算法自动生�?&bull;</div>
        <div style="margin-top: 8px; opacity: 0.7;">报告路径：reports/{{.Time}}</div>
    </div>
</div>

<!-- 存储原始 Markdown -->
<div id="markdown-raw" style="display:none;">{{.FinalContent}}</div>
<div id="rationale-raw" style="display:none;">{{.Rationale}}</div>
<div id="comparison-raw" style="display:none;">{{.Comparison}}</div>

<script>
// 初始化渲�?document.addEventListener('DOMContentLoaded', () => { 
    const raw = document.getElementById('markdown-raw').textContent;
    document.getElementById('final-content').innerHTML = marked.parse(raw);

    const rationaleRaw = document.getElementById('rationale-raw')?.textContent;
    if (rationaleRaw) {
        document.getElementById('rationale-content').innerHTML = marked.parse(rationaleRaw);
    }

    const comparisonRaw = document.getElementById('comparison-raw')?.textContent;
    if (comparisonRaw) {
        document.getElementById('comparison-content').innerHTML = marked.parse(comparisonRaw);
    }
});

function copyContent() {
    const content = document.getElementById('final-content').innerText;
    navigator.clipboard.writeText(content).then(() => {
        const btn = document.querySelector('.copy-btn');
        const originalText = btn.innerText;
        btn.innerText = '已复�?';
        btn.style.borderColor = '#22c55e';
        btn.style.color = '#16a34a';
        setTimeout(() => {
            btn.innerText = originalText;
            btn.style.borderColor = '';
            btn.style.color = '';
        }, 2000);
    });
}
</script>
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
	Rationale     string
	Comparison    string
}

const subReportTemplate = `<!doctype html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{{.Title}} - {{.Model}}</title>
<script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
<style>
body { font-family: -apple-system, sans-serif; background: #f8fafc; color: #334155; padding: 20px; line-height: 1.6; }
.card { background: white; border-radius: 12px; padding: 24px; box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1); max-width: 900px; margin: 0 auto; border: 1px solid #e2e8f0; }
h1 { font-size: 20px; color: #0f172a; margin-top: 0; border-bottom: 2px solid #e2e8f0; padding-bottom: 10px; }
.meta { font-size: 14px; color: #64748b; margin-bottom: 20px; background: #f1f5f9; padding: 12px; border-radius: 8px; }
.markdown-body { font-size: 15px; }
.markdown-body h2 { font-size: 18px; color: #1e293b; border-bottom: 1px solid #cbd5e1; padding-bottom: 5px; }
</style>
</head>
<body>
<div class="card">
    <h1>{{.Title}}</h1>
    <div class="meta">
        <div><strong>模型�?/strong>{{.Model}}</div>
        <div><strong>耗时�?/strong>{{.Duration}}</div>
        {{if .Error}}<div style="color: #ef4444;"><strong>错误�?/strong>{{.Error}}</div>{{end}}
    </div>
    <div id="content" class="markdown-body"></div>
</div>
<div id="raw" style="display:none;">{{.Content}}</div>
<script>
    document.addEventListener('DOMContentLoaded', () => {
        const raw = document.getElementById('raw').textContent;
        document.getElementById('content').innerHTML = marked.parse(raw);
    });
</script>
</body>
</html>`

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

func findExistingModelResultToday(t time.Time, modelName string) (*ModelResult, bool) {
	todayDir := filepath.Join("reports", t.Format("2006-01-02"))
	sanitizedName := sanitizeFileName(modelName)
	path := filepath.Join(todayDir, sanitizedName+".html")

	data, err := os.ReadFile(path)
	if err != nil {
		return nil, false
	}

	content := string(data)
	// 解析 HTML 中的原始 Markdown
	startTag := `<div id="raw" style="display:none;">`
	endTag := `</div>`

	startIdx := strings.Index(content, startTag)
	if startIdx == -1 {
		return nil, false
	}
	startIdx += len(startTag)
	endIdx := strings.Index(content[startIdx:], endTag)
	if endIdx == -1 {
		return nil, false
	}

	res := &ModelResult{
		Model:    modelName,
		Provider: "existing-report",
		Content:  strings.TrimSpace(content[startIdx : startIdx+endIdx]),
	}

	// 提取耗时（从元数据中提取�?	metaStart := strings.Index(content, "<strong>耗时�?/strong>")
	if metaStart != -1 {
		metaStart += len("<strong>耗时�?/strong>")
		metaEnd := strings.Index(content[metaStart:], "</div>")
		if metaEnd != -1 {
			dStr := strings.TrimSpace(content[metaStart : metaStart+metaEnd])
			d, _ := time.ParseDuration(dStr)
			res.CallDuration = d
		}
	}

	if res.Content != "" {
		return res, true
	}

	return nil, false
}

func saveSingleModelReport(reportDir string, t time.Time, result ModelResult) error {
	filename := sanitizeFileName(result.Model) + ".html"
	tmpl, _ := template.New("sub").Parse(subReportTemplate)

	data := struct {
		Title    string
		Model    string
		Duration string
		Error    string
		Content  string
	}{
		Title:    "模型输出报告",
		Model:    result.Model,
		Duration: result.CallDuration.Round(time.Millisecond).String(),
		Content:  result.Content,
	}
	if result.Err != nil {
		data.Error = result.Err.Error()
	}

	f, _ := os.Create(filepath.Join(reportDir, filename))
	defer f.Close()
	return tmpl.Execute(f, data)
}

func saveJudgeReport(reportDir string, t time.Time, judgeResult JudgeResult) error {
	tmpl, _ := template.New("sub").Parse(subReportTemplate)
	data := struct {
		Title    string
		Model    string
		Duration string
		Error    string
		Content  string
	}{
		Title:    "裁判决策报告",
		Model:    judgeResult.Model,
		Duration: judgeResult.CallDuration.Round(time.Millisecond).String(),
		Content:  judgeResult.Content,
	}
	if judgeResult.Err != nil {
		data.Error = judgeResult.Err.Error()
	}

	f, _ := os.Create(filepath.Join(reportDir, "judge.html"))
	defer f.Close()
	return tmpl.Execute(f, data)
}

func saveSummaryReport(reportDir string, t time.Time, prompt string, results []ModelResult, judgeResult JudgeResult, totalDuration time.Duration) error {
	var sb strings.Builder
	sb.WriteString("# 多模型汇总报告\n\n")
	sb.WriteString(fmt.Sprintf("- 时间�?s\n- 总耗时�?s\n\n", t.Format("2006-01-02 15:04:05"), totalDuration.Round(time.Millisecond)))

	for _, r := range results {
		sb.WriteString(fmt.Sprintf("## 模型�?s\n", r.Model))
		if r.Err != nil {
			sb.WriteString(fmt.Sprintf("- 错误�?v\n\n", r.Err))
		} else {
			sb.WriteString(fmt.Sprintf("- 耗时�?s\n\n%s\n\n", r.CallDuration.Round(time.Millisecond), r.Content))
		}
	}

	tmpl, _ := template.New("sub").Parse(subReportTemplate)
	data := struct {
		Title    string
		Model    string
		Duration string
		Error    string
		Content  string
	}{
		Title:    "汇总摘要报�?,
		Model:    "System Orchestrator",
		Duration: totalDuration.Round(time.Millisecond).String(),
		Content:  sb.String(),
	}

	f, _ := os.Create(filepath.Join(reportDir, "summary.html"))
	defer f.Close()
	return tmpl.Execute(f, data)
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
	finalContent := buildFinalContentWithoutScore(judgeResult.Content)
	if judgeResult.Err != nil {
		finalContent = fmt.Sprintf("⚠️ **裁判模型整合失败**\n\n- 错误信息: %v\n\n请查看各模型原始输出以获取推演结论�?, judgeResult.Err)
	}

	data := HTMLData{
		Time:          t.Format("2006-01-02 15:04:05"),
		Prompt:        prompt,
		Duration:      totalDuration.Round(time.Millisecond).String(),
		Score:         score,
		ScoreReason:   reason,
		ScoreClass:    fortuneScoreClass(score),
		Summary:       extractFinalConclusionSummary(judgeResult.Content),
		SuccessModels: successModels,
		FinalContent:  finalContent,
		Rationale:     extractSection(judgeResult.Content, "决策依据"),
		Comparison:    extractSection(judgeResult.Content, "各模型对�?),
	}

	if data.Comparison == "" {
		data.Comparison = extractSection(judgeResult.Content, "模型对比")
	}
	if data.Rationale == "" {
		data.Rationale = extractSection(judgeResult.Content, "采用逻辑")
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
	score := "未识�?
	reason := ""

	// 评分正则
	scoreRegexes := []*regexp.Regexp{
		regexp.MustCompile(`(?i)(?:评分|气场评分|综合评分)[�?]\s*([0-9.]+)(?:\s*/\s*10)?`),
		regexp.MustCompile(`([0-9.]+)\s*/\s*10`),
	}

	// 理由正则
	reasonRegex := regexp.MustCompile(`(?i)(?:气场点评|气场分析|核心点评|综合点评|运势点评|点评)[�?]\s*(.*)`)

	// 1. 尝试在文中定位评分和紧随其后的点�?	for i, line := range lines {
		foundScore := false
		for _, reg := range scoreRegexes {
			if reg.MatchString(line) {
				matches := reg.FindStringSubmatch(line)
				if len(matches) > 1 {
					score = matches[1] + " / 10"
					foundScore = true
					break
				}
			}
		}

		if foundScore {
			// 找到评分后，尝试在后面几行寻找点�?			for j := i; j < len(lines) && j < i+6; j++ {
				l := strings.TrimSpace(lines[j])
				if reasonRegex.MatchString(l) {
					rm := reasonRegex.FindStringSubmatch(l)
					reason = strings.Trim(rm[1], "*_ >#")
					break
				}
			}
			if reason != "" {
				return score, reason
			}
		}
	}

	// 2. 如果没在评分附近找到，尝试在全文中寻找第一个出现的点评标签
	if reason == "" {
		for _, line := range lines {
			if reasonRegex.MatchString(line) {
				rm := reasonRegex.FindStringSubmatch(line)
				reason = strings.Trim(rm[1], "*_ >#")
				break
			}
		}
	}

	// 3. 如果还是没有，使用摘要作为理由（避免显示错误或空值）
	if reason == "" {
		reason = extractFinalConclusionSummary(jr.Content)
	}

	if reason == "" {
		reason = "建议查看详细推演结论"
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
	if content == "" {
		return ""
	}
	// 尝试寻找第一段非标题内容，避免摘要和正文开头完全重�?	lines := strings.Split(content, "\n")
	var firstPara string
	for _, line := range lines {
		trimmed := strings.TrimSpace(line)
		if trimmed == "" || strings.HasPrefix(trimmed, "#") {
			continue
		}
		firstPara = trimmed
		break
	}

	// 简单的正则，去�?Markdown 符号
	re := regexp.MustCompile(`[#*` + "`" + `>_-]`)
	clean := re.ReplaceAllString(firstPara, " ")

	// 合并多余空格
	reSpace := regexp.MustCompile(`\s+`)
	clean = strings.TrimSpace(reSpace.ReplaceAllString(clean, " "))

	if len([]rune(clean)) > 140 {
		return string([]rune(clean)[:140]) + "..."
	}
	return clean
}

func buildFinalContentWithoutScore(content string) string {
	if content == "" {
		return "（裁判模型未返回有效推演结论，请检查各模型原始输出�?
	}
	lines := strings.Split(content, "\n")
	var result []string

	scoreRegex := regexp.MustCompile(`(?i)(?:评分|气场评分|综合评分)[�?]\s*[0-9.]+|[0-9.]+\s*/\s*10`)
	reasonRegex := regexp.MustCompile(`(?i)(?:气场点评|气场分析|核心点评|综合点评|运势点评|点评)[�?]`)

	for _, line := range lines {
		trimmed := strings.TrimSpace(line)
		if trimmed == "" {
			result = append(result, line)
			continue
		}
		// 过滤评分行及其点评行（这些已在卡片中显示�?		if scoreRegex.MatchString(line) || reasonRegex.MatchString(trimmed) ||
			strings.Contains(trimmed, "审计评分") || strings.Contains(trimmed, "最终评�?) ||
			strings.Contains(trimmed, "气场点评") || strings.Contains(trimmed, "气场分析") {
			continue
		}
		result = append(result, line)
	}

	final := strings.TrimSpace(strings.Join(result, "\n"))
	if (final == "" || len(final) < 50) && content != "" {
		return content // 过滤得太干净了就回退，确保有内容显示
	}
	return final
}

func extractSection(content, sectionName string) string {
	lines := strings.Split(content, "\n")
	var result []string
	found := false

	for _, line := range lines {
		trimmed := strings.TrimSpace(line)
		if strings.HasPrefix(trimmed, "## ") && (strings.Contains(trimmed, sectionName) || strings.Contains(sectionName, trimmed)) {
			found = true
			continue
		}
		if found {
			if strings.HasPrefix(trimmed, "## ") {
				break
			}
			result = append(result, line)
		}
	}
	return strings.TrimSpace(strings.Join(result, "\n"))
}