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
<title>能量管理最终结论</title>
<script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
<style>
:root {
    --primary: #0f172a;
    --accent: #6366f1;
    --accent-light: #e0e7ff;
    --bg: #f8fafc;
    --card-bg: #ffffff;
    --text-main: #1e293b;
    --text-muted: #64748b;
    --border: #e2e8f0;
    --shadow: 0 10px 15px -3px rgb(0 0 0 / 0.1), 0 4px 6px -4px rgb(0 0 0 / 0.1);
    --success: #10b981;
    --warning: #f59e0b;
    --danger: #ef4444;
}

body {
    font-family: 'JetBrains Mono', 'Fira Code', ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "PingFang SC", "Microsoft YaHei", monospace;
    background: var(--bg);
    color: var(--text-main);
    margin: 0;
    padding: 24px 16px;
    line-height: 1.6;
    -webkit-font-smoothing: antialiased;
}

.container {
    max-width: 1000px;
    margin: 0 auto;
}

.card {
    background: var(--card-bg);
    border-radius: 12px;
    padding: 28px;
    box-shadow: var(--shadow);
    margin-bottom: 24px;
    border: 1px solid var(--border);
    position: relative;
    overflow: hidden;
}

.card::before {
    content: "";
    position: absolute;
    top: 0;
    left: 0;
    width: 4px;
    height: 100%;
    background: var(--accent);
}

.header-card {
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
    color: white;
    border: none;
}
.header-card::before { display: none; }

.header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 20px;
}

h1 {
    margin: 0;
    font-size: 24px;
    font-weight: 800;
    letter-spacing: -0.025em;
    display: flex;
    align-items: center;
    gap: 12px;
}
h1::before {
    content: "⚡";
    font-size: 20px;
}

.badge {
    display: inline-flex;
    align-items: center;
    padding: 4px 12px;
    border-radius: 6px;
    background: rgba(99, 102, 241, 0.2);
    color: #a5b4fc;
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    border: 1px solid rgba(99, 102, 241, 0.3);
}

.meta {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 16px;
    font-size: 13px;
    color: #94a3b8;
    border-top: 1px solid rgba(255,255,255,0.1);
    padding-top: 16px;
}

.meta-item strong {
    color: #e2e8f0;
    font-weight: 600;
    margin-right: 4px;
}

.score-card {
    text-align: center;
    padding: 40px 20px;
    display: flex;
    flex-direction: column;
    align-items: center;
}

.score-good { border-color: var(--success); }
.score-good::before { background: var(--success); }
.score-mid { border-color: var(--warning); }
.score-mid::before { background: var(--warning); }
.score-low { border-color: var(--danger); }
.score-low::before { background: var(--danger); }

.score-label { 
    font-size: 12px; 
    font-weight: 800; 
    text-transform: uppercase; 
    letter-spacing: 0.2em; 
    color: var(--text-muted);
    margin-bottom: 8px;
}
.score-value { 
    font-size: 84px; 
    font-weight: 900; 
    line-height: 1; 
    margin: 12px 0; 
    font-variant-numeric: tabular-nums;
    letter-spacing: -0.05em;
    font-family: 'JetBrains Mono', monospace;
}
.score-good .score-value { color: var(--success); }
.score-mid .score-value { color: var(--warning); }
.score-low .score-value { color: var(--danger); }

.score-reason {
    font-size: 16px;
    max-width: 650px;
    margin: 20px auto 0;
    font-weight: 600;
    padding: 16px 24px;
    background: var(--bg);
    border-radius: 8px;
    color: var(--text-main);
    border: 1px dashed var(--border);
}

.content-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 20px;
    border-bottom: 2px solid var(--border);
    padding-bottom: 12px;
}

h2 { font-size: 18px; margin: 0; color: var(--primary); font-weight: 800; text-transform: uppercase; letter-spacing: 0.05em; }

.copy-btn {
    padding: 6px 14px;
    background: white;
    border: 1px solid var(--border);
    border-radius: 6px;
    font-size: 12px;
    font-weight: 700;
    color: var(--text-muted);
    cursor: pointer;
    transition: all 0.2s;
    text-transform: uppercase;
}
.copy-btn:hover { border-color: var(--accent); color: var(--accent); background: var(--accent-light); }

.markdown-body {
    font-size: 15px;
    color: var(--text-main);
    line-height: 1.8;
}

.markdown-body h1, .markdown-body h2, .markdown-body h3 { 
    color: var(--primary); 
    margin-top: 1.8em; 
    margin-bottom: 0.8em; 
    font-weight: 800;
}
.markdown-body h2 { 
    font-size: 1.15rem; 
    border-bottom: 1px solid var(--border); 
    padding-bottom: 0.4em;
    display: flex;
    align-items: center;
}
.markdown-body h3 { font-size: 1rem; color: var(--text-muted); }

.markdown-body ul, .markdown-body ol { padding-left: 1.4em; margin-bottom: 1.2em; }
.markdown-body li { margin-bottom: 0.6em; }
.markdown-body strong { color: var(--primary); font-weight: 700; background: rgba(99, 102, 241, 0.05); padding: 0 4px; }

.markdown-body blockquote {
    margin: 1.5em 0;
    padding: 0.8em 1.5em;
    color: #475569;
    border-left: 4px solid var(--accent);
    background: #f1f5f9;
    font-style: italic;
    border-radius: 0 8px 8px 0;
}

.markdown-body code {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.9em;
    background: #e2e8f0;
    padding: 0.2em 0.5em;
    border-radius: 4px;
    color: #334155;
    font-weight: 600;
}

.summary-card {
    background: #f0f9ff;
    border-color: #bae6fd;
}
.summary-card::before { background: #0ea5e9; }
.summary-title { color: #0369a1; font-size: 12px; font-weight: 800; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 8px; }
.summary-text { font-size: 15px; color: #0c4a6e; font-weight: 500; }

.section-title {
    font-size: 12px;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 12px;
    display: flex;
    align-items: center;
    gap: 8px;
}

.rationale-card::before { background: var(--success); }
.rationale-title { color: var(--success); }
.comparison-card::before { background: var(--warning); }
.comparison-title { color: var(--warning); }

.model-badge {
    background: #f1f5f9;
    padding: 4px 10px;
    border-radius: 6px;
    font-size: 11px;
    font-weight: 600;
    color: #64748b;
    border: 1px solid var(--border);
    font-family: 'JetBrains Mono', monospace;
}

.footer {
    text-align: center;
    font-size: 12px;
    color: var(--text-muted);
    margin-top: 60px;
    padding-bottom: 40px;
    border-top: 1px solid var(--border);
    padding-top: 24px;
}

@media (max-width: 640px) {
    .card { padding: 20px; }
    h1 { font-size: 20px; }
    .score-value { font-size: 64px; }
}
</style>
</head>
<body>
<div class="container">
    <div class="card header-card">
        <div class="header">
            <h1>SYSTEM ENERGY REPORT</h1>
            <div class="badge">Kernel v2.5.0-STABLE</div>
        </div>
        <div class="meta">
            <div class="meta-item"><strong>TARGET_DATE:</strong>{{.Prompt}}</div>
            <div class="meta-item"><strong>TIMESTAMP:</strong>{{.Time}}</div>
            <div class="meta-item"><strong>DURATION:</strong>{{.Duration}}</div>
        </div>
    </div>

    <div class="card score-card {{.ScoreClass}}">
        <div class="score-label">Global Energy Score</div>
        <div class="score-value">{{.Score}}</div>
        {{if .ScoreReason}}
        <div class="score-reason">{{.ScoreReason}}</div>
        {{end}}
    </div>

    {{if .Summary}}
    <div class="card summary-card">
        <div class="summary-title">Executive Summary</div>
        <div class="summary-text">{{.Summary}}</div>
    </div>
    {{end}}

    <div class="card">
        <div class="content-header">
            <h2>Final Orchestration Conclusions</h2>
            <button class="copy-btn" onclick="copyContent()">Copy Raw</button>
        </div>
        <div id="final-content" class="markdown-body"></div>
    </div>

    {{if .Rationale}}
    <div class="card rationale-card">
        <div class="section-title rationale-title">💡 Orchestration Rationale</div>
        <div id="rationale-content" class="markdown-body" style="font-size: 14px; opacity: 0.9;"></div>
    </div>
    {{end}}

    {{if .Comparison}}
    <div class="card comparison-card">
        <div class="section-title comparison-title">📊 Multi-Model Diff Analysis</div>
        <div id="comparison-content" class="markdown-body" style="font-size: 14px; opacity: 0.9;"></div>
    </div>
    {{end}}

    <div class="card">
        <div class="section-title" style="color: var(--text-muted);">Active Compute Nodes</div>
        <div style="display: flex; flex-wrap: wrap; gap: 8px;">
            {{range .SuccessModels}}
            <span class="model-badge">{{.}}</span>
            {{else}}
            <span class="model-badge">NO_ACTIVE_NODES</span>
            {{end}}
        </div>
    </div>

    <div class="footer">
        <div>ORCHESTRATION_HASH: {{.Time}} | LOG_PATH: reports/{{.Time}}</div>
        <div style="margin-top: 8px; opacity: 0.6; font-size: 10px;">&copy; 2026 Energy Management System. All rights reserved.</div>
    </div>
</div>

<div id="markdown-raw" style="display:none;">{{.FinalContent}}</div>
<div id="rationale-raw" style="display:none;">{{.Rationale}}</div>
<div id="comparison-raw" style="display:none;">{{.Comparison}}</div>

<script>
document.addEventListener('DOMContentLoaded', () => { 
    const mdOptions = { gfm: true, breaks: true };

    const render = (id, rawId) => {
        const el = document.getElementById(id);
        const rawEl = document.getElementById(rawId);
        if (el && rawEl) {
            el.innerHTML = marked.parse(rawEl.textContent, mdOptions);
        }
    };

    render('final-content', 'markdown-raw');
    render('rationale-content', 'rationale-raw');
    render('comparison-content', 'comparison-raw');
});

function copyContent() {
    const content = document.getElementById('markdown-raw').textContent;
    navigator.clipboard.writeText(content).then(() => {
        const btn = document.querySelector('.copy-btn');
        btn.innerText = 'COPIED!';
        setTimeout(() => btn.innerText = 'COPY RAW', 2000);
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
body { font-family: 'JetBrains Mono', monospace; background: #f8fafc; color: #334155; padding: 20px; line-height: 1.6; }
.card { background: white; border-radius: 8px; padding: 24px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); max-width: 900px; margin: 0 auto; border: 1px solid #e2e8f0; }
h1 { font-size: 18px; color: #0f172a; margin-top: 0; border-bottom: 2px solid #6366f1; padding-bottom: 10px; text-transform: uppercase; }
.meta { font-size: 12px; color: #64748b; margin-bottom: 20px; background: #f1f5f9; padding: 12px; border-radius: 4px; border-left: 4px solid #6366f1; }
.markdown-body { font-size: 14px; }
.markdown-body h2 { font-size: 16px; color: #1e293b; border-bottom: 1px solid #cbd5e1; padding-bottom: 5px; margin-top: 1.5em; }
</style>
</head>
<body>
<div class="card">
    <h1>{{.Title}}</h1>
    <div class="meta">
        <div><strong>NODE:</strong> {{.Model}}</div>
        <div><strong>LATENCY:</strong> {{.Duration}}</div>
        {{if .Error}}<div style="color: #ef4444;"><strong>FAULT:</strong> {{.Error}}</div>{{end}}
    </div>
    <div id="content" class="markdown-body"></div>
</div>
<div id="raw" style="display:none;">{{.Content}}</div>
<script>
    document.addEventListener('DOMContentLoaded', () => {
        document.getElementById('content').innerHTML = marked.parse(document.getElementById('raw').textContent);
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

	metaStart := strings.Index(content, "<strong>LATENCY:</strong>")
	if metaStart == -1 {
		metaStart = strings.Index(content, "<strong>耗时：</strong>")
	}
	if metaStart != -1 {
		metaStart = strings.Index(content[metaStart:], "</strong>") + metaStart + len("</strong>")
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

func findExistingJudgeResultToday(t time.Time) (*JudgeResult, bool) {
	todayDir := filepath.Join("reports", t.Format("2006-01-02"))
	path := filepath.Join(todayDir, "judge.html")

	data, err := os.ReadFile(path)
	if err != nil {
		return nil, false
	}

	content := string(data)
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

	res := &JudgeResult{
		Content: strings.TrimSpace(content[startIdx : startIdx+endIdx]),
		Enabled: true,
	}

	modelStart := strings.Index(content, "<strong>NODE:</strong>")
	if modelStart == -1 {
		modelStart = strings.Index(content, "<strong>模型：</strong>")
	}
	if modelStart != -1 {
		modelStart = strings.Index(content[modelStart:], "</strong>") + modelStart + len("</strong>")
		modelEnd := strings.Index(content[modelStart:], "</div>")
		if modelEnd != -1 {
			res.Model = strings.TrimSpace(content[modelStart : modelStart+modelEnd])
		}
	}

	metaStart := strings.Index(content, "<strong>LATENCY:</strong>")
	if metaStart == -1 {
		metaStart = strings.Index(content, "<strong>耗时：</strong>")
	}
	if metaStart != -1 {
		metaStart = strings.Index(content[metaStart:], "</strong>") + metaStart + len("</strong>")
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
		Title:    "Node Compute Output",
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
		Title:    "Orchestrator Audit Report",
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
	sb.WriteString("# 任务执行摘要\n\n")
	sb.WriteString(fmt.Sprintf("- 时间：%s\n- 总耗时：%s\n\n", t.Format("2006-01-02 15:04:05"), totalDuration.Round(time.Millisecond)))

	for _, r := range results {
		sb.WriteString(fmt.Sprintf("## 节点：%s\n", r.Model))
		if r.Err != nil {
			sb.WriteString(fmt.Sprintf("- 状态：ERROR\n- 错误：%v\n\n", r.Err))
		} else {
			sb.WriteString(fmt.Sprintf("- 状态：SUCCESS\n- 耗时：%s\n\n%s\n\n", r.CallDuration.Round(time.Millisecond), r.Content))
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
		Title:    "Cluster Summary Report",
		Model:    "System-Orchestrator",
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
		finalContent = fmt.Sprintf("⚠️ **ORCHESTRATION_FAILED**\n\n- Error: %v\n\nPlease refer to individual node reports for details.", judgeResult.Err)
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
		Rationale:     extractSection(judgeResult.Content, "决策依据", "Rationale", "采用逻辑"),
		Comparison:    extractSection(judgeResult.Content, "模型对比", "Comparison", "优胜模型"),
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

func extractFortuneScore(jr JudgeResult) (string, string) {
	if jr.Err != nil || jr.Content == "" {
		return "N/A", "FAULT_DURING_EXTRACTION"
	}
	lines := strings.Split(jr.Content, "\n")
	score := "UNDEFINED"
	reason := ""

	scoreRegexes := []*regexp.Regexp{
		regexp.MustCompile(`(?i)(?:评分|气场评分|综合评分|SCORE)[：:]\s*([0-9.]+)(?:\s*/\s*10)?`),
		regexp.MustCompile(`([0-9.]+)\s*/\s*10`),
	}

	reasonRegex := regexp.MustCompile(`(?i)(?:气场点评|气场分析|核心点评|综合点评|运势点评|点评|STATUS)[：:]\s*(.*)`)

	for i, line := range lines {
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
			for j := i; j < len(lines) && j < i+6; j++ {
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

	if reason == "" {
		for _, line := range lines {
			if reasonRegex.MatchString(line) {
				rm := reasonRegex.FindStringSubmatch(line)
				reason = strings.Trim(rm[1], "*_ >#")
				break
			}
		}
	}

	if reason == "" {
		reason = extractFinalConclusionSummary(jr.Content)
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
	} else if val <= 4.5 {
		return "score-low"
	}
	return "score-mid"
}

func extractFinalConclusionSummary(content string) string {
	if content == "" {
		return ""
	}
	lines := strings.Split(content, "\n")
	var firstPara string
	for _, line := range lines {
		trimmed := strings.TrimSpace(line)
		if trimmed == "" || strings.HasPrefix(trimmed, "#") {
			continue
		}
		firstPara = trimmed
		break
	}

	re := regexp.MustCompile(`[#*` + "`" + `>_-]`)
	clean := re.ReplaceAllString(firstPara, " ")

	reSpace := regexp.MustCompile(`\s+`)
	clean = strings.TrimSpace(reSpace.ReplaceAllString(clean, " "))

	if len([]rune(clean)) > 140 {
		return string([]rune(clean)[:140]) + "..."
	}
	return clean
}

func buildFinalContentWithoutScore(content string) string {
	if content == "" {
		return "（ORCHESTRATOR_OUTPUT_EMPTY）"
	}
	lines := strings.Split(content, "\n")
	var result []string

	scoreRegex := regexp.MustCompile(`(?i)(?:评分|气场评分|综合评分|SCORE)[：:]\s*[0-9.]+|[0-9.]+\s*/\s*10`)
	reasonRegex := regexp.MustCompile(`(?i)(?:气场点评|气场分析|核心点评|综合点评|运势点评|点评|STATUS)[：:]`)

	skipSection := false
	headerCounter := 1
	reHeader := regexp.MustCompile(`^##\s+(.*)`)

	for _, line := range lines {
		trimmed := strings.TrimSpace(line)
		if trimmed == "" {
			if !skipSection {
				result = append(result, line)
			}
			continue
		}

		if strings.HasPrefix(trimmed, "## ") {
			title := strings.ToLower(trimmed)
			if strings.Contains(title, "决策依据") || strings.Contains(title, "rationale") ||
				strings.Contains(title, "模型对比") || strings.Contains(title, "comparison") ||
				strings.Contains(title, "采用逻辑") || strings.Contains(title, "优胜模型") {
				skipSection = true
				continue
			}
			skipSection = false

			if matches := reHeader.FindStringSubmatch(trimmed); len(matches) > 1 {
				titleText := strings.TrimSpace(matches[1])
				// 去除原有的编号（如 1. 2.）
				titleText = regexp.MustCompile(`^\d+[.、\s]+`).ReplaceAllString(titleText, "")
				line = fmt.Sprintf("## %d. %s", headerCounter, titleText)
				headerCounter++
			}
		}

		if skipSection {
			continue
		}

		if scoreRegex.MatchString(line) || reasonRegex.MatchString(trimmed) ||
			strings.Contains(trimmed, "审计评分") || strings.Contains(trimmed, "最终评分") {
			continue
		}
		result = append(result, line)
	}

	return strings.TrimSpace(strings.Join(result, "\n"))
}

func extractSection(content string, keywords ...string) string {
	lines := strings.Split(content, "\n")
	var result []string
	found := false

	for _, line := range lines {
		trimmed := strings.TrimSpace(line)
		if strings.HasPrefix(trimmed, "## ") {
			if found {
				break
			}
			for _, kw := range keywords {
				if strings.Contains(strings.ToLower(trimmed), strings.ToLower(kw)) {
					found = true
					break
				}
			}
			continue
		}
		if found {
			result = append(result, line)
		}
	}
	return strings.TrimSpace(strings.Join(result, "\n"))
}
