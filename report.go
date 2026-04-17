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
<title>CORE_LOGIC_AUDIT_REPORT</title>
<script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
<style>
:root {
    --primary: #0f172a;
    --accent: #6366f1;
    --accent-glow: rgba(99, 102, 241, 0.4);
    --bg: #0b0f1a;
    --card-bg: rgba(21, 27, 45, 0.8);
    --text-main: #f1f5f9;
    --text-muted: #94a3b8;
    --border: #1e293b;
    --success: #10b981;
    --warning: #f59e0b;
    --danger: #ef4444;
}

body {
    font-family: 'Inter', 'JetBrains Mono', 'PingFang SC', sans-serif;
    background: var(--bg);
    background-image: radial-gradient(circle at 50% 0%, #1e293b 0%, #0b0f1a 100%);
    color: var(--text-main);
    margin: 0;
    padding: 40px 20px;
    line-height: 1.8;
}

.container { max-width: 960px; margin: 0 auto; }

.card {
    background: var(--card-bg);
    backdrop-filter: blur(10px);
    border-radius: 12px;
    padding: 32px;
    margin-bottom: 24px;
    border: 1px solid var(--border);
    box-shadow: 0 10px 30px rgba(0,0,0,0.3);
}

.header-card {
    border-bottom: 4px solid var(--accent);
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
}

h1 {
    margin: 0;
    font-size: 24px;
    font-weight: 900;
    letter-spacing: -0.02em;
}
h1 span { color: var(--accent); }

.meta-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 16px;
    margin-top: 24px;
    font-size: 12px;
    font-family: 'JetBrains Mono';
    color: var(--text-muted);
}

.meta-item strong { color: var(--text-main); margin-right: 8px; }

.score-panel {
    display: grid;
    grid-template-columns: 180px 1fr;
    gap: 32px;
    align-items: center;
}

.score-circle {
    width: 150px;
    height: 150px;
    border-radius: 50%;
    border: 8px solid var(--accent);
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    box-shadow: 0 0 30px var(--accent-glow);
}

.score-value { font-size: 56px; font-weight: 900; color: var(--accent); line-height: 1; }
.score-label { font-size: 10px; color: var(--text-muted); text-transform: uppercase; margin-top: 4px; letter-spacing: 2px; }

.score-summary {
    font-size: 20px;
    font-weight: 700;
    color: #fff;
    line-height: 1.4;
    padding-left: 24px;
    border-left: 4px solid var(--accent);
}

.markdown-body { font-size: 15px; }
.markdown-body h2 {
    font-size: 18px;
    color: var(--accent);
    margin: 40px 0 20px 0;
    padding-bottom: 12px;
    border-bottom: 1px solid var(--border);
}

.markdown-body strong { color: #fff; background: rgba(99, 102, 241, 0.2); padding: 0 4px; border-radius: 4px; }
.markdown-body ul { padding-left: 20px; }
.markdown-body li { margin-bottom: 12px; }

.footer-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 24px;
    margin-top: 48px;
}

.sub-card {
    background: rgba(255,255,255,0.03);
    border-radius: 8px;
    padding: 20px;
    border: 1px solid var(--border);
}

.sub-card h3 {
    margin: 0 0 16px 0;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: var(--text-muted);
}

.success-tags { display: flex; flex-wrap: wrap; gap: 8px; }
.tag {
    font-size: 10px;
    padding: 4px 10px;
    background: rgba(16, 185, 129, 0.1);
    color: var(--success);
    border: 1px solid var(--success);
    border-radius: 20px;
    font-family: 'JetBrains Mono';
}

.score-good { border-left-color: var(--success); }
.score-mid { border-left-color: var(--warning); }
/* Pillar Board Styles */
.bazi-board {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
    margin-bottom: 32px;
    padding: 20px;
    background: rgba(0,0,0,0.2);
    border-radius: 8px;
    border: 1px solid var(--border);
}

.pillar {
    text-align: center;
    padding: 12px;
    background: rgba(255,255,255,0.03);
    border-radius: 6px;
    border-top: 3px solid var(--accent);
}

.pillar-label { font-size: 10px; color: var(--text-muted); text-transform: uppercase; margin-bottom: 8px; }
.pillar-value { font-size: 20px; font-weight: 900; color: #fff; letter-spacing: 2px; }
.pillar-nayin { font-size: 10px; color: var(--text-muted); margin-top: 8px; }

.bazi-meta {
    display: flex;
    justify-content: space-between;
    font-size: 12px;
    margin-bottom: 16px;
    color: var(--text-muted);
}
.bazi-meta strong { color: var(--accent); margin-right: 4px; }
</style>
</head>
<body>
<div class="container">
    <div class="card header-card">
        <div>
            <h1>SYSTEM<span>_CORE_OUTPUT</span></h1>
            <div class="meta-grid">
                <div class="meta-item"><strong>TIMESTAMP</strong> {{.Time}}</div>
                <div class="meta-item"><strong>LATENCY</strong> {{.Duration}}</div>
                <div class="meta-item"><strong>AUDITOR</strong> {{.JudgeModel}}</div>
            </div>
        </div>
        <div style="text-align: right;">
            <div style="font-size: 10px; color: var(--text-muted); letter-spacing: 1px;">STATUS</div>
            <div style="color: var(--success); font-weight: 900; font-size: 14px;">● OPERATIONAL</div>
        </div>
    </div>

    <div class="card {{.ScoreClass}}" style="border-left-width: 8px;">
        <div class="score-panel">
            <div class="score-circle">
                <div class="score-value">{{.Score}}</div>
                <div class="score-label">Efficiency</div>
            </div>
            <div class="score-summary">
                {{.ScoreReason}}
            </div>
        </div>
    </div>

    <div class="card">
        <div class="bazi-meta">
            <div><strong>SOLAR</strong> {{.Bazi.SolarDate}}</div>
            <div><strong>LUNAR</strong> {{.Bazi.LunarDate}}</div>
            <div><strong>USER_MASTER</strong> {{.Bazi.UserGanzhi}}</div>
            <div><strong>DAY_MASTER</strong> {{.Bazi.TodayMaster}}</div>
            <div><strong>TERM</strong> {{.Bazi.SolarTerm}}</div>
        </div>

        <div class="bazi-board">
            <div class="pillar">
                <div class="pillar-label">YEAR (年)</div>
                <div class="pillar-value">{{index .Bazi.TodayGanzhi 0}}</div>
                <div class="pillar-nayin">{{index .Bazi.Nayins 0}}</div>
            </div>
            <div class="pillar">
                <div class="pillar-label">MONTH (月)</div>
                <div class="pillar-value">{{index .Bazi.TodayGanzhi 1}}</div>
                <div class="pillar-nayin">{{index .Bazi.Nayins 1}}</div>
            </div>
            <div class="pillar">
                <div class="pillar-label">DAY (日)</div>
                <div class="pillar-value">{{index .Bazi.TodayGanzhi 2}}</div>
                <div class="pillar-nayin">{{index .Bazi.Nayins 2}}</div>
            </div>
            <div class="pillar">
                <div class="pillar-label">HOUR (时)</div>
                <div class="pillar-value">{{index .Bazi.TodayGanzhi 3}}</div>
                <div class="pillar-nayin">{{index .Bazi.Nayins 3}}</div>
            </div>
        </div>

        <div style="font-size: 10px; color: var(--text-muted); margin-bottom: 24px; display: flex; gap: 20px;">
            <div><strong>XUNKONG (旬空)</strong> {{range .Bazi.Xunkongs}}{{.}} {{end}}</div>
        </div>

        <div id="content" class="markdown-body"></div>
    </div>

    <div class="footer-grid">
        <div class="sub-card">
            <h3>Decision Rationale</h3>
            <div id="rationale" class="markdown-body" style="font-size: 12px; color: var(--text-muted);"></div>
        </div>
        <div class="sub-card">
            <h3>Compute Nodes</h3>
            <div class="success-tags">
                {{range .SuccessModels}}<div class="tag">{{.}}</div>{{end}}
            </div>
            <h3 style="margin-top:24px">Node Analysis</h3>
            <div id="comparison" class="markdown-body" style="font-size: 12px; color: var(--text-muted);"></div>
        </div>
    </div>

    <details style="margin-top: 48px; border-top: 1px solid var(--border); padding-top: 24px;">
        <summary style="font-size: 11px; color: var(--text-muted); cursor: pointer; text-transform: uppercase; letter-spacing: 2px;">Technical Audit Trace (Raw)</summary>
        <div class="sub-card" style="margin-top: 16px; background: rgba(0,0,0,0.2);">
            <div id="raw-trace-display" class="markdown-body" style="font-size: 11px; font-family: 'JetBrains Mono'; color: #64748b;"></div>
        </div>
    </details>
</div>

<div id="raw" style="display:none;">{{.FinalContent}}</div>
<div id="raw-rationale" style="display:none;">{{.Rationale}}</div>
<div id="raw-comparison" style="display:none;">{{.Comparison}}</div>
<div id="raw-trace" style="display:none;">{{.RawTrace}}</div>

<script>
    document.addEventListener('DOMContentLoaded', () => {
        const renderer = new marked.Renderer();
        marked.setOptions({ renderer: renderer, gfm: true, breaks: true });
        
        document.getElementById('content').innerHTML = marked.parse(document.getElementById('raw').textContent);
        document.getElementById('rationale').innerHTML = marked.parse(document.getElementById('raw-rationale').textContent);
        document.getElementById('comparison').innerHTML = marked.parse(document.getElementById('raw-comparison').textContent);
        document.getElementById('raw-trace-display').innerHTML = marked.parse(document.getElementById('raw-trace').textContent);
    });
</script>
</body>
</html>`

const subReportTemplate = `<!doctype html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{{.Title}} - {{.Model}}</title>
<script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
<style>
body { font-family: 'JetBrains Mono', monospace; background: #0b0f1a; color: #e2e8f0; padding: 20px; line-height: 1.6; }
.card { background: #151b2d; border-radius: 8px; padding: 24px; box-shadow: 0 0 20px rgba(0,0,0,0.5); max-width: 900px; margin: 0 auto; border: 1px solid #1e293b; }
h1 { font-size: 18px; color: #fff; margin-top: 0; border-bottom: 2px solid #6366f1; padding-bottom: 10px; text-transform: uppercase; }
.meta { font-size: 12px; color: #94a3b8; margin-bottom: 20px; background: #1e293b; padding: 12px; border-radius: 4px; border-left: 4px solid #6366f1; }
.markdown-body { font-size: 14px; }
.markdown-body h2 { font-size: 16px; color: #6366f1; border-bottom: 1px solid #1e293b; padding-bottom: 5px; margin-top: 1.5em; }
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

type BaziInfo struct {
	SolarDate string
	LunarDate string
	UserGanzhi string
	TodayGanzhi []string // [年, 月, 日, 时]
	TodayMaster string
	SolarTerm   string
	Nayins      []string
	Xunkongs    []string
}

type HTMLData struct {
	Time          string
	Prompt        string
	Duration      string
	JudgeModel    string
	Score         string
	ScoreReason   string
	ScoreClass    string
	Summary       string
	SuccessModels []string
	FinalContent  string
	Rationale     string
	Comparison    string
	RawTrace      string
	Bazi          BaziInfo
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

func saveFinalConclusionHTML(reportDir string, t time.Time, prompt string, results []ModelResult, judgeResult JudgeResult, totalDuration time.Duration, bazi BaziInfo) (string, error) {
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
		JudgeModel:    judgeResult.Model,
		Score:         score,
		ScoreReason:   reason,
		ScoreClass:    fortuneScoreClass(score),
		Summary:       extractFinalConclusionSummary(judgeResult.Content),
		SuccessModels: successModels,
		FinalContent:  finalContent,
		Rationale:     extractSection(judgeResult.Content, "决策依据", "Rationale", "采用逻辑"),
		Comparison:    extractSection(judgeResult.Content, "模型对比", "Comparison", "优胜模型"),
		RawTrace:      judgeResult.Content,
		Bazi:          bazi,
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
		regexp.MustCompile(`(?i)(?:评分|能效评分|气场评分|综合评分|SCORE)[：:]\s*([0-9.]+)(?:\s*/\s*10)?`),
		regexp.MustCompile(`([0-9.]+)\s*/\s*10`),
	}

	reasonRegex := regexp.MustCompile(`(?i)(?:气场点评|气场分析|核心点评|综合点评|STATUS|气场判断)[：:]\s*(.*)`)

	for i, line := range lines {
		foundScore := false
		for _, reg := range scoreRegexes {
			if reg.MatchString(line) {
				matches := reg.FindStringSubmatch(line)
				if len(matches) > 1 {
					score = matches[1]
					foundScore = true
					break
				}
			}
		}

		if foundScore {
			// Search downwards for reason
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

	scoreRegex := regexp.MustCompile(`(?i)(?:评分|能效评分|气场评分|综合评分|SCORE)[：:]\s*[0-9.]+|[0-9.]+\s*/\s*10`)
	reasonRegex := regexp.MustCompile(`(?i)(?:气场点评|气场分析|核心点评|综合点评|STATUS|气场判断)[：:]`)
	// 匹配标题的正则：支持 #, ##, ### 或 **标题**
	headerRegex := regexp.MustCompile(`^(?:#{1,3}\s+|\*\*)(.*?)(?:\*\*|$)`)

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

		if headerRegex.MatchString(trimmed) {
			matches := headerRegex.FindStringSubmatch(trimmed)
			title := strings.ToLower(matches[1])

			if strings.Contains(title, "决策依据") || strings.Contains(title, "rationale") ||
				strings.Contains(title, "模型对比") || strings.Contains(title, "comparison") ||
				strings.Contains(title, "采用逻辑") || strings.Contains(title, "优胜模型") ||
				strings.Contains(title, "评分") || strings.Contains(title, "气场") ||
				strings.Contains(title, "核心结论") || strings.Contains(title, "结论") {
				skipSection = true
				continue
			}
			skipSection = false

			if strings.HasPrefix(trimmed, "## ") {
				if matches := reHeader.FindStringSubmatch(trimmed); len(matches) > 1 {
					titleText := strings.TrimSpace(matches[1])
					titleText = regexp.MustCompile(`^\d+[.、\s]+`).ReplaceAllString(titleText, "")
					line = fmt.Sprintf("## %d. %s", headerCounter, titleText)
					headerCounter++
				}
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

	// 匹配标题的正则：支持 #, ##, ### 或 **标题**
	headerRegex := regexp.MustCompile(`^(?:#{1,3}\s+|\*\*)(.*?)(?:\*\*|$)`)

	for _, line := range lines {
		trimmed := strings.TrimSpace(line)
		if trimmed == "" {
			if found {
				result = append(result, line)
			}
			continue
		}

		// 检查是否是标题行
		if headerRegex.MatchString(trimmed) {
			matches := headerRegex.FindStringSubmatch(trimmed)
			title := strings.ToLower(matches[1])

			isTargetHeader := false
			for _, kw := range keywords {
				if strings.Contains(title, strings.ToLower(kw)) {
					isTargetHeader = true
					break
				}
			}

			if isTargetHeader {
				found = true
				continue
			} else if found {
				// 如果已经找到了目标章节，现在遇到了另一个标题，说明当前章节结束
				break
			}
		}

		if found {
			result = append(result, line)
		}
	}
	return strings.TrimSpace(strings.Join(result, "\n"))
}
