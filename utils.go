package main

import (
	"fmt"
	"os/exec"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
)

func uniqueStrings(values []string) []string {
	seen := make(map[string]struct{}, len(values))
	result := make([]string, 0, len(values))
	for _, value := range values {
		v := strings.TrimSpace(value)
		if v == "" {
			continue
		}
		if _, ok := seen[v]; ok {
			continue
		}
		seen[v] = struct{}{}
		result = append(result, v)
	}
	return result
}

func applyModelSelectionRules(models []string, cfg Config) []string {
	selected := make([]string, 0, len(models))
	for _, model := range uniqueStrings(models) {
		if !isLikelyChatModel(model) {
			continue
		}
		if len(cfg.ModelFilter) > 0 && !containsAnyKeyword(model, cfg.ModelFilter) {
			continue
		}
		if len(cfg.ModelSkip) > 0 && containsAnyKeyword(model, cfg.ModelSkip) {
			continue
		}
		selected = append(selected, model)
	}
	sort.Strings(selected)
	if cfg.ModelLimit > 0 && len(selected) > cfg.ModelLimit {
		selected = selected[:cfg.ModelLimit]
	}
	return selected
}

func isLikelyChatModel(modelName string) bool {
	name := strings.ToLower(strings.TrimSpace(modelName))
	blockedKeywords := []string{
		"embed", "embedding", "bge", "rerank", "reranker", "vision", "vl",
		"llava", "minicpm-v", "moondream", "clip", "whisper", "asr", "tts",
		"stable-diffusion", "sdxl",
	}
	for _, keyword := range blockedKeywords {
		if strings.Contains(name, keyword) {
			return false
		}
	}
	return true
}

func containsAnyKeyword(modelName string, keywords []string) bool {
	name := strings.ToLower(strings.TrimSpace(modelName))
	for _, keyword := range keywords {
		k := strings.ToLower(strings.TrimSpace(keyword))
		if k != "" && strings.Contains(name, k) {
			return true
		}
	}
	return false
}

func sanitizeFileName(name string) string {
	replacer := strings.NewReplacer(
		"\\", "_", "/", "_", ":", "_", "*", "_", "?", "_", "\"", "_", "<", "_", ">", "_", "|", "_", " ", "_",
	)
	return replacer.Replace(name)
}

func openInDefaultBrowser(path string) error {
	absPath, err := filepath.Abs(path)
	if err != nil {
		return err
	}
	target := "file:///" + filepath.ToSlash(absPath)
	fmt.Println("准备打开文件:", target)
	switch runtime.GOOS {
	case "windows":
		return exec.Command("cmd", "/c", "start", "", target).Run()
	case "darwin":
		return exec.Command("open", target).Run()
	default:
		return exec.Command("xdg-open", target).Run()
	}
}
