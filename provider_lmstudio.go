package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

// LM Studio API structures
type LMStudioModelResponse struct {
	Data []struct {
		ID string `json:"id"`
	} `json:"data"`
}

type LMStudioNativeModel struct {
	ID    string `json:"id"`
	Path  string `json:"path"`
	State string `json:"state"` // "loaded" or "not_loaded"
}

type LMStudioNativeModelResponse struct {
	Data []LMStudioNativeModel `json:"data"`
}

type LMStudioChatRequest struct {
	Model    string    `json:"model"`
	Messages []Message `json:"messages"`
}

type LMStudioChatResponse struct {
	Choices []struct {
		Message Message `json:"message"`
	} `json:"choices"`
}

type LMStudioLoadRequest struct {
	Model string `json:"model"`
}

type LMStudioUnloadRequest struct {
	InstanceID string `json:"instance_id"`
}

// resolveLMStudioModels fetches all models, trying both Native and V1 APIs for maximum compatibility
func resolveLMStudioModels(baseURL string) ([]string, error) {
	client := &http.Client{Timeout: 5 * time.Second}

	// 1. Try Native API first to get rich info (including Path)
	apiBase := strings.Replace(baseURL, "/v1", "/api/v1", 1)
	nativeUrl := fmt.Sprintf("%s/models", strings.TrimRight(apiBase, "/"))

	var allModels []string
	nativeResp, err := client.Get(nativeUrl)
	if err == nil {
		defer nativeResp.Body.Close()
		var res LMStudioNativeModelResponse
		body, _ := io.ReadAll(nativeResp.Body)
		if json.Unmarshal(body, &res) == nil && len(res.Data) > 0 {
			for _, m := range res.Data {
				if m.ID != "" {
					allModels = append(allModels, m.ID)
				}
			}
		} else {
			fmt.Printf("   [调试] Native API 返回数据为空或解析失败: %s\n", string(body))
		}
	} else {
		fmt.Printf("   [调试] 尝试 Native API 失败: %v\n", err)
	}

	// 2. Always try V1 API as fallback or to supplement IDs
	v1Url := fmt.Sprintf("%s/models", strings.TrimRight(baseURL, "/"))
	v1Resp, err := client.Get(v1Url)
	if err == nil {
		defer v1Resp.Body.Close()
		var res LMStudioModelResponse
		if json.NewDecoder(v1Resp.Body).Decode(&res) == nil {
			for _, m := range res.Data {
				if m.ID != "" {
					allModels = append(allModels, m.ID)
				}
			}
		}
	}

	finalModels := uniqueStrings(allModels)
	if len(finalModels) == 0 {
		fmt.Printf("   [警告] 未能从任何接口获取到 LM Studio 模型列表。\n")
	}
	return finalModels, nil
}

// isLMStudioModelLoaded checks if a model is already in memory using Native API
func isLMStudioModelLoaded(baseURL, modelKey string) bool {
	client := &http.Client{Timeout: 3 * time.Second}
	apiBase := strings.Replace(baseURL, "/v1", "/api/v1", 1)
	url := fmt.Sprintf("%s/models", strings.TrimRight(apiBase, "/"))

	resp, err := client.Get(url)
	if err != nil {
		return false
	}
	defer resp.Body.Close()

	var res LMStudioNativeModelResponse
	if err := json.NewDecoder(resp.Body).Decode(&res); err != nil {
		return false
	}

	for _, m := range res.Data {
		if (m.ID == modelKey || m.Path == modelKey) && m.State == "loaded" {
			return true
		}
	}
	return false
}

// loadLMStudioModel explicitly loads a model into memory
func loadLMStudioModel(baseURL, modelKey string) error {
	if isLMStudioModelLoaded(baseURL, modelKey) {
		fmt.Printf("   模型 %s 当前已在内存中，跳过加载操作。\n", modelKey)
		return nil
	}

	apiBase := strings.Replace(baseURL, "/v1", "/api/v1", 1)
	url := fmt.Sprintf("%s/models/load", strings.TrimRight(apiBase, "/"))

	// 1. First attempt with provided modelKey
	err := attemptLoad(url, modelKey)
	if err == nil {
		return nil
	}

	// 2. If 404 and looks like an ID, try to find the Path
	if strings.Contains(err.Error(), "404") || strings.Contains(err.Error(), "model_not_found") {
		fmt.Printf("   直接通过 ID 加载失败，正在尝试查找模型路径...\n")
		path, findErr := findModelPathByID(baseURL, modelKey)
		if findErr == nil && path != "" && path != modelKey {
			fmt.Printf("   找到对应路径: %s，正在重新尝试加载...\n", path)
			return attemptLoad(url, path)
		}
	}

	return err
}

func attemptLoad(url, identifier string) error {
	reqBody := LMStudioLoadRequest{
		Model: identifier,
	}
	data, _ := json.Marshal(reqBody)
	client := &http.Client{Timeout: 10 * time.Minute}

	resp, err := client.Post(url, "application/json", bytes.NewBuffer(data))
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		if strings.Contains(string(body), "already loaded") {
			return nil
		}
		return fmt.Errorf("LM Studio 加载失败 (%d): %s", resp.StatusCode, string(body))
	}
	return nil
}

func findModelPathByID(baseURL, id string) (string, error) {
	client := &http.Client{Timeout: 5 * time.Second}
	apiBase := strings.Replace(baseURL, "/v1", "/api/v1", 1)
	url := fmt.Sprintf("%s/models", strings.TrimRight(apiBase, "/"))

	resp, err := client.Get(url)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	var res LMStudioNativeModelResponse
	if err := json.NewDecoder(resp.Body).Decode(&res); err != nil {
		return "", err
	}

	for _, m := range res.Data {
		if m.ID == id {
			return m.Path, nil
		}
	}
	return "", fmt.Errorf("model not found")
}

// unloadLMStudioModel unloads a model from memory
func unloadLMStudioModel(baseURL, identifier string) error {
	apiBase := strings.Replace(baseURL, "/v1", "/api/v1", 1)
	url := fmt.Sprintf("%s/models/unload", strings.TrimRight(apiBase, "/"))

	reqBody := LMStudioUnloadRequest{
		InstanceID: identifier,
	}

	data, _ := json.Marshal(reqBody)
	client := &http.Client{Timeout: 30 * time.Second}

	resp, err := client.Post(url, "application/json", bytes.NewBuffer(data))
	if err != nil {
		fmt.Printf("   卸载请求网络失败: %v\n", err)
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		fmt.Printf("   卸载指令被拒绝 (%d): %s\n", resp.StatusCode, string(body))
		return fmt.Errorf("unload failed: %s", string(body))
	}

	fmt.Printf("   模型 %s 已成功发出卸载指令。\n", identifier)
	return nil
}

// chatWithLMStudio sends a chat completion request to LM Studio
func chatWithLMStudio(baseURL, model, system, user string, timeout time.Duration) (string, error) {
	reqBody := LMStudioChatRequest{
		Model: model,
		Messages: []Message{
			{Role: "system", Content: system},
			{Role: "user", Content: user},
		},
	}

	data, _ := json.Marshal(reqBody)
	client := &http.Client{Timeout: timeout}

	resp, err := client.Post(
		fmt.Sprintf("%s/chat/completions", strings.TrimRight(baseURL, "/")),
		"application/json",
		bytes.NewBuffer(data),
	)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return "", fmt.Errorf("LM Studio 响应错误 (%d): %s", resp.StatusCode, string(body))
	}

	var res LMStudioChatResponse
	body, _ := io.ReadAll(resp.Body)
	if err := json.Unmarshal(body, &res); err != nil {
		fmt.Printf("   [调试] LM Studio 响应解析失败: %v\n", err)
		fmt.Printf("   [调试] 原始响应内容: %s\n", string(body))
		return "", err
	}

	if len(res.Choices) > 0 {
		content := res.Choices[0].Message.Content
		if content == "" {
			fmt.Printf("   [调试] LM Studio 返回内容为空。完整响应: %s\n", string(body))
		}
		return content, nil
	}

	fmt.Printf("   [调试] LM Studio 返回 choices 为空。完整响应: %s\n", string(body))
	return "", fmt.Errorf("LM Studio 返回了空内容")
}
