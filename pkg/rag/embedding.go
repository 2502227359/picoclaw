package rag

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/sipeed/picoclaw/pkg/config"
)

type EmbeddingClient struct {
	apiKey     string
	apiBase    string
	model      string
	batchSize  int
	httpClient *http.Client
}

func NewEmbeddingClient(cfg config.RagEmbeddingConfig) (*EmbeddingClient, error) {
	if cfg.APIBase == "" {
		return nil, fmt.Errorf("embedding api_base is required")
	}
	if cfg.Model == "" {
		return nil, fmt.Errorf("embedding model is required")
	}
	batchSize := cfg.BatchSize
	if batchSize <= 0 {
		batchSize = 16
	}
	timeout := cfg.TimeoutSeconds
	if timeout <= 0 {
		timeout = 60
	}
	return &EmbeddingClient{
		apiKey:     cfg.APIKey,
		apiBase:    strings.TrimRight(cfg.APIBase, "/"),
		model:      cfg.Model,
		batchSize:  batchSize,
		httpClient: &http.Client{Timeout: time.Duration(timeout) * time.Second},
	}, nil
}

func (c *EmbeddingClient) BatchSize() int {
	return c.batchSize
}

func (c *EmbeddingClient) Model() string {
	return c.model
}

func (c *EmbeddingClient) EmbedBatch(ctx context.Context, inputs []string) ([][]float64, error) {
	if len(inputs) == 0 {
		return nil, nil
	}

	requestBody := map[string]interface{}{
		"model": c.model,
		"input": inputs,
	}

	jsonData, err := json.Marshal(requestBody)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal embedding request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, "POST", c.apiBase+"/embeddings", bytes.NewReader(jsonData))
	if err != nil {
		return nil, fmt.Errorf("failed to create embedding request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	if c.apiKey != "" {
		req.Header.Set("Authorization", "Bearer "+c.apiKey)
	}

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("embedding request failed: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read embedding response: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("embedding API error: %d %s", resp.StatusCode, string(body))
	}

	var apiResponse struct {
		Data []struct {
			Embedding []float64 `json:"embedding"`
			Index     int       `json:"index"`
		} `json:"data"`
	}

	if err := json.Unmarshal(body, &apiResponse); err != nil {
		return nil, fmt.Errorf("failed to parse embedding response: %w", err)
	}

	if len(apiResponse.Data) == 0 {
		return nil, fmt.Errorf("embedding response missing data")
	}

	embeddings := make([][]float64, len(apiResponse.Data))
	for _, item := range apiResponse.Data {
		if item.Index < 0 || item.Index >= len(embeddings) {
			continue
		}
		embeddings[item.Index] = item.Embedding
	}

	return embeddings, nil
}
