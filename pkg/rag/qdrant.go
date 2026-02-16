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

type QdrantClient struct {
	baseURL    string
	collection string
	httpClient *http.Client
}

type QdrantPoint struct {
	ID      string                 `json:"id"`
	Vector  []float64              `json:"vector"`
	Payload map[string]interface{} `json:"payload"`
}

func NewQdrantClient(cfg config.RagVectorDBConfig) (*QdrantClient, error) {
	if cfg.URL == "" {
		return nil, fmt.Errorf("vector_db url is required")
	}
	if cfg.Collection == "" {
		return nil, fmt.Errorf("vector_db collection is required")
	}
	timeout := cfg.TimeoutSeconds
	if timeout <= 0 {
		timeout = 30
	}
	return &QdrantClient{
		baseURL:    strings.TrimRight(cfg.URL, "/"),
		collection: cfg.Collection,
		httpClient: &http.Client{Timeout: time.Duration(timeout) * time.Second},
	}, nil
}

func (c *QdrantClient) Collection() string {
	return c.collection
}

func (c *QdrantClient) EnsureCollection(ctx context.Context, dimension int, recreate bool) error {
	if dimension <= 0 {
		return fmt.Errorf("invalid vector dimension: %d", dimension)
	}

	if recreate {
		_ = c.deleteCollection(ctx)
		return c.createCollection(ctx, dimension)
	}

	exists, currentDim, err := c.getCollectionDimension(ctx)
	if err != nil {
		return err
	}
	if !exists {
		return c.createCollection(ctx, dimension)
	}
	if currentDim > 0 && currentDim != dimension {
		if err := c.deleteCollection(ctx); err != nil {
			return err
		}
		return c.createCollection(ctx, dimension)
	}
	return nil
}

func (c *QdrantClient) Upsert(ctx context.Context, points []QdrantPoint) error {
	if len(points) == 0 {
		return nil
	}
	reqBody := map[string]interface{}{
		"points": points,
	}
	return c.doRequest(ctx, "PUT", fmt.Sprintf("/collections/%s/points?wait=true", c.collection), reqBody, nil)
}

func (c *QdrantClient) DeleteByPath(ctx context.Context, path string) error {
	if path == "" {
		return nil
	}
	reqBody := map[string]interface{}{
		"filter": map[string]interface{}{
			"must": []map[string]interface{}{
				{
					"key": "path",
					"match": map[string]interface{}{
						"value": path,
					},
				},
			},
		},
	}
	return c.doRequest(ctx, "POST", fmt.Sprintf("/collections/%s/points/delete?wait=true", c.collection), reqBody, nil)
}

func (c *QdrantClient) Search(ctx context.Context, vector []float64, limit int, minSimilarity float64) ([]SearchResult, error) {
	if len(vector) == 0 {
		return nil, fmt.Errorf("empty query vector")
	}
	if limit <= 0 {
		limit = 5
	}
	reqBody := map[string]interface{}{
		"vector":         vector,
		"limit":          limit,
		"with_payload":   true,
		"score_threshold": minSimilarity,
	}

	var resp struct {
		Result []struct {
			Score   float64                `json:"score"`
			Payload map[string]interface{} `json:"payload"`
		} `json:"result"`
	}

	if err := c.doRequest(ctx, "POST", fmt.Sprintf("/collections/%s/points/search", c.collection), reqBody, &resp); err != nil {
		return nil, err
	}

	results := make([]SearchResult, 0, len(resp.Result))
	for _, item := range resp.Result {
		payload := item.Payload
		res := SearchResult{
			Score: item.Score,
		}
		if v, ok := payload["path"].(string); ok {
			res.Path = v
		}
		if v, ok := payload["heading"].(string); ok {
			res.Heading = v
		}
		if v, ok := payload["content"].(string); ok {
			res.Content = v
		}
		if v, ok := payload["start_line"].(float64); ok {
			res.StartLine = int(v)
		}
		if v, ok := payload["end_line"].(float64); ok {
			res.EndLine = int(v)
		}
		results = append(results, res)
	}
	return results, nil
}

func (c *QdrantClient) getCollectionDimension(ctx context.Context) (bool, int, error) {
	var resp struct {
		Result struct {
			Config struct {
				Params struct {
					Vectors struct {
						Size int `json:"size"`
					} `json:"vectors"`
				} `json:"params"`
			} `json:"config"`
		} `json:"result"`
	}

	err := c.doRequest(ctx, "GET", fmt.Sprintf("/collections/%s", c.collection), nil, &resp)
	if err != nil {
		if strings.Contains(err.Error(), "404") {
			return false, 0, nil
		}
		return false, 0, err
	}

	return true, resp.Result.Config.Params.Vectors.Size, nil
}

func (c *QdrantClient) createCollection(ctx context.Context, dimension int) error {
	reqBody := map[string]interface{}{
		"vectors": map[string]interface{}{
			"size":     dimension,
			"distance": "Cosine",
		},
	}
	return c.doRequest(ctx, "PUT", fmt.Sprintf("/collections/%s", c.collection), reqBody, nil)
}

func (c *QdrantClient) deleteCollection(ctx context.Context) error {
	return c.doRequest(ctx, "DELETE", fmt.Sprintf("/collections/%s", c.collection), nil, nil)
}

func (c *QdrantClient) doRequest(ctx context.Context, method, path string, body interface{}, out interface{}) error {
	var reader io.Reader
	if body != nil {
		data, err := json.Marshal(body)
		if err != nil {
			return fmt.Errorf("failed to marshal qdrant request: %w", err)
		}
		reader = bytes.NewReader(data)
	}

	req, err := http.NewRequestWithContext(ctx, method, c.baseURL+path, reader)
	if err != nil {
		return fmt.Errorf("failed to create qdrant request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return fmt.Errorf("qdrant request failed: %w", err)
	}
	defer resp.Body.Close()

	data, err := io.ReadAll(resp.Body)
	if err != nil {
		return fmt.Errorf("failed to read qdrant response: %w", err)
	}

	if resp.StatusCode >= 300 {
		return fmt.Errorf("qdrant API error: %d %s", resp.StatusCode, string(data))
	}

	if out == nil {
		return nil
	}
	if err := json.Unmarshal(data, out); err != nil {
		return fmt.Errorf("failed to parse qdrant response: %w", err)
	}
	return nil
}
