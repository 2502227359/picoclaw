package rag

import (
	"encoding/json"
	"os"
	"path/filepath"
	"time"
)

type indexState struct {
	Version            int              `json:"version"`
	UpdatedAt          string           `json:"updated_at"`
	Collection         string           `json:"collection"`
	EmbeddingModel     string           `json:"embedding_model"`
	EmbeddingDimension int              `json:"embedding_dimension"`
	ChunkSize          int              `json:"chunk_size"`
	ChunkOverlap       int              `json:"chunk_overlap"`
	IncludePatterns    []string         `json:"include_patterns"`
	ExcludePatterns    []string         `json:"exclude_patterns"`
	Files              map[string]int64 `json:"files"`
}

func loadIndexState(path string) (*indexState, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var state indexState
	if err := json.Unmarshal(data, &state); err != nil {
		return nil, err
	}
	if state.Files == nil {
		state.Files = map[string]int64{}
	}
	return &state, nil
}

func saveIndexState(path string, state *indexState) error {
	state.UpdatedAt = time.Now().Format(time.RFC3339)
	data, err := json.MarshalIndent(state, "", "  ")
	if err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(path), 0755); err != nil {
		return err
	}
	return os.WriteFile(path, data, 0644)
}
