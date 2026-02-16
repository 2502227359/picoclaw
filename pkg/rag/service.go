package rag

import (
	"context"
	"fmt"
	"strings"

	"github.com/sipeed/picoclaw/pkg/config"
)

type Service struct {
	cfg       config.RagConfig
	workspace string
	embedder  *EmbeddingClient
	qdrant    *QdrantClient
}

func NewService(cfg *config.Config, workspace string) (*Service, error) {
	if !cfg.RAG.Enabled {
		return nil, fmt.Errorf("rag is disabled")
	}
	embedder, err := NewEmbeddingClient(cfg.RAG.Embedding)
	if err != nil {
		return nil, err
	}
	qdrant, err := NewQdrantClient(cfg.RAG.VectorDB)
	if err != nil {
		return nil, err
	}
	return &Service{
		cfg:       cfg.RAG,
		workspace: workspace,
		embedder:  embedder,
		qdrant:    qdrant,
	}, nil
}

func (s *Service) Config() config.RagConfig {
	return s.cfg
}

func (s *Service) TriggerDecision(message string) TriggerDecision {
	return DecideTrigger(message, s.cfg.Trigger)
}

func (s *Service) Search(ctx context.Context, query string) ([]SearchResult, error) {
	query = strings.TrimSpace(query)
	if query == "" {
		return nil, nil
	}
	embeddings, err := s.embedder.EmbedBatch(ctx, []string{query})
	if err != nil {
		return nil, err
	}
	if len(embeddings) == 0 || len(embeddings[0]) == 0 {
		return nil, fmt.Errorf("embedding returned empty vector")
	}
	return s.qdrant.Search(ctx, embeddings[0], s.cfg.TopK, s.cfg.MinSimilarity)
}

func (s *Service) Index(ctx context.Context, opts IndexOptions) (*IndexSummary, error) {
	indexer := newIndexer(s.cfg, s.workspace, s.embedder, s.qdrant)
	return indexer.run(ctx, opts)
}

func (s *Service) FormatContext(results []SearchResult) string {
	if len(results) == 0 {
		return ""
	}
	var sb strings.Builder
	sb.WriteString("## Knowledge Base Notes\n")
	sb.WriteString("Use the notes below to answer the question. If the notes do not contain the answer, say so explicitly.\n\n")
	for idx, r := range results {
		label := idx + 1
		sb.WriteString(fmt.Sprintf("[%d] %s\n", label, formatSource(r)))
		snippet := strings.TrimSpace(r.Content)
		if s.cfg.SnippetMaxChars > 0 && len(snippet) > s.cfg.SnippetMaxChars {
			snippet = snippet[:s.cfg.SnippetMaxChars] + "...(truncated)"
		}
		sb.WriteString(snippet)
		sb.WriteString("\n\n")
	}
	sb.WriteString("When you answer, cite sources like [1], [2] and include a Sources section listing the cited entries.\n")
	return sb.String()
}

func (s *Service) FormatSources(results []SearchResult) string {
	if len(results) == 0 {
		return ""
	}
	var sb strings.Builder
	sb.WriteString("Sources:\n")
	for idx, r := range results {
		label := idx + 1
		sb.WriteString(fmt.Sprintf("[%d] %s\n", label, formatSource(r)))
	}
	return strings.TrimSpace(sb.String())
}

func formatSource(r SearchResult) string {
	if r.Heading != "" {
		return fmt.Sprintf("%s#%s L%d-L%d", r.Path, r.Heading, r.StartLine, r.EndLine)
	}
	return fmt.Sprintf("%s L%d-L%d", r.Path, r.StartLine, r.EndLine)
}
