package rag

import (
	"context"
	"crypto/sha1"
	"encoding/hex"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"regexp"
	"strings"

	"github.com/sipeed/picoclaw/pkg/config"
)

type indexer struct {
	cfg       config.RagConfig
	workspace string
	embedder  *EmbeddingClient
	qdrant    *QdrantClient
}

func newIndexer(cfg config.RagConfig, workspace string, embedder *EmbeddingClient, qdrant *QdrantClient) *indexer {
	return &indexer{
		cfg:       cfg,
		workspace: workspace,
		embedder:  embedder,
		qdrant:    qdrant,
	}
}

func (i *indexer) run(ctx context.Context, opts IndexOptions) (*IndexSummary, error) {
	vaultPath := expandHome(i.cfg.VaultPath)
	if vaultPath == "" {
		return nil, fmt.Errorf("rag.vault_path is required")
	}
	info, err := os.Stat(vaultPath)
	if err != nil || !info.IsDir() {
		return nil, fmt.Errorf("vault path not found: %s", vaultPath)
	}

	statePath := filepath.Join(i.workspace, "rag", "index_state.json")
	state, _ := loadIndexState(statePath)

	reindexAll := opts.ReindexAll
	if state == nil {
		reindexAll = true
	}

	if state != nil && !reindexAll {
		if state.EmbeddingModel != i.embedder.Model() {
			reindexAll = true
		}
		if state.ChunkSize != i.cfg.ChunkSize || state.ChunkOverlap != i.cfg.ChunkOverlap {
			reindexAll = true
		}
		if !stringSliceEqual(state.IncludePatterns, i.cfg.IncludePatterns) ||
			!stringSliceEqual(state.ExcludePatterns, i.cfg.ExcludePatterns) {
			reindexAll = true
		}
		if state.Collection != i.qdrant.Collection() {
			reindexAll = true
		}
	}

	files, err := listMarkdownFiles(vaultPath, i.cfg.IncludePatterns, i.cfg.ExcludePatterns)
	if err != nil {
		return nil, err
	}

	currentFiles := make(map[string]int64, len(files))
	for _, f := range files {
		currentFiles[f.RelPath] = f.MTime
	}

	if state == nil {
		state = &indexState{
			Version: 1,
			Files:   map[string]int64{},
		}
	}

	dimension := state.EmbeddingDimension
	if dimension == 0 && i.cfg.Embedding.Dimension > 0 {
		dimension = i.cfg.Embedding.Dimension
	}

	ensureCollection := func(dim int) error {
		if dim <= 0 {
			return fmt.Errorf("invalid embedding dimension")
		}
		if err := i.qdrant.EnsureCollection(ctx, dim, reindexAll); err != nil {
			return err
		}
		state.EmbeddingDimension = dim
		return nil
	}

	if dimension > 0 {
		if err := ensureCollection(dimension); err != nil {
			return nil, err
		}
	}

	summary := &IndexSummary{TotalFiles: len(files)}

	if reindexAll {
		state.Files = map[string]int64{}
	}

	for path := range state.Files {
		if _, ok := currentFiles[path]; !ok {
			if err := i.qdrant.DeleteByPath(ctx, path); err != nil {
				return nil, err
			}
			delete(state.Files, path)
			summary.RemovedFiles++
		}
	}

	for _, file := range files {
		mt := file.MTime
		if !reindexAll {
			if prev, ok := state.Files[file.RelPath]; ok && prev == mt {
				summary.SkippedFiles++
				continue
			}
		}

		content, err := os.ReadFile(file.AbsPath)
		if err != nil {
			return nil, fmt.Errorf("failed to read %s: %w", file.AbsPath, err)
		}

		chunks := chunkMarkdown(file.RelPath, string(content), i.cfg.ChunkSize, i.cfg.ChunkOverlap)
		if len(chunks) == 0 {
			state.Files[file.RelPath] = mt
			continue
		}

		if err := i.qdrant.DeleteByPath(ctx, file.RelPath); err != nil {
			return nil, err
		}

		batchSize := i.embedder.BatchSize()
		for start := 0; start < len(chunks); start += batchSize {
			end := start + batchSize
			if end > len(chunks) {
				end = len(chunks)
			}
			batch := chunks[start:end]
			texts := make([]string, len(batch))
			for idx, ch := range batch {
				texts[idx] = ch.Content
			}
			embeddings, err := i.embedder.EmbedBatch(ctx, texts)
			if err != nil {
				return nil, err
			}
			if len(embeddings) != len(batch) {
				return nil, fmt.Errorf("embedding result size mismatch")
			}
			if state.EmbeddingDimension == 0 {
				dimension = len(embeddings[0])
				if i.cfg.Embedding.Dimension > 0 && i.cfg.Embedding.Dimension != dimension {
					return nil, fmt.Errorf("embedding dimension mismatch: got %d expected %d", dimension, i.cfg.Embedding.Dimension)
				}
				if err := ensureCollection(dimension); err != nil {
					return nil, err
				}
			}

			points := make([]QdrantPoint, 0, len(batch))
			for idx, ch := range batch {
				emb := embeddings[idx]
				pointID := hashPointID(file.RelPath, ch.StartLine, ch.EndLine)
				points = append(points, QdrantPoint{
					ID:     pointID,
					Vector: emb,
					Payload: map[string]interface{}{
						"path":       ch.Path,
						"heading":    ch.Heading,
						"start_line": ch.StartLine,
						"end_line":   ch.EndLine,
						"content":    ch.Content,
						"mtime":      mt,
					},
				})
				summary.Chunks++
			}
			if err := i.qdrant.Upsert(ctx, points); err != nil {
				return nil, err
			}
		}

		if _, ok := state.Files[file.RelPath]; ok && !reindexAll {
			summary.UpdatedFiles++
		} else {
			summary.IndexedFiles++
		}
		state.Files[file.RelPath] = mt
	}

	state.Collection = i.qdrant.Collection()
	state.EmbeddingModel = i.embedder.Model()
	state.ChunkSize = i.cfg.ChunkSize
	state.ChunkOverlap = i.cfg.ChunkOverlap
	state.IncludePatterns = append([]string{}, i.cfg.IncludePatterns...)
	state.ExcludePatterns = append([]string{}, i.cfg.ExcludePatterns...)

	if err := saveIndexState(statePath, state); err != nil {
		return nil, err
	}

	return summary, nil
}

type fileEntry struct {
	AbsPath string
	RelPath string
	MTime   int64
}

func listMarkdownFiles(root string, includePatterns, excludePatterns []string) ([]fileEntry, error) {
	root = filepath.Clean(root)
	includeRegex := compilePatterns(includePatterns)
	excludeRegex := compilePatterns(excludePatterns)

	var files []fileEntry
	err := filepath.WalkDir(root, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() {
			return nil
		}
		if filepath.Ext(path) != ".md" {
			return nil
		}
		rel, err := filepath.Rel(root, path)
		if err != nil {
			return err
		}
		rel = filepath.ToSlash(rel)
		if matchesAny(rel, excludeRegex) {
			return nil
		}
		if len(includeRegex) > 0 && !matchesAny(rel, includeRegex) {
			return nil
		}
		info, err := d.Info()
		if err != nil {
			return err
		}
		files = append(files, fileEntry{
			AbsPath: path,
			RelPath: rel,
			MTime:   info.ModTime().UnixNano(),
		})
		return nil
	})
	if err != nil {
		return nil, err
	}
	return files, nil
}

func compilePatterns(patterns []string) []*regexp.Regexp {
	var res []*regexp.Regexp
	for _, pat := range patterns {
		pat = strings.TrimSpace(pat)
		if pat == "" {
			continue
		}
		re, err := globToRegex(pat)
		if err != nil {
			continue
		}
		res = append(res, re)
	}
	return res
}

func globToRegex(pattern string) (*regexp.Regexp, error) {
	pattern = filepath.ToSlash(pattern)
	var sb strings.Builder
	sb.WriteString("^")
	for i := 0; i < len(pattern); i++ {
		ch := pattern[i]
		switch ch {
		case '*':
			if i+1 < len(pattern) && pattern[i+1] == '*' {
				sb.WriteString(".*")
				i++
			} else {
				sb.WriteString("[^/]*")
			}
		case '?':
			sb.WriteString(".")
		case '.':
			sb.WriteString("\\.")
		case '+', '(', ')', '[', ']', '{', '}', '|', '^', '$', '\\':
			sb.WriteString("\\")
			sb.WriteByte(ch)
		default:
			sb.WriteByte(ch)
		}
	}
	sb.WriteString("$")
	return regexp.Compile(sb.String())
}

func matchesAny(path string, patterns []*regexp.Regexp) bool {
	for _, re := range patterns {
		if re.MatchString(path) {
			return true
		}
	}
	return false
}

func hashPointID(path string, startLine, endLine int) string {
	sum := sha1.Sum([]byte(fmt.Sprintf("%s:%d:%d", path, startLine, endLine)))
	return hex.EncodeToString(sum[:])
}

func stringSliceEqual(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func expandHome(path string) string {
	if path == "" {
		return path
	}
	if path[0] == '~' {
		home, _ := os.UserHomeDir()
		if len(path) > 1 && path[1] == '/' {
			return filepath.Join(home, path[2:])
		}
		return home
	}
	return path
}
