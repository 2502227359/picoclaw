package rag

type SearchResult struct {
	Path      string
	Heading   string
	StartLine int
	EndLine   int
	Content   string
	Score     float64
}

type IndexSummary struct {
	TotalFiles   int
	IndexedFiles int
	UpdatedFiles int
	RemovedFiles int
	SkippedFiles int
	Chunks       int
}

type IndexOptions struct {
	ReindexAll bool
}
