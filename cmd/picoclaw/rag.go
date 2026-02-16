package main

import (
	"context"
	"fmt"
	"os"
	"time"

	"github.com/sipeed/picoclaw/pkg/rag"
)

func ragCmd() {
	if len(os.Args) < 3 || os.Args[2] == "--help" || os.Args[2] == "-h" {
		ragHelp()
		return
	}

	subcommand := os.Args[2]
	switch subcommand {
	case "index":
		ragIndexCmd(os.Args[3:])
	default:
		fmt.Printf("Unknown rag command: %s\n", subcommand)
		ragHelp()
	}
}

func ragHelp() {
	fmt.Println("\nRAG commands:")
	fmt.Println("  index        Build or update the knowledge base index")
	fmt.Println()
	fmt.Println("Options:")
	fmt.Println("  --full       Rebuild all vectors from scratch")
	fmt.Println()
	fmt.Println("Examples:")
	fmt.Println("  picoclaw rag index")
	fmt.Println("  picoclaw rag index --full")
}

func ragIndexCmd(args []string) {
	reindexAll := false
	for _, arg := range args {
		if arg == "--full" {
			reindexAll = true
		}
	}

	cfg, err := loadConfig()
	if err != nil {
		fmt.Printf("Error loading config: %v\n", err)
		return
	}

	if !cfg.RAG.Enabled {
		fmt.Println("RAG is disabled in config.")
		return
	}

	service, err := rag.NewService(cfg, cfg.WorkspacePath())
	if err != nil {
		fmt.Printf("RAG initialization failed: %v\n", err)
		return
	}

	fmt.Println("Indexing knowledge base...")
	start := time.Now()

	summary, err := service.Index(context.Background(), rag.IndexOptions{ReindexAll: reindexAll})
	if err != nil {
		fmt.Printf("Index failed: %v\n", err)
		return
	}

	fmt.Printf("✓ Done in %s\n", time.Since(start).Truncate(time.Second))
	fmt.Printf("  Files: %d total, %d new, %d updated, %d removed, %d skipped\n",
		summary.TotalFiles, summary.IndexedFiles, summary.UpdatedFiles, summary.RemovedFiles, summary.SkippedFiles)
	fmt.Printf("  Chunks: %d\n", summary.Chunks)
}
