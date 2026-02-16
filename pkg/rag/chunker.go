package rag

import (
	"path/filepath"
	"strings"
)

type chunk struct {
	Path      string
	Heading   string
	StartLine int
	EndLine   int
	Content   string
}

func chunkMarkdown(path string, content string, chunkSize int, chunkOverlap int) []chunk {
	if chunkSize <= 0 {
		chunkSize = 800
	}
	if chunkOverlap < 0 {
		chunkOverlap = 0
	}
	if chunkOverlap >= chunkSize {
		chunkOverlap = chunkSize / 2
	}

	lines := strings.Split(content, "\n")
	headings := headingsByLine(lines)

	var chunks []chunk
	i := 0
	for i < len(lines) {
		start := i
		charCount := 0
		for i < len(lines) {
			lineLen := len(lines[i]) + 1
			if charCount > 0 && charCount+lineLen > chunkSize {
				break
			}
			charCount += lineLen
			i++
		}
		end := i - 1
		if end < start {
			break
		}
		heading := headings[start]
		if heading == "" {
			heading = strings.TrimSuffix(filepath.Base(path), filepath.Ext(path))
		}
		text := strings.TrimSpace(strings.Join(lines[start:i], "\n"))
		if text != "" {
			chunks = append(chunks, chunk{
				Path:      path,
				Heading:   heading,
				StartLine: start + 1,
				EndLine:   end + 1,
				Content:   text,
			})
		}

		if i >= len(lines) {
			break
		}

		if chunkOverlap > 0 {
			overlapChars := 0
			j := i - 1
			for j >= start {
				overlapChars += len(lines[j]) + 1
				if overlapChars >= chunkOverlap {
					break
				}
				j--
			}
			if j < start {
				j = start
			}
			if j < i {
				i = j
			}
		}
	}

	return chunks
}

func headingsByLine(lines []string) []string {
	headings := make([]string, len(lines))
	stack := make([]string, 6)
	for i, line := range lines {
		trimmed := strings.TrimSpace(line)
		if strings.HasPrefix(trimmed, "#") {
			level := 0
			for level < len(trimmed) && trimmed[level] == '#' {
				level++
			}
			if level > 0 && level <= 6 {
				title := strings.TrimSpace(trimmed[level:])
				if title != "" {
					stack[level-1] = title
					for j := level; j < len(stack); j++ {
						stack[j] = ""
					}
				}
			}
		}
		headings[i] = joinHeading(stack)
	}
	return headings
}

func joinHeading(stack []string) string {
	var parts []string
	for _, h := range stack {
		if h != "" {
			parts = append(parts, h)
		}
	}
	return strings.Join(parts, " > ")
}
