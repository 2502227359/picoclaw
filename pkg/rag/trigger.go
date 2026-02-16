package rag

import (
	"strings"

	"github.com/sipeed/picoclaw/pkg/config"
)

type TriggerDecision struct {
	CleanedMessage string
	ShouldSearch   bool
	Forced         bool
	Skipped        bool
	MatchedKeyword string
}

func DecideTrigger(message string, cfg config.RagTriggerConfig) TriggerDecision {
	trimmed := strings.TrimSpace(message)
	if trimmed == "" {
		return TriggerDecision{CleanedMessage: message}
	}

	if prefix, ok := matchPrefix(trimmed, cfg.ForcePrefixes); ok {
		clean := strings.TrimSpace(strings.TrimPrefix(trimmed, prefix))
		return TriggerDecision{
			CleanedMessage: clean,
			ShouldSearch:   true,
			Forced:         true,
		}
	}
	if prefix, ok := matchPrefix(trimmed, cfg.SkipPrefixes); ok {
		clean := strings.TrimSpace(strings.TrimPrefix(trimmed, prefix))
		return TriggerDecision{
			CleanedMessage: clean,
			ShouldSearch:   false,
			Skipped:        true,
		}
	}

	clean := trimmed
	if !cfg.Auto {
		return TriggerDecision{CleanedMessage: clean}
	}

	keyword := matchKeyword(clean, cfg.AutoKeywords)
	if keyword != "" {
		return TriggerDecision{
			CleanedMessage: clean,
			ShouldSearch:   true,
			MatchedKeyword: keyword,
		}
	}

	return TriggerDecision{CleanedMessage: clean}
}

func matchPrefix(message string, prefixes []string) (string, bool) {
	for _, prefix := range prefixes {
		if prefix == "" {
			continue
		}
		if strings.HasPrefix(message, prefix) {
			return prefix, true
		}
	}
	return "", false
}

func matchKeyword(message string, keywords []string) string {
	if len(keywords) == 0 {
		return ""
	}
	lower := strings.ToLower(message)
	for _, kw := range keywords {
		if kw == "" {
			continue
		}
		if strings.Contains(lower, strings.ToLower(kw)) {
			return kw
		}
	}
	return ""
}
