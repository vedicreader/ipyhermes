---
description: karma — Repository memory and decision tracking for development context.
---

# karma

Provides persistent development context via SQLite + FTS5.

## Available Functions

- `dev_context(task, path)` — Get relevant context for a development task
- `search_code(query)` — Full-text search across indexed code
- `index_repo(path)` — Index a repository for search
- `index_env()` — Index the current environment
- `add_practice(text)` — Record a development practice
- `log_decision(why)` — Log a decision with rationale
- `query_practices()` — Query recorded practices
- `search_decisions(query)` — Search logged decisions

## Workflow

1. On session start: `index_repo('.')` + `index_env()` (auto-called)
2. Before any code: `dev_context('<task>', '.')`
3. After implementation: `log_decision('<why>')`
