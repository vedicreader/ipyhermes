---
name: ipyhermes-help
description: "Help and documentation for ipyhermes — the IPython AI extension powered by hermes-agent."
---

# ipyhermes Help

## Quick Reference

| Command | Effect |
|---|---|
| `.your question` | Send prompt to AI (dot-prompt mode) |
| `%ipyhermes` | Show current settings |
| `%ipyhermes model <name>` | Switch execution model |
| `%ipyhermes plan_model <name>` | Switch planning model |
| `%ipyhermes provider <name>` | Switch provider |
| `%ipyhermes think l\|m\|h` | Set thinking level |
| `%ipyhermes search l\|m\|h` | Set search level |
| `%ipyhermes code_theme <name>` | Syntax theme |
| `%ipyhermes reset` | Clear AI history in session |
| `%ipyhermes save <path>` | Save session notebook |
| `%ipyhermes load <path>` | Load session notebook |
| `%ipyhermes sessions` | List resumable sessions |
| `%ipyhermes prompt` | Toggle prompt mode |
| `%ipyhermes caveman` | Toggle caveman mode (~75% fewer tokens) |
| `%ipyhermes memory on\|off` | Toggle karma ConversationLog |
| `%ipyhermes route` | Show provider quota status (bhoga) |
| `%ipyhermes route auto` | Auto-select best provider (bhoga) |
| `%ipyhermes mcp` | List connected MCP servers |
| `%ipyhermes mcp connect <url>` | Connect to an MCP server |
| `%ipyhermes mcp disconnect <url>` | Disconnect from MCP server |

## Keybindings

| Binding | Action |
|---|---|
| `Alt-.` | AI inline completion |
| `Alt-W` | Paste all code blocks from last response |
| `Alt-1..9` | Paste nth code block |
| `Alt-Shift-↑/↓` | Cycle through code blocks |
| `Alt-↑/↓` | Jump through history entries |
| `Alt-P` | Toggle prompt mode |

## Backtick References

- `&`func`` — expose a callable as a tool
- `$`var`` — inject a variable's value
- `!`cmd`` — run shell command and inject output
