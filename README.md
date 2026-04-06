# ipyhermes

`ipyhermes` is an IPython extension that turns any input starting with `.` into an AI prompt, powered by [hermes-agent](https://github.com/NousResearch/hermes-agent) with added tooling — [karma](https://github.com/vedicreader/karma) (repo memory), [webba](https://github.com/vedicreader/webba) (web search), and [shortcutpy](https://github.com/AnswerDotAI/shortcutpy) (macOS Shortcuts).

It is aimed at terminal IPython, not notebook frontends.

## Install

```bash
pip install ipyhermes
```

## CLI

`ipyhermes` provides a standalone command that launches IPython with `ipyhermes`, `ipythonng`, `safepyrun`, and `pshnb` extensions pre-loaded:

```bash
ipyhermes
```

Resume a previous session:

```bash
ipyhermes -r        # interactive session picker
ipyhermes -r 43     # resume session 43 directly
```

On exit, `ipyhermes` prints the session ID so you can resume later.

## Load As Extension

```python
%load_ext ipyhermes
```

If you change the package in a running shell:

```python
%reload_ext ipyhermes
```

## How To Auto-Load `ipyhermes`

Add this to an `ipython_config.py` file used by terminal `ipython`:

```python
c.TerminalIPythonApp.extensions = ["ipyhermes"]
```

Good places for that file include:

- env-local: `{sys.prefix}/etc/ipython/ipython_config.py`
- user-local: `~/.ipython/profile_default/ipython_config.py`

## Usage

Only the leading period is special. There is no closing delimiter.

Single line:

```python
.write a haiku about sqlite
```

Multiline paste:

```python
.summarize this module:
focus on state management
and persistence behavior
```

Backslash-Enter continuation in the terminal:

```python
.draft a migration plan \
with risks and rollback steps
```

`ipyhermes` also provides a line and cell magic named `%ipyhermes` / `%%ipyhermes`.

Plan mode — routes the prompt to a dedicated planning model:

```python
.plan refactor the auth module
```

Note: `.01 * 3` and similar expressions starting with `.` followed by a digit will be interpreted as prompts. Write `0.01 * 3` instead.

## Notes

Any IPython cell containing only a string literal is treated as a "note". Notes provide context to the AI without being executable code:

```python
"This is a note explaining what I'm about to do"
```

Notes appear in the AI context as `<note>` blocks rather than `<code>` blocks. When saving a session, notes are stored as markdown cells in the notebook.

## `%ipyhermes` Commands

```python
%ipyhermes
%ipyhermes model gpt-5.4
%ipyhermes plan_model claude-opus-4-6
%ipyhermes complete_model gpt-5.4-mini
%ipyhermes provider openai-codex
%ipyhermes think m
%ipyhermes search h
%ipyhermes code_theme monokai
%ipyhermes log_exact true
%ipyhermes save mysession
%ipyhermes load mysession
%ipyhermes reset
%ipyhermes sessions
%ipyhermes help
%ipyhermes prompt
%ipyhermes caveman
%ipyhermes memory on
%ipyhermes route
%ipyhermes route auto
```

- `%ipyhermes` — show current settings and config file paths
- `%ipyhermes model ...` / `plan_model ...` / `complete_model ...` / `provider ...` — change model or provider for the current session (hot-swaps the agent)
- `%ipyhermes think ...` / `search ...` / `code_theme ...` / `log_exact ...` — change settings
- `%ipyhermes save <file>` — save the current session to a `.ipynb` notebook
- `%ipyhermes load <file>` — restore a session from a notebook
- `%ipyhermes reset` — clear AI prompt history for the current session
- `%ipyhermes sessions` — list resumable sessions for the current directory
- `%ipyhermes prompt` — toggle prompt mode (all input → AI, `;` escapes to Python)
- `%ipyhermes caveman` — toggle caveman mode (~75% fewer response tokens)
- `%ipyhermes memory on|off` — toggle karma ConversationLog persistence (on by default when karma is installed)
- `%ipyhermes route` — show provider quota status (requires bhoga)
- `%ipyhermes route auto` — auto-select best provider based on quota (requires bhoga)
- `%ipyhermes route <provider>` — force a specific provider
- `%ipyhermes help` — show command reference

## Tools

Expose a function from the active IPython namespace as a tool by referencing it with `` &`name` `` in the prompt:

```python
def weather(city): return f"Sunny in {city}"

.use &`weather` to answer the question about Brisbane
```

Callable objects and async callables are also supported.

Tools are discovered from multiple sources:

- **Backtick refs**: `` &`name` `` in prompts
- **Skills**: tools listed in `allowed-tools` frontmatter
- **Notes**: string-literal cells can contain `` &`name` `` references or YAML frontmatter with `allowed-tools`
- **Auto-injected**: `pyrun`, `bash`, `ex`, `sed` are always added if they exist in the namespace

### Variable References

Expose a variable's value with `` $`name` ``:

```python
data = [1, 2, 3]
.analyze $`data` for outliers
```

The variable's current value is serialized as XML and included in the prompt.

### Shell References

Include shell command output with `` !`cmd` ``:

```python
.explain the output of !`git status`
```

The command runs via subprocess and its output is injected into the prompt.

## HITL Approval Gate

Dangerous tool calls (`terminal`, `execute_code`, `patch`, `write_file`, `create_file`) trigger a Rich confirmation prompt before execution:

```
⚡ terminal
docker run ...
Run? [Y/n]
```

## Skills

`ipyhermes` supports [Agent Skills](https://agentskills.io/) — reusable instruction sets that the AI can load on demand. Skills are discovered at extension load time from:

- `.agents/skills/` in the current directory and every parent directory
- `~/.config/agents/skills/`
- Bundled skills in the `ipyhermes/_skills/` package directory

Each skill is a directory containing a `SKILL.md` file with YAML frontmatter (`name`, `description`) and markdown instructions.

At the start of each conversation, the AI sees a list of available skill names and descriptions. When a request matches a skill, the AI calls the `load_skill` tool to read its full instructions before responding.

Python code blocks in skills that start with `#| eval: true` are executed in the IPython namespace when the skill is loaded:

````markdown
```python
#| eval: true
def my_tool(x):
    "A skill-provided tool"
    return x * 2
```
````

### Bundled Skills

| Skill | Description |
|---|---|
| `ex` | `ex` editor commands for surgical file editing |
| `exhash` | ex command hashing for edit tracking |
| `ipyhermes-help` | Help and docs for ipyhermes |
| `ipython-unicode` | Unicode symbol insertion via LaTeX names |
| `sed` | Stream editing tool |

### Hermes Skills

Bundled hermes-format skills are installed to `~/.hermes/skills/` on first import:

| Skill | Description |
|---|---|
| `karma` | Repository memory and decision tracking |
| `webba` | Web search and content fetching |
| `shortcutpy` | Compile Python DSL into Apple Shortcuts |
| `ipython-context` | Automatic IPython session context injection |

## Keyboard Shortcuts

`ipyhermes` registers prompt_toolkit keybindings:

| Shortcut | Action |
|---|---|
| **Alt-.** | AI inline completion (calls fast model, shows as greyed suggestion — accept with right arrow, or **Alt-f** to accept one word at a time) |
| **Alt-Up/Down** | Jump through complete history entries (skips line-by-line in multiline inputs) |
| **Alt-Shift-W** | Insert all Python code blocks from the last AI response |
| **Alt-Shift-1** through **Alt-Shift-9** | Insert the Nth code block |
| **Alt-Shift-Up/Down** | Cycle through code blocks one at a time |
| **Alt-P** | Toggle prompt mode |

Code blocks are extracted from fenced markdown blocks tagged as `python` or `py`. Blocks tagged with other languages (bash, json, etc.) or untagged blocks are skipped.

Syntax highlighting is disabled while typing `.` prompts and `%%ipyhermes` cells so natural language isn't coloured as Python.

## Output Rendering

Responses are streamed and rendered as markdown in the terminal via Rich. Thinking indicators (🧠) are displayed during model reasoning and removed once the response begins. Tool calls are compacted to a short `🔧` summary form.

## Three-Agent Architecture

`ipyhermes` runs three hermes-agent instances:

| Agent | Model | Toolsets | Purpose |
|---|---|---|---|
| `_exec` | `IPYHERMES_MODEL` (default: gpt-5.4) | terminal, web, execute_code, browser | Main execution agent |
| `_plan` | `IPYHERMES_PLAN` (default: claude-opus-4-6) | web | Planning agent (`.plan` prefix) |
| `_fast` | `IPYHERMES_COMPLETE` (default: gpt-5.4-mini) | (none) | Alt-. inline completions |

## Namespace Injection

On load, `ipyhermes` injects these into the IPython namespace (when available):

- **karma**: `dev_context`, `search_code`, `index_repo`, `index_env`, `add_practice`, `log_decision`, `query_practices`, `search_decisions`
- **webba**: `web_search`, `web_fetch`
- **shortcutpy**: `shortcut`, `ask_for_text`, `choose_from_menu`, `show_result`, `compile_file`, `compile_source`
- **bgterm**: `start_bgterm`, `write_stdin`, `close_bgterm` — persistent background shell sessions for multi-step workflows
- **exhash**: `lnhashview_file`, `exhash_file` — hash-addressed file editing immune to line number drift
- **bhoga**: `bhoga_router`, `apply_to_hermes` — quota-aware provider routing
- **safecmd**: `bash`, `ex`, `sed`
- **pyskills**: `doc`

## Karma ConversationLog (Always-On)

When [karma](https://github.com/vedicreader/karma) is installed, `ipyhermes` automatically creates a `ConversationLog` at extension load time. All AI prompts and responses are persisted via karma's session system. Toggle with `%ipyhermes memory on|off`.

## Configuration

Config files live under `~/.config/ipyhermes/` and are created on demand:

| File | Purpose |
|---|---|
| `config.json` | Model, provider, think/search level, code theme, log flag |
| `sysp.txt` | System prompt |
| `exact-log.jsonl` | Raw prompt/response log (when `log_exact` is enabled) |

`config.json` supports:

```json
{
  "model": "gpt-5.4",
  "plan_model": "claude-opus-4-6",
  "complete_model": "gpt-5.4-mini",
  "provider": "openai-codex",
  "think": "l",
  "search": "l",
  "code_theme": "monokai",
  "log_exact": false,
  "prompt_mode": false
}
```

Environment variables override defaults when the config is first created:

| Variable | Default |
|---|---|
| `IPYHERMES_MODEL` | `gpt-5.4` |
| `IPYHERMES_PLAN` | `claude-opus-4-6` |
| `IPYHERMES_COMPLETE` | `gpt-5.4-mini` |
| `IPYHERMES_PROVIDER` | `openai-codex` |
| `IPYHERMES_PLAN_PROV` | `copilot` |

## Hermes Agent Configuration

On first import, `ipyhermes` writes `~/.hermes/cli-config.yaml` with sensible defaults (Docker terminal backend, smart model routing, compression). It never overwrites an existing config.

## Startup Replay

`%ipyhermes save` snapshots the current session to a `.ipynb` notebook:

- code cells are saved as code cells (notes become markdown cells)
- AI prompts are saved with the response as markdown and the prompt in cell metadata

When loaded via `%ipyhermes load`, saved code is replayed and saved prompts are restored into the conversation history.

## Development

Run the test suite:

```bash
pip install -e ".[dev]"
pytest tests/
```

The tests mock `run_agent.AIAgent` and run entirely offline — no API keys needed.
