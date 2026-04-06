import asyncio,io,json,os,sqlite3,sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from IPython.core.inputtransformer2 import TransformerManager

import ipyhermes.core as core
from ipyhermes.core import (EXTENSION_NS, LAST_PROMPT, LAST_RESPONSE, RESET_LINE_NS,
    DEFAULT_CODE_THEME, DEFAULT_LOG_EXACT, DEFAULT_SEARCH, DEFAULT_SYSTEM_PROMPT, DEFAULT_THINK,
    DEFAULT_MODEL, DEFAULT_PLAN_MODEL, DEFAULT_COMPLETE_MODEL, DEFAULT_PROVIDER, DEFAULT_CAVEMAN,
    HermesExtension, astream_to_stdout, compact_tool_display, prompt_from_lines, transform_dots,
    _parse_skill, _allowed_tools, _tool_results, _tool_refs,
    _var_names, _var_refs, _format_var_xml,
    _shell_names, _shell_refs, _run_shell_refs,
    transform_prompt_mode,
    _discover_skills, _skills_xml, _build_sysp, _strip_thinking, _extract_code_blocks, _eval_code_blocks, load_skill,
    _git_repo_root, _list_sessions, resume_session, _hermes_session_id, _mk_convlog)


# ── Test doubles ─────────────────────────────────────────────────────────────

class TTYStringIO(io.StringIO):
    def isatty(self): return True

class DummyMarkdown:
    def __init__(self, text, **kwargs): self.text,self.kwargs = text,kwargs

class DummyConsole:
    instances = []
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.printed = []
        type(self).instances.append(self)

    def print(self, obj):
        self.printed.append(obj)
        self.kwargs["file"].write(f"RICH:{obj.text}")

class DummyLive:
    instances = []
    def __init__(self, renderable, **kwargs):
        self.kwargs = kwargs
        self.renderables = [renderable]
        type(self).instances.append(self)

    def __enter__(self): return self
    def __exit__(self, exc_type, exc, tb): self.kwargs["console"].print(self.renderables[-1])
    def update(self, renderable, refresh=False): self.renderables.append(renderable)

class DummyHistory:
    def __init__(self, session_number=1):
        self.session_number = session_number
        self.db = sqlite3.connect(":memory:")
        self.entries = {}
        self.input_hist_parsed = [""]
        self.input_hist_raw = [""]

    def add(self, line, source, output=None): self.entries[line] = (source, output)
    def get_range(self, session=0, start=1, stop=None, raw=True, output=False):
        if stop is None: stop = max(self.entries, default=0) + 1
        for i in range(start, stop):
            if i not in self.entries: continue
            src,out = self.entries[i]
            yield (0, i, (src, out) if output else src)

class DummyDisplayPublisher:
    def __init__(self): self._is_publishing = False

class DummyInputTransformerManager:
    def __init__(self): self.cleanup_transforms = []

class DummyShell:
    def __init__(self):
        self.input_transformer_manager = DummyInputTransformerManager()
        self.user_ns = {}
        self.magics = []
        self.history_manager = DummyHistory()
        self.display_pub = DummyDisplayPublisher()
        self.execution_count = 2
        self.ran_cells = []
        self.loop_runner = asyncio.run
        self.prompts = None

    def register_magics(self, magics): self.magics.append(magics)
    def set_custom_exc(self, *args): pass

    def run_cell(self, source, store_history=False):
        self.ran_cells.append((source, store_history))
        if store_history:
            self.history_manager.add(self.execution_count, source)
            self.execution_count += 1
        try: exec(compile(source, f'<cell-{self.execution_count}>', 'exec'), self.user_ns)
        except Exception: pass
        return SimpleNamespace(success=True)

    async def run_cell_async(self, source, store_history=False, transformed_cell=None):
        return self.run_cell(transformed_cell or source, store_history=store_history)


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _config_paths(monkeypatch, tmp_path):
    cfg_dir = tmp_path/"ipyhermes"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(core, "CONFIG_DIR", cfg_dir)
    monkeypatch.setattr(core, "CONFIG_PATH", cfg_dir/"config.json")
    monkeypatch.setattr(core, "SYSP_PATH", cfg_dir/"sysp.txt")
    monkeypatch.setattr(core, "LOG_PATH", cfg_dir/"exact-log.jsonl")
    monkeypatch.setattr(core, "_discover_skills", lambda cwd=None: [])
    # Always stub _mk_agent so HermesExtension can be created without hermes-agent
    class StubAgent:
        def __init__(self, **kw): self.kw = kw
        async def astream(self, prompt, sp=None, hist=None):
            yield ""
    monkeypatch.setattr(core, "_mk_agent", lambda *a, **kw: StubAgent(**kw))
    # Stub karma so tests don't require it
    monkeypatch.setattr(core, "_mk_convlog", lambda sid: None)


@pytest.fixture
def dummy_agent(monkeypatch):
    """Patch _mk_agent and astream_to_stdout so run_prompt can complete without real LLM calls."""
    calls = []
    class FakeAgent:
        def __init__(self, **kw): self.kw = kw
        async def astream(self, prompt, sp=None, hist=None):
            calls.append(dict(prompt=prompt, sp=sp, hist=hist))
            for chunk in ["first ", "second"]: yield chunk
    monkeypatch.setattr(core, "_mk_agent", lambda *a, **kw: FakeAgent(**kw))

    async def _fake_astream(chunks, **kwargs):
        return "".join([c async for c in chunks])
    monkeypatch.setattr(core, "astream_to_stdout", _fake_astream)
    return calls


# ── Helpers ──────────────────────────────────────────────────────────────────

async def _chunks(*items):
    for o in items: yield o


def run_stream(*items, **kwargs):
    return asyncio.run(astream_to_stdout(_chunks(*items), **kwargs))


def _strip_ids(nb):
    return {**nb, "cells": [{k:v for k,v in c.items() if k != "id"} for c in nb.get("cells", [])]}


def mk_ext(load=True, **kwargs):
    shell = DummyShell()
    ext = HermesExtension(shell=shell, **kwargs)
    return shell, ext.load() if load else ext


# ── Tests: Streaming ─────────────────────────────────────────────────────────

def test_astream_to_stdout_collects_streamed_text():
    out = io.StringIO()
    text = run_stream("a", "b", out=out)
    assert text == "ab"
    assert out.getvalue() == "ab\n"


def test_astream_to_stdout_uses_live_markdown_for_tty_and_returns_full_text():
    DummyConsole.instances = []
    DummyLive.instances = []
    out = TTYStringIO()
    text = run_stream("hello", " world", out=out, console_cls=DummyConsole, markdown_cls=DummyMarkdown, live_cls=DummyLive)
    assert text == "hello world"


def test_astream_to_stdout_uses_rich_markdown_options_for_live_updates():
    DummyConsole.instances = []
    DummyLive.instances = []
    out = TTYStringIO()
    text = run_stream("`x`", out=out, code_theme="github-dark", console_cls=DummyConsole, markdown_cls=DummyMarkdown, live_cls=DummyLive)
    assert text == "`x`"
    md = DummyLive.instances[-1].renderables[-1]
    assert md.text == "`x`"
    assert md.kwargs == dict(code_theme="github-dark", inline_code_theme="github-dark", inline_code_lexer="python")


def test_astream_to_stdout_updates_live_markdown_as_chunks_arrive():
    DummyConsole.instances = []
    DummyLive.instances = []
    out = TTYStringIO()
    text = run_stream("a", "b", out=out, console_cls=DummyConsole, markdown_cls=DummyMarkdown, live_cls=DummyLive)
    assert text == "ab"
    assert [o.text for o in DummyLive.instances[-1].renderables] == ["a", "ab"]
    assert out.getvalue() == "RICH:ab"


# ── Tests: Transforms ────────────────────────────────────────────────────────

def test_prompt_from_lines_drops_continuation_backslashes():
    lines = [".plan this work\\\n", "with two lines\n"]
    assert prompt_from_lines(lines) == "plan this work\nwith two lines\n"


def test_transform_dots_executes_ai_magic_call():
    seen = {}
    class DummyIPython:
        def run_cell_magic(self, magic, line, cell): seen.update(magic=magic, line=line, cell=cell)
    code = "".join(transform_dots([".hello\n", "world\n"]))
    exec(code, {"get_ipython": lambda: DummyIPython()})
    assert seen == dict(magic="ipyhermes", line="", cell="hello\nworld\n")


def test_cleanup_transform_prevents_help_syntax_interference():
    tm = TransformerManager()
    tm.cleanup_transforms.insert(1, transform_dots)
    code = tm.transform_cell(".I am testing my new AI prompt system.\\\nTell me do you see a newline in this prompt?")
    assert code == "get_ipython().run_cell_magic('ipyhermes', '', 'I am testing my new AI prompt system.\\nTell me do you see a newline in this prompt?\\n')\n"
    assert tm.check_complete(".I am testing my new AI prompt system.\\") == ("incomplete", 0)
    assert tm.check_complete(".I am testing my new AI prompt system.\\\nTell me do you see a newline in this prompt?") == ("complete", None)


# ── Tests: Display Processing ────────────────────────────────────────────────

def test_compact_tool_display():
    text = "Before <tool_call>{\"id\":1}</tool_call> After"
    res = compact_tool_display(text)
    assert "🔧" in res
    assert "<tool_call>" not in res
    assert "Before" in res and "After" in res


def test_strip_thinking_shows_brains_while_thinking():
    assert _strip_thinking("🧠🧠🧠") == "🧠🧠🧠"

def test_strip_thinking_removes_brains_once_content_arrives():
    assert _strip_thinking("🧠🧠🧠\n\nHello world") == "Hello world"

def test_strip_thinking_handles_no_brains():
    assert _strip_thinking("Hello world") == "Hello world"


def test_live_stream_strips_thinking_from_display():
    DummyConsole.instances = []
    DummyLive.instances = []
    out = TTYStringIO()
    text = run_stream("🧠🧠🧠", "\n\n", "Hello", out=out, console_cls=DummyConsole, markdown_cls=DummyMarkdown, live_cls=DummyLive)
    assert text == "🧠🧠🧠\n\nHello"
    rendered = [o.text for o in DummyLive.instances[-1].renderables]
    assert rendered[0] == "🧠🧠🧠"
    assert rendered[-1] == "Hello"


# ── Tests: Code Block Extraction ─────────────────────────────────────────────

def test_extract_code_blocks_python_only():
    text = "Here's some code:\n```python\nx = 1\ny = 2\n```\nAnd more:\n```\nz = 3\n```\nBash:\n```bash\necho hi\n```\nPy:\n```py\na = 4\n```"
    assert _extract_code_blocks(text) == ["x = 1\ny = 2", "a = 4"]


def test_extract_code_blocks_empty_response():
    assert _extract_code_blocks("") == []
    assert _extract_code_blocks("no code here") == []


# ── Tests: Extension Load / Config ───────────────────────────────────────────

async def test_extension_load_is_idempotent_and_tracks_last_response(dummy_agent):
    shell,ext = mk_ext()
    ext.load()
    assert shell.input_transformer_manager.cleanup_transforms == [transform_dots]
    assert len(shell.magics) == 1
    assert shell.user_ns[EXTENSION_NS] is ext

    await ext.run_prompt("tell me something")

    assert shell.user_ns[LAST_PROMPT] == "tell me something"
    assert shell.user_ns[LAST_RESPONSE] == "first second"
    assert ext.prompt_rows() == [("tell me something", "first second")]
    assert ext.prompt_records()[0]['history_line'] == 1


async def test_run_prompt_suppresses_ipython_output_history_while_streaming(monkeypatch, dummy_agent):
    shell,ext = mk_ext(load=False)
    seen = []

    async def _fake_astream(chunks, **kwargs):
        seen.append(shell.display_pub._is_publishing)
        return "".join([c async for c in chunks])

    monkeypatch.setattr(core, "astream_to_stdout", _fake_astream)

    await ext.run_prompt("tell me something")

    assert seen == [True]
    assert shell.display_pub._is_publishing is False


async def test_run_prompt_stores_cleaned_response_for_output_history(monkeypatch, dummy_agent):
    shell,ext = mk_ext(load=False)
    ng = SimpleNamespace(_pty_output=None)
    shell._ipythonng_extension = ng

    async def _fake_astream(chunks, **kwargs): return "🧠🧠🧠\n\nHello world"
    monkeypatch.setattr(core, "astream_to_stdout", _fake_astream)

    await ext.run_prompt("test")

    assert ng._pty_output == "Hello world"


def test_config_file_is_created_and_loaded():
    _,ext = mk_ext(load=False)

    assert core.CONFIG_PATH.exists()
    assert core.SYSP_PATH.exists()
    data = json.loads(core.CONFIG_PATH.read_text())
    assert data["model"] == ext.model
    assert data["think"] == DEFAULT_THINK
    assert data["search"] == DEFAULT_SEARCH
    assert data["code_theme"] == DEFAULT_CODE_THEME
    assert data["log_exact"] == DEFAULT_LOG_EXACT
    assert core.SYSP_PATH.read_text() == DEFAULT_SYSTEM_PROMPT
    assert ext.system_prompt == DEFAULT_SYSTEM_PROMPT


def test_existing_sysp_file_is_loaded():
    sysp_path = core.SYSP_PATH
    sysp_path.write_text("custom sysp")
    _,ext = mk_ext(load=False)
    assert ext.system_prompt == "custom sysp"


async def test_config_values_drive_model_think_and_search(dummy_agent):
    core.CONFIG_PATH.write_text(json.dumps(dict(model="cfg-model", think="m", search="h", log_exact=True)))
    shell,ext = mk_ext()

    await ext.run_prompt("tell me something")

    assert ext.model == "cfg-model"
    assert ext.think == "m"
    assert ext.search == "h"
    assert ext.log_exact is True


def test_handle_line_can_report_and_set_model(capsys):
    _,ext = mk_ext(load=False, model="old-model", think="m", search="h", code_theme="github-dark", log_exact=True)

    ext.handle_line("")
    out = capsys.readouterr().out
    assert "self.model='old-model'" in out
    assert f"CONFIG_PATH=" in out

    ext.handle_line("model new-model")
    assert ext.model == "new-model"
    assert capsys.readouterr().out == "self.model='new-model'\n"

    ext.handle_line("think l")
    assert ext.think == "l"
    assert capsys.readouterr().out == "self.think='l'\n"

    ext.handle_line("search m")
    assert ext.search == "m"
    assert capsys.readouterr().out == "self.search='m'\n"

    ext.handle_line("code_theme ansi_dark")
    assert ext.code_theme == "ansi_dark"
    assert capsys.readouterr().out == "self.code_theme='ansi_dark'\n"

    ext.handle_line("log_exact false")
    assert ext.log_exact is False
    assert capsys.readouterr().out == "self.log_exact=False\n"


# ── Tests: Dialog History ────────────────────────────────────────────────────

async def test_second_prompt_uses_in_memory_prompt_history(dummy_agent):
    shell,ext = mk_ext()

    await ext.run_prompt("first prompt")
    shell.execution_count = 3
    await ext.run_prompt("second prompt")

    assert ext.prompt_rows() == [
        ("first prompt", "first second"),
        ("second prompt", "first second")]


async def test_interrupted_prompt_with_no_output_has_valid_history(dummy_agent):
    """When a prompt is interrupted before any output, the saved response should
    still produce a valid assistant message in the history for the next prompt."""
    shell,ext = mk_ext()
    ext.save_prompt("interrupted prompt", "", 1)
    shell.execution_count = 3
    await ext.run_prompt("follow up")

    hist, recs = ext.dialog_history()
    assert len(recs) == 2
    # Empty response gets substituted in dialog_history
    assert recs[0]["response"] == "<system>user interrupted</system>"
    # The history list should have non-empty assistant message
    assert hist[1] != "", "Empty assistant response in history causes prefill errors"


async def test_second_prompt_replays_prior_context_in_chat_history(dummy_agent):
    shell,ext = mk_ext()
    shell.history_manager.add(1, "print('a')", "a")
    shell.history_manager.add(2, "print(1)", "1")
    shell.history_manager.add(3, "1+1", "2")
    shell.execution_count = 5

    await ext.run_prompt("What code history?")

    shell.history_manager.add(5, "from IPython.display import HTML")
    shell.execution_count = 7

    await ext.run_prompt("Do you see prints?")

    # The first prompt should have included context from code history
    first_call = dummy_agent[0]
    assert "<code>print('a')</code>" in first_call["prompt"]


def test_reset_clears_in_memory_prompts(capsys):
    shell,ext = mk_ext()

    ext.save_prompt("p1", "r1", 1)
    ext.save_prompt("p2", "r2", 7)

    ext.handle_line("reset")

    assert capsys.readouterr().out == "Deleted 2 AI prompts from session 1.\n"
    assert ext.prompt_rows() == []
    assert shell.user_ns[RESET_LINE_NS] == 1  # current_prompt_line() = execution_count - 1 = 2 - 1 = 1


# ── Tests: Context XML ───────────────────────────────────────────────────────

def test_context_xml_includes_code_and_outputs_since_last_prompt():
    shell = DummyShell()
    shell.history_manager.add(1, "a = 1")
    shell.history_manager.add(2, "a", "1")
    ext = HermesExtension(shell=shell).load()

    ctx = ext.code_context(1, 3)
    assert "<context><code>a = 1</code><code>a</code><output>1</output></context>\n" == ctx


def test_code_context_uses_note_tag_for_string_literals():
    shell = DummyShell()
    shell.history_manager.add(1, '"This is a note"')
    shell.history_manager.add(2, 'x = 1')
    shell.history_manager.add(3, '"""multi\nline"""')
    ext = HermesExtension(shell=shell).load()

    ctx = ext.code_context(1, 4)
    assert ctx == '<context><note>This is a note</note><code>x = 1</code><note>multi\nline</note></context>\n'


def test_history_context_uses_lines_since_last_prompt_only():
    shell = DummyShell()
    shell.history_manager.add(1, "before = 1")
    shell.history_manager.add(2, ".first prompt")
    shell.history_manager.add(3, "after = 2")
    shell.execution_count = 3
    ext = HermesExtension(shell=shell).load()
    ext.save_prompt("first prompt", "first response", 2)

    prompt = ext.format_prompt("second prompt", ext.last_prompt_line()+1, 4)
    assert "before = 1" not in prompt
    assert "after = 2" in prompt


# ── Tests: Notebook Save/Load ────────────────────────────────────────────────

def test_save_notebook_converts_notes_to_markdown_cells(tmp_path):
    shell = DummyShell()
    shell.history_manager.add(1, '"# My note"')
    shell.history_manager.add(2, 'x = 1')
    shell.execution_count = 3
    ext = HermesExtension(shell=shell).load()

    path, _, _ = ext.save_notebook(tmp_path / "test")
    assert path.suffix == ".ipynb"
    nb = json.loads(path.read_text())
    c0 = {k:v for k,v in nb["cells"][0].items() if k != "id"}
    assert c0 == dict(cell_type="markdown", source="# My note",
        metadata=dict(ipyhermes=dict(kind="code", line=1, source='"# My note"')))
    assert nb["cells"][1]["cell_type"] == "code"
    assert nb["cells"][1]["source"] == "x = 1"


def test_notebook_roundtrip_preserves_notes(tmp_path):
    shell = DummyShell()
    shell.history_manager.add(1, '"a note"')
    shell.history_manager.add(2, 'x = 1')
    shell.execution_count = 3
    ext = HermesExtension(shell=shell).load()
    ext.save_notebook(tmp_path / "test")

    shell2 = DummyShell()
    shell2.execution_count = 1
    ext2 = HermesExtension(shell=shell2).load()
    ext2.load_notebook(tmp_path / "test")
    assert shell2.ran_cells == [('"a note"', True), ('x = 1', True)]


def test_load_notebook_replays_code_and_restores_prompts(tmp_path):
    cells = [
        dict(cell_type="code", source="import math", metadata=dict(ipyhermes=dict(kind="code", line=1)), outputs=[], execution_count=None),
        dict(cell_type="markdown", source="hello", metadata=dict(ipyhermes=dict(kind="prompt", line=3, history_line=2, prompt="hi"))),
        dict(cell_type="code", source="x = 1", metadata=dict(ipyhermes=dict(kind="code", line=3)), outputs=[], execution_count=None),
    ]
    nb_path = tmp_path / "test.ipynb"
    nb_path.write_text(json.dumps(dict(cells=cells, metadata=dict(ipyhermes_version=1), nbformat=4, nbformat_minor=5)))
    shell = DummyShell()
    shell.execution_count = 1
    ext = HermesExtension(shell=shell).load()
    ext.load_notebook(nb_path)

    assert shell.ran_cells == [("import math", True), ("x = 1", True)]
    assert ext.prompt_rows() == [("hi", "hello")]
    assert ext.prompt_records()[0]['history_line'] == 2
    assert ext.dialog_history()[0][0] == "<context><code>import math</code></context>\n<user-request>hi</user-request>"
    assert shell.execution_count == 4


def test_save_writes_notebook(tmp_path, capsys):
    shell = DummyShell()
    shell.history_manager.add(1, "import math")
    shell.history_manager.add(2, ".first prompt")
    shell.history_manager.add(3, "x = 1")
    shell.execution_count = 4
    ext = HermesExtension(shell=shell).load()
    ext.save_prompt("first prompt", "first response", 1)

    ext.handle_line(f"save {tmp_path / 'mysession'}")

    nb_path = tmp_path / "mysession.ipynb"
    assert f"Saved 2 code cells and 1 prompts to {nb_path}.\n" in capsys.readouterr().out
    nb = json.loads(nb_path.read_text())
    assert all("id" in c for c in nb["cells"])
    assert _strip_ids(nb) == dict(
        cells=[
            dict(cell_type="code", source="import math", metadata=dict(ipyhermes=dict(kind="code", line=1)), outputs=[], execution_count=None),
            dict(cell_type="markdown", source="first response",
                metadata=dict(ipyhermes=dict(kind="prompt", line=2, history_line=1, prompt="first prompt"))),
            dict(cell_type="code", source="x = 1", metadata=dict(ipyhermes=dict(kind="code", line=3)), outputs=[], execution_count=None),
        ],
        metadata=dict(ipyhermes_version=1), nbformat=4, nbformat_minor=5)


# ── Tests: Log Exact ─────────────────────────────────────────────────────────

async def test_log_exact_writes_full_prompt_and_response(dummy_agent):
    log_path = core.LOG_PATH
    shell = DummyShell()
    shell.history_manager.add(1, "a = 1")
    shell.execution_count = 3
    ext = HermesExtension(shell=shell, log_exact=True).load()

    await ext.run_prompt("tell me something")

    rec = json.loads(log_path.read_text().strip())
    assert rec["session"] == 1
    assert "tell me something" in rec["prompt"]
    assert rec["response"] == "first second"


# ── Tests: Tool/Variable/Shell References ────────────────────────────────────

def test_tools_resolve_from_ampersand_backticks():
    def demo():
        "Demo tool."
        return "ok"

    shell = DummyShell()
    shell.user_ns["demo"] = demo
    ext = HermesExtension(shell=shell).load()

    tools, bad = ext.resolve_tools("please call &`demo` now", [])
    assert any(t["function"]["name"] == "demo" for t in tools)
    assert bad == []


def test_tools_resolve_callable_objects_by_namespace_name():
    class Demo:
        def __call__(self):
            "Demo tool."
            return "ok"

    shell = DummyShell()
    shell.user_ns["demo"] = Demo()
    ext = HermesExtension(shell=shell).load()

    tools, bad = ext.resolve_tools("please call &`demo` now", [])
    assert any(t["function"]["name"] == "demo" for t in tools)
    assert bad == []


def test_var_names_extracts_dollar_backtick():
    assert _var_names("use $`x` and $`y`") == {"x", "y"}

def test_var_names_empty_on_no_match():
    assert _var_names("no vars here") == set()
    assert _var_names("") == set()

def test_var_refs_from_prompt_and_history():
    refs = _var_refs("use $`a`", [dict(prompt="use $`b`")])
    assert refs == {"a", "b"}

def test_var_refs_from_skills():
    skills = [dict(name="s", path="/s", description="", tools=[], vars=["x"])]
    refs = _var_refs("use $`a`", [], skills=skills)
    assert refs == {"a", "x"}

def test_var_refs_from_notes():
    refs = _var_refs("", [], notes=["---\nexposed-vars: x y\n---\nuse $`z`"])
    assert refs == {"x", "y", "z"}

def test_format_var_xml():
    ns = dict(x=42, name="hello")
    xml = _format_var_xml({"x", "name"}, ns)
    assert '<variable name="x" type="int">42</variable>' in xml
    assert '<variable name="name" type="str">hello</variable>' in xml

def test_format_var_xml_missing_returns_empty():
    assert _format_var_xml({"missing"}, {}) == ""


def test_shell_names_extracts_bang_backtick():
    assert _shell_names("check !`uname -a` and !`ls`") == {"uname -a", "ls"}

def test_shell_names_empty_on_no_match():
    assert _shell_names("no shell here") == set()

def test_shell_refs_from_prompt_and_history():
    refs = _shell_refs("run !`echo hi`", [dict(prompt="run !`date`")])
    assert refs == {"echo hi", "date"}

def test_shell_refs_from_skills():
    skills = [dict(name="s", path="/s", description="", tools=[], vars=[], shell_cmds=["git status"])]
    refs = _shell_refs("", [], skills=skills)
    assert refs == {"git status"}

def test_shell_refs_from_notes():
    refs = _shell_refs("", [], notes=["---\nshell-cmds: git status\n---\nrun !`ls`"])
    assert refs == {"git status", "ls"}

def test_run_shell_refs_runs_commands():
    xml = _run_shell_refs({"echo hello"})
    assert '<shell cmd="echo hello">' in xml
    assert 'hello' in xml

def test_run_shell_refs_empty_for_no_cmds():
    assert _run_shell_refs(set()) == ""


async def test_resolve_tools_missing_tool_not_raised(dummy_agent):
    shell,ext = mk_ext()
    await ext.run_prompt("call &`nonexistent`")
    call = dummy_agent[-1]
    assert '<warnings>' in call["prompt"]
    assert 'nonexistent' in call["prompt"]


async def test_var_in_prompt_adds_variable_xml(dummy_agent):
    shell,ext = mk_ext()
    shell.user_ns["myval"] = 99
    await ext.run_prompt("check $`myval`")
    call = dummy_agent[-1]
    assert '<variable name="myval" type="int">99</variable>' in call["prompt"]


async def test_missing_var_in_prompt_adds_note(dummy_agent):
    shell,ext = mk_ext()
    await ext.run_prompt("check $`missing_var`")
    call = dummy_agent[-1]
    assert '<warnings>' in call["prompt"]
    assert 'missing_var' in call["prompt"]


async def test_shell_in_prompt_adds_shell_xml(dummy_agent):
    shell,ext = mk_ext()
    await ext.run_prompt("check !`echo test123`")
    call = dummy_agent[-1]
    assert '<shell cmd="echo test123">' in call["prompt"]
    assert 'test123' in call["prompt"]


# ── Tests: Skills ────────────────────────────────────────────────────────────

def _mk_skill(root, name, description="A test skill."):
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "SKILL.md").write_text(f"---\nname: {name}\ndescription: {description}\n---\nInstructions here.\n")
    return d


def test_parse_skill(tmp_path):
    d = _mk_skill(tmp_path, "my-skill", "Does things.")
    s = _parse_skill(d)
    assert s == dict(name="my-skill", path=str(d), description="Does things.", tools=[], vars=[], shell_cmds=[])


def test_parse_skill_missing_file(tmp_path): assert _parse_skill(tmp_path / "nope") is None


def test_parse_skill_missing_name(tmp_path):
    d = tmp_path / "bad"
    d.mkdir()
    (d / "SKILL.md").write_text("---\ndescription: no name\n---\n")
    assert _parse_skill(d) is None


def test_discover_skills_walks_parents(tmp_path):
    skills_dir = tmp_path / "a" / "b" / ".agents" / "skills"
    _mk_skill(skills_dir, "deep-skill", "Deep.")
    parent_skills = tmp_path / ".agents" / "skills"
    _mk_skill(parent_skills, "top-skill", "Top.")
    cwd = tmp_path / "a" / "b"
    cwd.mkdir(parents=True, exist_ok=True)

    skills = _discover_skills(cwd=cwd)
    names = [s["name"] for s in skills]
    assert "deep-skill" in names
    assert "top-skill" in names
    assert names.index("deep-skill") < names.index("top-skill")


def test_discover_skills_deduplicates(tmp_path):
    skills_dir = tmp_path / ".agents" / "skills"
    _mk_skill(skills_dir, "only-once")
    skills = _discover_skills(cwd=tmp_path)
    assert sum(s["name"] == "only-once" for s in skills) == 1


def test_skills_xml_empty(): assert _skills_xml([]) == ""


def test_skills_xml_formats_correctly():
    skills = [dict(name="my-skill", path="/tmp/my-skill", description="Does things.")]
    xml = _skills_xml(skills)
    assert "<skills>" in xml
    assert 'name="my-skill"' in xml
    assert 'path="/tmp/my-skill"' in xml
    assert "Does things." in xml
    assert "load_skill" in xml


async def test_load_skill_reads_skill_md(tmp_path):
    d = _mk_skill(tmp_path, "test-skill")
    result = await load_skill(str(d))
    assert "Instructions here." in result
    assert "name: test-skill" in result


async def test_load_skill_missing_returns_error(tmp_path):
    result = await load_skill(str(tmp_path / "nope"))
    assert "Error" in result


async def test_eval_code_blocks_runs_eval_true():
    shell,ext = mk_ext()
    text = "# Intro\n\n```python\n#|eval: true\ndef foo(): return 42\n```\n\n```python\ndef bar(): return 99\n```"
    await _eval_code_blocks(text, shell)
    assert shell.user_ns["foo"]() == 42
    assert "bar" not in shell.user_ns


async def test_eval_code_blocks_space_after_pipe():
    shell,ext = mk_ext()
    text = "```python\n#| eval: true\nx = 7\n```"
    await _eval_code_blocks(text, shell)
    assert shell.user_ns["x"] == 7


def test_allowed_tools_from_frontmatter():
    text = "---\nallowed-tools: foo bar\n---\nbody"
    assert _allowed_tools(text) == {"foo", "bar"}

def test_allowed_tools_from_backtick_refs():
    assert _allowed_tools("use &`mytool` here") == {"mytool"}

def test_allowed_tools_combined():
    text = "---\nallowed-tools: foo\n---\nuse &`bar` here"
    assert _allowed_tools(text) == {"foo", "bar"}

def test_skill_allowed_tools(tmp_path):
    d = tmp_path / "my-skill"
    d.mkdir()
    (d / "SKILL.md").write_text("---\nname: my-skill\ndescription: x\nallowed-tools: analyze\n---\nuse &`fmt`\n")
    s = _parse_skill(d)
    assert set(s["tools"]) == {"analyze", "fmt"}


def test_tool_refs_includes_notes_but_not_unloaded_skills():
    skills = [dict(name="s", path="/s", description="", tools=["analyze"])]
    notes = ["---\nallowed-tools: fmt\n---\nuse &`helper`"]
    refs = _tool_refs("use &`main`", [], skills=skills, notes=notes)
    assert refs == {"main", "load_skill", "fmt", "helper"}


def test_skill_vars_parsed(tmp_path):
    d = tmp_path / "my-skill"
    d.mkdir()
    (d / "SKILL.md").write_text("---\nname: my-skill\ndescription: x\nexposed-vars: data\n---\nuse $`count`\n")
    s = _parse_skill(d)
    assert set(s["vars"]) == {"data", "count"}


def test_skill_shell_cmds_parsed(tmp_path):
    d = tmp_path / "my-skill"
    d.mkdir()
    (d / "SKILL.md").write_text("---\nname: my-skill\ndescription: x\nshell-cmds: git status\n---\nrun !`ls`\n")
    s = _parse_skill(d)
    assert set(s["shell_cmds"]) == {"git status", "ls"}


# ── Tests: Prompt Mode ───────────────────────────────────────────────────────

def test_prompt_mode_wraps_input_as_magic():
    lines = transform_prompt_mode(["hello world\n"])
    assert "run_cell_magic" in lines[0]
    assert "ipyhermes" in lines[0]
    assert "hello world" in lines[0]

def test_prompt_mode_passes_through_semicolon_as_python():
    lines = transform_prompt_mode([";x = 42\n"])
    assert lines == ["x = 42\n"]

def test_prompt_mode_passes_through_bang_as_shell():
    lines = transform_prompt_mode(["!ls\n"])
    assert lines == ["!ls\n"]

def test_prompt_mode_passes_through_percent_as_magic():
    lines = transform_prompt_mode(["%timeit 1+1\n"])
    assert lines == ["%timeit 1+1\n"]

def test_prompt_mode_passes_through_double_percent():
    lines = transform_prompt_mode(["%%bash\n", "echo hi\n"])
    assert lines == ["%%bash\n", "echo hi\n"]

def test_prompt_mode_multiline():
    lines = transform_prompt_mode(["tell me\\\n", "about python\n"])
    assert "run_cell_magic" in lines[0]
    assert "tell me" in lines[0] and "about python" in lines[0]

def test_prompt_mode_empty_passthrough():
    assert transform_prompt_mode(["\n"]) == ["\n"]
    assert transform_prompt_mode([]) == []

def test_prompt_mode_toggle(dummy_agent):
    shell,ext = mk_ext()
    assert not ext.prompt_mode
    ext.handle_line("prompt")
    assert ext.prompt_mode
    ext.handle_line("prompt")
    assert not ext.prompt_mode

def test_prompt_mode_flag():
    shell = DummyShell()
    ext = HermesExtension(shell=shell, prompt_mode=True).load()
    assert ext.prompt_mode

def test_prompt_mode_config_default(monkeypatch, tmp_path):
    cfg = tmp_path / "config.json"
    cfg.write_text('{"prompt_mode": true}')
    monkeypatch.setattr(core, "CONFIG_PATH", cfg)
    shell = DummyShell()
    ext = HermesExtension(shell=shell).load()
    assert ext.prompt_mode

def test_prompt_mode_config_with_flag_toggles(monkeypatch, tmp_path):
    cfg = tmp_path / "config.json"
    cfg.write_text('{"prompt_mode": true}')
    monkeypatch.setattr(core, "CONFIG_PATH", cfg)
    shell = DummyShell()
    ext = HermesExtension(shell=shell, prompt_mode=True).load()
    assert not ext.prompt_mode

def test_prompt_mode_registered_transformer(dummy_agent):
    shell,ext = mk_ext()
    ext.handle_line("prompt")
    cts = shell.input_transformer_manager.cleanup_transforms
    assert transform_prompt_mode in cts
    ext.handle_line("prompt")
    assert transform_prompt_mode not in cts


def test_sysprompt_mentions_variables_and_shell():
    assert '$`' in DEFAULT_SYSTEM_PROMPT
    assert '!`' in DEFAULT_SYSTEM_PROMPT


# ── Tests: Session Persistence ───────────────────────────────────────────────

def _mk_sessions_db():
    "Create an in-memory DB with the IPython sessions and history tables."
    db = sqlite3.connect(":memory:")
    db.execute("CREATE TABLE sessions (session INTEGER PRIMARY KEY AUTOINCREMENT, start TEXT, end TEXT, num_cmds INTEGER, remark TEXT)")
    db.execute("CREATE TABLE history (session INTEGER, line INTEGER, source TEXT, source_raw TEXT)")
    return db

def test_git_repo_root(tmp_path):
    (tmp_path / ".git").mkdir()
    sub = tmp_path / "a" / "b"
    sub.mkdir(parents=True)
    assert _git_repo_root(str(sub)) == str(tmp_path)

def test_git_repo_root_none(tmp_path):
    assert _git_repo_root(str(tmp_path)) is None or _git_repo_root(str(tmp_path)) != str(tmp_path)

def test_list_sessions_exact_match():
    db = _mk_sessions_db()
    db.execute("INSERT INTO sessions VALUES (1, '2025-01-01', '2025-01-01', 5, '/home/user/project')")
    db.execute("INSERT INTO sessions VALUES (2, '2025-01-02', '2025-01-02', 3, '/home/user/other')")
    rows = _list_sessions(db, "/home/user/project")
    assert len(rows) == 1
    assert rows[0][0] == 1

def test_list_sessions_returns_matching_rows():
    db = _mk_sessions_db()
    db.execute("INSERT INTO sessions VALUES (1, '2025-01-01', NULL, 5, '/proj')")
    rows = _list_sessions(db, "/proj")
    assert len(rows) == 1
    assert rows[0][0] == 1

def test_list_sessions_git_fallback(tmp_path):
    (tmp_path / ".git").mkdir()
    db = _mk_sessions_db()
    sub = str(tmp_path / "sub")
    db.execute("INSERT INTO sessions VALUES (1, '2025-01-01', NULL, 5, ?)", (str(tmp_path),))
    db.execute("INSERT INTO sessions VALUES (2, '2025-01-02', NULL, 3, ?)", (sub,))
    rows = _list_sessions(db, sub)
    assert len(rows) == 1 and rows[0][0] == 2
    rows = _list_sessions(db, str(tmp_path / "newsub"))
    assert len(rows) == 1 and rows[0][0] == 1

def test_resume_session():
    db = _mk_sessions_db()
    db.execute("INSERT INTO sessions VALUES (5, '2025-01-01', '2025-01-01 12:00', 10, '/proj')")
    db.execute("INSERT INTO history VALUES (5, 1, 'x=1', 'x=1')")
    db.execute("INSERT INTO history VALUES (5, 2, 'y=2', 'y=2')")
    db.execute("INSERT INTO sessions VALUES (6, '2025-01-02', NULL, NULL, '')")
    shell = DummyShell()
    shell.history_manager.db = db
    shell.history_manager.session_number = 6
    shell.history_manager.input_hist_parsed = [""]
    shell.history_manager.input_hist_raw = [""]
    resume_session(shell, 5)
    assert shell.history_manager.session_number == 5
    assert shell.execution_count == 3
    assert db.execute("SELECT * FROM sessions WHERE session=6").fetchone() is None
    row = db.execute("SELECT end FROM sessions WHERE session=5").fetchone()
    assert row[0] is None
    assert len(shell.history_manager.input_hist_parsed) == 3

def test_resume_session_not_found():
    db = _mk_sessions_db()
    db.execute("INSERT INTO sessions VALUES (1, '2025-01-01', NULL, NULL, '')")
    shell = DummyShell()
    shell.history_manager.db = db
    shell.history_manager.session_number = 1
    with pytest.raises(ValueError, match="Session 99 not found"): resume_session(shell, 99)


def test_handle_line_sessions(dummy_agent):
    shell,ext = mk_ext()
    hm = shell.history_manager
    hm.db.execute("CREATE TABLE IF NOT EXISTS sessions (session INTEGER PRIMARY KEY, start TEXT, end TEXT, num_cmds INTEGER, remark TEXT)")
    hm.db.execute("INSERT INTO sessions VALUES (1, '2025-01-01', NULL, 5, ?)", (os.getcwd(),))
    import io as _io
    buf = _io.StringIO()
    import sys as _sys
    old = _sys.stdout
    _sys.stdout = buf
    try: ext.handle_line("sessions")
    finally: _sys.stdout = old
    out = buf.getvalue()
    assert "1" in out


# ── Tests: Hermes Session ID Bridge ──────────────────────────────────────────

def test_hermes_session_id_format():
    assert _hermes_session_id(1) == "ipyhermes-1"
    assert _hermes_session_id(42) == "ipyhermes-42"
    assert _hermes_session_id(0) == "ipyhermes-0"


def test_agents_receive_session_id(monkeypatch):
    """All three agents should be created with a session_id derived from the IPython session number."""
    created = []
    class TrackingAgent:
        def __init__(self, **kw):
            self.kw = kw
            created.append(kw)
        async def astream(self, prompt, sp=None, hist=None):
            yield ""
    monkeypatch.setattr(core, "_mk_agent", lambda *a, **kw: TrackingAgent(**kw))
    shell = DummyShell()
    shell.history_manager.session_number = 7
    ext = HermesExtension(shell=shell)
    assert len(created) == 3
    for agent_kw in created:
        assert agent_kw.get("session_id") == "ipyhermes-7"


def test_hot_swap_preserves_session_id(monkeypatch, capsys):
    """Agent hot-swap on model change should pass the current session_id."""
    created = []
    class TrackingAgent:
        def __init__(self, **kw):
            self.kw = kw
            created.append(kw)
        async def astream(self, prompt, sp=None, hist=None):
            yield ""
    monkeypatch.setattr(core, "_mk_agent", lambda *a, **kw: TrackingAgent(**kw))
    shell = DummyShell()
    shell.history_manager.session_number = 3
    ext = HermesExtension(shell=shell)
    ext.load()
    created.clear()
    ext.handle_line("model gpt-4o")
    assert len(created) == 1
    assert created[0].get("session_id") == "ipyhermes-3"


def test_reset_clears_prompts():
    """reset_session_history should clear the in-memory _prompts list."""
    shell, ext = mk_ext()
    ext._prompts = [dict(prompt="hi", response="hello", history_line=1)]
    ext.reset_session_history()
    assert ext._prompts == []


# ── Tests: Handle Line Commands ──────────────────────────────────────────────

def test_handle_line_help(capsys):
    _,ext = mk_ext(load=False)
    ext.handle_line("help")
    out = capsys.readouterr().out
    assert "Usage:" in out
    assert "save" in out
    assert "load" in out
    assert "reset" in out
    assert "sessions" in out
    assert "memory" in out


def test_handle_line_memory(capsys, monkeypatch):
    monkeypatch.setattr(core, "_mk_convlog", lambda sid: None)  # karma not installed
    _,ext = mk_ext(load=False)
    ext.handle_line("memory on")
    out = capsys.readouterr().out
    assert "in-memory only" in out
    ext.handle_line("memory off")
    assert "Memory off" in capsys.readouterr().out
    ext.handle_line("memory")
    assert "memory=off" in capsys.readouterr().out


def test_handle_line_unknown(capsys):
    _,ext = mk_ext(load=False)
    ext.handle_line("nonexistent_cmd")
    out = capsys.readouterr().out
    assert "Unknown command" in out


# ── Tests: Plan mode ─────────────────────────────────────────────────────────

async def test_plan_prefix_is_stripped(dummy_agent):
    shell,ext = mk_ext()
    await ext.run_prompt(".plan refactor auth")
    call = dummy_agent[-1]
    assert "refactor auth" in call["prompt"]
    # .plan prefix is stripped
    assert ".plan" not in call["prompt"]


# ── Tests: Unload ────────────────────────────────────────────────────────────

def test_unload_cleans_up():
    shell,ext = mk_ext()
    assert ext.loaded
    assert shell.user_ns.get(EXTENSION_NS) is ext

    ext.unload()
    assert not ext.loaded
    assert EXTENSION_NS not in shell.user_ns


# ── Tests: Caveman Mode ──────────────────────────────────────────────────────

def test_caveman_default_off():
    _,ext = mk_ext()
    assert ext.caveman is False

def test_caveman_toggle(capsys):
    _,ext = mk_ext()
    assert ext.caveman is False
    ext.handle_line("caveman")
    assert ext.caveman is True
    assert "Caveman mode ON" in capsys.readouterr().out
    ext.handle_line("caveman")
    assert ext.caveman is False
    assert "Caveman mode OFF" in capsys.readouterr().out

def test_caveman_explicit_on_off(capsys):
    _,ext = mk_ext()
    ext.handle_line("caveman true")
    assert ext.caveman is True
    assert capsys.readouterr().out == "self.caveman=True\n"
    ext.handle_line("caveman false")
    assert ext.caveman is False
    assert capsys.readouterr().out == "self.caveman=False\n"

def test_caveman_init_flag():
    _,ext = mk_ext(caveman=True)
    assert ext.caveman is True

def test_caveman_config_default(monkeypatch, tmp_path):
    cfg = tmp_path / "config.json"
    cfg.write_text('{"caveman": true}')
    monkeypatch.setattr(core, "CONFIG_PATH", cfg)
    shell = DummyShell()
    ext = HermesExtension(shell=shell).load()
    assert ext.caveman is True

def test_caveman_in_status_output(capsys):
    _,ext = mk_ext()
    ext.handle_line("")
    out = capsys.readouterr().out
    assert "self.caveman=" in out

def test_build_sysp_caveman_off():
    sp = _build_sysp("base", [], caveman=False)
    assert "<caveman>" not in sp

def test_build_sysp_caveman_on():
    sp = _build_sysp("base", [], caveman=True)
    assert "<caveman>" in sp
    assert "caveman mode" in sp.lower()


# ── Tests: Phase 1 - bgterm & exhash injection ──────────────────────────────

def test_inject_bgterm_adds_functions(monkeypatch):
    """_inject_bgterm should add start_bgterm, write_stdin, close_bgterm to namespace."""
    import types
    fake_mod = types.ModuleType('bgterm')
    def _start_bgterm(name): return f"started {name}"
    def _write_stdin(name, text): return f"wrote {text}"
    def _close_bgterm(name): return f"closed {name}"
    _start_bgterm.__name__ = 'start_bgterm'
    _write_stdin.__name__ = 'write_stdin'
    _close_bgterm.__name__ = 'close_bgterm'
    fake_mod.start_bgterm = _start_bgterm
    fake_mod.write_stdin = _write_stdin
    fake_mod.close_bgterm = _close_bgterm
    monkeypatch.setitem(sys.modules, 'bgterm', fake_mod)

    ns = {}
    core._inject_bgterm(ns)
    assert 'start_bgterm' in ns
    assert 'write_stdin' in ns
    assert 'close_bgterm' in ns
    assert ns['start_bgterm']('test') == "started test"


def test_inject_bgterm_noop_when_missing(monkeypatch):
    """_inject_bgterm should silently skip when bgterm not installed."""
    import builtins
    _orig_import = builtins.__import__
    def _raise_for_bgterm(name, *a, **kw):
        if name == 'bgterm' or name.startswith('bgterm.'):
            raise ImportError(f"No module named '{name}'")
        return _orig_import(name, *a, **kw)
    monkeypatch.delitem(sys.modules, 'bgterm', raising=False)
    monkeypatch.setattr(builtins, '__import__', _raise_for_bgterm)
    ns = {}
    core._inject_bgterm(ns)
    assert 'start_bgterm' not in ns


def test_inject_exhash_adds_functions(monkeypatch):
    """_inject_exhash should add lnhashview_file, exhash_file to namespace."""
    import types
    fake_mod = types.ModuleType('exhash')
    def _lnhashview_file(path): return f"viewed {path}"
    def _exhash_file(path, cmds): return f"edited {path}"
    _lnhashview_file.__name__ = 'lnhashview_file'
    _exhash_file.__name__ = 'exhash_file'
    fake_mod.lnhashview_file = _lnhashview_file
    fake_mod.exhash_file = _exhash_file
    monkeypatch.setitem(sys.modules, 'exhash', fake_mod)

    ns = {}
    core._inject_exhash(ns)
    assert 'lnhashview_file' in ns
    assert 'exhash_file' in ns
    assert ns['lnhashview_file']('test.py') == "viewed test.py"


def test_inject_bhoga_adds_router(monkeypatch):
    """_inject_bhoga should add bhoga_router and apply_to_hermes to namespace."""
    import types
    fake_mod = types.ModuleType('bhoga')
    class FakeRouter:
        def quotas(self): return {}
        def best_for(self, model): return None
    fake_mod.Router = FakeRouter
    fake_mod.apply_to_hermes = lambda r, m: None
    monkeypatch.setitem(sys.modules, 'bhoga', fake_mod)

    ns = {}
    core._inject_bhoga(ns)
    assert 'bhoga_router' in ns
    assert 'apply_to_hermes' in ns
    assert isinstance(ns['bhoga_router'], FakeRouter)


# ── Tests: Phase 1 - system prompt includes new tool docs ───────────────────

def test_build_sysp_includes_bgterm():
    sp = _build_sysp("base", [])
    assert "<bgterm>" in sp
    assert "start_bgterm" in sp

def test_build_sysp_includes_exhash():
    sp = _build_sysp("base", [])
    assert "<exhash>" in sp
    assert "exhash_file" in sp

def test_build_sysp_includes_bhoga():
    sp = _build_sysp("base", [])
    assert "<bhoga>" in sp
    assert "bhoga_router" in sp


# ── Tests: Phase 2a - Always-on karma ConversationLog ────────────────────────

def test_convlog_always_initialized(monkeypatch):
    """ConversationLog should be initialized automatically (always-on)."""
    class FakeConvLog:
        def __init__(self, session_id): self.session_id = session_id
        def get_session(self, n=50): return []
    monkeypatch.setattr(core, "_mk_convlog", lambda sid: FakeConvLog(sid))
    shell = DummyShell()
    ext = HermesExtension(shell=shell)
    assert ext._convlog is not None
    assert ext._convlog.session_id == "ipyhermes-1"


def test_convlog_none_when_karma_missing(monkeypatch):
    """ConversationLog should be None if karma is not installed."""
    monkeypatch.setattr(core, "_mk_convlog", lambda sid: None)
    shell = DummyShell()
    ext = HermesExtension(shell=shell)
    assert ext._convlog is None


def test_memory_on_off_still_works(capsys, monkeypatch):
    """User can still toggle memory on/off for backward compat."""
    monkeypatch.setattr(core, "_mk_convlog", lambda sid: None)
    _,ext = mk_ext(load=False)
    ext.handle_line("memory on")
    out = capsys.readouterr().out
    assert "in-memory only" in out
    ext.handle_line("memory off")
    assert "Memory off" in capsys.readouterr().out



# ── Tests: Phase 3 - bhoga route command ─────────────────────────────────────

def test_handle_route_no_bhoga(capsys, monkeypatch):
    """route command should show error when bhoga not installed."""
    monkeypatch.delitem(sys.modules, 'bhoga', raising=False)
    def raise_import(name, *a, **kw):
        if name == 'bhoga' or name.startswith('bhoga.'):
            raise ImportError("No module named 'bhoga'")
        return original_import(name, *a, **kw)
    import builtins
    original_import = builtins.__import__
    monkeypatch.setattr(builtins, '__import__', raise_import)
    _,ext = mk_ext(load=False)
    ext._handle_route("")
    out = capsys.readouterr().out
    assert "bhoga not installed" in out


def test_handle_route_show_quotas(capsys, monkeypatch):
    """route command with no args should show quota status."""
    import types
    fake_bhoga = types.ModuleType('bhoga')
    class FakeQuota:
        remaining_pct = 0.75
        status = 'OK'
    class FakeRouter:
        def quotas(self): return {'anthropic_api': FakeQuota()}
        def best_for(self, model): return None
    fake_bhoga.Router = FakeRouter
    fake_bhoga.apply_to_hermes = lambda r, m: None
    monkeypatch.setitem(sys.modules, 'bhoga', fake_bhoga)

    _,ext = mk_ext(load=False)
    ext.shell.user_ns['bhoga_router'] = FakeRouter()
    ext._handle_route("")
    out = capsys.readouterr().out
    assert "anthropic_api" in out
    assert "75%" in out


def test_handle_route_force_provider(capsys, monkeypatch):
    """route command with provider name should force that provider."""
    import types
    fake_bhoga = types.ModuleType('bhoga')
    fake_bhoga.Router = type('Router', (), {'quotas': lambda self: {}})
    fake_bhoga.apply_to_hermes = lambda r, m: None
    monkeypatch.setitem(sys.modules, 'bhoga', fake_bhoga)

    _,ext = mk_ext(load=False)
    ext._handle_route("openai-codex")
    assert ext.provider == "openai-codex"
    out = capsys.readouterr().out
    assert "Provider forced to" in out


# ── Tests: Help text includes new commands ───────────────────────────────────

def test_help_includes_route(capsys):
    _,ext = mk_ext(load=False)
    ext._show_help()
    out = capsys.readouterr().out
    assert "route" in out


# ── Tests: Status output ────────────────────────────────────────────────────

def test_status_shows_memory_state(capsys, monkeypatch):
    """Status should show memory=on when convlog is active."""
    class FakeConvLog:
        def __init__(self, sid): pass
    monkeypatch.setattr(core, "_mk_convlog", lambda sid: FakeConvLog(sid))
    shell = DummyShell()
    ext = HermesExtension(shell=shell).load()
    ext.handle_line("")
    out = capsys.readouterr().out
    assert "memory=on" in out
