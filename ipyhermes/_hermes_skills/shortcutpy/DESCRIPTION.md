---
description: shortcutpy — compile Python DSL into Apple Shortcuts (.shortcut files) on macOS.
---

# shortcutpy

Build macOS Shortcuts from Python source using a declarative DSL.

## Usage

```python
from shortcutpy.dsl import shortcut, ask_for_text, choose_from_menu, show_result

@shortcut(color="yellow", glyph="hand")
def my_shortcut():
    name = ask_for_text("Your name?")
    show_result(f"Hello {name}")
```

Compile and sign: `compile_file('my_shortcut.py')` or `compile_source(source_code)`

## Available DSL Actions
- `ask_for_text(prompt)` — prompt user for text input
- `choose_from_menu(prompt, options)` — present a choice menu
- `show_result(value)` — display output
- `save_file(value)`, `get_files()`, `resize_image(img, width=, height=)`
- `shortcut_input()` — get Share Sheet input
- See `shortcutpy.dsl` for full catalog
