---
description: ipython-context — Automatic IPython session context injection for AI prompts.
---

# ipython-context

This skill automatically provides the AI with context from the current IPython session:

- Recent code cells and their outputs
- Variable values referenced with `$`name``
- Shell command outputs referenced with `!`cmd``
- Note cells (string literals used as documentation)

The context is injected as XML blocks in the prompt, giving the AI full awareness of what
has been executed in the session.
