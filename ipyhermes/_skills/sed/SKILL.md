---
name: sed
description: "Stream editor for filtering and transforming text — use sed() for quick text processing."
allowed-tools: sed
---

# sed — stream editing tool

`sed(text, cmds)` applies sed-style transformations to text.

## Common patterns

```python
sed(text, 's/old/new/g')         # global replace
sed(text, '/pattern/d')          # delete matching lines
sed(text, '1,5d')                # delete first 5 lines
sed(text, '/start/,/end/p')      # print range
```

Use `sed` for quick text transformations without writing Python loops.
