---
name: ex
description: "Run ex editor commands on a file — great for reading files, search/replace, indent/dedent, and surgical multi-line edits."
allowed-tools: ex
---

# ex — file editing tool

`ex(path, cmds)` runs ex editor commands on a file.

## Common patterns

```
ex('file.py', '1,5p')           # print lines 1-5
ex('file.py', '/pattern/p')     # print matching lines
ex('file.py', 's/old/new/g')    # global search/replace
ex('file.py', '10,20d')         # delete lines 10-20
ex('file.py', '5a\nnew line')   # append after line 5
```

Use `ex` for surgical edits instead of rewriting whole files.
