from pathlib import Path
import shutil,os

_PKG = Path(__file__).parent
_HH  = Path(os.environ.get('HERMES_HOME', '~/.hermes')).expanduser()

def _install_hermes_skills():
    "Copy bundled hermes-format skills to ~/.hermes/skills/ (once per skill)."
    src = _PKG / '_hermes_skills'
    if not src.is_dir(): return
    dst = _HH / 'skills'
    dst.mkdir(parents=True, exist_ok=True)
    for sd in src.iterdir():
        d = dst / sd.name
        if not d.exists(): shutil.copytree(sd, d)

def _write_hermes_cfg():
    "Write ~/.hermes/cli-config.yaml once. Never overwrites."
    p = _HH / 'cli-config.yaml'
    if p.exists(): return
    p.parent.mkdir(parents=True, exist_ok=True)
    try: import yaml
    except ImportError: return
    cfg = dict(
        model=dict(default='gpt-5.4', provider='openai-codex'),
        smart_model_routing=dict(enabled=True, cheap_model='gpt-5.4-mini',
                                 cheap_provider='openai-codex', threshold=0.30),
        terminal=dict(backend='docker', cwd='/workspace',
                      docker_image='nikolaik/python-nodejs:python3.11-nodejs20',
                      docker_mount_cwd_to_workspace=True,
                      container_memory=5120, container_persistent=True, lifetime_seconds=600),
        compression=dict(enabled=True, threshold=0.50,
                         summary_provider='openai-codex', summary_model='gpt-5.4-mini'),
    )
    p.write_text(yaml.dump(cfg, default_flow_style=False))

_install_hermes_skills()
_write_hermes_cfg()
try: import karma
except ImportError: pass
try: import webba
except ImportError: pass

from .core import (EXTENSION_ATTR, EXTENSION_NS, LAST_PROMPT, LAST_RESPONSE, MAGIC_NAME,
    PROMPTS_TABLE, RESET_LINE_NS, HermesExtension, create_extension,
    load_ipython_extension, unload_ipython_extension)
