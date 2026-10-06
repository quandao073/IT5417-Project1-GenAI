"""The embed extra must pin exactly what the worker image installs.

A model revision is only "pinned" if the library that loads it is too: the gate
measured on the dev venv must describe the model the Docker worker will run.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PINNED = ("torch", "torchvision", "open_clip_torch", "transformers")


def _pins(requirements):
    pins = {}
    for req in requirements:
        if match := re.fullmatch(r"([A-Za-z0-9_.-]+)==([0-9][0-9A-Za-z.+-]*)", req.strip()):
            pins[match.group(1)] = match.group(2)
    return pins


def test_embed_extra_matches_the_worker_dockerfile():
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    extra = _pins(pyproject["project"]["optional-dependencies"]["embed"])

    dockerfile = (ROOT / "docker" / "worker.Dockerfile").read_text(encoding="utf-8")
    docker = dict(
        re.findall(r"\b(torch|torchvision|open_clip_torch|transformers)==([\w.+-]+)", dockerfile)
    )

    assert set(PINNED) <= set(docker), "Dockerfile no longer pins every embed library"
    assert {name: extra.get(name) for name in PINNED} == {name: docker[name] for name in PINNED}
