"""The embed extra must pin exactly what the worker image installs.

A model revision is only "pinned" if the library that loads it is too: the gate
measured on the dev venv must describe the model the Docker worker will run.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PINNED = ("torch", "torchvision", "open_clip_torch", "transformers")


def _pins(requirements):
    pins = {}
    for req in requirements:
        if match := re.fullmatch(r"([A-Za-z0-9_.-]+)==([0-9][0-9A-Za-z.+-]*)", req.strip()):
            pins[match.group(1)] = match.group(2)
    return pins


def _docker_pins(name: str) -> dict[str, str]:
    text = (ROOT / "docker" / name).read_text(encoding="utf-8")
    return dict(
        re.findall(r"\b(torch|torchvision|open_clip_torch|transformers)==([\w.+-]+)", text)
    )


@pytest.mark.parametrize(
    "extra, dockerfile",
    [("embed", "worker.Dockerfile"), ("api", "api.Dockerfile")],
)
def test_extra_matches_its_dockerfile(extra, dockerfile):
    """A model revision is only pinned if the library that loads it is too, in
    every image that loads it: the worker embeds, the API encodes queries."""
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    pins = _pins(pyproject["project"]["optional-dependencies"][extra])
    docker = _docker_pins(dockerfile)

    assert set(PINNED) <= set(docker), f"{dockerfile} no longer pins every model library"
    assert {name: pins.get(name) for name in PINNED} == {name: docker[name] for name in PINNED}


def test_api_dockerfile_installs_torch_from_the_cpu_index():
    """On Linux, plain `pip install torch` pulls the CUDA build and several GB of
    nvidia packages into an image that only encodes short text queries."""
    text = (ROOT / "docker" / "api.Dockerfile").read_text(encoding="utf-8")
    assert "download.pytorch.org/whl/cpu" in text


def test_compose_qdrant_image_matches_the_client_minor():
    """qdrant-client 1.19 against a 1.9 server is 10 minors apart; keep them aligned."""
    import importlib.metadata as md

    client_minor = ".".join(md.version("qdrant-client").split(".")[:2])
    compose = (ROOT / "compose.yaml").read_text(encoding="utf-8")
    match = re.search(r"image:\s*qdrant/qdrant:v(\d+\.\d+)\.\d+", compose)
    assert match, "compose.yaml must pin a qdrant/qdrant:vX.Y.Z image"
    assert match.group(1) == client_minor
