"""A throwaway Neo4j with the tiny KG loaded (tests/kg/tiny_kg.py), for the KG integration tests.

The container is started here and removed at the end of the session, so the real graph that
`cxr kg-load-neo4j` loaded into the compose service is never touched. Skipped when Docker or the
pinned image is not there (nothing is pulled).
"""

from __future__ import annotations

import shutil
import subprocess
import time
import uuid

import pytest

from src.kg import neo4j_load as nl
from tests.kg import tiny_kg

IMAGE = "neo4j:5.26-community"
PASSWORD = "testtest123"
TOP_N = 3


def _docker(*args: str) -> str:
    return subprocess.run(["docker", *args], capture_output=True, text=True, timeout=120,
                          check=True).stdout.strip()


@pytest.fixture(scope="session")
def kg_neo4j():
    """(driver, manifest) of a fresh Neo4j holding the tiny KG with top-3 PREDICTED edges."""
    if shutil.which("docker") is None:
        pytest.skip("docker is not installed")
    try:
        _docker("image", "inspect", IMAGE)
        name = f"cxr-test-neo4j-{uuid.uuid4().hex[:8]}"
        _docker("run", "-d", "--rm", "--name", name, "-p", "127.0.0.1::7687",
                "-e", f"NEO4J_AUTH=neo4j/{PASSWORD}", IMAGE)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        pytest.skip(f"cannot start {IMAGE}: {exc}")
    from neo4j import GraphDatabase

    driver = None
    try:
        port = _docker("port", name, "7687").splitlines()[0].rsplit(":", 1)[1]
        driver = GraphDatabase.driver(f"bolt://127.0.0.1:{port}", auth=("neo4j", PASSWORD))
        for _ in range(60):
            try:
                driver.verify_connectivity()
                break
            except Exception:  # noqa: BLE001 - still starting
                time.sleep(2)
        else:
            pytest.skip("neo4j did not come up")
        with driver.session() as session:
            manifest = nl.load_all(
                session, nodes=tiny_kg.nodes(), quality=tiny_kg.quality(),
                types=tiny_kg.semantic_types(), aliases=tiny_kg.aliases(),
                edges=tiny_kg.edges(), scores=tiny_kg.scores(), images=tiny_kg.images(),
                concept_columns=tiny_kg.COLUMNS, top_n=TOP_N, batch=5,
                build={"kg_build_id": "kg_test", "matrix_version": "matrix_test@0"})
        yield driver, manifest
    finally:
        if driver is not None:
            driver.close()
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)
