"""Config loading, the build config hash, and runtime settings."""

from __future__ import annotations

import pytest

from src.common import config


@pytest.fixture
def config_dir(tmp_path):
    """A throwaway configs/ directory with two small files."""
    (tmp_path / "data.yaml").write_text("corpus:\n  target_studies: 25000\n", encoding="utf-8")
    (tmp_path / "models.yaml").write_text("llm:\n  temperature: 0.0\n", encoding="utf-8")
    return tmp_path


class TestLoadConfig:
    def test_reads_a_config_by_bare_name(self, config_dir):
        assert config.load_config("data", config_dir)["corpus"]["target_studies"] == 25000

    def test_reads_the_real_project_configs(self):
        assert len(config.load_config("data")["labels"]["pathologies"]) == 14

    def test_reports_a_missing_config(self, config_dir):
        with pytest.raises(FileNotFoundError, match="nope"):
            config.load_config("nope", config_dir)

    @pytest.mark.parametrize("name", ["../secrets", "sub/data", r"sub\data", "/etc/passwd"])
    def test_rejects_a_name_that_is_not_a_bare_name(self, name, config_dir):
        with pytest.raises(ValueError):
            config.load_config(name, config_dir)


class TestConfigHash:
    def test_is_deterministic(self, config_dir):
        first = config.config_hash(("data", "models"), config_dir)
        assert first == config.config_hash(("data", "models"), config_dir)

    def test_changes_when_a_value_changes(self, config_dir):
        before = config.config_hash(("data", "models"), config_dir)
        (config_dir / "data.yaml").write_text(
            "corpus:\n  target_studies: 10000\n", encoding="utf-8"
        )
        assert config.config_hash(("data", "models"), config_dir) != before

    def test_ignores_comments_and_formatting(self, config_dir):
        """A comment edit must not invalidate an index that is still correct."""
        before = config.config_hash(("data", "models"), config_dir)
        (config_dir / "data.yaml").write_text(
            "# explain the corpus size\ncorpus:\n\n  target_studies:   25000\n",
            encoding="utf-8",
        )
        assert config.config_hash(("data", "models"), config_dir) == before

    def test_ignores_the_order_the_names_are_given_in(self, config_dir):
        assert config.config_hash(("models", "data"), config_dir) == config.config_hash(
            ("data", "models"), config_dir
        )

    def test_defaults_to_the_configs_that_affect_a_build(self):
        """logging and evaluation must not invalidate a built index."""
        assert set(config.BUILD_CONFIGS) == {"data", "models", "retrieval", "concepts"}
        assert len(config.config_hash()) == 64


class TestSettings:
    def test_has_local_defaults_that_need_no_env_file(self):
        settings = config.Settings(_env_file=None)
        assert settings.qdrant_url.startswith("http")
        assert settings.canonical_dir.name == "canonical"
        assert settings.build_id is None

    def test_reads_values_from_the_environment(self, monkeypatch):
        monkeypatch.setenv("QDRANT_URL", "http://qdrant:6333")
        monkeypatch.setenv("BUILD_ID", "20260921_chexpert25k_v1")
        settings = config.Settings(_env_file=None)
        assert settings.qdrant_url == "http://qdrant:6333"
        assert settings.build_id == "20260921_chexpert25k_v1"

    def test_coerces_numbers_and_booleans(self, monkeypatch):
        monkeypatch.setenv("API_PORT", "9000")
        monkeypatch.setenv("LLM_DISABLED", "true")
        settings = config.Settings(_env_file=None)
        assert settings.api_port == 9000
        assert settings.llm_disabled is True

    def test_treats_a_blank_build_id_as_unset(self, monkeypatch):
        """.env.example ships BUILD_ID= so the API falls back to build_manifest.json."""
        monkeypatch.setenv("BUILD_ID", "")
        assert config.Settings(_env_file=None).build_id is None


def test_every_setting_is_documented_in_the_env_example():
    with open(".env.example", encoding="utf-8") as handle:
        documented = {
            line.split("=", 1)[0].strip()
            for line in handle
            if "=" in line and not line.lstrip().startswith("#")
        }
    undocumented = {
        name.upper() for name in config.Settings.model_fields if name.upper() not in documented
    }
    assert not undocumented, f"add these to .env.example: {sorted(undocumented)}"
