"""Ollama adapter: request shape, probe states, output validation, prompt. No network."""

from __future__ import annotations

import json

import httpx
import pytest

from src.common.config import load_config
from src.data.labels import CHEXPERT_LABELS
from src.query import llm_parser as llm


def client_with(handler) -> llm.OllamaClient:
    return llm.OllamaClient("http://ollama.test", "qwen-test", 5,
                            transport=httpx.MockTransport(handler))


def ok_chat(content: dict | str) -> httpx.Response:
    body = content if isinstance(content, str) else json.dumps(content)
    return httpx.Response(200, json={"message": {"role": "assistant", "content": body},
                                     "total_duration": 1_500_000_000, "eval_count": 40})


class TestOutputSchema:
    def test_concepts_are_enums_of_the_fourteen_labels(self):
        schema = llm.output_schema()
        include = schema["properties"]["include"]["items"]["properties"]
        assert include["concept"]["enum"] == list(CHEXPERT_LABELS)
        assert include["status"]["enum"] == ["PRESENT", "UNCERTAIN"]
        assert schema["properties"]["exclude"]["items"]["enum"] == list(CHEXPERT_LABELS)
        assert schema["properties"]["views"]["items"]["enum"] == ["AP", "PA"]
        assert set(schema["required"]) == {"findings", "include", "exclude", "views", "language"}


class TestChat:
    def test_sends_the_schema_temperature_zero_and_keep_alive(self):
        seen: dict = {}

        def handler(request):
            seen.update(json.loads(request.content))
            seen["path"] = request.url.path
            return ok_chat({"findings": "x"})

        client_with(handler).chat([{"role": "user", "content": "q"}], llm.output_schema())
        assert seen["path"] == "/api/chat"
        assert seen["model"] == "qwen-test" and seen["stream"] is False
        assert seen["format"] == llm.output_schema()
        assert seen["options"] == {"temperature": 0, "seed": 20260101, "num_predict": 300}
        assert seen["keep_alive"] == "30m"
        assert seen["messages"] == [{"role": "user", "content": "q"}]

    def test_returns_the_content_and_server_milliseconds(self):
        def handler(request):
            return ok_chat({"a": 1})

        content, ms = client_with(handler).chat([], {})
        assert json.loads(content) == {"a": 1} and ms == 1500.0

    @pytest.mark.parametrize("failure", [httpx.ReadTimeout("slow"), httpx.ConnectError("down")])
    def test_transport_failures_become_llm_error(self, failure):
        def handler(request):
            raise failure

        with pytest.raises(llm.LLMError, match=type(failure).__name__):
            client_with(handler).chat([], {})

    def test_http_errors_become_llm_error(self):
        def handler(request):
            return httpx.Response(500, text="boom")

        with pytest.raises(llm.LLMError, match="500"):
            client_with(handler).chat([], {})

    def test_malformed_200_body_is_invalid_output(self):
        def not_json(request):
            return httpx.Response(200, text="not json")

        def no_message(request):
            return httpx.Response(200, json={"done": True})

        with pytest.raises(llm.LLMInvalidOutput, match="malformed"):
            client_with(not_json).chat([], {})
        with pytest.raises(llm.LLMInvalidOutput, match="malformed"):
            client_with(no_message).chat([], {})


class TestProbe:
    def test_ok_when_the_model_is_listed(self):
        def handler(request):
            assert request.url.path == "/api/tags"
            return httpx.Response(200, json={"models": [{"name": "qwen-test"}]})

        assert client_with(handler).probe() == "ok"

    def test_missing_model_when_it_is_not(self):
        def handler(request):
            return httpx.Response(200, json={"models": [{"name": "other:latest"}]})

        assert client_with(handler).probe() == "missing_model"

    def test_unreachable_on_any_transport_or_http_failure(self):
        def refused(request):
            raise httpx.ConnectError("refused")

        def broken(request):
            return httpx.Response(502, text="bad gateway")

        assert client_with(refused).probe() == "unreachable"
        assert client_with(broken).probe() == "unreachable"

    def test_version_is_read_from_the_server_or_none(self):
        def handler(request):
            return httpx.Response(200, json={"version": "0.40.3"})

        def down(request):
            raise httpx.ConnectError("refused")

        assert client_with(handler).version() == "0.40.3"
        assert client_with(down).version() is None


class TestParseOutput:
    def test_valid_output_is_canonicalised(self):
        out = llm.parse_output(json.dumps({
            "findings": "edema", "include": [{"concept": "edema", "status": "PRESENT"}],
            "exclude": ["Pleural Effusion"], "views": [], "language": "en"}))
        assert out.include[0].concept == "Edema" and out.exclude == ["Pleural Effusion"]

    def test_not_json_is_invalid_output(self):
        with pytest.raises(llm.LLMInvalidOutput, match="not JSON"):
            llm.parse_output("{oops")

    def test_unknown_concept_is_invalid_output(self):
        bad = {"findings": "", "include": [{"concept": "Cardiomegalyy", "status": "PRESENT"}],
               "exclude": [], "views": [], "language": "en"}
        with pytest.raises(llm.LLMInvalidOutput, match="unknown concept"):
            llm.parse_output(json.dumps(bad))

    def test_missing_language_is_invalid_output(self):
        with pytest.raises(llm.LLMInvalidOutput, match="language"):
            llm.parse_output(json.dumps(
                {"findings": "", "include": [], "exclude": [], "views": []}))

    def test_a_json_list_is_invalid_output(self):
        with pytest.raises(llm.LLMInvalidOutput):
            llm.parse_output("[1, 2]")

    def test_unknown_exclude_concept_is_invalid_output(self):
        bad = {"findings": "", "include": [], "exclude": ["Pleural Efusion"],
               "views": [], "language": "en"}
        with pytest.raises(llm.LLMInvalidOutput, match="unknown concept"):
            llm.parse_output(json.dumps(bad))

    def test_exclude_is_canonicalised(self):
        out = llm.parse_output(json.dumps({
            "findings": "", "include": [], "exclude": ["pleural effusion"],
            "views": [], "language": "en"}))
        assert out.exclude == ["Pleural Effusion"]


class TestPrompt:
    def test_renders_every_concept_with_both_languages(self):
        text = llm.render_prompt("Concepts:\n{{CONCEPTS}}\nEnd", load_config("concepts"))
        for concept in CHEXPERT_LABELS:
            assert f"- {concept}:" in text
        assert "tràn dịch màng phổi" in text and "{{CONCEPTS}}" not in text

    def test_the_shipped_prompt_renders_and_has_examples(self):
        prompt = llm.load_prompt(load_config("models")["llm"]["prompt_file"],
                                 load_config("concepts"))
        examples = llm.prompt_examples(prompt)
        assert len(examples) >= 4 and "{{CONCEPTS}}" not in prompt
        assert all(example.strip() for example in examples)
