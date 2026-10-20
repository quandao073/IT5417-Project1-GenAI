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


# --- OpenAI-compatible adapter: OpenAI, and Gemini through its OpenAI endpoint ---


def compat_with(handler, base_url="https://api.test/v1") -> llm.OpenAICompatClient:
    return llm.OpenAICompatClient(base_url, "model-test", 5, api_key="sk-test",
                                  transport=httpx.MockTransport(handler))


def ok_completion(content: str | None) -> httpx.Response:
    return httpx.Response(200, json={"choices": [
        {"message": {"role": "assistant", "content": content}, "finish_reason": "stop"}]})


def _objects(node):
    """Every JSON-schema object node, depth first."""
    if isinstance(node, dict):
        if node.get("type") == "object":
            yield node
        for value in node.values():
            yield from _objects(value)
    elif isinstance(node, list):
        for value in node:
            yield from _objects(value)


class TestOpenAICompatChat:
    def test_posts_chat_completions_under_the_base_path_with_a_bearer_key(self):
        seen: dict = {}

        def handler(request):
            seen["url"] = str(request.url)
            seen["auth"] = request.headers.get("authorization")
            return ok_completion("{}")

        compat_with(handler).chat([{"role": "user", "content": "q"}], llm.output_schema())
        assert seen["url"] == "https://api.test/v1/chat/completions"
        assert seen["auth"] == "Bearer sk-test"

    def test_sends_the_same_schema_strictly_at_temperature_zero(self):
        seen: dict = {}

        def handler(request):
            seen.update(json.loads(request.content))
            return ok_completion("{}")

        messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "q"}]
        compat_with(handler).chat(messages, llm.output_schema())
        assert seen["model"] == "model-test" and seen["messages"] == messages
        # Thinking models count reasoning against the cap: gemini-2.5-flash spent
        # 212-221 thinking tokens on one parse and hit `length` 2/6 times under a 300 cap.
        # The GPT-5 family rejects `max_tokens` outright, so the newer name is used.
        assert seen["temperature"] == 0 and seen["max_completion_tokens"] == 2048
        assert "max_tokens" not in seen
        # Gemini's OpenAI endpoint rejects `seed` with a 400 (measured 2026-10-20), and on
        # a hosted API it is best-effort anyway; temperature 0 is what is portable.
        assert "seed" not in seen
        fmt = seen["response_format"]
        assert fmt["type"] == "json_schema" and fmt["json_schema"]["strict"] is True
        sent = fmt["json_schema"]["schema"]
        # Strict mode needs additionalProperties: false on every object; nothing else changes,
        # so the enum of the 14 concepts still constrains decoding as it does for Ollama.
        assert all(o.get("additionalProperties") is False for o in _objects(sent))
        stripped = json.loads(json.dumps(sent).replace(', "additionalProperties": false', ""))
        assert stripped == llm.output_schema()

    def test_model_specific_options_replace_the_default_temperature(self):
        """gpt-5-nano rejects temperature 0 and burns 1,536 reasoning tokens unless told
        reasoning_effort=minimal (measured 2026-10-20); such knobs come from config."""
        seen: dict = {}

        def handler(request):
            seen.update(json.loads(request.content))
            return ok_completion("{}")

        client = llm.OpenAICompatClient("https://api.test/v1", "m", 5, api_key="k",
                                        options={"reasoning_effort": "minimal"},
                                        transport=httpx.MockTransport(handler))
        client.chat([], llm.output_schema())
        assert seen["reasoning_effort"] == "minimal" and "temperature" not in seen

    def test_does_not_mutate_the_callers_schema(self):
        schema = llm.output_schema()
        compat_with(lambda r: ok_completion("{}")).chat([], schema)
        assert schema == llm.output_schema()

    def test_returns_the_message_content(self):
        content, ms = compat_with(lambda r: ok_completion('{"a": 1}')).chat([], {})
        assert json.loads(content) == {"a": 1} and ms >= 0

    @pytest.mark.parametrize("failure", [httpx.ReadTimeout("slow"), httpx.ConnectError("down")])
    def test_transport_failures_become_llm_error(self, failure):
        def handler(request):
            raise failure

        with pytest.raises(llm.LLMError, match=type(failure).__name__):
            compat_with(handler).chat([], {})

    @pytest.mark.parametrize("code", [400, 401, 429, 500])
    def test_http_errors_become_llm_error_with_the_status(self, code):
        def handler(request):
            return httpx.Response(code, json={"error": {"message": "nope"}})

        with pytest.raises(llm.LLMError, match=str(code)):
            compat_with(handler).chat([], {})

    def test_a_malformed_or_empty_reply_is_invalid_output(self):
        def no_choices(request):
            return httpx.Response(200, json={"id": "x"})

        def refused(request):
            return ok_completion(None)

        with pytest.raises(llm.LLMInvalidOutput, match="malformed"):
            compat_with(no_choices).chat([], {})
        with pytest.raises(llm.LLMInvalidOutput, match="no content"):
            compat_with(refused).chat([], {})


class TestOpenAICompatProbe:
    def test_ok_when_the_model_is_listed(self):
        def handler(request):
            assert str(request.url) == "https://api.test/v1/models"
            return httpx.Response(200, json={"data": [{"id": "model-test"}]})

        assert compat_with(handler).probe() == "ok"

    def test_gemini_style_ids_with_a_models_prefix_also_match(self):
        def handler(request):
            return httpx.Response(200, json={"data": [{"id": "models/model-test"}]})

        assert compat_with(handler).probe() == "ok"

    def test_missing_model_when_it_is_not_listed(self):
        def handler(request):
            return httpx.Response(200, json={"data": [{"id": "other"}]})

        assert compat_with(handler).probe() == "missing_model"

    def test_unreachable_on_a_bad_key_or_any_failure(self):
        def unauthorized(request):
            return httpx.Response(401, json={"error": {"message": "bad key"}})

        def refused(request):
            raise httpx.ConnectError("refused")

        assert compat_with(unauthorized).probe() == "unreachable"
        assert compat_with(refused).probe() == "unreachable"

    def test_has_no_server_version(self):
        assert compat_with(lambda r: httpx.Response(200)).version() is None


class TestMakeClient:
    def test_ollama_is_the_default_provider(self):
        client = llm.make_client("ollama", "http://ollama.test", "qwen", 5, api_key=None,
                                 num_predict=300, seed=1, keep_alive="30m")
        assert isinstance(client, llm.OllamaClient)

    def test_openai_builds_the_compat_client(self):
        client = llm.make_client("openai", "https://api.test/v1", "m", 5, api_key="k",
                                 num_predict=300, seed=1, keep_alive="30m")
        assert isinstance(client, llm.OpenAICompatClient) and client.model == "m"

    def test_openai_without_a_key_is_refused(self):
        with pytest.raises(ValueError, match="LLM_API_KEY"):
            llm.make_client("openai", "https://api.test/v1", "m", 5, api_key=None,
                            num_predict=300, seed=1, keep_alive="30m")

    def test_an_unknown_provider_is_refused(self):
        with pytest.raises(ValueError, match="provider"):
            llm.make_client("claude", "x", "m", 5, api_key="k",
                            num_predict=300, seed=1, keep_alive="30m")


class TestRequestOptions:
    CFG = {"hosted_request": {"default": {"temperature": 0},
                              "gpt-x-nano-2026-01-01": {"reasoning_effort": "none"}}}

    def test_a_listed_model_gets_its_own_options(self):
        options = llm.request_options(self.CFG, "gpt-x-nano-2026-01-01")
        assert options == {"reasoning_effort": "none"}

    def test_an_unlisted_model_gets_the_default(self):
        assert llm.request_options(self.CFG, "other") == {"temperature": 0}

    def test_returns_a_copy_so_callers_cannot_edit_the_config(self):
        llm.request_options(self.CFG, "other")["temperature"] = 1
        assert self.CFG["hosted_request"]["default"] == {"temperature": 0}

    def test_the_shipped_config_has_a_default_entry(self):
        assert "default" in load_config("models")["llm"]["hosted_request"]

    def test_make_client_forwards_the_options(self):
        client = llm.make_client("openai", "https://api.test/v1", "m", 5, api_key="k",
                                 num_predict=300, seed=1, keep_alive="30m",
                                 options={"reasoning_effort": "none"})
        assert client.options == {"reasoning_effort": "none"}
