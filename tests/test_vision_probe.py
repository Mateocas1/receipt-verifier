"""Vision-probe tests: one synthetic PNG, fake transports, no network."""

from __future__ import annotations

from receipt_verifier.extractors.llm import SYSTEM_PROMPT, LlmCompletion, LlmTransportError
from receipt_verifier.vision_probe import (
    NON_CHAT_MODELS,
    PROBE_MARKER,
    VisionProbe,
    build_probe_image,
    chat_candidates,
    looks_like_amount,
    parse_model_ids,
    run_vision_probe,
)


class AnsweringTransport:
    """Returns one queued answer and remembers the request."""

    def __init__(self, answer: str = PROBE_MARKER) -> None:
        self.answer = answer
        self.calls: list[dict[str, object]] = []

    def complete(
        self,
        *,
        model: str,
        system: str,
        prompt: str,
        image_base64: str,
        media_type: str,
        timeout_seconds: float,
    ) -> LlmCompletion:
        self.calls.append(
            {
                "model": model,
                "system": system,
                "prompt": prompt,
                "image_base64": image_base64,
                "media_type": media_type,
                "timeout_seconds": timeout_seconds,
            }
        )
        return LlmCompletion(text=self.answer, prompt_tokens=10, completion_tokens=2)


class FailingTransport:
    def complete(self, **kwargs: object) -> LlmCompletion:
        raise LlmTransportError("bad request: model does not support images")


class TestInventory:
    def test_parses_the_openai_envelope(self) -> None:
        body = {"data": [{"id": "gemma4"}, {"id": "qwen3.6"}], "object": "list"}
        assert parse_model_ids(body) == ("gemma4", "qwen3.6")

    def test_parses_a_bare_list(self) -> None:
        assert parse_model_ids(["a", "b"]) == ("a", "b")

    def test_ignores_entries_without_an_id(self) -> None:
        body = {"data": [{"id": "a"}, {"object": "model"}, {"id": "  "}, "b", 7]}
        assert parse_model_ids(body) == ("a", "b")

    def test_malformed_body_is_empty(self) -> None:
        assert parse_model_ids({"data": "nope"}) == ()
        assert parse_model_ids(None) == ()

    def test_non_chat_models_are_not_candidates(self) -> None:
        listed = ("gemma4", *NON_CHAT_MODELS, "minimax-h3")
        assert chat_candidates(listed) == ("gemma4", "minimax-h3")


class TestProbeImage:
    def test_image_is_a_real_png(self) -> None:
        data = build_probe_image()
        assert data.startswith(b"\x89PNG\r\n\x1a\n")
        assert len(data) > 500


class TestAnswerVerdict:
    def test_digits_only(self) -> None:
        assert looks_like_amount("25000")

    def test_printed_form(self) -> None:
        assert looks_like_amount("25.000,00")

    def test_with_surrounding_words(self) -> None:
        assert looks_like_amount("The total is 25.000,00 ARS")

    def test_refusal_is_not_vision(self) -> None:
        assert not looks_like_amount("I cannot read the image")

    def test_wrong_number_is_not_vision(self) -> None:
        assert not looks_like_amount("1000")


class TestRunProbe:
    def test_readable_image_is_vision_capable(self) -> None:
        transport = AnsweringTransport("25.000")
        probe = run_vision_probe(
            model="gemma4", transport=transport, image=build_probe_image(), timeout_seconds=30.0
        )
        assert probe.vision is True
        assert probe.error == ""
        assert transport.calls[0]["model"] == "gemma4"
        assert transport.calls[0]["media_type"] == "image/png"
        assert transport.calls[0]["image_base64"]

    def test_probe_uses_the_real_extraction_prompt(self) -> None:
        transport = AnsweringTransport('{"amount": "25.000,00"}')
        probe = run_vision_probe(
            model="qwen3.6", transport=transport, image=build_probe_image(), timeout_seconds=30.0
        )
        assert probe.vision is True
        assert transport.calls[0]["system"] == SYSTEM_PROMPT
        assert "JSON" in str(transport.calls[0]["prompt"])

    def test_refusal_is_recorded_not_raised(self) -> None:
        probe = run_vision_probe(
            model="whisper",
            transport=AnsweringTransport("no"),
            image=build_probe_image(),
            timeout_seconds=30.0,
        )
        assert probe.vision is False
        assert probe.error == ""

    def test_transport_failure_becomes_a_result(self) -> None:
        probe = run_vision_probe(
            model="gemma4",
            transport=FailingTransport(),
            image=build_probe_image(),
            timeout_seconds=30.0,
        )
        assert probe.vision is False
        assert "does not support images" in probe.error
        assert isinstance(probe, VisionProbe)
