"""Tests for Fireworks retry and fallback behavior."""

import unittest
from unittest.mock import patch

import httpx
from openai import APIConnectionError, InternalServerError

from src import config, llm


def status_error(status_code: int) -> InternalServerError:
    """Create an OpenAI status error without making a network request."""
    request = httpx.Request("POST", "https://api.fireworks.ai/inference/v1/chat/completions")
    response = httpx.Response(status_code, request=request)
    return InternalServerError("server error", response=response, body=None)


class GatewayReliabilityTests(unittest.TestCase):
    """Keep permanent deployment faults fast and transient failures retryable."""

    def test_retry_policy_matches_documented_transient_errors(self) -> None:
        request = httpx.Request("POST", "https://api.fireworks.ai/inference/v1")

        self.assertFalse(llm._is_retryable(status_error(500)))
        self.assertTrue(llm._is_retryable(status_error(503)))
        self.assertTrue(llm._is_retryable(APIConnectionError(request=request)))

    def test_chat_falls_back_to_serverless_after_deployment_500(self) -> None:
        messages = [{"role": "user", "content": "hello"}]

        with (
            patch.object(llm, "MOCK", False),
            patch.object(
                llm,
                "_call",
                side_effect=[status_error(500), ("fallback response", 12, 4)],
            ) as call,
            patch.object(llm.budget, "check"),
            patch.object(llm.budget, "record") as record,
        ):
            result = llm.chat("dedicated-model", messages, "perceive")

        self.assertEqual(result, "fallback response")
        self.assertEqual(call.call_args_list[0].args[0], "dedicated-model")
        self.assertEqual(call.call_args_list[1].args[0], config.SERVERLESS_FALLBACK_MODEL)
        record.assert_called_once_with(
            config.SERVERLESS_FALLBACK_MODEL,
            12,
            4,
            "perceive",
        )

    def test_chat_uses_only_evaluator_allowed_models(self) -> None:
        messages = [{"role": "user", "content": "hello"}]
        allowed = ["accounts/fireworks/models/allowed-a", "allowed-b"]

        with (
            patch.object(llm, "MOCK", False),
            patch.object(config, "ALLOWED_MODELS", allowed),
            patch.object(
                llm,
                "_call",
                side_effect=[status_error(500), ("allowed response", 7, 3)],
            ) as call,
            patch.object(llm.budget, "check"),
            patch.object(llm.budget, "record"),
        ):
            result = llm.chat("private-deployment", messages, "perceive")

        self.assertEqual(result, "allowed response")
        self.assertEqual(
            [item.args[0] for item in call.call_args_list],
            allowed,
        )


class AllowedModelConfigurationTests(unittest.TestCase):
    def test_parses_json_and_comma_separated_allow_lists(self) -> None:
        self.assertEqual(
            config.parse_allowed_models('["model-a", "model-b", "model-a"]'),
            ["model-a", "model-b"],
        )
        self.assertEqual(
            config.parse_allowed_models("model-a, model-b"),
            ["model-a", "model-b"],
        )


if __name__ == "__main__":
    unittest.main()
