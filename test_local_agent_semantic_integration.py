import ast
import contextlib
import importlib.machinery
import importlib.util
import io
import json
import unittest
from pathlib import Path
from unittest.mock import patch

CANDIDATE = Path("local_agent.py")


def load_candidate():
    loader = importlib.machinery.SourceFileLoader(
        "candidate_agent_under_test", str(CANDIDATE.resolve())
    )
    spec = importlib.util.spec_from_loader(loader.name, loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


class SemanticIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.agent = load_candidate()
        self.path = "src/main/java/JvmMetricsProvider.java"
        self.claim = (
            "The provider runs asynchronously and never blocks the UI thread."
        )
        self.report = (
            self.claim + f" ({self.path}:1-3)"
        )
        self.evidence = {
            self.path: {
                "total_lines": 3,
                "lines": {1, 2, 3},
                "line_text": {
                    1: "public interface JvmMetricsProvider {",
                    2: "    JvmMetricsSnapshot collect();",
                    3: "}",
                },
            }
        }

    def model_response(self, status):
        return {
            "choices": [{
                "finish_reason": "stop",
                "message": {
                    "content": json.dumps({
                        "claims": [{
                            "claim": self.claim,
                            "status": status,
                            "reason": (
                                "The interface declaration does not establish "
                                "thread scheduling or UI-thread behavior."
                                if status != "supported"
                                else "Direct evidence supports the claim."
                            ),
                        }]
                    })
                },
            }]
        }

    def test_unsupported_claim_is_rejected_and_report_suppressed(self):
        output = io.StringIO()
        errors = io.StringIO()
        with patch.object(
            self.agent, "request_model",
            return_value=self.model_response("unsupported"),
        ) as mock_request:
            with contextlib.redirect_stdout(output), contextlib.redirect_stderr(errors):
                accepted = self.agent.accept_final_report(
                    self.report, self.evidence,
                    "Inspect source and cite exact file and line references",
                )

        self.assertFalse(accepted)
        self.assertNotIn(self.report, output.getvalue())
        self.assertIn("semantic claim-evidence audit rejected", errors.getvalue())
        self.assertEqual(mock_request.call_count, 1)
        kwargs = mock_request.call_args.kwargs
        self.assertIs(kwargs["allow_tools"], False)
        self.assertEqual(kwargs["response_format"], {"type": "json_object"})
        self.assertEqual(kwargs["max_tokens"], 1200)

    def test_supported_claim_is_printed(self):
        output = io.StringIO()
        with patch.object(
            self.agent, "request_model",
            return_value=self.model_response("supported"),
        ):
            with contextlib.redirect_stdout(output):
                accepted = self.agent.accept_final_report(
                    self.report, self.evidence,
                    "Inspect source and cite exact file and line references",
                )

        self.assertTrue(accepted)
        self.assertIn(self.report, output.getvalue())

    def test_non_citation_task_keeps_existing_behavior(self):
        output = io.StringIO()
        with patch.object(
            self.agent, "request_model",
            side_effect=AssertionError("Audit should not run for this task"),
        ):
            with contextlib.redirect_stdout(output):
                accepted = self.agent.accept_final_report(
                    "A general answer.", {}, "Explain what a JVM is"
                )

        self.assertTrue(accepted)
        self.assertIn("A general answer.", output.getvalue())

    def test_all_three_final_paths_use_shared_gate(self):
        tree = ast.parse(CANDIDATE.read_text())
        main = next(
            node for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "main"
        )
        calls = [
            node for node in ast.walk(main)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "accept_final_report"
        ]
        self.assertEqual(len(calls), 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)


class RequestModelPayloadTests(unittest.TestCase):
    def setUp(self):
        self.agent = load_candidate()
        self.messages = [{"role": "user", "content": "test"}]

    def invoke_with_mock_response(self, **kwargs):
        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self):
                return b'{"choices": []}'

        with patch.object(
            self.agent.urllib.request,
            "urlopen",
            return_value=FakeResponse(),
        ) as mock_urlopen:
            result = self.agent.request_model(self.messages, **kwargs)

        request = mock_urlopen.call_args.args[0]
        timeout = mock_urlopen.call_args.kwargs["timeout"]
        payload = json.loads(request.data.decode("utf-8"))
        return result, request, timeout, payload

    def test_default_payload_preserves_existing_behavior(self):
        _, request, timeout, payload = self.invoke_with_mock_response()

        self.assertEqual(request.get_method(), "POST")
        self.assertEqual(request.get_header("Content-type"), "application/json")
        self.assertEqual(timeout, 300)
        self.assertEqual(payload["model"], self.agent.MODEL)
        self.assertEqual(payload["messages"], self.messages)
        self.assertEqual(payload["temperature"], 0)
        self.assertEqual(payload["max_tokens"], 512)
        self.assertNotIn("response_format", payload)
        self.assertEqual(payload["tools"], self.agent.TOOLS
                         if self.agent.POLICY_MODE == "edit"
                         else [
                             tool for tool in self.agent.TOOLS
                             if tool["function"]["name"] in {"list_files", "read_file"}
                         ])
        self.assertEqual(payload["tool_choice"], "auto")

    def test_response_format_is_omitted_when_unspecified(self):
        _, _, _, payload = self.invoke_with_mock_response(
            allow_tools=False,
        )
        self.assertNotIn("response_format", payload)
        self.assertNotIn("tools", payload)
        self.assertNotIn("tool_choice", payload)

    def test_semantic_audit_payload_disables_tools_and_requests_json(self):
        _, _, _, payload = self.invoke_with_mock_response(
            allow_tools=False,
            response_format={"type": "json_object"},
            max_tokens=1200,
        )

        self.assertEqual(payload["response_format"], {"type": "json_object"})
        self.assertEqual(payload["max_tokens"], 1200)
        self.assertNotIn("tools", payload)
        self.assertNotIn("tool_choice", payload)

    def test_explicit_tool_disabling_does_not_change_message_payload(self):
        _, _, _, payload = self.invoke_with_mock_response(
            allow_tools=False,
            max_tokens=1200,
        )

        self.assertEqual(payload["messages"], self.messages)
        self.assertEqual(payload["temperature"], 0)
        self.assertEqual(payload["max_tokens"], 1200)
        self.assertNotIn("tools", payload)
        self.assertNotIn("tool_choice", payload)
