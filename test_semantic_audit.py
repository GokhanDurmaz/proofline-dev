import json
import unittest

from semantic_audit import (
    audit_report,
    collect_cited_source,
    extract_cited_report_claims,
    validate_audit_json,
)


class SemanticAuditTests(unittest.TestCase):
    def setUp(self):
        self.path = "src/main/java/JvmMetricsProvider.java"
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
        self.claim = (
            "The provider runs asynchronously and never blocks the UI thread."
        )
        self.report = (
            self.claim + " (src/main/java/JvmMetricsProvider.java:1-3)"
        )

    def response(self, payload, finish_reason="stop"):
        return {
            "choices": [{
                "finish_reason": finish_reason,
                "message": {"content": json.dumps(payload)},
            }]
        }

    def test_interface_declaration_does_not_support_async_claim(self):
        def mock_model(messages, **kwargs):
            self.assertIs(kwargs.get("allow_tools"), False)
            self.assertEqual(kwargs.get("response_format"), {"type": "json_object"})
            sent = json.loads(messages[1]["content"])
            self.assertEqual(sent["cited_source_lines"][1]["text"],
                             "    JvmMetricsSnapshot collect();")
            return self.response({
                "claims": [{
                    "claim": self.claim,
                    "status": "unsupported",
                    "reason": "The interface signature does not establish scheduling or UI-thread behavior.",
                }]
            })

        accepted, reason = audit_report(self.report, self.evidence, mock_model)
        self.assertFalse(accepted)
        self.assertIn("unsupported", reason)

    def test_uncertain_claim_fails_closed(self):
        def mock_model(messages, **kwargs):
            return self.response({
                "claims": [{
                    "claim": self.claim,
                    "status": "uncertain",
                    "reason": "No implementation evidence is provided.",
                }]
            })

        accepted, _ = audit_report(self.report, self.evidence, mock_model)
        self.assertFalse(accepted)

    def test_truncated_model_response_fails_closed(self):
        def mock_model(messages, **kwargs):
            return self.response({"claims": []}, finish_reason="length")

        accepted, reason = audit_report(self.report, self.evidence, mock_model)
        self.assertFalse(accepted)
        self.assertIn("truncated", reason)

    def test_malformed_json_fails_closed(self):
        def mock_model(messages, **kwargs):
            return {"choices": [{
                "finish_reason": "stop",
                "message": {"content": '{"claims":'},
            }]}

        accepted, _ = audit_report(self.report, self.evidence, mock_model)
        self.assertFalse(accepted)

    def test_unread_cited_line_is_rejected_before_model_call(self):
        incomplete = {
            self.path: {
                "total_lines": 3,
                "lines": {1, 2},
                "line_text": {1: "interface", 2: "collect();"},
            }
        }

        def must_not_run(*args, **kwargs):
            self.fail("Model should not be called for unavailable source evidence.")

        accepted, reason = audit_report(self.report, incomplete, must_not_run)
        self.assertFalse(accepted)
        self.assertIn("was not read", reason)

    def test_unknown_or_non_verbatim_claim_is_rejected(self):
        with self.assertRaises(ValueError):
            validate_audit_json(json.dumps({
                "claims": [{
                    "claim": "An invented claim.",
                    "status": "supported",
                    "reason": "Looks plausible.",
                }]
            }), self.report)

    def test_collects_only_cited_exact_lines(self):
        source = collect_cited_source(self.report, self.evidence)
        self.assertEqual([item["line"] for item in source], [1, 2, 3])


    def test_omitted_report_claim_fails_closed(self):
        second_claim = "The collector uses a background thread."
        report = (
            self.claim + " (src/main/java/JvmMetricsProvider.java:1-3) "
            + second_claim + " (src/main/java/JvmMetricsProvider.java:1-3)"
        )

        def incomplete_model(messages, **kwargs):
            return self.response({
                "claims": [{
                    "claim": self.claim,
                    "status": "supported",
                    "reason": "The model claims this is supported.",
                }]
            })

        accepted, reason = audit_report(report, self.evidence, incomplete_model)
        self.assertFalse(
            accepted,
            "Audit must fail closed when the model omits a substantive report claim.",
        )


    def test_sentence_period_before_citation_is_preserved(self):
        report = (
            "The provider runs asynchronously and never blocks the UI thread. "
            "(src/main/java/JvmMetricsProvider.java:1-3)"
        )
        self.assertEqual(
            extract_cited_report_claims(report),
            ["The provider runs asynchronously and never blocks the UI thread."],
        )

    def test_multiple_cited_sentences_are_extracted_individually(self):
        report = (
            "The provider runs asynchronously. "
            "(src/main/java/JvmMetricsProvider.java:1-3) "
            "The collector uses a background thread. "
            "(src/main/java/JvmMetricsProvider.java:1-3)"
        )
        self.assertEqual(
            extract_cited_report_claims(report),
            [
                "The provider runs asynchronously.",
                "The collector uses a background thread.",
            ],
        )



if __name__ == "__main__":
    unittest.main()
