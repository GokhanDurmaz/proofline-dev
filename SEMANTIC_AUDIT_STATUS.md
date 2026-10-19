# Semantic Audit — Development Status

**Date:** 2026-10-19
**Status:** Unit/integration tests pass; real-model end-to-end validation remains unresolved.

## Goal

Prevent the coding agent from presenting a source-inspection report unless its cited claims are checked against the exact source lines successfully read during the task.

## Implemented

- Added semantic claim extraction and claim-to-evidence audit helpers.
- The audit receives the exact cited source evidence collected by the agent.
- The audit requests structured JSON and disables tools during the audit-model request.
- Malformed or truncated audit responses, uncertain claims, unsupported claims, omitted claims, and citations outside the successfully read evidence are rejected.
- The shared final-report gate is used by all three final-report paths.
- Integration tests target `local_agent.py` directly rather than a separate semantic-candidate copy.

## Automated test results

On 2026-10-19, Python syntax compilation completed successfully and all **18 unit/integration tests passed**.

## Real-model end-to-end test

The agent was run against JMon in read-only mode, requesting 3–5 concrete findings with verified project-relative file paths and line references.

Observed result:

- The agent successfully read the project guidance files.
- The source-reading tool reached its limit of 8 distinct successfully read source files.
- The initial final report did not include the required citations.
- During correction, the model produced citations to files/lines that were not present in the successfully read evidence.
- The source-reference validator rejected the response.
- The final report was suppressed, and the run ended with an unresolved source-reference validation error.

**Outcome:** Fail-closed behavior was observed in this run, but a successful end-to-end report has not yet been demonstrated. The JMon project was intended to be read-only; no JMon edits were part of this test.

## Known issues and follow-up work

1. **Evidence budget and selection:** Inspect how the 8-file read limit interacts with the task and how the agent chooses source files.
2. **Citation correction:** Inspect the correction prompt and evidence passed to it. The model must not invent citations to unread files or lines.
3. **Evidence reuse:** Ensure retries use the exact successfully read evidence and do not imply that an attempted or blocked read succeeded.
4. **End-to-end regression test:** Re-run the real local model against JMon in read-only mode and verify that any accepted findings cite exact lines that were successfully read.
5. **Development scripts:** Review `apply_policy_patch.py`, `build_policy_candidate.py`, and `patch_read_file.py` before treating them as supported workflows. They use source-text replacements and can generate or modify code; check match counts and inspect diffs before running them.
6. **Server setup:** Document the required Python dependencies and launch procedure for `server.py` so the local model API can be reproduced on another development machine.

## Safety and interpretation

- A rejected report is preferable to publishing unsupported findings, but it is not proof that the audit model is semantically correct in every case.
- Passing unit/integration tests is not equivalent to successful real-model end-to-end validation.
- Do not claim that the end-to-end issue is fixed until a real run demonstrates a report with verified citations.
- Keep JMon read-only during these regression tests unless a separate change is explicitly authorized.
