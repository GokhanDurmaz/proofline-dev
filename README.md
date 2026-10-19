# MLX GSQ — Local Coding Agent with Evidence-Based Source Validation

## Overview

MLX GSQ is a local coding agent designed to inspect software projects using an OpenAI-compatible API served by a local MLX model. The current setup uses:

- **Model:** `mlx-community/Mellum2-12B-A2.5B-Instruct-4bit`
- **API endpoint:** `http://127.0.0.1:8000/v1/chat/completions`
- **Runtime:** Python
- **Test project:** JMon, a JVM monitoring project

The primary goal is to produce reliable, evidence-based source-code reviews with verified project-relative file paths and line references, while avoiding fabricated implementation details, unsupported claims, and unverified test results.

The agent is intended to run locally, inspect source code through tools, and produce concise architectural findings grounded in actual source lines.

## Current Architecture

The agent uses an iterative model/tool interaction loop.

Available tools:

- `list_files` — discovers project files and directories.
- `read_file` — reads source files and returns numbered lines.
- `write_file` — writes project files when edit mode permits it.
- `run_command` — executes permitted commands when edit mode permits it.

In read-only mode, the agent can inspect files but cannot use the write or command tools to modify the target project or execute project commands.

### Source evidence collection

The agent extracts evidence from successful `read_file` results. Each evidence entry contains:

- Project-relative file path
- Total source-file line count
- Line numbers actually returned
- Source text associated with each returned line

Current limits include:

- Maximum 20 model iterations
- Final 4 iterations reserved for finalization
- Maximum 8 source files in the normal evidence collection flow
- Maximum 30,000 characters per source file
- Maximum 12,000 characters for tool output

These limits help constrain inspection and model context usage.

### Citation validation

The current `validate_line_references()` function performs structural checks on citations. It checks whether:

- A cited file exists in the collected evidence.
- A file reference is ambiguous.
- The cited line range is within the file's reported bounds.
- Every cited line was actually returned by `read_file`.
- The returned evidence includes the source text for cited lines.
- Required citations are present in tasks that explicitly request source references.

The validator recognizes several citation formats, including:

`src/main/java/io/jmon/core/JvmMetricsProvider.java:3-5`

and:

`File: JvmMetricsProvider.java, lines 3-5`

## Confirmed Validation Gap

A test using synthetic source evidence demonstrated that the structural validator can accept a report containing an unsupported semantic claim.

Example source evidence:

```java
public interface JvmMetricsProvider {
    JvmMetricsSnapshot collect();
}
```

The report claimed that the provider runs asynchronously and never blocks the UI thread, citing the interface declaration.

The cited lines establish that the interface declares a `collect()` method returning a `JvmMetricsSnapshot`. They do not establish asynchronous execution, thread scheduling, executor usage, or UI-thread behavior.

Nevertheless, the current structural validator returned an empty issue list:

```text
Validation issues: []
```

This result confirms a distinction between two kinds of validation:

1. **Structural validation:** Does the citation refer to source lines that were actually read?
2. **Semantic validation:** Do those source lines genuinely support the claim made in the report?

The first is implemented. The second remains an identified gap.

## Development Objective: Semantic Claim-Evidence Audit

The next development phase is to add a semantic validation step before the final report is printed.

The proposed flow is:

1. The agent inspects the target project and collects source evidence.
2. The model generates a draft report with file and line references.
3. The existing structural citation validator checks the references.
4. A separate audit call sends the draft and collected source evidence to the same local model.
5. The audit evaluates whether the report's implementation claims are supported by the supplied source lines.
6. Unsupported or uncertain claims must not be treated as verified.
7. The report is either corrected and revalidated or rejected when the validation process cannot establish sufficient support.

The semantic auditor should call the existing API through `request_model(messages, allow_tools=False)`, so it cannot request project tools during the audit.

The audit should use a strict, machine-parseable response format. Malformed output, missing verdicts, or uncertain results must not silently count as successful validation.

### Important limitation

The auditor uses the same model as the report generator. It is therefore a defense-in-depth mechanism, not an independent proof of correctness. It may overlook unsupported claims or make incorrect judgments itself.

The design should favor conservative failure handling, preserve the original evidence, and never present an audit result as a guarantee of correctness.

## Additional Issues to Address

### Evidence file limit

The normal evidence collection flow defines `MAX_SOURCE_FILES = 8`. However, the existing `prefetch_missing_read_evidence()` path can add cited source evidence without enforcing the same file-count limit.

This should be reviewed so that citation prefetching cannot bypass the intended source-file budget.

### Final response paths

The final-response flow contains multiple paths that print a report and return success, including paths reached after citation revalidation.

The semantic audit must cover **every successful final-report path**, rather than only the main path. Otherwise, some reports could bypass semantic validation.

### Regression tests

Tests should cover at least:

- A supported claim with a valid citation.
- An unsupported claim with a structurally valid citation.
- A claim that exceeds what an interface or declaration proves.
- A malformed semantic-audit response.
- A missing or ambiguous verdict.
- An audit that reports uncertainty.
- A model response that requests tools during the audit.
- A report that fails semantic validation and must not be printed as a successful result.
- Citation-prefetch behavior at the source-file limit.
- All final-response paths, including correction and retry paths.

Tests should use synthetic evidence where possible and should not modify the JMon test project.

## Working Rules

The development process follows these constraints:

- Modify `local_agent.py` only when testing or implementing agent changes.
- Keep JMon read-only during agent-validation experiments.
- Preserve a verified backup before substantive changes.
- Run Python syntax checks and relevant regression tests after modifications.
- Verify outcomes from actual command output.
- Do not claim that a test passed unless it was executed and its output confirms success.
- Keep changes focused and reversible.
- Validate file references and semantic support separately.

## Current Status

| Area | Status |
|---|---|
| Local MLX API integration | Existing |
| File inspection tools | Existing |
| Source evidence extraction | Existing |
| Structural citation validation | Existing |
| Reproduction of unsupported-claim acceptance | Confirmed |
| Semantic claim-evidence audit | Planned; not yet implemented |
| Enforcement of file limits during citation prefetch | Needs review |
| Regression suite for semantic validation | Pending |

## Definition of Done

The semantic validation work will be considered ready for further evaluation when:

- Unsupported claims with otherwise valid citations are detected in regression tests.
- Valid claims supported by the supplied evidence are not systematically rejected.
- Invalid, malformed, or uncertain audit results cannot bypass validation.
- Every successful final-report path runs the semantic audit when required.
- Source evidence limits are consistently enforced.
- Syntax checks and regression tests have been executed and their results recorded.
- JMon remains unchanged throughout the agent-validation process.

The overall objective is not simply to generate citations, but to make the agent distinguish between **a citation that points to real source code** and **a claim that the source code actually proves**.
