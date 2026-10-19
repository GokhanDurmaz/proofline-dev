#!/usr/bin/env python3
"""Local coding agent for the MLX GSQ-RCO OpenAI-compatible API."""

import json
import shlex
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

API_URL = "http" + "://127.0.0.1:8000/v1/chat/completions"
MODEL = "mlx-community/Mellum2-12B-A2.5B-Instruct-4bit"
MAX_ITERATIONS = 20
FINALIZATION_ITERATIONS = 4
MAX_SOURCE_FILES = 8
MAX_FILE_CHARS = 30_000
MAX_OUTPUT_CHARS = 12_000

IGNORED = {
    ".git", ".venv", "venv", "node_modules", "build",
    "dist", ".gradle", "__pycache__", ".idea",
}

SYSTEM_PROMPT = """
You are a coding agent working on the user's local project.

Rules:
- Inspect relevant files before proposing changes.
- Use tools to read actual source; never invent file contents or test results.
- All file paths passed to tools must be relative to the project root.
- Use paths exactly as returned by list_files; never pass absolute paths.
- If read_file returns File not found, use list_files to locate the file, then retry with the verified relative path.
- Before reading implementation files, use list_files to discover the actual project layout instead of guessing package paths.
- Never emit tool-call JSON or pseudo-JSON as assistant text. Use native tool_calls, and wait for tool results before making claims.
- Make small, focused changes that satisfy the user's task.
- Explain the purpose of changes briefly.
- Run relevant tests/build checks after editing, when appropriate.
- Read command output carefully. If a check fails, diagnose and try a
  focused fix, respecting the iteration limit.
- Never claim a command or test succeeded unless its output confirms it.
- Do not attempt destructive commands, access secrets, or read files outside
  the project root.
- Ask the user for clarification if requirements are materially ambiguous.
- File writes and terminal commands require user approval.
- Finish with a concise summary of changes and checks performed.
- You MUST invoke tools through native tool_calls, not by writing tool-call
  JSON or pseudo-JSON in the assistant's text.
- Available tools are list_files, read_file, write_file, and run_command.
- When you need a tool, make an actual tool call with the exact tool name
  and valid JSON arguments matching its schema.
- Never pretend that a tool was called. Wait for the actual tool result
  before describing what you learned from it.
- Source reviews: inspect the actual source before making claims.
- When citations are requested, inspect relevant implementation and test files, not only project documentation. Cite only lines returned by read_file. If source evidence is missing, say the claim is unverified.
- For full-file reviews, verify that consecutive returned source ranges cover the file without gaps or overlap.
- Source line numbers are the numeric prefixes in read_file content. Never use tool arguments, start_line values, or iteration numbers as source line numbers.
- Before citing a line, verify that the actual source line supports the claim. Use the narrowest accurate range.
- Answer every requested review topic separately. If a behavior is absent, say so; do not infer that it exists.
- Distinguish requesting process termination from confirming that the process has exited.
- Separate confirmed facts from risks and assumptions. Do not present risks as proven defects without source evidence.
- Never claim tests or commands ran unless tool output confirms execution.
- Before answering, verify every requested topic and every cited source line number.
- Before finalizing a source review, compare each claimed line number with the actual numbered source content returned by read_file. Tool arguments and chunk boundaries are not source evidence. If a claim cannot be verified against the returned source lines, correct it or state that it is unverified. Never infer behavior from a closing brace or from a line that does not contain the claimed operation.
- Treat the output of list_files as the authoritative project file inventory. Never guess a path that is not listed; if a listed path fails, search the inventory once and then move on.
- For a source-architecture review requesting 3-5 findings, prioritize 3-5 directly relevant implementation files and read only the additional files needed to verify specific claims.
- Do not read unrelated tests, documentation, or neighboring classes once the selected findings have sufficient source evidence.
- After each tool result, assess whether the requested findings can already be supported. If they can, stop calling tools and produce the final report.
- Reserve the final four iterations for synthesizing findings and checking citations; do not start a new file inspection in those iterations.
- If evidence remains incomplete near the iteration limit, report only verified findings and explicitly identify what could not be verified.
"""


def project_path(root: Path, relative: str) -> Path:
    candidate = (root / relative).resolve()
    if candidate != root and root not in candidate.parents:
        raise ValueError("Path escapes the project root.")
    return candidate


def list_files(root: Path, path: str = ".") -> str:
    base = project_path(root, path)
    if not base.is_dir():
        return f"Not a directory: {path}"

    results = []
    for item in sorted(base.rglob("*")):
        if any(part in IGNORED for part in item.relative_to(root).parts):
            continue
        if item.is_file():
            results.append(str(item.relative_to(root)))
        if len(results) >= 250:
            results.append("... listing truncated at 250 files")
            break

    return "\n".join(results) if results else "(no files found)"


def read_file(
    root: Path,
    path: str,
    start_line: int = 1,
    max_lines: int = 200,
) -> str:
    file_path = project_path(root, path)

    if not file_path.is_file():
        return f"File not found: {path}"
    if file_path.stat().st_size > 1_000_000:
        return "File is over 1 MB; inspect a smaller file instead."
    if start_line < 1:
        return "Invalid start_line: must be at least 1."
    if not 1 <= max_lines <= 500:
        return "Invalid max_lines: must be between 1 and 500."

    try:
        lines = file_path.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError:
        return "File is not UTF-8 text."

    total_lines = len(lines)
    if start_line > total_lines and not (
        total_lines == 0 and start_line == 1
    ):
        return (
            f"start_line {start_line} exceeds the file's "
            f"{total_lines} lines."
        )

    end_index = min(start_line - 1 + max_lines, total_lines)
    selected = lines[start_line - 1:end_index]
    numbered = [
        f"{index}: {line}"
        for index, line in enumerate(selected, start=start_line)
    ]

    header = (
        f"File: {path}\n"
        f"Total lines: {total_lines}\n"
        f"Returned lines: "
        f"{start_line}-{end_index if selected else start_line - 1}\n"
    )
    result = header + "\n".join(numbered)

    if end_index < total_lines:
        result += (
            f"\n[TRUNCATED: more lines remain. "
            f"Read again with start_line={end_index + 1}.]"
        )

    if len(result) > MAX_FILE_CHARS:
        result = result[:MAX_FILE_CHARS]
        result += (
            "\n[TRUNCATED BY CHARACTER LIMIT: "
            "request a smaller max_lines range.]"
        )

    return result


def approve(action: str) -> bool:
    answer = input(f"\nApprove {action}? [y/N] ").strip().lower()
    return answer in {"y", "yes"}


def write_file(root: Path, path: str, content: str) -> str:
    target = project_path(root, path)

    if target == root:
        return "Refusing to overwrite the project root."
    if len(content) > 200_000:
        return "File content is too large (maximum 200,000 characters)."

    action = "create" if not target.exists() else "overwrite"
    print(f"\nFile {action}: {target.relative_to(root)}")
    print(f"New content size: {len(content)} characters")

    if not approve("this file change"):
        return "User declined the file change."

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return f"Saved {target.relative_to(root)} ({len(content)} characters)."


def command_is_allowed(argv: list[str]) -> bool:
    if not argv:
        return False

    # Never invoke a shell; also reject shell-control characters explicitly.
    if any(c in argv[0] for c in ";&|><`$"):
        return False

    executable = argv[0]
    args = argv[1:]

    if any(c in arg for c in ";&|><`" for arg in args):
        return False

    if executable == "git":
        return bool(args) and args[0] in {
            "status", "diff", "log", "show", "branch", "ls-files"
        }

    if executable in {"./gradlew", "gradle"}:
        allowed_tasks = {
            "test", "check", "build", "assemble", "lint",
            "testDebugUnitTest", "testReleaseUnitTest",
        }
        return any(arg in allowed_tasks for arg in args) and not any(
            arg in {"clean", "--init-script", "-I"} for arg in args
        )

    if executable in {"./mvnw", "mvn"}:
        return bool(args) and args[0] in {"test", "verify", "package", "compile"}

    if executable == "npm":
        return len(args) >= 2 and args[0] == "run" and args[1] in {
            "test", "build", "lint", "typecheck"
        }

    if executable == "pytest":
        return True

    if executable in {"python", "python3"}:
        return len(args) >= 2 and args[0] == "-m" and args[1] in {
            "pytest", "unittest", "compileall"
        }

    if executable == "make":
        return not args or args[0] in {"test", "build", "check"}

    return False


POLICY_MODE = "read-only"
EXPLICIT_COMMANDS = []


def run_command(root: Path, command: str) -> str:
    try:
        argv = shlex.split(command)
    except ValueError as exc:
        return f"Could not parse command: {exc}"

    if not command_is_allowed(argv):
        return (
            "Command blocked by the agent's command allowlist. "
            "Allowed examples: git status, git diff, ./gradlew test, "
            "./gradlew build, pytest, python -m pytest, npm run test."
        )

    try:
        permitted = [shlex.split(item) for item in EXPLICIT_COMMANDS]
    except ValueError as exc:
        return f"Invalid --allow-command value: {exc}"

    if argv not in permitted:
        return ("Command blocked: authorize this exact command with "
                "--allow-command.")

    print(f"\nProject: {root}")
    print(f"Command: {shlex.join(argv)}")

    if not approve("this terminal command"):
        return "User declined the command."

    try:
        result = subprocess.run(
            argv,
            cwd=root,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        output = (
            f"Exit code: {result.returncode}\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )
        if len(output) > MAX_OUTPUT_CHARS:
            output = output[:MAX_OUTPUT_CHARS] + "\n... [output truncated]"
        return output
    except subprocess.TimeoutExpired:
        return "Command timed out after 120 seconds."
    except OSError as exc:
        return f"Could not run command: {exc}"


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List project files under a relative directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Relative directory"}
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": (
                "Read a UTF-8 text file inside the project. "
                "The path must be relative to the project root, not absolute. "
                "Use paths returned by list_files. "
                "Use start_line and max_lines to inspect file chunks; "
                "continue from the next line when output says TRUNCATED."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Relative file path"},
                    "start_line": {
                        "type": "integer",
                        "description": "1-based first line to read; defaults to 1",
                    },
                    "max_lines": {
                        "type": "integer",
                        "description": "Number of lines to read, from 1 to 500; defaults to 200",
                    },
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Create or replace a UTF-8 text file inside the project. User approval is required.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": "Run an allowlisted project inspection, build, or test command. User approval is required.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "One command, without shell chaining",
                    }
                },
                "required": ["command"],
            },
        },
    },
]


def execute_tool(root: Path, name: str, args: dict) -> str:
    try:
        if name == "list_files":
            return list_files(root, args["path"])
        if name == "read_file":
            return read_file(
                root,
                args["path"],
                args.get("start_line", 1),
                args.get("max_lines", 200),
            )
        if name == "write_file":
            if POLICY_MODE != "edit":
                return "Write blocked: agent is running in read-only mode."
            return write_file(root, args["path"], args["content"])
        if name == "run_command":
            if POLICY_MODE != "edit":
                return "Command blocked: terminal commands are disabled in read-only mode."
            return run_command(root, args["command"])
        return f"Unknown tool: {name}"
    except (KeyError, TypeError, ValueError, OSError) as exc:
        return f"Tool error: {type(exc).__name__}: {exc}"


def request_model(messages: list[dict], allow_tools: bool = True) -> dict:
    available_tools = TOOLS if POLICY_MODE == "edit" else [tool for tool in TOOLS if tool["function"]["name"] in {"list_files", "read_file"}]
    payload = {
        "model": MODEL,
        "messages": messages,
        "temperature": 0,
        "max_tokens": 512,
    }
    if allow_tools:
        payload["tools"] = available_tools
        payload["tool_choice"] = "auto"
    request = urllib.request.Request(
        API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Could not reach {API_URL}. Is the model server running? {exc}"
        ) from exc


def extract_read_evidence(result: str) -> dict:
    """Extract complete source lines returned by read_file."""
    import re

    header = re.search(
        r"(?m)^File: (.+)\nTotal lines: (\d+)\n"
        r"Returned lines: (\d+)-(\d+)\n",
        result,
    )
    if not header:
        return {}

    path, total, returned_start, returned_end = header.groups()
    total = int(total)
    returned_start = int(returned_start)
    returned_end = int(returned_end)

    if (
        total < 0
        or returned_start < 1
        or returned_end < returned_start - 1
        or returned_end > total
    ):
        return {}

    body = result[header.end():]
    char_truncated = (
        "[TRUNCATED BY CHARACTER LIMIT:" in body
    )

    matches = list(re.finditer(r"(?m)^(\d+): (.*)$", body))
    if char_truncated and matches:
        # The last source line may have been cut off mid-line.
        matches.pop()

    parsed_numbers = [int(match.group(1)) for match in matches]
    if len(parsed_numbers) != len(set(parsed_numbers)):
        return {}

    line_text = {
        int(match.group(1)): match.group(2)
        for match in matches
        if returned_start <= int(match.group(1)) <= returned_end
        and 1 <= int(match.group(1)) <= total
    }

    if not char_truncated:
        expected_numbers = set(range(returned_start, returned_end + 1))
        if set(line_text) != expected_numbers:
            return {}

    return {
        "path": Path(path).as_posix(),
        "total_lines": total,
        "lines": set(line_text),
        "line_text": line_text,
    }


def task_requires_line_references(task: str) -> bool:
    """Detect tasks that explicitly require file and line citations."""
    import re

    patterns = (
        r"\bfile\s*:\s*line\b",
        r"\bline\s+references?\b",
        r"\bsource\s+(?:code\s+)?citations?\b",
        r"\bcite\s+(?:the\s+)?(?:exact\s+)?(?:source|file|line)",
        r"\bexact\s+file\s+and\s+line\b",
        r"\bline\s+ranges?\b",
        r"\bsource\s+references?\b",
        r"\bsatır\s+aralıkları?\b",
        r"\bsatır\s+aralığı\b",
        r"\bsatır\s+referansları?\b",
        r"\bkaynak\s+referansları?\b",
        r"\bdosya\s+ve\s+satır\b",
        r"\bsatır\s+numaraları?\b",
        r"\bkaynak\s+göster",
    )
    return any(re.search(p, task, re.IGNORECASE) for p in patterns)


def prefetch_missing_read_evidence(
    root: Path, issues: list[str], read_evidence: dict
) -> int:
    """Read only citation ranges explicitly reported as missing evidence."""
    import re

    patterns = (
        re.compile(
            r"^(?P<path>(?:[\w.-]+/)*[\w.-]+\.[A-Za-z0-9]+):"
            r"(?P<start>\d+)-(?P<end>\d+): source file was not found "
            r"among successfully read evidence\.$"
        ),
        re.compile(
            r"^(?P<path>(?:[\w.-]+/)*[\w.-]+\.[A-Za-z0-9]+):"
            r"(?P<start>\d+)-(?P<end>\d+): these lines were not returned "
            r"by read_file: \[(?P<missing>[0-9, ]*)\]\.$"
        ),
    )

    requests = {}
    for issue in issues:
        for pattern in patterns:
            match = pattern.fullmatch(issue)
            if not match:
                continue
            cited_path = Path(match.group("path")).as_posix()
            start = int(match.group("start"))
            end = int(match.group("end"))
            if start < 1 or end < start:
                break

            # Do not guess which file a basename-only citation refers to.
            if "/" not in cited_path:
                break

            try:
                target = project_path(root, cited_path)
            except (ValueError, OSError):
                break
            if not target.is_file():
                break

            if "missing" in match.groupdict():
                missing = {
                    int(number)
                    for number in match.group("missing").split(",")
                    if number.strip()
                }
                if not missing or any(line < start or line > end for line in missing):
                    break
                start, end = min(missing), max(missing)

            requests[(cited_path, start, end)] = None
            break

    added = 0
    for (cited_path, start, end) in requests:
        try:
            result = read_file(
                root, cited_path, start_line=start,
                max_lines=min(end - start + 1, 500),
            )
        except (ValueError, OSError):
            continue

        item = extract_read_evidence(result)
        if not item or item["path"] != cited_path:
            continue
        if item["lines"] != set(range(start, min(end, start + 499) + 1)):
            continue

        key = item["path"]
        if key not in read_evidence:
            read_evidence[key] = {
                "total_lines": item["total_lines"],
                "lines": set(),
                "line_text": {},
            }
        read_evidence[key]["total_lines"] = item["total_lines"]
        read_evidence[key]["lines"].update(item["lines"])
        read_evidence[key]["line_text"].update(item["line_text"])
        added += len(item["lines"])

    return added


def validate_line_references(
    content: str, evidence: dict, require_references: bool = False
) -> list[str]:
    """Validate supported source citations against lines actually read."""
    import re

    issues = []
    files = evidence
    patterns = [
        re.compile(
            r"(?P<path>(?:[\w.-]+/)*[\w.-]+\.[A-Za-z0-9]+)"
            r"\s*:\s*(?P<start>\d+)"
            r"(?:\s*[-–—]\s*(?P<end>\d+))?"
        ),
        re.compile(
            r"(?i)\bFile:\s*"
            r"(?P<path>(?:[\w.-]+/)*[\w.-]+\.[A-Za-z0-9]+)"
            r"\s*,?\s*lines?\s+"
            r"(?P<start>\d+)"
            r"(?:\s*[-–—]\s*(?P<end>\d+))?"
        ),
        re.compile(
            r"(?i)[\"']file[\"']\s*:\s*[\"']"
            r"(?P<path>(?:[\w.-]+/)*[\w.-]+\.[A-Za-z0-9]+)"
            r"[\"']\s*,\s*[\"']line[\"']\s*:\s*[\"']?"
            r"(?P<start>\d+)[\"']?"
            r"(?:\s*,\s*[\"']end[\"']\s*:\s*[\"']?"
            r"(?P<end>\d+)[\"']?)?"
        ),
    ]

    matches = []
    for pattern in patterns:
        matches.extend(pattern.finditer(content))
    matches.sort(key=lambda match: match.start())

    # Do not let a valid citation hide malformed JSON-style citations.
    json_file_mentions = list(
        re.finditer(r"""(?i)["']file["']\s*:""", content)
    )
    json_citations = list(patterns[2].finditer(content))
    valid_json_starts = {match.start() for match in json_citations}

    for mention in json_file_mentions:
        if mention.start() not in valid_json_starts:
            issues.append(
                "Malformed or unsupported JSON-style file citation near "
                f"character {mention.start()}."
            )

    for match in matches:
        cited_path = Path(match.group("path")).as_posix()
        start = int(match.group("start"))
        end = int(match.group("end") or start)

        candidates = [
            item for path, item in files.items()
            if path == cited_path
            or path.endswith("/" + cited_path)
            or Path(path).name == cited_path
        ]

        if not candidates:
            issues.append(
                f"{cited_path}:{start}-{end}: "
                "source file was not found among successfully read evidence."
            )
            continue

        if len(candidates) > 1:
            issues.append(
                f"{cited_path}:{start}-{end}: "
                "source path is ambiguous among read files."
            )
            continue

        item = candidates[0]
        if start < 1 or end < start or end > item["total_lines"]:
            issues.append(
                f"{cited_path}:{start}-{end}: line range is outside file bounds."
            )
            continue

        missing = [
            line for line in range(start, end + 1)
            if line not in item["lines"]
        ]
        missing_text = [
            line for line in range(start, end + 1)
            if line not in item.get("line_text", {})
        ]

        if missing:
            issues.append(
                f"{cited_path}:{start}-{end}: "
                f"these lines were not returned by read_file: {missing}."
            )

        if missing_text:
            issues.append(
                f"{cited_path}:{start}-{end}: "
                f"source text is unavailable for lines: {missing_text}."
            )

    if not matches:
        if require_references:
            issues.append(
                "The task requested source line references, but the final "
                "response contains no supported file-and-line citations."
            )
        elif re.search(
            r"(?i)\bFile:\s*[\w./-]+\.[A-Za-z0-9]+"
            r"|[\w./-]+\.[A-Za-z0-9]+\s*:\s*\d+",
            content,
        ):
            issues.append(
                "Source-like references were present, but none matched "
                "the supported citation formats; references were not verified."
            )

    if require_references and matches:
        # Remove citation spans before checking whether the response
        # contains explanatory prose rather than a bibliography alone.
        spans = sorted((m.start(), m.end()) for m in matches)
        merged_spans = []
        for start, end in spans:
            if merged_spans and start <= merged_spans[-1][1]:
                merged_spans[-1] = (
                    merged_spans[-1][0],
                    max(merged_spans[-1][1], end),
                )
            else:
                merged_spans.append((start, end))

        remaining_parts = []
        cursor = 0
        for start, end in merged_spans:
            remaining_parts.append(content[cursor:start])
            cursor = end
        remaining_parts.append(content[cursor:])
        explanatory_text = " ".join(remaining_parts)

        words = re.findall(r"\b[\w'-]+\b", explanatory_text)
        if len(words) < 25:
            issues.append(
                "The response contains citations but insufficient "
                "explanatory content. Provide 3-5 concrete findings "
                "in complete sentences, with a verified citation "
                "immediately after each finding."
            )

    return issues


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Local Qwen coding agent")
    parser.add_argument(
        "--root",
        default=".",
        help="Project directory to work in (default: current directory)",
    )
    parser.add_argument(
        "--mode", choices=("read-only", "edit"), default="read-only",
        help="Tool policy (default: read-only)",
    )
    parser.add_argument(
        "--allow-command", action="append", default=[], metavar="COMMAND",
        help="Authorize one exact command; repeat this option as needed",
    )
    parser.add_argument("task", nargs="*", help="Task to perform")
    args = parser.parse_args()
    global POLICY_MODE, EXPLICIT_COMMANDS
    POLICY_MODE = args.mode
    EXPLICIT_COMMANDS = args.allow_command

    root = Path(args.root).expanduser().resolve()
    if not root.is_dir():
        print(f"Project directory does not exist: {root}", file=sys.stderr)
        return 2

    task = " ".join(args.task).strip()
    if not task:
        task = input("What should the agent do? ").strip()
    if not task:
        print("No task provided.", file=sys.stderr)
        return 2

    print(f"\nProject root: {root}")
    print(f"API: {API_URL}")
    print(f"Task: {task}\n")

    read_evidence = {}

    task_message = f"Project root: {root}\nTask: {task}"
    if task_requires_line_references(task):
        task_message += (
            "\n\nMandatory evidence workflow for this task: "
            "First call list_files to discover the actual source layout. "
            "Then use native read_file tool calls to inspect implementation "
            "files relevant to each requested finding. Do not draft the final "
            "report until those source files have actually been read and their "
            "returned numbered lines support the findings. Documentation alone "
            "is not sufficient evidence for implementation claims. If you "
            "cannot call a tool natively, do not imitate it as text; state that "
            "the evidence could not be collected."
        )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": task_message},
    ]

    citation_retry_count = 0
    length_retry_count = 0

    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"\n--- Agent iteration {iteration}/{MAX_ITERATIONS} ---")

        try:
            allow_tools = iteration <= MAX_ITERATIONS - FINALIZATION_ITERATIONS
            response = request_model(messages, allow_tools=allow_tools)
            choice = response["choices"][0]
            usage = response.get("usage") or {}
            print(
                "[API diagnostics] "
                f"finish_reason={choice.get('finish_reason')!r}, "
                f"prompt_tokens={usage.get('prompt_tokens')!r}, "
                f"completion_tokens={usage.get('completion_tokens')!r}",
                file=sys.stderr,
            )
            message = choice["message"]
        except (RuntimeError, KeyError, IndexError, ValueError) as exc:
            print(f"API error: {exc}", file=sys.stderr)
            return 1

        messages.append(message)
        calls = message.get("tool_calls") or []

        if not calls:
            if choice.get("finish_reason") == "length":
                if length_retry_count < 1:
                    length_retry_count += 1
                    messages.append({
                        "role": "user",
                        "content": (
                            "Your previous response was cut off because it "
                            "reached the output token limit. Produce a complete, "
                            "concise final report in at most 300 tokens. Do not "
                            "output JSON-style citations. Use only these supported "
                            "citation formats, with paths and lines actually "
                            "returned by read_file: "
                            "`src/main/java/io/jmon/core/File.java:28-40` or "
                            "`File: File.java, lines 28-40`. Cite each "
                            "implementation claim. In read-only mode, describe "
                            "observed architecture only; do not claim files were "
                            "changed and do not include a files_changed list. "
                            "Do not invent evidence."
                        ),
                    })
                    print(
                        "Response reached the token limit; requesting one "
                        "concise replacement."
                    )
                    continue

                print(
                    "Agent failed: response reached the token limit again; "
                    "incomplete report suppressed.",
                    file=sys.stderr,
                )
                return 1

            content = message.get("content") or ""
            print(
                f"[Response diagnostics] content_length={len(content)}, "
                f"content_preview={content[:240]!r}",
                file=sys.stderr,
            )
            stripped = content.strip()
            lowered = stripped.lower()

            tool_names = ("list_files", "read_file", "write_file", "run_command")
            looks_like_tool_call = (
                stripped.startswith("{")
                and "arguments" in lowered
                and any(name in lowered for name in tool_names)
            )

            if looks_like_tool_call:
                print(
                    "\\nModel emitted a tool-call-like text response. "
                    "Requesting a proper tool call..."
                )
                messages.append({
                    "role": "user",
                    "content": (
                        "Your previous response attempted to describe a tool call "
                        "as plain text, so no tool was executed. Do not output "
                        "tool-call JSON as text. If you need a tool, invoke it using "
                        "the native tool_calls mechanism and valid arguments. "
                        "Otherwise, provide your normal final answer."
                    ),
                })
                continue

            print("\n=== Agent response ===\n")
            if not content or not content.strip():
                print("Agent failed: empty final response.", file=sys.stderr)
                return 1
            issues = validate_line_references(
                content,
                read_evidence,
                require_references=task_requires_line_references(task),
            )
            if issues:
                print("\n=== Source-reference verification warnings ===")
                for issue in issues:
                    print(f"- {issue}")

                if citation_retry_count < 1:
                    prefetched = prefetch_missing_read_evidence(
                        root, issues, read_evidence
                    )
                    if prefetched:
                        print(
                            f"Prefetched {prefetched} source line(s) "
                            "for citation re-validation."
                        )
                        issues = validate_line_references(
                            content,
                            read_evidence,
                            require_references=task_requires_line_references(task),
                        )
                        if not issues:
                            print("\n=== Agent response ===\n")
                            print(content)
                            return 0

                    citation_retry_count += 1
                    messages.append({
                        "role": "user",
                        "content": (
                            "Your draft failed source-reference validation. "
                            "Revise the report to address every issue below. "
                            "Use only file paths and line numbers supported by "
                            "read_file results already present in this conversation. "
                            "Every implementation claim must have a supported citation. "
                            "Use plain-text citations in one of these exact formats: "
                            "`src/main/java/io/jmon/core/JvmMonitoringService.java:28-40` "
                            "or `File: JvmMonitoringService.java, lines 28-40`. "
                            "Do not use JSON-style file citations. These examples "
                            "show formatting only; use paths and line numbers actually "
                            "returned by read_file. "
                            "If any validation issue says a source file was not found "
                            "among successfully read evidence, or that cited lines "
                            "were not returned by read_file, you MUST first call "
                            "read_file for the exact cited project-relative path and "
                            "relevant line range. Do not produce a revised final "
                            "report until those tool calls have returned. After reading, "
                            "revise claims and citations using only the returned evidence. "
                            "If read_file reports that the path does not exist or cannot "
                            "be read, state that limitation and remove unsupported claims. "
                            "Do not invent citations or test results. "
                            + (
                                "This task is running in read-only mode: do not claim "
                                "that files were changed, implemented, or modified; "
                                "do not include a files_changed list. Report only "
                                "observed architecture and evidence. "
                                if POLICY_MODE == "read-only" else ""
                            )
                            + "Return 3-5 concrete architecture findings in complete "
                              "sentences. Explain what each component does and how "
                              "the components relate, with a verified citation "
                              "immediately after each finding. Do not return a "
                              "citation-only list. Keep the report concise and "
                              "evidence-based, not an implementation summary.\n\n"
                            "Validation issues:\n- "
                            + "\n- ".join(issues)
                        ),
                    })
                    print("Requesting one evidence-based correction...")
                    continue

                prefetched = prefetch_missing_read_evidence(
                    root, issues, read_evidence
                )
                if prefetched:
                    print(
                        f"Prefetched {prefetched} source line(s) "
                        "after the correction attempt."
                    )
                    issues = validate_line_references(
                        content,
                        read_evidence,
                        require_references=task_requires_line_references(task),
                    )
                    if not issues:
                        print("\n=== Agent response ===\n")
                        print(content)
                        return 0

                print(
                    "Agent failed: source-reference validation still has "
                    "unresolved issues after the correction attempt. "
                    "Rejected report suppressed.",
                    file=sys.stderr,
                )
                return 1

            print("\n=== Agent response ===\n")
            print(content)
            return 0

        if calls and not allow_tools:
            print(
                "Agent requested tools during the finalization phase; "
                "rejecting those calls and asking for a final report.",
                file=sys.stderr,
            )
            for call in calls:
                messages.append({
                    "role": "tool",
                    "tool_call_id": call.get("id", ""),
                    "name": call.get("function", {}).get("name", ""),
                    "content": "Tool call rejected: tools are disabled during finalization. Use only evidence already collected.",
                })
            messages.append({
                "role": "user",
                "content": "Do not call tools again. Produce the concise final report using only verified evidence already collected. If evidence is insufficient, state that limitation.",
            })
            continue

        for call in calls:
            function = call.get("function", {})
            name = function.get("name", "")
            raw_args = function.get("arguments", "{}")

            print(f"\nTool requested: {name}")

            try:
                tool_args = json.loads(raw_args)
                if not isinstance(tool_args, dict):
                    raise ValueError("Tool arguments must be a JSON object.")
                if name == "read_file":
                    print("Read request:", {k: tool_args.get(k) for k in ("path", "start_line", "max_lines")})
                    requested_path = Path(str(tool_args.get("path", ""))).as_posix()
                    existing = read_evidence.get(requested_path)
                    if existing is None and len(read_evidence) >= MAX_SOURCE_FILES:
                        result = (
                            f"Read blocked: the limit of {MAX_SOURCE_FILES} distinct successfully read source files has been reached. "
                            "Use the evidence already collected or report what remains unverified."
                        )
                    elif existing is not None:
                        start = tool_args.get("start_line", 1)
                        count = tool_args.get("max_lines", 200)
                        if isinstance(start, int) and isinstance(count, int) and start >= 1 and count >= 1:
                            end = min(start + count - 1, existing["total_lines"])
                            requested_lines = set(range(start, end + 1)) if start <= end else set()
                            if requested_lines and requested_lines.issubset(existing["lines"]):
                                result = (
                                    f"Read skipped: {requested_path} lines {start}-{end} are already in the evidence collected. "
                                    "Use the previous read_file result; do not repeat this read."
                                )
                            else:
                                result = execute_tool(root, name, tool_args)
                        else:
                            result = execute_tool(root, name, tool_args)
                    else:
                        result = execute_tool(root, name, tool_args)
                else:
                    result = execute_tool(root, name, tool_args)
                if name == "read_file":
                    item = extract_read_evidence(result)
                    if item:
                        key = item["path"]
                        if key not in read_evidence:
                            read_evidence[key] = {
                                "total_lines": item["total_lines"],
                                "lines": set(),
                                "line_text": {},
                            }
                        read_evidence[key]["total_lines"] = item["total_lines"]
                        read_evidence[key]["lines"].update(item["lines"])
                        read_evidence[key]["line_text"].update(item["line_text"])
            except (json.JSONDecodeError, ValueError) as exc:
                result = f"Invalid tool arguments: {exc}"

            print(f"Tool result:\n{result[:2000]}")

            messages.append({
                "role": "tool",
                "tool_call_id": call.get("id", ""),
                "name": name,
                "content": result,
            })

    print(
        f"\nStopped after {MAX_ITERATIONS} iterations. "
        "Review the changes and start a new task if needed."
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
