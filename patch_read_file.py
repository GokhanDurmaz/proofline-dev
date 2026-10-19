from pathlib import Path

source = Path("local_agent.py")
candidate = Path("local_agent.py.candidate")
text = source.read_text(encoding="utf-8")

replacements = [
(
'''def read_file(root: Path, path: str) -> str:
    file_path = project_path(root, path)
    if not file_path.is_file():
        return f"File not found: {path}"
    if file_path.stat().st_size > 1_000_000:
        return "File is over 1 MB; inspect a smaller file instead."

    try:
        text = file_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return "File is not UTF-8 text."

    if len(text) > MAX_FILE_CHARS:
        text = text[:MAX_FILE_CHARS] + "\\n... [truncated]"
    return text
''',
'''def read_file(
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
        f"File: {path}\\n"
        f"Total lines: {total_lines}\\n"
        f"Returned lines: "
        f"{start_line}-{end_index if selected else start_line - 1}\\n"
    )
    result = header + "\\n".join(numbered)

    if end_index < total_lines:
        result += (
            f"\\n[TRUNCATED: more lines remain. "
            f"Read again with start_line={end_index + 1}.]"
        )

    if len(result) > MAX_FILE_CHARS:
        result = result[:MAX_FILE_CHARS]
        result += (
            "\\n[TRUNCATED BY CHARACTER LIMIT: "
            "request a smaller max_lines range.]"
        )

    return result
'''
),
(
'''            "description": "Read a UTF-8 text file inside the project.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Relative file path"}
                },
                "required": ["path"],
''',
'''            "description": (
                "Read a UTF-8 text file inside the project. "
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
'''
),
(
'            return read_file(root, args["path"])',
'''            return read_file(
                root,
                args["path"],
                args.get("start_line", 1),
                args.get("max_lines", 200),
            )'''
),
]

for index, (old, new) in enumerate(replacements, start=1):
    count = text.count(old)
    if count != 1:
        raise SystemExit(
            f"ABORT: patch section {index}: expected 1 match, found {count}. "
            "No candidate was written."
        )
    text = text.replace(old, new, 1)

candidate.write_text(text, encoding="utf-8")
print(f"Candidate written: {candidate}")
print("Original local_agent.py was not modified.")
