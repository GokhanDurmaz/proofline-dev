from pathlib import Path

source_path = Path("local_agent.py")
candidate_path = Path("local_agent.py.candidate")
s = source_path.read_text(encoding="utf-8")
original = s

def replace_once(old, new, label):
    global s
    count = s.count(old)
    if count != 1:
        raise SystemExit(
            f"ABORT: {label}: expected 1 match, found {count}. "
            "Original file unchanged."
        )
    s = s.replace(old, new, 1)
    print(f"OK: {label}")

# Add policy state without changing existing function signatures.
replace_once(
    "def run_command(root: Path, command: str) -> str:",
    'POLICY_MODE = "read-only"\n'
    "EXPLICIT_COMMANDS = []\n\n\n"
    "def run_command(root: Path, command: str) -> str:",
    "policy state",
)

# Preserve the existing allowlist and require an exact explicit permission too.
replace_once(
    '    print(f"\\nProject: {root}")',
    '    try:\n'
    "        permitted = [shlex.split(item) for item in EXPLICIT_COMMANDS]\n"
    "    except ValueError as exc:\n"
    '        return f"Invalid --allow-command value: {exc}"\n'
    "\n"
    "    if argv not in permitted:\n"
    '        return ("Command blocked: authorize this exact command with "\n'
    '                "--allow-command.")\n'
    "\n"
    '    print(f"\\nProject: {root}")',
    "explicit command authorization",
)

# Enforce read-only at the tool boundary, before write_file or its approval prompt.
replace_once(
    '        if name == "write_file":\n'
    '            return write_file(root, args["path"], args["content"])',
    '        if name == "write_file":\n'
    '            if POLICY_MODE != "edit":\n'
    '                return "Write blocked: agent is running in read-only mode."\n'
    '            return write_file(root, args["path"], args["content"])',
    "read-only write protection",
)

# Add CLI controls.
replace_once(
    '    parser.add_argument("task", nargs="*", help="Task to perform")',
    '    parser.add_argument(\n'
    '        "--mode", choices=("read-only", "edit"), default="read-only",\n'
    '        help="Tool policy (default: read-only)",\n'
    '    )\n'
    '    parser.add_argument(\n'
    '        "--allow-command", action="append", default=[], metavar="COMMAND",\n'
    '        help="Authorize one exact command; repeat this option as needed",\n'
    '    )\n'
    '    parser.add_argument("task", nargs="*", help="Task to perform")',
    "CLI options",
)

replace_once(
    "    args = parser.parse_args()",
    "    args = parser.parse_args()\n"
    "    global POLICY_MODE, EXPLICIT_COMMANDS\n"
    "    POLICY_MODE = args.mode\n"
    "    EXPLICIT_COMMANDS = args.allow_command",
    "policy configuration",
)

replace_once(
    '            print(content or "(empty response)")\n'
    "            return 0",
    '            if not content or not content.strip():\n'
    '                print("Agent failed: empty final response.", file=sys.stderr)\n'
    "                return 1\n"
    "            print(content)\n"
    "            return 0",
    "empty response handling",
)

if s == original:
    raise SystemExit("ABORT: no changes generated.")

# Compile before writing the candidate.
compile(s, str(candidate_path), "exec")
candidate_path.write_text(s, encoding="utf-8")
print(f"\nCandidate created: {candidate_path}")
print("Original local_agent.py was not changed.")
