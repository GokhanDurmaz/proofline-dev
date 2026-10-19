from pathlib import Path
import re
import py_compile

p = Path(__file__).with_name("local_agent.py")
s = p.read_text(encoding="utf-8")
original = s

def sub(pattern, replacement, label):
    global s
    updated, count = re.subn(pattern, replacement, s, count=1, flags=re.MULTILINE | re.DOTALL)
    if count != 1:
        raise SystemExit(f"İptal: {label}; eşleşme sayısı={count}. Dosya yazılmadı.")
    s = updated

# Replace run_command with exact per-task authorization.
sub(
    r"def run_command\(root: Path, command: str\) -> str:\n.*?(?=\n\nTOOLS = \[)",
    '''def run_command(
    root: Path, command: str, allowed_commands: list[str]
) -> str:
    try:
        argv = shlex.split(command)
        permitted = [shlex.split(item) for item in allowed_commands]
    except ValueError as exc:
        return f"Could not parse command: {exc}"

    if not command_is_allowed(argv):
        return "Command blocked by the agent's safety allowlist."

    if argv not in permitted:
        return (
            "Command blocked: this exact command was not explicitly "
            "authorized with --allow-command."
        )

    print(f"\\\\nProject: {root}")
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
            f"Exit code: {result.returncode}\\\\n"
            f"STDOUT:\\\\n{result.stdout}\\\\n"
            f"STDERR:\\\\n{result.stderr}"
        )
        if len(output) > MAX_OUTPUT_CHARS:
            output = output[:MAX_OUTPUT_CHARS] + "\\\\n... [output truncated]"
        return output
    except subprocess.TimeoutExpired:
        return "Command timed out after 120 seconds."
    except OSError as exc:
        return f"Could not run command: {exc}"''',
    "run_command",
)

# Enforce the mode at the tool-execution boundary.
sub(
    r"def execute_tool\(root: Path, name: str, args: dict\) -> str:\n.*?(?=\n\ndef request_model\()",
    '''def execute_tool(
    root: Path,
    name: str,
    args: dict,
    mode: str,
    allowed_commands: list[str],
) -> str:
    try:
        if name == "list_files":
            return list_files(root, args["path"])
        if name == "read_file":
            return read_file(root, args["path"])
        if name == "write_file":
            if mode != "edit":
                return "Write blocked: agent is running in read-only mode."
            return write_file(root, args["path"], args["content"])
        if name == "run_command":
            return run_command(root, args["command"], allowed_commands)
        return f"Unknown tool: {name}"
    except (KeyError, TypeError, ValueError, OSError) as exc:
        return f"Tool error: {type(exc).__name__}: {exc}"''',
    "execute_tool",
)

sub(
    r'    parser\.add_argument\("task", nargs="\*", help="Task to perform"\)',
    '''    parser.add_argument(
        "--mode",
        choices=("read-only", "edit"),
        default="read-only",
        help="Tool policy; defaults to read-only",
    )
    parser.add_argument(
        "--allow-command",
        action="append",
        default=[],
        metavar="COMMAND",
        help="Authorize one exact command; repeatable, approval still required",
    )
    parser.add_argument("task", nargs="*", help="Task to perform")''',
    "CLI options",
)

sub(
    r'    print\(f"Task: \{task\}\\\\n"\)',
    '''    print(f"Mode: {args.mode}")
    print(f"Explicit command permissions: {args.allow_command or '(none)'}")
    print(f"Task: {task}\\\\n")''',
    "startup policy display",
)

sub(
    r'        \{"role": "user", "content": f"Project root: \{root\}\\\\nTask: \{task\}"\},',
    '''        {
            "role": "user",
            "content": (
                f"Project root: {root}\\\\n"
                f"Mode: {args.mode}\\\\n"
                f"Explicitly authorized commands: {args.allow_command or '(none)'}\\\\n"
                f"Task: {task}"
            ),
        },''',
    "task policy message",
)

sub(
    r'            print\(content or "\(empty response\)"\)\n            return 0',
    '''            if not stripped:
                print(
                    "Agent failed: model returned an empty final response.",
                    file=sys.stderr,
                )
                return 1
            print(content)
            return 0''',
    "empty final response",
)

sub(
    r'                result = execute_tool\(root, name, tool_args\)',
    '                result = execute_tool(root, name, tool_args, args.mode, args.allow_command)',
    "tool dispatch",
)

if s == original:
    raise SystemExit("İptal: hiçbir değişiklik oluşmadı.")

# Compile the candidate before touching the original file.
candidate = p.with_name("local_agent.py.candidate")
candidate.write_text(s, encoding="utf-8")
try:
    py_compile.compile(str(candidate), doraise=True)
except Exception:
    candidate.unlink(missing_ok=True)
    raise SystemExit("İptal: aday dosyanın sözdizimi kontrolü başarısız; ana dosya değişmedi.")

p.write_text(s, encoding="utf-8")
candidate.unlink(missing_ok=True)
print("Yama uygulandı ve Python sözdizimi kontrolü geçti.")
