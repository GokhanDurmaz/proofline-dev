import json
import re
import time
import uuid
import threading
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

MODEL_ID = "uqer1244/Qwen3.8-27B-GSQ-RCO-MLX"

model_obj = None
tokenizer_obj = None
model_lock = threading.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model_obj, tokenizer_obj

    from mlx_gsq import load

    print(f"Model yükleniyor: {MODEL_ID}", flush=True)
    model_obj, tokenizer_obj = load(MODEL_ID)
    print("Model hazır.", flush=True)

    yield

    model_obj = None
    tokenizer_obj = None


app = FastAPI(
    title="MLX GSQ-RCO OpenAI API",
    lifespan=lifespan,
)


class ChatRequest(BaseModel):
    model: str | None = None
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]] | None = None
    tool_choice: Any = "auto"
    parallel_tool_calls: bool | None = None
    stream: bool = False
    max_tokens: int = Field(default=512, ge=1, le=8192)
    max_completion_tokens: int | None = Field(default=None, ge=1, le=8192)
    temperature: float | None = Field(default=None, ge=0, le=2)
    top_p: float | None = Field(default=None, gt=0, le=1)


def normalize_tools(req: ChatRequest) -> list[dict[str, Any]]:
    tools = req.tools or []

    if req.tool_choice == "none":
        return []

    normalized = []

    for tool in tools:
        if not isinstance(tool, dict):
            continue

        if tool.get("type") == "function" and isinstance(tool.get("function"), dict):
            normalized.append(tool)
        elif isinstance(tool.get("name"), str):
            # Also accept a plain function definition.
            normalized.append({
                "type": "function",
                "function": tool,
            })

    if isinstance(req.tool_choice, dict):
        selected = req.tool_choice.get("function", {}).get("name")
        if selected:
            normalized = [
                tool for tool in normalized
                if tool["function"].get("name") == selected
            ]

    return normalized


def make_prompt(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
) -> str:
    normalized = []

    for message in messages:
        role = message.get("role", "user")
        content = message.get("content", "")

        if role == "developer":
            role = "system"

        # Reconstruct prior assistant tool calls in the model's XML format.
        if role == "assistant" and message.get("tool_calls"):
            calls = []

            for call in message["tool_calls"]:
                function = call.get("function", {})
                name = function.get("name", "")
                arguments = function.get("arguments", "{}")

                try:
                    arguments = json.loads(arguments) if isinstance(arguments, str) else arguments
                except json.JSONDecodeError:
                    arguments = {}

                if not isinstance(arguments, dict):
                    arguments = {}

                params = "".join(
                    f"<parameter={key}>\n{value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)}\n</parameter>\n"
                    for key, value in arguments.items()
                )
                calls.append(
                    f"<tool_call>\n<function={name}>\n{params}</function>\n</tool_call>"
                )

            content = (content or "") + "\n" + "\n".join(calls)

        # Qwen's template may not accept OpenAI's role=tool directly.
        # Present tool results as user-readable context instead.
        if role == "tool":
            tool_name = message.get("name", "tool")
            content = f"[Tool result: {tool_name}]\n{content}"
            role = "user"

        if isinstance(content, list):
            content = "\n".join(
                part["text"]
                for part in content
                if isinstance(part, dict)
                and part.get("type") == "text"
                and isinstance(part.get("text"), str)
            )

        if not isinstance(content, str):
            content = str(content)

        normalized.append({"role": role, "content": content})

    kwargs = {
        "tokenize": False,
        "add_generation_prompt": True,
        "enable_thinking": False,
    }

    if tools:
        kwargs["tools"] = tools

    return tokenizer_obj.apply_chat_template(normalized, **kwargs)


def clean_output(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = text.replace("<|im_end|>", "")
    text = text.replace("<|endoftext|>", "")
    return text.strip()


def parse_tool_calls(text: str, tools: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    """Parse the XML-like function-call syntax emitted by this model."""
    available = {
        tool["function"]["name"]: tool["function"]
        for tool in tools
        if isinstance(tool.get("function"), dict)
        and isinstance(tool["function"].get("name"), str)
    }

    calls = []
    pattern = re.compile(
        r"<tool_call>\s*<function=([A-Za-z_][\w.-]*)>(.*?)</function>\s*</tool_call>",
        re.DOTALL,
    )

    for match in pattern.finditer(text):
        name = match.group(1)
        body = match.group(2)

        if name not in available:
            continue

        arguments = {}
        param_pattern = re.compile(
            r"<parameter=([A-Za-z_][\w.-]*)>\s*(.*?)\s*</parameter>",
            re.DOTALL,
        )

        schema = available[name].get("parameters", {})
        properties = schema.get("properties", {}) if isinstance(schema, dict) else {}

        for param in param_pattern.finditer(body):
            key = param.group(1)
            value = param.group(2).strip()

            if key not in properties:
                continue

            prop_type = properties[key].get("type") if isinstance(properties[key], dict) else None

            if prop_type in {"object", "array", "integer", "number", "boolean"}:
                try:
                    value = json.loads(value)
                except (json.JSONDecodeError, TypeError):
                    pass

            arguments[key] = value

        calls.append({
            "id": "call_" + uuid.uuid4().hex[:24],
            "type": "function",
            "function": {
                "name": name,
                "arguments": json.dumps(arguments, ensure_ascii=False),
            },
        })

    if not calls:
        return clean_output(text), []

    # Remove parsed tool-call markup from the user-visible message.
    visible = pattern.sub("", text).strip()
    return clean_output(visible), calls


def make_sampler_options(req: ChatRequest) -> dict[str, Any]:
    if req.temperature is None and req.top_p is None:
        return {}

    from mlx_lm.sample_utils import make_sampler

    # make_sampler defaults to greedy decoding at temp=0.
    # top_p=0 disables nucleus sampling in mlx-lm.
    return {
        "sampler": make_sampler(
            temp=req.temperature if req.temperature is not None else 0.0,
            top_p=req.top_p if req.top_p is not None else 0.0,
        )
    }


@app.get("/")
def root():
    return {"status": "ok", "model": MODEL_ID}


@app.get("/health")
def health():
    return {
        "status": "ready" if model_obj is not None else "loading",
        "model": MODEL_ID,
    }


@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [{
            "id": MODEL_ID,
            "object": "model",
            "created": int(time.time()),
            "owned_by": "local",
        }],
    }


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest):
    if model_obj is None or tokenizer_obj is None:
        raise HTTPException(status_code=503, detail="Model is loading")

    if not req.messages:
        raise HTTPException(status_code=400, detail="messages cannot be empty")

    from mlx_lm.generate import generate, stream_generate

    tools = normalize_tools(req)
    prompt = make_prompt(req.messages, tools)
    max_tokens = req.max_completion_tokens or req.max_tokens
    sampler_options = make_sampler_options(req)

    completion_id = "chatcmpl-" + uuid.uuid4().hex
    created = int(time.time())

    # Buffering here lets us distinguish normal text from tool calls reliably.
    # SSE remains OpenAI-compatible, though stream chunks arrive after generation.
    def run_generation():
        with model_lock:
            return generate(
                model_obj,
                tokenizer_obj,
                prompt=prompt,
                max_tokens=max_tokens,
                **sampler_options,
            )

    def build_response(raw: str):
        content, tool_calls = parse_tool_calls(raw, tools)
        message = {"role": "assistant", "content": content or None}

        if tool_calls:
            message["tool_calls"] = tool_calls

        prompt_count = len(tokenizer_obj.encode(prompt))
        completion_count = len(tokenizer_obj.encode(raw))

        return {
            "id": completion_id,
            "object": "chat.completion",
            "created": created,
            "model": MODEL_ID,
            "choices": [{
                "index": 0,
                "message": message,
                "finish_reason": "tool_calls" if tool_calls else "stop",
            }],
            "usage": {
                "prompt_tokens": prompt_count,
                "completion_tokens": completion_count,
                "total_tokens": prompt_count + completion_count,
            },
        }

    if not req.stream:
        raw = run_generation()
        return build_response(raw)

    def event(data):
        return "data: " + json.dumps(data, ensure_ascii=False) + "\n\n"

    def stream_events():
        # stream_generate uses the same sampler API as generate().
        pieces = []

        with model_lock:
            for response in stream_generate(
                model_obj,
                tokenizer_obj,
                prompt=prompt,
                max_tokens=max_tokens,
                **sampler_options,
            ):
                pieces.append(response.text)

        result = build_response("".join(pieces))
        choice = result["choices"][0]
        message = choice["message"]

        yield event({
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": MODEL_ID,
            "choices": [{
                "index": 0,
                "delta": {"role": "assistant"},
                "finish_reason": None,
            }],
        })

        if message.get("tool_calls"):
            for index, call in enumerate(message["tool_calls"]):
                yield event({
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": MODEL_ID,
                    "choices": [{
                        "index": 0,
                        "delta": {
                            "tool_calls": [{
                                "index": index,
                                "id": call["id"],
                                "type": "function",
                                "function": call["function"],
                            }]
                        },
                        "finish_reason": None,
                    }],
                })
        elif message.get("content"):
            yield event({
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": MODEL_ID,
                "choices": [{
                    "index": 0,
                    "delta": {"content": message["content"]},
                    "finish_reason": None,
                }],
            })

        yield event({
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": MODEL_ID,
            "choices": [{
                "index": 0,
                "delta": {},
                "finish_reason": choice["finish_reason"],
            }],
        })
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        stream_events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
