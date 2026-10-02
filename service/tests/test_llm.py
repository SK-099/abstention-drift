"""OpenAICompatChat against a fake OpenAI-compatible server (what Ollama /v1, vLLM, Groq, ... expose)."""

import json
import socket
import threading
import time

import pytest
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from abstain_api.llm import OpenAICompatChat
from abstain_api.settings import Settings

fake = FastAPI()
seen = []


@fake.post("/v1/chat/completions")
async def completions(req: Request):
    body = await req.json()
    seen.append(body)
    text = "Paris is the capital."
    if body.get("stream"):

        def gen():
            for w in text.split(" "):
                chunk = {
                    "id": "x",
                    "object": "chat.completion.chunk",
                    "created": 0,
                    "model": body["model"],
                    "choices": [{"index": 0, "delta": {"content": w + " "}, "finish_reason": None}],
                }
                yield f"data: {json.dumps(chunk)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(gen(), media_type="text/event-stream")
    return JSONResponse(
        {
            "id": "x",
            "object": "chat.completion",
            "created": 0,
            "model": body["model"],
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
    )


@pytest.fixture(scope="module")
def base_url():
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    server = uvicorn.Server(uvicorn.Config(fake, host="127.0.0.1", port=port, log_level="warning"))
    th = threading.Thread(target=server.run, daemon=True)
    th.start()
    while not server.started:
        time.sleep(0.05)
    yield f"http://127.0.0.1:{port}/v1"
    server.should_exit = True


def test_chat_and_stream(base_url):
    s = Settings(llm_base_url=base_url, llm_model="qwen2.5:3b", llm_max_tokens=32)
    llm = OpenAICompatChat(s)
    assert llm.chat([{"role": "user", "content": "capital of France?"}]) == "Paris is the capital."
    assert seen[-1]["temperature"] == 0 and seen[-1]["max_tokens"] == 32 and seen[-1]["model"] == "qwen2.5:3b"

    # inside a LangGraph node with token streaming on, tokens go to the custom stream
    from langgraph.graph import END, START, StateGraph
    from typing_extensions import TypedDict

    from abstain.agent import streaming_tokens

    class S(TypedDict):
        out: str

    def node(_):
        with streaming_tokens():
            return {"out": llm.chat([{"role": "user", "content": "q"}], max_new_tokens=8)}

    g = StateGraph(S)
    g.add_node("n", node)
    g.add_edge(START, "n")
    g.add_edge("n", END)
    chunks = list(g.compile().stream({"out": ""}, stream_mode=["custom", "values"]))
    toks = [c[1]["text"] for c in chunks if c[0] == "custom"]
    assert "".join(toks).strip() == "Paris is the capital." and len(toks) == 4
    assert chunks[-1][1]["out"].strip() == "Paris is the capital."
    assert seen[-1]["stream"] is True and seen[-1]["max_tokens"] == 8
