"""Gradio demo: a multi-turn RAG assistant that shows its sources and says "I don't know"
when the retrieved evidence can't answer.

    python app/app.py                                  # full system (GPU)
    LLM=Qwen/Qwen2.5-1.5B-Instruct DENSE=0 python app/app.py   # lighter, runs on a small GPU / CPU

Deploy: push this repo to a Hugging Face Space (SDK: gradio, app_file: app/app.py) with the
indexes/ folder and checkpoints/checker/best uploaded (or built at startup for one collection).
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import gradio as gr  # noqa: E402

from abstain.config import load_config  # noqa: E402
from abstain.data import Task  # noqa: E402
from abstain.llm import ChatLLM  # noqa: E402
from abstain.pipeline import AbstainRAG  # noqa: E402
from abstain.retrieval import load_retriever  # noqa: E402
from abstain.rewrite import QueryRewriter  # noqa: E402

cfg = load_config(os.environ.get("CONFIG", "configs/default.yaml"))
COLLECTIONS = [c for c in ("fiqa", "govt", "cloud", "clapnq") if (Path(cfg["data"]["index_dir"]) / c).exists()]
llm = ChatLLM(os.environ.get("LLM", cfg["llm"]["model"]), cfg["llm"]["load_in_4bit"], cfg["llm"]["max_new_tokens"])
retrievers = {c: load_retriever(c, cfg, use_dense=os.environ.get("DENSE", "1") == "1",
                                use_rerank=os.environ.get("RERANK", "1") == "1") for c in COLLECTIONS}
checker = None
ckpt = os.environ.get("CHECKER", cfg["checker"]["output_dir"] + "/best")
if Path(ckpt).exists():
    from abstain.checker import AnswerabilityChecker
    checker = AnswerabilityChecker(ckpt, device=cfg["checker"].get("infer_device"))
system = AbstainRAG(QueryRewriter("llm", llm=llm), retrievers, llm=llm, checker=checker, k=cfg["retrieval"]["final_k"])


def meter(p):
    if p is None:
        return "_No checker loaded: abstention is prompt-only._"
    bar = "█" * round(p * 20) + "░" * (20 - round(p * 20))
    th = checker.threshold
    return f"**P(unanswerable)** `{bar}` {p:.2f} (abstains at ≥ {th:.2f})"


def respond(message, chat, collection):
    history = []
    for m in chat:
        history.append({"speaker": "user" if m["role"] == "user" else "agent", "text": m["content"]})
    turn = sum(h["speaker"] == "user" for h in history) + 1
    task = Task("demo", "demo", turn, collection, history, message, "UNKNOWN")
    out = system(task)
    chat = chat + [{"role": "user", "content": message}, {"role": "assistant", "content": out.answer}]
    src = "\n\n".join(f"**[{i + 1}]** `{p['document_id']}` {p['text'][:300].strip()}…" for i, p in enumerate(out.passages))
    info = (f"**Turn {turn}** · rewritten query: _{out.query}_\n\n{meter(out.p_unanswerable)}\n\n"
            f"Abstained: **{out.abstained}** ({out.abstained_by})")
    return "", chat, info, src


with gr.Blocks(title="Knowing When Not to Answer") as demo:
    gr.Markdown("# Knowing When Not to Answer\nMulti-turn RAG that refuses when its evidence can't support an answer.")
    with gr.Row():
        with gr.Column(scale=3):
            coll = gr.Dropdown(COLLECTIONS, value=COLLECTIONS[0] if COLLECTIONS else None, label="Document collection")
            chat = gr.Chatbot(type="messages", height=460)
            box = gr.Textbox(placeholder="Ask a question, then a follow-up…", show_label=False)
            gr.ClearButton([box, chat])
        with gr.Column(scale=2):
            info = gr.Markdown()
            gr.Markdown("### Retrieved passages")
            src = gr.Markdown()
    box.submit(respond, [box, chat, coll], [box, chat, info, src])

if __name__ == "__main__":
    demo.launch()
