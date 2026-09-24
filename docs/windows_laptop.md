# Running on a Windows laptop (GTX 1650, 4 GB VRAM, 16 GB RAM)

Everything lives on **F:** (repo, data, virtual environment, model cache), so C: stays clean.
All commands are for **PowerShell**, run from `F:\NLP\abstention-drift`. Use `configs/laptop.yaml` in every command.

## What changes on 4 GB

| Part | Default (T4, 16 GB) | Laptop (4 GB) |
|---|---|---|
| LLM (rewrite + answer) | Qwen2.5-7B-Instruct, 4-bit | **Qwen2.5-3B-Instruct**, 4-bit (~2.2 GB VRAM) |
| Precision | bf16 | **fp16** (GTX 16xx has no bf16; chosen automatically) |
| Re-ranker | bge-reranker-base, top 50 | same model in fp16, top 30, 384 tokens |
| Checker | DeBERTa-v3-base, batch 8 × 2 | same model, batch 4 × 4, 384 tokens; inference on CPU |
| Dense index build | batch 256 | batch 64 |

The research questions and systems S0–S4 are unchanged. If you later get a bigger GPU (Kaggle or Colab), rerun the final tables with `configs/default.yaml` (7B).

## 0. One-time prerequisites

1. **NVIDIA driver** (recent Game Ready or Studio driver). Check: `nvidia-smi` shows the GTX 1650.
2. **Python 3.11** from python.org. Tick "Add python.exe to PATH".
3. **Git for Windows** from git-scm.com.
4. Optional: in *Power & sleep*, set "Sleep" to *Never* while long runs are going.

## 1. Environment (about 15 minutes, about 6 GB download)

```powershell
F:
cd F:\NLP\abstention-drift

# keep model and pip caches on F: (permanent for new windows + set for this window)
setx HF_HOME "F:\NLP\hf_cache"
setx PIP_CACHE_DIR "F:\NLP\pip_cache"
setx PYTHONUTF8 1
$env:HF_HOME = "F:\NLP\hf_cache"; $env:PIP_CACHE_DIR = "F:\NLP\pip_cache"; $env:PYTHONUTF8 = "1"

py -3.11 -m venv .venv
# if the next line is blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned   (answer Y)
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
pip install -e .

# sanity checks: expect "True NVIDIA GeForce GTX 1650", then a bitsandbytes version
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python -c "import bitsandbytes, faiss, bm25s; print('bitsandbytes', bitsandbytes.__version__)"
```

Every new PowerShell window: `F:` → `cd F:\NLP\abstention-drift` → `.\.venv\Scripts\Activate.ps1`.

## 1b. Ollama for the LLM (recommended on 4 GB)

With transformers + bitsandbytes 4-bit, the 3B model managed about 160 s per task on a GTX 1650, which is too slow for ~430 tasks x 5 systems. `configs/laptop.yaml` therefore uses **Ollama** (llama.cpp, GGUF 4-bit) for rewriting and answering. Retrieval, re-ranking and the checker are unchanged.

1. Install Ollama for Windows from https://ollama.com/download. It runs in the background (tray icon).
2. Move its model store to F: (optional, ~2 GB): `setx OLLAMA_MODELS "F:\NLP\ollama_models"`, then quit and restart Ollama from the tray.
3. In any prompt: `ollama pull qwen2.5:3b`, then check it with `ollama run qwen2.5:3b "Say hi"` (type `/bye` to exit).

To use the Hugging Face backend instead, add `--set llm.backend=hf`.

## 2. Data and quick checks (5 minutes)

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup_data.ps1
python scripts\dataset_stats.py --config configs\laptop.yaml
pytest -q                                            # expect: 14 passed
```

## 3. Smoke test on one domain (about 15 minutes, first model download ~2.5 GB)

Run this first. It exercises every stage on FiQA with 20 tasks, so problems show up in minutes rather than hours.

```powershell
$C = "--config", "configs\laptop.yaml"
python scripts\build_indexes.py @C --collections fiqa
python scripts\make_checker_data.py @C --rewrite oracle --collections fiqa
python scripts\train_checker.py @C --set checker.epochs=1
python scripts\run_pipeline.py @C --system S4 --split dev --collections fiqa --limit 20
python scripts\evaluate_abstention.py outputs\S4_dev.jsonl @C
```

If a step runs out of memory, see Troubleshooting.

## 4. Full run

Times are rough estimates for a GTX 1650. The generation runs are the long part, so start them before bed. Log files go to `logs\`.

```powershell
$C = "--config", "configs\laptop.yaml"
New-Item -ItemType Directory -Force logs | Out-Null

# 4.1 Indexes for all 4 domains (~366K passages)            ~1-2 h (FiQA alone ~10-20 min)
python scripts\build_indexes.py @C *> logs\indexes.log

# 4.2 Checker data with LLM rewrites                          ~30-45 min
#     (add --verify-llm for cleaner synthetic labels: +~40 min)
python scripts\make_checker_data.py @C --rewrite llm *> logs\checker_data.log

# 4.3 Train checkers: S4 (real + synthetic) and S3 (real only)  ~15-25 min each
python scripts\train_checker.py @C *> logs\train_S4.log
python scripts\train_checker.py @C --real-only *> logs\train_S3.log

# 4.4 Systems on the held-out MTRAG-UN test set               ~1-1.5 h per system
foreach ($S in "S0", "S1", "S2", "S3", "S4") {
    python scripts\run_pipeline.py @C --system $S --split mtragun *> "logs\run_$S.log"
}
# optional upper bound with human rewrites:
python scripts\run_pipeline.py @C --system S4-oracle --split mtragun *> logs\run_S4-oracle.log

# 4.5 Controlled drift probe (RQ1 with question type held fixed)
python scripts\run_synthetic_probe.py @C --who checker
python scripts\run_synthetic_probe.py @C --who generator --limit 300   # ~40 min

# 4.6 Results: results\abstention.md + results\drift.png
python scripts\evaluate_abstention.py "outputs\S*_mtragun.jsonl" "outputs\probe_*.jsonl" @C
```

Follow a running job from a second window with `Get-Content logs\run_S2.log -Wait -Tail 5`, and GPU use with `nvidia-smi -l 5`.

## 5. Demo

```powershell
$env:CONFIG = "configs/laptop.yaml"
python app\app.py
# open http://127.0.0.1:7860
```

## 6. Answer-quality metrics (official MTRAG scripts)

The algorithmic metrics (ROUGE-L, BERTScore, recall against the reference answer) run locally.

```powershell
pip install evaluate rouge-score bert_score beautifulsoup4 lxml
python data\mt-rag-benchmark\scripts\evaluation\run_algorithmic.py -i outputs\S4_mtragun.jsonl -o results\S4_mtragun_algo.jsonl -e data\mt-rag-benchmark\scripts\evaluation\config.yaml
```

The LLM-judge metrics (RB_llm, IDK-conditioned scores) need vLLM and an 8B judge, which don't fit a 4 GB GPU or run natively on Windows. Upload the `outputs\*.jsonl` files to a Kaggle notebook (free T4 x2) and run `run_generation_eval.py --provider hf` there.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `CUDA out of memory` in `run_pipeline.py` | `--set retrieval.rerank_top_n=15` or `--set retrieval.reranker_device=cpu`. Last resort: `--set llm.model=Qwen/Qwen2.5-1.5B-Instruct` (report it) |
| OOM while building the dense index | `--set retrieval.encode_batch_size=16` |
| OOM or NaN loss while training the checker | `--set checker.batch_size=2 checker.grad_accum=8`, or `--set checker.model=microsoft/deberta-v3-small` |
| `torch.cuda.is_available()` is False | You installed the CPU build: `pip uninstall -y torch` then reinstall with the cu126 line |
| bitsandbytes error on import | `pip install -U bitsandbytes`. Otherwise run without 4-bit: `--set llm.load_in_4bit=false llm.model=Qwen/Qwen2.5-1.5B-Instruct` |
| `UnicodeDecodeError` | Make sure `$env:PYTHONUTF8 = "1"` is set in the window |
| Hugging Face download is slow or times out | Rerun the same command: downloads resume from `F:\NLP\hf_cache` |
| Laptop is slow or hot during long runs | Normal. Keep it plugged in on a hard surface. Runs can be stopped and restarted per system |
