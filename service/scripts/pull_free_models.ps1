# Pull free, open-weight models into Ollama so they appear in the notebook's model picker.
#   powershell -ExecutionPolicy Bypass -File service\scripts\pull_free_models.ps1          (fit a 4 GB GPU)
#   powershell -ExecutionPolicy Bypass -File service\scripts\pull_free_models.ps1 -Large   (+ 7-8B models, partly on CPU)
param([switch]$Large)

$small = @(
  "qwen2.5:3b",        # the research model (already calibrated)
  "qwen2.5:1.5b",
  "qwen3:4b",          # thinks before answering (slower)
  "qwen3:1.7b",
  "llama3.2:3b",
  "llama3.2:1b",
  "gemma3:4b",
  "phi4-mini",
  "granite4:3b",
  "deepseek-r1:1.5b",  # reasoning model (slow, thinks first)
  "smollm2:1.7b"
)
$bigModels = @("qwen2.5:7b", "qwen3:8b", "llama3.1:8b", "deepseek-r1:7b", "mistral:7b", "olmo-3:7b", "granite4.2:8b")

$models = if ($Large) { $small + $bigModels } else { $small }
foreach ($m in $models) {
  Write-Host "==> ollama pull $m"
  ollama pull $m
}
Write-Host "Done. In the notebook, open the model menu and press the refresh icon."
