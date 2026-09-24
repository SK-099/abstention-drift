# Windows equivalent of setup_data.sh: download MTRAG + MTRAG-UN (Apache-2.0) into data\
New-Item -ItemType Directory -Force -Path data | Out-Null
if (-not (Test-Path "data\mt-rag-benchmark")) {
    git clone --depth 1 https://github.com/IBM/mt-rag-benchmark.git data\mt-rag-benchmark
}
Write-Host "MTRAG ready at data\mt-rag-benchmark"
