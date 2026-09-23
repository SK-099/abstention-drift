#!/usr/bin/env bash
# Download MTRAG + MTRAG-UN (Apache-2.0) into data/. Corpora stay zipped; our loader reads them in place.
set -euo pipefail
mkdir -p data
if [ ! -d data/mt-rag-benchmark ]; then
  git clone --depth 1 https://github.com/IBM/mt-rag-benchmark.git data/mt-rag-benchmark
fi
echo "MTRAG ready at data/mt-rag-benchmark"
