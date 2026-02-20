# Lambda ONNX Inference (Repo Scaffold)

This folder is a standalone scaffold you can publish as a separate repository for AWS Lambda inference against an ONNX-exported CS336 model.

## What It Does

- Loads ONNX model + tokenizer on cold start.
- Runs autoregressive generation (`top-p` + temperature) on CPU using ONNX Runtime.
- Supports local artifacts or S3-hosted artifacts.

## Expected Artifacts

- `model.onnx`
- `model.onnx.metadata.json` (produced by `cs336_basics.export_onnx`)
- Tokenizer files:
  - `vocab.json`
  - `merges.pkl`

## Quick Start (Container Image)

1. Build artifacts in this repo:

```bash
uv run python -m cs336_basics.export_onnx \
  --checkpoint checkpoints/optimizer_sweeps/<run_id>/best.pt \
  --out artifacts/model.onnx
```

2. Copy artifacts into this scaffold:

```bash
cp artifacts/model.onnx lambda-onnx-inference/model.onnx
cp artifacts/model.onnx.metadata.json lambda-onnx-inference/model.onnx.metadata.json
mkdir -p lambda-onnx-inference/tokenizer
cp tokenizers/tinystories/vocab.json lambda-onnx-inference/tokenizer/
cp tokenizers/tinystories/merges.pkl lambda-onnx-inference/tokenizer/
```

3. Build and push image:

```bash
cd lambda-onnx-inference
docker build -t cs336-lambda-onnx:latest .
```

4. Deploy Lambda from the image and set optional environment variables:

- `MODEL_ONNX_PATH` (default: `/var/task/model.onnx`)
- `MODEL_METADATA_PATH` (default: `/var/task/model.onnx.metadata.json`)
- `TOKENIZER_DIR` (default: `/var/task/tokenizer`)
- `MODEL_ONNX_S3_URI` (optional override)
- `MODEL_METADATA_S3_URI` (optional override)
- `TOKENIZER_S3_URI` (optional override, prefix containing `vocab.json` and `merges.pkl`)

## Event Format

```json
{
  "prompt": "Once upon a time",
  "max_new_tokens": 64,
  "temperature": 1.0,
  "top_p": 0.9
}
```

If invoked via API Gateway proxy, the same payload can be placed under `body` as JSON.

## Response Format

```json
{
  "text": "Once upon a time ...",
  "prompt": "Once upon a time",
  "completion": " ...",
  "tokens_generated": 64,
  "context_length": 256
}
```

## Publishing as a Separate Repo

```bash
cp -R lambda-onnx-inference /tmp/lambda-onnx-inference
cd /tmp/lambda-onnx-inference
git init
git add .
git commit -m "Initial Lambda ONNX inference scaffold"
```
