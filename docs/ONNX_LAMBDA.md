# ONNX Export and Lambda Inference

This repo now includes:

- `cs336_basics/export_onnx.py` to export checkpoints in eval mode to ONNX.
- `lambda-onnx-inference/` as a standalone AWS Lambda scaffold using ONNX Runtime.

If you want GPU serving on Modal L4 instead, see `docs/ONNX_MODAL.md` and `modal-onnx-inference/`.

## 1. Export Checkpoint to ONNX

```bash
uv run python -m cs336_basics.export_onnx \
  --checkpoint checkpoints/optimizer_sweeps/<run_id>/best.pt \
  --out artifacts/model.onnx
```

Outputs:

- `artifacts/model.onnx`
- `artifacts/model.onnx.metadata.json`

Notes:

- Export runs in eval mode.
- By default, dynamic batch and sequence axes are enabled.
- If `onnxruntime` is installed, a parity check runs automatically.

## 2. Prepare Lambda Scaffold

```bash
mkdir -p lambda-onnx-inference/tokenizer
cp artifacts/model.onnx lambda-onnx-inference/model.onnx
cp artifacts/model.onnx.metadata.json lambda-onnx-inference/model.onnx.metadata.json
cp tokenizers/tinystories/vocab.json lambda-onnx-inference/tokenizer/
cp tokenizers/tinystories/merges.pkl lambda-onnx-inference/tokenizer/
```

Build image:

```bash
cd lambda-onnx-inference
docker build -t cs336-lambda-onnx:latest .
```

## 3. Runtime Options

The Lambda handler accepts artifacts from:

- Local image paths (`MODEL_ONNX_PATH`, `TOKENIZER_DIR`, `MODEL_METADATA_PATH`)
- Or S3 (`MODEL_ONNX_S3_URI`, `TOKENIZER_S3_URI`, `MODEL_METADATA_S3_URI`)

Request payload:

```json
{
  "prompt": "Once upon a time",
  "max_new_tokens": 64,
  "temperature": 1.0,
  "top_p": 0.9
}
```
