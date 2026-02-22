# ONNX Inference on Modal L4

This repo includes a Modal L4 scaffold at:

- `/Users/juliusmopper/Dev/stanford-cs336/assignment1-basics/modal-onnx-inference`

## 1. Export ONNX

```bash
cd /Users/juliusmopper/Dev/stanford-cs336/assignment1-basics
uv run python -m cs336_basics.export_onnx \
  --checkpoint checkpoints/optimizer_sweeps/<run_id>/best.pt \
  --out artifacts/model.onnx
```

## 2. Provide Artifacts to Modal

Option A: local files in deploy context

```bash
mkdir -p modal-onnx-inference/tokenizer
cp artifacts/model.onnx modal-onnx-inference/model.onnx
cp artifacts/model.onnx.metadata.json modal-onnx-inference/model.onnx.metadata.json
cp tokenizers/tinystories/vocab.json modal-onnx-inference/tokenizer/
cp tokenizers/tinystories/merges.pkl modal-onnx-inference/tokenizer/
```

Option B: S3 URIs in environment

- `MODEL_ONNX_S3_URI`
- `MODEL_METADATA_S3_URI` (optional)
- `TOKENIZER_S3_URI` (prefix with `vocab.json` + `merges.pkl`)

## 3. Deploy to Modal L4

```bash
cd /Users/juliusmopper/Dev/stanford-cs336/assignment1-basics/modal-onnx-inference
modal deploy app.py
```

`app.py` uses `modal.gpu.L4(count=1)` in the model class.

## 4. Smoke Test

```bash
cd /Users/juliusmopper/Dev/stanford-cs336/assignment1-basics/modal-onnx-inference
modal run app.py::smoke --prompt "Hello"
```
