# Modal L4 ONNX Inference (Repo Scaffold)

Run CS336 autoregressive inference on a Modal L4 GPU using ONNX Runtime GPU.

## What This Scaffold Includes

- `app.py`: Modal app with:
  - `@app.cls(gpu=modal.gpu.L4(...))` model container
  - `@modal.web_endpoint` HTTP generation endpoint
  - local CLI entrypoint for smoke tests
- `tokenizer.py`: Minimal BPE runtime (same format as this repo tokenizer assets)
- Optional S3 artifact loading (model + metadata + tokenizer)

## Artifacts Needed

- `model.onnx`
- `model.onnx.metadata.json` (from `cs336_basics.export_onnx`)
- tokenizer files:
  - `vocab.json`
  - `merges.pkl`

## Export ONNX First

From the main repo:

```bash
uv run python -m cs336_basics.export_onnx \
  --checkpoint checkpoints/optimizer_sweeps/<run_id>/best.pt \
  --out artifacts/model.onnx
```

## Local Files Option (baked into deploy context)

Copy artifacts into this scaffold:

```bash
mkdir -p modal-onnx-inference/tokenizer
cp artifacts/model.onnx modal-onnx-inference/model.onnx
cp artifacts/model.onnx.metadata.json modal-onnx-inference/model.onnx.metadata.json
cp tokenizers/tinystories/vocab.json modal-onnx-inference/tokenizer/
cp tokenizers/tinystories/merges.pkl modal-onnx-inference/tokenizer/
```

Deploy:

```bash
cd modal-onnx-inference
modal deploy app.py
```

## S3 Option (recommended for larger artifacts)

Set environment variables in Modal for artifact locations:

- `MODEL_ONNX_S3_URI` (e.g. `s3://bucket/path/model.onnx`)
- `MODEL_METADATA_S3_URI` (optional)
- `TOKENIZER_S3_URI` (prefix containing `vocab.json` and `merges.pkl`)

Then deploy:

```bash
cd modal-onnx-inference
modal deploy app.py
```

## Endpoint Payload

```json
{
  "prompt": "Once upon a time",
  "max_new_tokens": 64,
  "temperature": 1.0,
  "top_p": 0.9
}
```

## Local Smoke Test via Modal

```bash
cd modal-onnx-inference
modal run app.py::smoke --prompt "Hello"
```

## Publish as Separate Repo

```bash
cp -R modal-onnx-inference /tmp/modal-onnx-inference
cd /tmp/modal-onnx-inference
git init
git add .
git commit -m "Initial Modal L4 ONNX inference scaffold"
```
