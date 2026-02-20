"""AWS Lambda handler for ONNX-based autoregressive inference."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import boto3
import numpy as np
import onnxruntime as ort

from tokenizer import Tokenizer, load_bpe

S3 = boto3.client("s3")

_SESSION: ort.InferenceSession | None = None
_TOKENIZER: Tokenizer | None = None
_META: dict[str, Any] = {}
_INPUT_NAME: str | None = None


def _parse_s3_uri(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme != "s3":
        raise ValueError(f"Expected s3:// URI, got: {uri}")
    return parsed.netloc, parsed.path.lstrip("/")


def _download_file_s3(s3_uri: str, local_path: Path) -> None:
    bucket, key = _parse_s3_uri(s3_uri)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    S3.download_file(bucket, key, str(local_path))


def _download_tokenizer_s3(prefix_uri: str, local_dir: Path) -> None:
    prefix = prefix_uri.rstrip("/")
    local_dir.mkdir(parents=True, exist_ok=True)
    _download_file_s3(f"{prefix}/vocab.json", local_dir / "vocab.json")
    _download_file_s3(f"{prefix}/merges.pkl", local_dir / "merges.pkl")


def _load_metadata(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_local_artifacts() -> tuple[Path, Path, Path | None]:
    # Defaults for image-bundled artifacts.
    model_path = Path(os.environ.get("MODEL_ONNX_PATH", "/var/task/model.onnx"))
    tokenizer_dir = Path(os.environ.get("TOKENIZER_DIR", "/var/task/tokenizer"))
    metadata_path = Path(os.environ.get("MODEL_METADATA_PATH", "/var/task/model.onnx.metadata.json"))

    # Optional S3 overrides.
    model_s3 = os.environ.get("MODEL_ONNX_S3_URI")
    tokenizer_s3 = os.environ.get("TOKENIZER_S3_URI")
    metadata_s3 = os.environ.get("MODEL_METADATA_S3_URI")

    if model_s3:
        model_path = Path("/tmp/model.onnx")
        _download_file_s3(model_s3, model_path)
    if tokenizer_s3:
        tokenizer_dir = Path("/tmp/tokenizer")
        _download_tokenizer_s3(tokenizer_s3, tokenizer_dir)
    if metadata_s3:
        metadata_path = Path("/tmp/model.onnx.metadata.json")
        _download_file_s3(metadata_s3, metadata_path)

    metadata_file = metadata_path if metadata_path.exists() else None
    return model_path, tokenizer_dir, metadata_file


def _ensure_loaded() -> None:
    global _SESSION, _TOKENIZER, _META, _INPUT_NAME
    if _SESSION is not None and _TOKENIZER is not None and _INPUT_NAME is not None:
        return

    model_path, tokenizer_dir, metadata_path = _resolve_local_artifacts()

    if not model_path.exists():
        raise FileNotFoundError(
            "ONNX model not found. Provide MODEL_ONNX_PATH or MODEL_ONNX_S3_URI."
        )
    if not (tokenizer_dir / "vocab.json").exists() or not (tokenizer_dir / "merges.pkl").exists():
        raise FileNotFoundError(
            "Tokenizer files not found. Provide TOKENIZER_DIR or TOKENIZER_S3_URI."
        )

    _META = _load_metadata(metadata_path)
    vocab, merges = load_bpe(tokenizer_dir)
    special_tokens = _META.get("special_tokens", ["<|endoftext|>"])
    _TOKENIZER = Tokenizer(vocab, merges, special_tokens=special_tokens)

    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    _SESSION = ort.InferenceSession(
        str(model_path),
        sess_options=sess_opts,
        providers=["CPUExecutionProvider"],
    )
    _INPUT_NAME = _SESSION.get_inputs()[0].name


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    exp = np.exp(shifted)
    return exp / np.clip(np.sum(exp), 1e-12, None)


def _sample_top_p(logits: np.ndarray, temperature: float, top_p: float) -> int:
    t = max(float(temperature), 1e-5)
    probs = _softmax(logits / t)
    order = np.argsort(-probs)
    sorted_probs = probs[order]
    cdf = np.cumsum(sorted_probs)
    cutoff = int(np.searchsorted(cdf, top_p, side="left")) + 1
    cutoff = min(max(cutoff, 1), sorted_probs.shape[0])
    kept_idx = order[:cutoff]
    kept_probs = probs[kept_idx]
    kept_probs = kept_probs / np.clip(kept_probs.sum(), 1e-12, None)
    return int(np.random.choice(kept_idx, p=kept_probs))


def _get_payload(event: dict[str, Any]) -> dict[str, Any]:
    body = event.get("body")
    if isinstance(body, str):
        try:
            return json.loads(body)
        except json.JSONDecodeError:
            return {}
    if isinstance(body, dict):
        return body
    return event


def _context_length() -> int:
    model_cfg = _META.get("model_settings", {})
    return int(model_cfg.get("context_length", os.environ.get("DEFAULT_CONTEXT_LENGTH", 256)))


def _generate(
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> dict[str, Any]:
    assert _SESSION is not None
    assert _TOKENIZER is not None
    assert _INPUT_NAME is not None

    eot_token = os.environ.get("EOT_TOKEN", "<|endoftext|>")
    ctx = _context_length()

    output = prompt
    generated = 0

    start = time.perf_counter()
    for _ in range(max_new_tokens):
        ids = _TOKENIZER.encode(output)[-ctx:]
        model_in = np.asarray([ids], dtype=np.int64)
        logits = _SESSION.run(None, {_INPUT_NAME: model_in})[0]
        next_logits = logits[0, -1, :]
        next_id = _sample_top_p(next_logits, temperature=temperature, top_p=top_p)
        piece = _TOKENIZER.decode([next_id])
        if piece == eot_token:
            break
        output += piece
        generated += 1

    elapsed_ms = (time.perf_counter() - start) * 1000.0
    completion = output[len(prompt):]
    return {
        "text": output,
        "prompt": prompt,
        "completion": completion,
        "tokens_generated": generated,
        "context_length": ctx,
        "latency_ms": elapsed_ms,
    }


def lambda_handler(event: dict[str, Any], _context: Any) -> dict[str, Any]:
    try:
        _ensure_loaded()
        payload = _get_payload(event)

        prompt = str(payload.get("prompt", ""))
        if not prompt:
            return {
                "statusCode": 400,
                "headers": {"Content-Type": "application/json"},
                "body": json.dumps({"error": "Missing required field: prompt"}),
            }

        max_new_tokens = int(payload.get("max_new_tokens", 64))
        max_new_tokens = max(1, min(max_new_tokens, int(os.environ.get("MAX_NEW_TOKENS_CAP", 256))))
        temperature = float(payload.get("temperature", 1.0))
        top_p = float(payload.get("top_p", 0.9))
        top_p = min(max(top_p, 1e-4), 1.0)

        result = _generate(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        return {
            "statusCode": 200,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps(result),
        }
    except Exception as exc:
        return {
            "statusCode": 500,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"error": str(exc)}),
        }
