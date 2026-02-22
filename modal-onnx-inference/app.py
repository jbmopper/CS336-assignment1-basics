"""Modal app for ONNX autoregressive inference on L4."""

from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import boto3
import modal
import numpy as np
import onnxruntime as ort

from tokenizer import Tokenizer, load_bpe

APP_NAME = "cs336-onnx-l4"
ROOT_DIR = Path(__file__).resolve().parent

image = modal.Image.from_registry(
    "nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04",
    add_python="3.11",
).pip_install_from_requirements(str(ROOT_DIR / "requirements.txt"))

# If local artifacts are present at deploy time, bundle them into the image.
local_model = ROOT_DIR / "model.onnx"
local_metadata = ROOT_DIR / "model.onnx.metadata.json"
local_tokenizer = ROOT_DIR / "tokenizer"
if local_model.exists():
    image = image.add_local_file(str(local_model), remote_path="/root/model.onnx")
if local_metadata.exists():
    image = image.add_local_file(str(local_metadata), remote_path="/root/model.onnx.metadata.json")
if local_tokenizer.exists():
    image = image.add_local_dir(str(local_tokenizer), remote_path="/root/tokenizer")

app = modal.App(APP_NAME, image=image)
_S3_CLIENT: Any | None = None


def _get_s3_client() -> Any:
    global _S3_CLIENT
    if _S3_CLIENT is None:
        _S3_CLIENT = boto3.client("s3")
    return _S3_CLIENT


def _parse_s3_uri(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme != "s3":
        raise ValueError(f"Expected s3:// URI, got: {uri}")
    return parsed.netloc, parsed.path.lstrip("/")


def _download_file_s3(s3_uri: str, local_path: Path) -> None:
    bucket, key = _parse_s3_uri(s3_uri)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    _get_s3_client().download_file(bucket, key, str(local_path))


def _download_tokenizer_s3(prefix_uri: str, local_dir: Path) -> None:
    prefix = prefix_uri.rstrip("/")
    local_dir.mkdir(parents=True, exist_ok=True)
    _download_file_s3(f"{prefix}/vocab.json", local_dir / "vocab.json")
    _download_file_s3(f"{prefix}/merges.pkl", local_dir / "merges.pkl")


def _load_metadata(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_artifacts() -> tuple[Path, Path, Path | None]:
    cwd = Path("/root")
    model_path = Path(os.environ.get("MODEL_ONNX_PATH", str(cwd / "model.onnx")))
    tokenizer_dir = Path(os.environ.get("TOKENIZER_DIR", str(cwd / "tokenizer")))
    metadata_path = Path(os.environ.get("MODEL_METADATA_PATH", str(cwd / "model.onnx.metadata.json")))

    model_s3 = os.environ.get("MODEL_ONNX_S3_URI")
    tokenizer_s3 = os.environ.get("TOKENIZER_S3_URI")
    metadata_s3 = os.environ.get("MODEL_METADATA_S3_URI")

    if model_s3 or tokenizer_s3 or metadata_s3:
        tmp = Path(tempfile.gettempdir()) / "cs336_modal_artifacts"
        tmp.mkdir(parents=True, exist_ok=True)
        if model_s3:
            model_path = tmp / "model.onnx"
            _download_file_s3(model_s3, model_path)
        if tokenizer_s3:
            tokenizer_dir = tmp / "tokenizer"
            _download_tokenizer_s3(tokenizer_s3, tokenizer_dir)
        if metadata_s3:
            metadata_path = tmp / "model.onnx.metadata.json"
            _download_file_s3(metadata_s3, metadata_path)

    metadata_file = metadata_path if metadata_path.exists() else None
    return model_path, tokenizer_dir, metadata_file


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


@app.cls(
    gpu=modal.gpu.L4(count=1),
    timeout=60 * 20,
    scaledown_window=60 * 10,
    container_idle_timeout=60 * 10,
)
class InferenceModel:
    _session: ort.InferenceSession
    _tokenizer: Tokenizer
    _input_name: str
    _meta: dict[str, Any]

    @modal.enter()
    def load(self) -> None:
        model_path, tokenizer_dir, metadata_path = _resolve_artifacts()
        if not model_path.exists():
            raise FileNotFoundError("ONNX model not found. Set MODEL_ONNX_PATH or MODEL_ONNX_S3_URI.")
        if not (tokenizer_dir / "vocab.json").exists() or not (tokenizer_dir / "merges.pkl").exists():
            raise FileNotFoundError("Tokenizer files missing. Set TOKENIZER_DIR or TOKENIZER_S3_URI.")

        self._meta = _load_metadata(metadata_path)
        vocab, merges = load_bpe(tokenizer_dir)
        special_tokens = self._meta.get("special_tokens", ["<|endoftext|>"])
        self._tokenizer = Tokenizer(vocab, merges, special_tokens=special_tokens)

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._session = ort.InferenceSession(str(model_path), sess_options=sess_opts, providers=providers)
        self._input_name = self._session.get_inputs()[0].name

    def _context_length(self) -> int:
        model_cfg = self._meta.get("model_settings", {})
        return int(model_cfg.get("context_length", os.environ.get("DEFAULT_CONTEXT_LENGTH", 256)))

    @modal.method()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 64,
        temperature: float = 1.0,
        top_p: float = 0.9,
    ) -> dict[str, Any]:
        if not prompt:
            raise ValueError("prompt must be non-empty")

        eot_token = os.environ.get("EOT_TOKEN", "<|endoftext|>")
        top_p = min(max(float(top_p), 1e-4), 1.0)
        max_new_tokens = max(1, min(int(max_new_tokens), int(os.environ.get("MAX_NEW_TOKENS_CAP", 512))))

        output = prompt
        generated = 0
        ctx = self._context_length()

        start = time.perf_counter()
        for _ in range(max_new_tokens):
            ids = self._tokenizer.encode(output)[-ctx:]
            model_in = np.asarray([ids], dtype=np.int64)
            logits = self._session.run(None, {self._input_name: model_in})[0]
            next_logits = logits[0, -1, :]
            next_id = _sample_top_p(next_logits, temperature=temperature, top_p=top_p)
            piece = self._tokenizer.decode([next_id])
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


@app.function(timeout=60 * 20)
@modal.web_endpoint(method="POST")
def generate_web(payload: dict[str, Any]) -> dict[str, Any]:
    model = InferenceModel()
    prompt = str(payload.get("prompt", ""))
    if not prompt:
        return {"error": "Missing required field: prompt"}
    return model.generate.remote(
        prompt=prompt,
        max_new_tokens=int(payload.get("max_new_tokens", 64)),
        temperature=float(payload.get("temperature", 1.0)),
        top_p=float(payload.get("top_p", 0.9)),
    )


@app.local_entrypoint()
def smoke(
    prompt: str = "Once upon a time",
    max_new_tokens: int = 32,
    temperature: float = 1.0,
    top_p: float = 0.9,
) -> None:
    result = InferenceModel().generate.remote(
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
    )
    print(json.dumps(result, indent=2))
