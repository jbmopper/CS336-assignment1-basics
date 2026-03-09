#!/usr/bin/env python3
"""Export a trained TransformerLM checkpoint to ONNX (eval mode).

Usage:
    uv run python -m cs336_basics.export_onnx \
        --checkpoint checkpoints/optimizer_sweeps/<run_id>/best.pt \
        --prefill_out artifacts/model_prefill.onnx \
        --decode_out artifacts/model_decode.onnx
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from cs336_basics import TransformerLM


def _load_checkpoint(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    # Explicitly disable weights_only to support checkpoints that also store config.
    return torch.load(path, map_location="cpu", weights_only=False)


def _build_model_from_checkpoint(ckpt_obj: dict[str, Any]) -> tuple[TransformerLM, dict[str, Any], dict[str, Any]]:
    if "config" not in ckpt_obj:
        raise ValueError("Checkpoint does not contain `config` needed for model reconstruction.")
    if "model" not in ckpt_obj:
        raise ValueError("Checkpoint does not contain `model` state dict.")

    config = ckpt_obj["config"]
    model_cfg = config.get("model_settings", config)

    required = ["vocab_size", "d_model", "num_heads", "num_layers", "d_ff", "context_length"]
    missing = [k for k in required if k not in model_cfg]
    if missing:
        raise ValueError(f"Missing required model config keys: {missing}")

    model = TransformerLM(
        model_cfg["vocab_size"],
        model_cfg["d_model"],
        model_cfg["num_heads"],
        model_cfg["num_layers"],
        model_cfg["d_ff"],
        model_cfg["context_length"],
        model_cfg.get("rope_theta", 10000.0),
        norm_mode=model_cfg.get("norm_mode", "pre"),
        use_rope=model_cfg.get("use_rope", True),
        ffn_type=model_cfg.get("ffn_type", "swiglu"),
        ffn_hidden_dim=model_cfg.get("ffn_hidden_dim"),
        final_norm=model_cfg.get("final_norm"),
    )
    model.load_state_dict(ckpt_obj["model"], strict=True)
    model.eval()
    return model, config, model_cfg


def _export_onnx(
    model: TransformerLM,
    prefill_out_path: Path,
    decode_out_path: Path,
    vocab_size: int,
    batch_size: int,
    seq_len: int,
    opset: int,
    dynamic: bool,
) -> tuple[dict[str, dict[int, str]] | None, torch.Tensor]:
    prefill_out_path.parent.mkdir(parents=True, exist_ok=True)
    decode_out_path.parent.mkdir(parents=True, exist_ok=True)

    d_head = model.d_model // model.num_heads
    dummy = torch.randint(low=0, high=vocab_size, size=(batch_size, seq_len), dtype=torch.long)
    kv_dummy = [
        (
            torch.rand(size=(batch_size, model.num_heads, seq_len, d_head), dtype=torch.float),
            torch.rand(size=(batch_size, model.num_heads, seq_len, d_head), dtype=torch.float)
        ) for _ in range(model.num_layers)
    ]
    
    dynamic_axes = None
    if dynamic:
        dynamic_axes = {
            "input_ids": {0: "batch", 1: "seq"},
            "logits": {0: "batch", 1: "seq"},
        }

    past_cache_names = [
        name
        for i in range(model.num_layers)
        for name in (f"past_k_{i}", f"past_v_{i}")
    ]
    present_cache_names = [
        name
        for i in range(model.num_layers)
        for name in (f"present_k_{i}", f"present_v_{i}")
    ]

    with torch.inference_mode():
        torch.onnx.export(
            model,
            dummy,
            str(prefill_out_path),
            export_params=True,
            opset_version=opset,
            do_constant_folding=True,
            input_names=["input_ids"],
            output_names=["logits"] + present_cache_names,
            dynamic_axes=dynamic_axes,
        )

        torch.onnx.export(
            model,
            (dummy, kv_dummy),
            str(decode_out_path),
            export_params=True,
            opset_version=opset,
            do_constant_folding=True,
            input_names=["input_ids"] + past_cache_names,
            output_names=["logits"] + present_cache_names,
            dynamic_axes=dynamic_axes,
        )

    return dynamic_axes, dummy


def _validate_export(onnx_path: Path, model: TransformerLM, sample_input: torch.Tensor) -> dict[str, float] | None:
    try:
        import onnxruntime as ort
    except Exception:
        print("Validation skipped: onnxruntime not installed.")
        return None

    with torch.inference_mode():
        torch_out = model(sample_input)[0].cpu().numpy()

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ort_inputs = {"input_ids": sample_input.cpu().numpy().astype(np.int64)}
    ort_out = sess.run(["logits"], ort_inputs)[0]

    abs_diff = np.abs(torch_out - ort_out)
    metrics = {
        "max_abs_diff": float(np.max(abs_diff)),
        "mean_abs_diff": float(np.mean(abs_diff)),
    }
    print(
        "Validation complete: "
        f"max_abs_diff={metrics['max_abs_diff']:.6g}, "
        f"mean_abs_diff={metrics['mean_abs_diff']:.6g}"
    )
    return metrics


def _write_metadata(
    metadata_path: Path,
    checkpoint: Path,
    prefill_onnx_path: Path,
    decode_onnx_path: Path,
    model_cfg: dict[str, Any],
    config: dict[str, Any],
    opset: int,
    dynamic_axes: dict[str, dict[int, str]] | None,
    validation_metrics: dict[str, float] | None,
) -> None:
    metadata = {
        "exported_at_utc": datetime.now(tz=timezone.utc).isoformat(),
        "checkpoint_path": str(checkpoint),
        "onnx_prefill_path": str(prefill_onnx_path),
        "onnx_decode_path": str(decode_onnx_path),
        "onnx_opset": opset,
        "model_settings": model_cfg,
        "tokenizer_dir": config.get("tokenizer_dir"),
        "special_tokens": config.get("special_tokens", ["<|endoftext|>"]),
        "dynamic_axes": dynamic_axes,
        "validation": validation_metrics,
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Wrote metadata: {metadata_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export TransformerLM checkpoint to ONNX.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to .pt checkpoint.")
    parser.add_argument("--prefill_out", type=Path, required=True, help="Output prefill ONNX path.")
    parser.add_argument("--decode_out", type=Path, required=True, help="Output decode ONNX path.")
    parser.add_argument(
        "--metadata-out",
        type=Path,
        default=None,
        help="Output metadata JSON path (default: <prefill_out>.metadata.json).",
    )
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version.")
    parser.add_argument("--batch-size", type=int, default=1, help="Dummy batch size used for export.")
    parser.add_argument("--seq-len", type=int, default=64, help="Dummy sequence length used for export.")
    parser.add_argument(
        "--static-shapes",
        action="store_true",
        help="Export fixed-shape graph (default uses dynamic batch/sequence axes).",
    )
    parser.add_argument(
        "--skip-validate",
        action="store_true",
        help="Skip ONNXRuntime parity validation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    ckpt_obj = _load_checkpoint(args.checkpoint)
    model, config, model_cfg = _build_model_from_checkpoint(ckpt_obj)

    seq_len = min(args.seq_len, int(model_cfg.get("context_length", args.seq_len)))
    if seq_len <= 0:
        raise ValueError("seq_len must be > 0")
    if args.batch_size <= 0:
        raise ValueError("batch_size must be > 0")

    dynamic_axes, dummy = _export_onnx(
        model=model,
        prefill_out_path=args.prefill_out,
        decode_out_path=args.decode_out,
        vocab_size=int(model_cfg["vocab_size"]),
        batch_size=args.batch_size,
        seq_len=seq_len,
        opset=args.opset,
        dynamic=not args.static_shapes,
    )
    print(f"Wrote ONNX: {args.prefill_out}, {args.decode_out}")

    validation = None
    if not args.skip_validate:
        validation = _validate_export(args.prefill_out, model, dummy)

    metadata_out = args.metadata_out or args.prefill_out.with_suffix(args.prefill_out.suffix + ".metadata.json")
    _write_metadata(
        metadata_path=metadata_out,
        checkpoint=args.checkpoint,
        onnx_prefill_path=args.prefill_out,
        onnx_decode_path=args.decode_out,
        model_cfg=model_cfg,
        config=config,
        opset=args.opset,
        dynamic_axes=dynamic_axes,
        validation_metrics=validation,
    )


if __name__ == "__main__":
    main()
