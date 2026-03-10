import numpy as np
import onnxruntime as rt
import argparse
import tokenizer as t
from typing import Any



ARTIFACT_BASEPATH = "./export/"
MODEL_BASEPATH = "models/"
TOKENIZER_BASEPATH = "tokenizer/"
PREFILL_SUFFIX = "_prefill_best.onnx"
DECODE_SUFFIX = "_decode_best.onnx"
MODEL_NAME = "baseline"

PREFILL_PATH = ARTIFACT_BASEPATH + MODEL_BASEPATH + MODEL_NAME + PREFILL_SUFFIX
DECODE_PATH = ARTIFACT_BASEPATH + MODEL_BASEPATH + MODEL_NAME + DECODE_SUFFIX
TOKENIZER_PATH = ARTIFACT_BASEPATH + TOKENIZER_BASEPATH

CONTEXT_LENGTH = 256
SPECIAL_TOKENS = ["<|endoftext|>"]
PROMPT = " "


def _print_snapshot_io(prefill: rt.InferenceSession, decode: rt.InferenceSession):
    print("prefill inputs")
    for inp in prefill.get_inputs():
        print(inp.name, inp.shape, inp.type)

    print("prefill outputs")
    for outp in prefill.get_outputs():
        print(outp.name, outp.shape, outp.type)

    print("decode inputs")
    for inp in decode.get_inputs():
        print(inp.name, inp.shape, inp.type)

    print("decode outputs")
    for outp in decode.get_outputs():
        print(outp.name, outp.shape, outp.type)

def _setup_tokenizer(path, special_tokens):
    vocab, merges = t.load_bpe(path)
    tokenizer = t.Tokenizer(vocab, merges, special_tokens)
    return tokenizer

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

def _generate(
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    tokenizer,
    prefill: rt.InferenceSession,
    decode: rt.InferenceSession
) -> dict[str, Any]:


    eot_token = SPECIAL_TOKENS[0]
    ctx = CONTEXT_LENGTH

    output = prompt
    generated = 0

    prefill_inputs = [i.name for i in prefill.get_inputs]
    decode_inputs = [i.name for i in decode.get_inputs]
    prefill_outputs = [i.name for i in prefill.get_outputs]
    decode_outputs = [i.name for i in decode.get_outputs]

    for _ in range(max_new_tokens):
        ids = tokenizer.encode(output)[-ctx:]
        model_in = np.asarray([ids], dtype=np.int64)
        prefill_logits = prefill.run(None, {prefill_inputs[0]: model_in})[0]
        next_logits = prefill_logits[0, -1, :]
        next_id = _sample_top_p(next_logits, temperature=temperature, top_p=top_p)
        piece = tokenizer.decode([next_id])
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






def main():
    tokenizer = _setup_tokenizer(TOKENIZER_PATH, SPECIAL_TOKENS)
    prefill = rt.InferenceSession(PREFILL_PATH, providers=["CPUExecutionProvider"])
    decode = rt.InferenceSession(DECODE_PATH, providers=["CPUExecutionProvider"])
    _print_snapshot_io(prefill, decode)
    



if __name__ == "__main__":
    main()