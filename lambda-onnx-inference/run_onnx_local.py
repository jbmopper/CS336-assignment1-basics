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
    tokenizer: t.Tokenizer,
    prefill: rt.InferenceSession,
    decode: rt.InferenceSession
) -> dict[str, Any]:

    eot_token = SPECIAL_TOKENS[0]
    ctx = CONTEXT_LENGTH

    output = prompt
    generated = 0

    prefill_input_names = [i.name for i in prefill.get_inputs()]
    decode_input_names = [i.name for i in decode.get_inputs()]
    prefill_output_names = [i.name for i in prefill.get_outputs()]
    decode_output_names = [i.name for i in decode.get_outputs()]

    encoded_prompt = tokenizer.encode(prompt)[-ctx:]
    model_in = np.asarray([encoded_prompt], dtype=np.int64)
    prefill_output = prefill.run(None, {prefill_input_names[0]: model_in})
    prefill_logits = prefill_output[0]
    kv = prefill_output[1:]
    next_logits = prefill_logits[0, -1, :]
    next_id = _sample_top_p(next_logits, temperature=temperature, top_p=top_p)
    piece = tokenizer.decode([next_id])
    generated += 1
    
    output = output + piece
    print(output)

    while generated < max_new_tokens:
        decode_input = {decode_input_names[0]: np.asarray([[next_id]], dtype=np.int64)}
        for i, name in enumerate(decode_input_names[1:]):
            decode_input[name] = kv[i]
        decode_output = decode.run(None, decode_input)
        decode_logits = decode_output[0]
        kv = decode_output[1:]
        next_logits = decode_logits[0, -1, :]
        next_id = _sample_top_p(next_logits, temperature=temperature, top_p=top_p)
        piece = tokenizer.decode([next_id])
        if piece == eot_token:
            break # pls don't spam endoftext lol
        print(piece)
        generated += 1

    completion = output[len(prompt):]
    return {
        "text": output,
        "prompt": prompt,
        "completion": completion,
        "tokens_generated": generated,
        "context_length": ctx,
    }



class Inferrer():
    def __init__(
        self,
        tokenzier_path: str,
        special_tokens: list[str],
        prefill_snapshot_path: str,
        decode_snapshot_path: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ):
        self.tokenizer = _setup_tokenizer(tokenzier_path, special_tokens=["<|endoftext|>"])
        self.prefill = rt.InferenceSession(prefill_snapshot_path, providers=["CPUExecutionProvider"])
        self.decode = rt.InferenceSession(decode_snapshot_path, providers=["CPUExecutionProvider"])
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
    def generate(self, prompt) -> dict[str, Any]:
        return _generate(
            prompt,
            self.max_new_tokens,
            self.temperature,
            self.top_p,
            self.tokenizer,
            self.prefill,
            self.decode
        )
    def print_snapshot_io(self):
        _print_snapshot_io(self.prefill, self.decode)



def main():
    tokenizer = _setup_tokenizer(TOKENIZER_PATH, SPECIAL_TOKENS)
    prefill = rt.InferenceSession(PREFILL_PATH, providers=["CPUExecutionProvider"])
    decode = rt.InferenceSession(DECODE_PATH, providers=["CPUExecutionProvider"])
    _print_snapshot_io(prefill, decode)

    max_new_tokens = 1024
    temperature = 1.
    top_p = .9
    
    _generate(PROMPT, max_new_tokens, temperature, top_p, tokenizer, prefill, decode)
    



if __name__ == "__main__":
    main()