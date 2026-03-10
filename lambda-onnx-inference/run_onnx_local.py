import numpy
import onnxruntime as rt
import argparse
import tokenizer as t



ARTIFACT_BASEPATH = "./export/"
MODEL_BASEPATH = "models/"
TOKENIZER_BASEPATH = "tokenizer/"
PREFILL_SUFFIX = "_prefill_best.onnx"
DECODE_SUFFIX = "_decode_best.onnx"
MODEL_NAME = "baseline"

PREFILL_PATH = ARTIFACT_BASEPATH + MODEL_BASEPATH + MODEL_NAME + PREFILL_SUFFIX
DECODE_PATH = ARTIFACT_BASEPATH + MODEL_BASEPATH + MODEL_NAME + DECODE_SUFFIX
TOKENIZER_PATH = ARTIFACT_BASEPATH + TOKENIZER_BASEPATH

SPECIAL_TOKENS = ["<|endoftext|>"]
PROMPT = " "


def _print_snapshot_io(prefill, decode):
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

def main():
    tokenizer = _setup_tokenizer(TOKENIZER_PATH, SPECIAL_TOKENS)
    prefill = rt.InferenceSession(PREFILL_PATH, providers=["CPUExecutionProvider"])
    decode = rt.InferenceSession(DECODE_PATH, providers=["CPUExecutionProvider"])
    _print_snapshot_io(prefill, decode)
    



if __name__ == "__main__":
    main()