from pathlib import Path
from urllib.parse import urlparse
from collections.abc import Iterable
import boto3
import run_onnx_local as o
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.sse import EventSourceResponse, ServerSentEvent
from pydantic import BaseModel


BASE_DIR = Path(__file__).resolve().parent
DOWNLOAD_ROOT = BASE_DIR / "tmp"

with (BASE_DIR / "inference_config.yaml").open("r", encoding="utf-8") as config_file:
    config = yaml.safe_load(config_file)

s3 = boto3.client("s3")

app = FastAPI()
app.state.inferrers = {}


class GenerateRequest(BaseModel):
    model: str
    prompt: str = "<|endoftext|>"
    temperature: float = 1.
    top_p: float = 0.9


class WarmupRequest(BaseModel):
    model: str


@app.post("/warmup")
def warmup(req: WarmupRequest):
    model_name = req.model
    model_config = config["models"].get(model_name)
    if model_config is None:
        raise HTTPException(status_code=404, detail=f"Model {model_name} not found")
    model_prefix = model_config["model_uri"].rstrip("/")
    tokenizer_prefix = model_config["tokenizer_uri"].rstrip("/")
    tokenizer_path = DOWNLOAD_ROOT / "tokenizer"
    prefill_path = DOWNLOAD_ROOT / "models" / f"{model_name}{o.PREFILL_SUFFIX}"
    decode_path = DOWNLOAD_ROOT / "models" / f"{model_name}{o.DECODE_SUFFIX}"

    downloads = [
        (f"{model_prefix}/{model_name}{o.PREFILL_SUFFIX}", prefill_path),
        (f"{model_prefix}/{model_name}{o.DECODE_SUFFIX}", decode_path),
        (f"{tokenizer_prefix}/vocab.json", tokenizer_path / "vocab.json"),
        (f"{tokenizer_prefix}/merges.pkl", tokenizer_path / "merges.pkl"),
    ]

    for s3_uri, local_path in downloads:
        parsed = urlparse(s3_uri)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        if not local_path.exists():
            try:
                s3.download_file(parsed.netloc, parsed.path.lstrip("/"), str(local_path))
            except Exception as exc:
                raise HTTPException(status_code=500, detail=f"S3 download error:\n\n{exc}") 

    special_tokens = model_config["special_tokens"]
    if isinstance(special_tokens, str):
        special_tokens = [special_tokens]

    app.state.inferrers[model_name] = o.Inferrer(
        tokenizer_path=str(tokenizer_path),
        special_tokens=special_tokens,
        prefill_snapshot_path=str(prefill_path),
        decode_snapshot_path=str(decode_path),
    )

    return {"status": "ready", "model_name": model_name}


@app.post("/generate", response_class=EventSourceResponse)
def generate(req: GenerateRequest) -> Iterable[ServerSentEvent]:
    for piece in app.state.inferrers[req.model].generate(
        req.prompt,
        1024, # max new tokens
        req.temperature,
        req.top_p,
    ):
        yield ServerSentEvent(data={"token": piece}, event="token")
        
    yield ServerSentEvent(data={"done": True}, event="done")