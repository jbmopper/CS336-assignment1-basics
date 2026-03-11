from fastapi import FastAPI
from pydantic import BaseModel
import run_onnx_local as o
import yaml
import boto3

config = yaml.safe_load(open("inference_config.yaml"))
s3 = boto3.client("s3")

app = FastAPI()

@app.get("/warmup")
def warmup_onnx():
    








