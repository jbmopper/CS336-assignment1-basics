# Running the Decoder on AWS Lambda

If you want the new ONNX-based path (recommended for Lambda), see `docs/ONNX_LAMBDA.md` and `lambda-onnx-inference/`.

The decoder (`cs336_basics/decode.py`) is currently CLI-only. Here are practical options for running it on **AWS Lambda** (the serverless function service).

---

## Lambda constraints (relevant to inference)

| Limit | Value |
|-------|--------|
| Deployment size (zip) | 50 MB zipped / 250 MB unzipped |
| Container image | Up to 10 GB |
| Memory | 128 MB – 10 GB |
| Timeout | Up to 15 minutes |
| Response payload (sync) | 6 MB |
| `/tmp` | 512 MB – 10 GB (configurable) |
| GPU | Not available on Lambda |

So: **CPU-only inference**, and the runtime + model + tokenizer must fit within the deployment and memory limits.

---

## Option 1: Lambda with container image (recommended)

**Use when:** You want a single deployable unit and can accept cold starts.

- **Build a Docker image** that includes:
  - Python 3.10+ and `torch` (CPU-only to keep size down: `pip install torch --index-url https://download.pytorch.org/whl/cpu`).
  - Your package (`cs336_basics`) and its dependencies.
  - Either:
    - **Checkpoint + tokenizer in the image** (simplest; image can be up to 10 GB), or
    - **Only code in the image** and **checkpoint + tokenizer in S3**; on cold start the handler downloads them to `/tmp` and loads the model (slower cold start, smaller image).
- **Handler:** HTTP (e.g. API Gateway REST or HTTP API) or direct invoke. Parse `prompt`, `max_new_tokens`, `temperature`, `top_p` from the event; call your decode logic; return `{"text": "..."}` (and optionally `{"tokens": [...]}`). Keep `max_new_tokens` modest (e.g. 64–256) so you stay under timeout and 6 MB response.
- **Streaming:** Lambda does not support true response streaming. You get one response when the function returns. For “word-by-word” UX you’d need a different design (e.g. Option 4).

**Pros:** No servers to manage, scales to zero, pay per request.  
**Cons:** Cold starts (model load can be 10–60+ seconds), CPU-only (slower than GPU), no native streaming.

---

## Option 2: Lambda with zip + layers + S3 model

**Use when:** You want to avoid maintaining a container and your total artifact fits Lambda’s zip limits.

- **Problem:** PyTorch CPU-only is still ~200+ MB; with your code and a small checkpoint you can exceed the 250 MB unzipped limit. So this is only viable for **very small** setups (e.g. minimal deps or a tiny “hello world” model).
- **Layout:**
  - **Layer:** PyTorch CPU-only + numpy (and any other large deps). Attach layer(s) to the function.
  - **Function package:** Only `cs336_basics` and a thin handler that reads `ckpt` path (or S3 URI) and tokenizer path from env.
  - **Model/tokenizer:** Stored in S3. On cold start (or first request), download checkpoint and tokenizer files to `/tmp`, then `load_model(...)` and load BPE. Cache in a global variable so later invocations reuse the same process.
- **Handler:** Same as Option 1: event in → decode → `return {"text": "..."}`.

**Pros:** No container build if you stay within size limits.  
**Cons:** Tight size limits; cold start includes S3 download + model load; still no streaming.

---

## Option 3: Lambda + Provisioned Concurrency (or frequent traffic)

**Use when:** You need lower latency and can afford always-warm capacity.

- Same as Option 1 or 2, but set **Provisioned Concurrency** to 1 (or more) so at least one instance stays warm.
- Cold start only happens when scaling beyond provisioned capacity.

**Pros:** Much more stable latency.  
**Cons:** You pay for idle capacity.

---

## Option 4: “Streaming” via polling or WebSocket

Lambda returns one response per invocation. To simulate word-by-word or chunk-by-chunk output:

- **Polling:** Lambda writes each token (or every N tokens) to **DynamoDB** or **S3** (e.g. “job_id” + “chunk_index” + “text”), then returns quickly with `job_id`. Client polls an API that reads from DynamoDB/S3 and returns the latest chunks. More moving parts and latency.
- **Step Functions / multiple invocations:** One invocation per token (or per chunk) is possible in theory but usually too slow and expensive; not recommended.
- **WebSocket API + Lambda:** Lambda can push to a WebSocket connection, but the connection is not long-lived from a single invocation; you’d still need another service (e.g. API Gateway WebSocket + something that holds the connection and calls Lambda or a long-running backend) to get true streaming. So “streaming” here usually implies a hybrid design (see Option 5).

---

## Option 5: Use a long-running service instead of (or in front of) Lambda

**Use when:** You need real streaming, lower latency, or GPU.

| Service | Use case |
|--------|----------|
| **SageMaker** | Real-time or async inference endpoints; GPU; built-in model hosting; can stream responses with certain endpoint types. |
| **ECS Fargate** or **App Runner** | Run a small **Flask/FastAPI** app that loads the model once and serves HTTP; stream with chunked transfer or SSE. |
| **EC2** | Small instance (or GPU) running the same decoder as a service; full control. |

Flow can be: **API Gateway → Lambda** for auth/routing, then Lambda **invokes SageMaker / ECS / EC2** (or returns a signed URL or job ID for polling). Lambda then acts as the “front door,” with the actual decoding and streaming done by the other service.

---

## Scale to zero and cost for 1–2 sessions/month

For **long-running** services (streaming, GPU, or “real” server), only a few options scale down to 0 and keep cost low for very sporadic use (e.g. 1–2 sessions per month).

| Service | Scale to 0? | Cost when idle | Notes |
|--------|-------------|----------------|-------|
| **Lambda** | Yes | $0 | Pay per request + duration. **Best cost profile** for 1–2/month; no streaming. |
| **SageMaker Serverless Inference** | Yes | $0 | Pay per invocation (compute + memory duration). Scales to 0 between requests. **Limits:** ~60 s max invocation, 4 MB payload; good fit if completions are short. |
| **SageMaker Real-Time Endpoint** (scale-to-zero) | Yes | $0 | Set `MinInstanceCount = 0` (Inference Components). Pay only when instances are up. **Cold start:** several minutes when scaling from 0; fine for 1–2/month if you accept that delay. |
| **App Runner** | No | Always paying | `MinSize` cannot be 0; you always have at least one provisioned instance (memory cost). **Not suitable** for 1–2/month. |
| **ECS Fargate** | 0 tasks possible | $0 if 0 tasks | You can have 0 running tasks, but you need something (e.g. Lambda) to start a task on first request; cold start is slow (1–2+ min). When a task runs, pay per vCPU/memory second. Possible but clunky. |
| **EC2** | No | Pay for instance or EBS | Doesn’t scale to 0 unless you stop the instance (and add custom “start on demand” logic). Overkill for 1–2/month. |

**Recommendation for 1–2 sessions/month:**

- **Single-shot completion (no streaming):** Use **Lambda** (container or zip). Scales to 0, pay only for those few invocations; cold start once per session is acceptable.
- **Streaming or GPU, still 1–2/month:** Use **SageMaker Serverless Inference** if your responses fit (≤60 s, ≤4 MB). If you need longer runs or streaming, use a **SageMaker Real-Time endpoint with scale-to-zero** (MinInstanceCount = 0); you pay only for the time instances are up during your 1–2 sessions, and accept a multi-minute cold start when scaling from 0.

Outside AWS, **Google Cloud Run** and **Azure Container Apps** also scale to 0 with pay-per-use; similar cost profile for very low traffic.

---

## Summary

- **Lambda alone:** Best for **single-shot completion**: one prompt in, one full text out. Use a **container image** (Option 1), optionally with model in S3 to keep the image smaller. Keep `max_new_tokens` small and timeouts in mind.
- **Word-by-word / streaming:** Not a good fit for Lambda alone. Use **SageMaker**, **ECS/App Runner**, or **EC2** (Option 5) for real streaming, and optionally put Lambda in front for auth and routing.
- **Cold start:** Mitigate with **Provisioned Concurrency** (Option 3) or by moving inference to a long-running service (Option 5).
- **1–2 sessions/month, scale to 0:** Prefer **Lambda** (best cost, no streaming) or **SageMaker Serverless / Real-Time scale-to-zero** if you need streaming or longer inference.
