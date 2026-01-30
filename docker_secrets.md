# Docker secrets setup (RunPod)

This repo's `Dockerfile.train` uses `docker/entrypoint.sh` to write credentials
to files at container startup. RunPod secrets must be injected as environment
variables, then the entrypoint converts them into:

- `~/.ssh/id_ed25519` and `~/.ssh/known_hosts`
- `~/.aws/credentials` and `~/.aws/config`

## RunPod secret mapping (recommended)

### Git (SSH)

- `SSH_PRIVATE_KEY={{ RUNPOD_SECRET_ssh_private_key }}`
  - Or use base64: `SSH_PRIVATE_KEY_B64={{ RUNPOD_SECRET_ssh_private_key_b64 }}`
- Optional: `GIT_SSH_HOST=github.com` (default)
- Optional: `SSH_KNOWN_HOSTS={{ RUNPOD_SECRET_known_hosts }}`

### S3 (AWS)

- `AWS_ACCESS_KEY_ID={{ RUNPOD_SECRET_aws_access_key_id }}`
- `AWS_SECRET_ACCESS_KEY={{ RUNPOD_SECRET_aws_secret_access_key }}`
- Optional: `AWS_SESSION_TOKEN={{ RUNPOD_SECRET_aws_session_token }}`
- Optional: `AWS_DEFAULT_REGION=us-west-2` (or `AWS_REGION`)

## Base64 SSH key helper

Create the base64 string on your machine:

```
base64 -w 0 ~/.ssh/id_ed25519
```

## Security notes

- Prefer short-lived AWS creds (STS) and least-privilege policies.
- Avoid long-lived root keys in third-party environments.
- Rotate and revoke credentials after runs.
