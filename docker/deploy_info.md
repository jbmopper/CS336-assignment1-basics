# Deployment info
## v1, runpod

### conainer image
`jbmopper/cs336-basics:train-25.03`

### start command
```bash
bash -lc 'set -e; REPO=/workspace/CS336-assignment1-basics; if [ -d "$REPO/.git" ]; then git -C "$REPO" pull --rebase; else git clone git@github.com:jbmopper/CS336-assignment1-basics.git "$REPO"; fi; cd "$REPO"; uv sync --frozen --python-preference system; export PS1='\''\u@\h:\w\$ '\''; bash'
```

### run commands
```bash
# One-time prep: sync data, add symlinks, start background S3 sync
./runpod/prep_train_compare.sh s3://YOUR-BUCKET/assignment1-basics/checkpoints/model_comp 300

# Train Model A vs B
uv run python -m cs336_basics.train_compare --precision bf16 --checkpoint-dir /workspace/checkpoints/model_comp
```

### runpod env vars
```
SSH_PRIVATE_KEY_B64={{ RUNPOD_SECRET_ssh_private_key_b64 }}
AWS_ACCESS_KEY_ID={{ RUNPOD_SECRET_aws_access_key_id }}
AWS_SECRET_ACCESS_KEY={{ RUNPOD_SECRET_aws_secret_access_key }}
AWS_DEFAULT_REGION=us-west-2
WANDB_API_KEY={{ RUNPOD_SECRET_wandb_api_key }}
```

### other settings
- container disk: 100GB (for larger outputs like traces)
- volume disk: 0 (S3 syncs ongoing)

