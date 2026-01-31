# Deployment info
## v1, runpod

### conainer image
`jbmopper/cs336-basics:train-25.03`

### start command
```bash
bash -lc 'set -e; REPO=/workspace/CS336-assignment1-basics; if [ -d "$REPO/.git" ]; then git -C "$REPO" pull --rebase; else git clone git@github.com:jbmopper/CS336-assignment1-basics.git "$REPO"; fi; cd "$REPO"; uv sync --frozen --python-preference system; export PS1='\''\u@\h:\w\$ '\''; bash'
```

### other settings
- container disk: 100GB (for larger outputs like traces)
- volume disk: 0 (S3 syncs ongoing)

