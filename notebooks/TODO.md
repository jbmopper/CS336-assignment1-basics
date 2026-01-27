# SVG Diagram Analysis & TODOs

## Current State

The `cs336_forward.svg` diagram captures the forward pass architecture well:

### What it captures well
- Tensor shapes at each stage with template variables (`{{B}}`, `{{seq_len}}`, etc.)
- Weight sizes for all major components (Embedding, QKV, O proj, SwiGLU, LM head)
- Compute (FLOPs) for each operation
- Both residual connections (the + circles)
- Q/K split for RoPE
- The transformer block boundary (`n_blocks`)

## TODOs

### High Priority (Memory Analysis Gaps)

- [ ] **Softmax intermediates**: SDPA shows `[B, h, S, S]` once, but softmax creates ~5 S² tensors (adjusted, exp, sum, output, masked). Add note or expand SDPA box.

- [ ] **SwiGLU intermediates**: Shows weights `3×[d_model, d_ff]` but not the 5 activation tensors `[B, S, d_ff]` (w1, w3, sigmoid, silu, silu*w3). Consider adding intermediate shapes.

- [ ] **"Saved for backward" markers**: Add visual indicator for which tensors autograd keeps alive during training. This is the main memory concern.

- [ ] **Accumulation across blocks**: The box shows 1 block; add emphasis that memory accumulates L× across all blocks.

- [ ] **Causal mask**: `[S, S]` boolean tensor not shown (minor memory impact but architecturally relevant).

### Medium Priority (Visual Improvements)

- [ ] **Memory hotspot indicator**: Add visual emphasis on SDPA as the S² bottleneck (e.g., red border, warning icon)

- [ ] **V tensor flow**: The V tensor path into SDPA is less clear than Q→RoPE, K→RoPE paths. Consider making this more explicit.

- [ ] **Intermediate activation counts**: Add counts inside each box showing how many tensors are created/saved

### Low Priority (Minor Fixes)

- [ ] **Typo**: "Ebmedding" → "Embedding" (line 6 of SVG)

## Notes

The `local_tiny.py` notebook's detailed memory accounting is more accurate for memory analysis since it explicitly lists all intermediate tensors saved for backward. The SVG is better suited for architectural understanding and compute flow visualization.

### Memory vs Compute Summary

| Component | Compute Scaling | Memory Scaling | Shown in SVG |
|-----------|-----------------|----------------|--------------|
| Embedding | O(B×S×d) | O(V×d) weights | ✅ |
| RMS Norm | O(B×S×d) | O(d) weights | ✅ |
| QKV Proj | O(B×S×d²) | O(d²) weights | ✅ |
| RoPE | O(B×h×S×dh) | O(S×dh) cached | ✅ |
| Attention (QK^T) | O(B×h×S²×dh) | O(B×h×S²) | ✅ |
| Softmax | O(B×h×S²) | O(B×h×S²) ×5 | ❌ (shows 1×) |
| Attention (×V) | O(B×h×S²×dh) | O(B×h×S×dh) | ✅ |
| O Proj | O(B×S×d²) | O(d²) weights | ✅ |
| SwiGLU | O(B×S×d×dff) | O(B×S×dff) ×5 | ❌ (weights only) |
| LM Head | O(B×S×d×V) | O(d×V) weights | ✅ |
