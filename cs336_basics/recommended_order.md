# Recommended Implementation Order for adapters.py

## Tier 1: Simple utilities (building blocks)
1. ✅ `run_linear` - Matrix multiplication layer
   - Test: `tests/test_model.py::test_linear`
2. ✅ `run_embedding` - Token embeddings
   - Test: `tests/test_model.py::test_embedding`
3. ✅ `run_silu` - SiLU/Swish activation
   - Test: `tests/test_model.py::test_silu_matches_pytorch`
4. ✅ `run_softmax` - Softmax normalization (watch for numerical stability!)
   - Test: `tests/test_nn_utils.py::test_softmax_matches_pytorch`
5. ✅ `run_rmsnorm` - RMS normalization layer
   - Test: `tests/test_model.py::test_rmsnorm`
6. ✅ `run_swiglu` - SwiGLU feedforward network
   - Test: `tests/test_model.py::test_swiglu`

## Tier 2: Attention mechanism
7.  ✅ `run_scaled_dot_product_attention` - Core attention computation
   - Test: `tests/test_model.py::test_scaled_dot_product_attention`
   - Test: `tests/test_model.py::test_4d_scaled_dot_product_attention`
8.  ✅ `run_rope` - Rotary positional embeddings
   - Test: `tests/test_model.py::test_rope`
9.  `run_multihead_self_attention` - Multi-head attention without RoPE
   - Test: `tests/test_model.py::test_multihead_self_attention`
10.  `run_multihead_self_attention_with_rope` - Multi-head attention with RoPE
    - Test: `tests/test_model.py::test_multihead_self_attention_with_rope`

## Tier 3: Full model components
11. ✅ `run_transformer_block` - Complete transformer block (uses everything above)
    - Test: `tests/test_model.py::test_transformer_block`
12. ✅ `run_transformer_lm` - Full transformer language model
    - Test: `tests/test_model.py::test_transformer_lm`
    - Test: `tests/test_model.py::test_transformer_lm_truncated_input`

## Tier 4: Training utilities (can do in any order)
13. ✅ `run_cross_entropy` - Cross-entropy loss function
    - Test: `tests/test_nn_utils.py::test_cross_entropy`
14. ✅ `run_get_batch` - Data loading/sampling
    - Test: `tests/test_data.py::test_get_batch`
15. ✅ `run_gradient_clipping` - Gradient clipping for training stability
    - Test: `tests/test_nn_utils.py::test_gradient_clipping`
16. `get_adamw_cls` - AdamW optimizer
    - Test: `tests/test_optimizer.py::test_adamw`
17. ✅ `run_get_lr_cosine_schedule` - Cosine learning rate schedule with warmup
    - Test: `tests/test_optimizer.py::test_get_lr_cosine_schedule`

## Tier 5: Serialization & Tokenization (independent)
18. `run_save_checkpoint` / `run_load_checkpoint` - Save/load model checkpoints
    - Test: `tests/test_serialization.py::test_checkpointing`
19. ✅ `get_tokenizer` - Create BPE tokenizer from vocab/merges
    - Tests: `tests/test_tokenizer.py::test_*` (many tests)
20. ✅ `run_train_bpe` - Train a BPE tokenizer
    - Test: `tests/test_train_bpe.py::test_train_bpe`
    - Test: `tests/test_train_bpe.py::test_train_bpe_special_tokens`
    - Test: `tests/test_train_bpe.py::test_train_bpe_speed`

