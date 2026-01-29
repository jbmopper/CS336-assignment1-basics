### Test 2

Test 2 will be focused on a more conventional width/depth tradeoff and will be more trainable.


| Parameter | Model A (Wide) | Model B (Deep) |
| :--- | :--- | :--- |
| **batch_size** | 32 | 32 |
| **context_length** | 256 | 256 |
| **d_model** | 768 | 384 |
| **num_layers** | 2 | 12 |
| **num_heads** | 12 | 12 |
| **d_head** | 64 | 32 |
| **d_ff** | 2048 | 1024 |
| **Parameters** | 29.52M | 28.92M |
| **Est. Throughput** | ?? tok/s | ~4,490 tok/s |
