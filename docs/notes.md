## Model selection
### (M4 traning)

To explore the properties of models with similar training characteristics but different architectures, two models that were closely matched for token throughput and memory consumption were selected:

Model A:
- batch_size: 64
- seq_len: 256
- d_model: 640
- d_head: 64
- num_heads: 10
- num_layers: 10
- d_ff: 1024

Model B:
- batch_size: 48
- seq_len: 256
- d_model: 384
- d_head: 32
- num_heads: 12
- num_layers: 12
- d_ff: 1728

Both of these used about 16GB of RAM and were observed to have 3800-3900 tokens/s throughput.