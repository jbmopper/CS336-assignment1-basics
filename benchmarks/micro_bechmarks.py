from cs336_basics.nn import scaled_dot_product_attention
import torch.utils.benchmark as benchmark
import torch
from itertools import product
import json
from datetime import datetime
import argparse

def main(args):
    # scaled_dot_product_attention()
    # the sdpa function take sthe Q, K, and V projection tensors
    # these are the inputs multiplied by the concatenated weights
    # each being [B, h, S, d_head]
    # the mask is a [B, h, S, S] tensor of booleans
    # the output is a [B, h, S, d_head] tensor

    # B can be varied to see how throughput scales... 4, 8, 16, 32, 64
    # d_head can be 32, 64, 96, 128, ... bigger?
    # h can be 4, 8, 16, 32
    # S can be 128, 256, 384, 512, 640, 768, 896, 1024
    

    parser = argparse.ArgumentParser()
    parser.add_argument("--smol", action="store_true", help="smol run")
    args = parser.parse_args()

    if args.smol:
        Bs = [8, 32]
        d_heads = [64, 128]
        hs = [8]
        Ss = [256, 512, 1024]
    else:
        Bs = [4, 8, 16, 32, 64]
        d_heads = [32, 64, 96, 128]
        hs = [4, 8, 16, 32]
        Ss = [128, 256, 384, 512, 640, 768, 896, 1024]

    device = "mps"
    configs = [
        {"B": B, "d_head": d_head, "h": h, "S": S}
        for B, d_head, h, S in product(Bs, d_heads, hs, Ss)
    ]

    results = []
    for config in configs:
        label = "scaled dot product attention"
        sublabel = "B={B}, d_head={d_head}, h={h}, S={S}".format(**config) 
        mask = torch.ones((config["S"], config["S"]), dtype=bool, device=device)
        mask = torch.tril(mask)
        Q, K, V = (torch.randn((config["B"], config["h"], config["S"], config["d_head"]),
            dtype=torch.float32, device=device, requires_grad=True) for _ in range(3))

        print(f"Running benchmark for {config['B']}x{config['h']}x{config['S']}x{config['d_head']}")
        torch.mps.synchronize()
        results.append(benchmark.Timer(
            stmt="scaled_dot_product_attention(Q, K, V, mask); torch.mps.synchronize()",
            globals={
                "Q": Q, "K": K, "V": V, "mask": mask,
                "torch": torch,
                "scaled_dot_product_attention": scaled_dot_product_attention,
            },
            label=label,
            sub_label=sublabel,
            description=f"Seq Len {config['S']}",
        ).blocked_autorange(min_run_time=2))

    rows = []
    for m in results:
        rows.append({
            "label": m.label,
            "sublabel": m.sub_label,
            "median_s": m.median,   # seconds
            "mean_s": m.mean,
            "iqr_s": m.iqr,
            "num_runs": len(m.times),
        })

    filename = f"micro_benchmarks_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, "w") as f:
        json.dump(rows, f)
        print(f"Saved results to {filename}")   


    compare = benchmark.Compare(results)
    compare.colorize().print()

if __name__ == "__main__":
    main()