# Hardware Realizability Report (MAIN STACK Experiment 2)

Are the sparse selections (oracle top-S at the 2% reference budget) hardware-friendly? Long-context cases (≥8k). Metrics describe contiguity, block structure, head/layer reuse, gather overhead, and variable softmax length.

## Classification: **GPU-HOSTILE / gather-bound (scattered selections: low block utilization, high block over-read, low cross-head overlap)**

- Mean selected tokens S = 294 (median 178); contiguous runs per selected token = 0.55 (1.0 = fully scattered, →0 = contiguous).
- Block utilization: size-32 **0.15**, size-64 **0.11**, size-128 0.08 (fraction of a touched block that is actually selected).
- Recent-window fraction 0.15, sink fraction 0.01 of selected tokens.
- Cross-head Jaccard overlap **0.24**, cross-layer **0.15** (shared selected tokens across heads / layers).

## Block size trade-off

`mean_overread` = tokens actually read / tokens needed (block padding cost); `meta_overhead_frac` = index bytes / value bytes read.

|   block_size |   mean_blocks |   mean_block_util |   mean_overread |   meta_overhead_frac |
|-------------:|--------------:|------------------:|----------------:|---------------------:|
|           16 |       111.759 |             0.202 |           6.176 |                0.002 |
|           32 |        84.931 |             0.146 |           9.399 |                0.001 |
|           64 |        62.589 |             0.108 |          13.964 |                0     |
|          128 |        41.704 |             0.082 |          18.798 |                0     |

## By regime (size-32 blocks)

| regime   |   mean_S |   recent_frac |   blockutil_32 |   nblocks_32 |
|:---------|---------:|--------------:|---------------:|-------------:|
| agent    |   328    |         0.141 |          0.147 |       93.324 |
| lenshift |   193.25 |         0.18  |          0.122 |       64.342 |
| longdoc  |   328    |         0.131 |          0.154 |       89.894 |
| needle   |   328    |         0.143 |          0.161 |       92.162 |

## Variable softmax length (selected tokens S) distribution

```
count    5376.000000
mean      294.312500
std       192.187631
min       100.000000
10%       100.000000
25%       164.000000
50%       178.500000
75%       341.000000
90%       656.000000
99%       656.000000
max       656.000000
```

## Reading & next step

- **Block locality:** with size-32/64 blocks, utilization (0.15/0.11) sets the padding cost; `mean_overread` quantifies wasted reads if you gather whole blocks. A recent/sink fraction of 0.16 is contiguous by construction and trivially block-friendly.
- **Head/layer reuse:** Jaccard overlap (0.24 head, 0.15 layer) indicates how much a shared block selection could be amortized across heads/layers — higher overlap favors a single gathered block set reused by many heads.
- **Decision:** if block-local and overlapping → the next step is a Triton block-sparse attention kernel. If highly irregular but *stable* across calls → specialized KV gather hardware becomes the more plausible bet. Current reading: **GPU-HOSTILE / gather-bound**.
- **Kernel gate: NOT cleared.** Block locality does *not* pass — at block size 32 you over-read ~9× (util 0.15) and cross-head overlap is only 0.24, so a naive block-sparse gather would spend most of the sparse saving on padding. The combined gate (adaptive budget ≥4× AND block locality friendly) is **half-met**: Exp 1 gives ≥4× at 16k, but selections are gather-bound. **Do not build a Triton kernel yet.**
- Next instead: (a) measure selection *stability* across query positions / decode steps (if scattered-but-stable, the gather set can be precomputed and amortized — a different kernel/hardware bet); (b) try learned or locality-regularized routers that prefer contiguous blocks, then re-measure utilization vs the rel-L2 cost.
