<p align="center">
  <h1 align="center">Vitrum</h1>
  <p align="center">
    A testbed for <em>observable</em> language models: hidden dimension equals vocabulary size, so every register state is a readable distribution over words.
    <br /><br />
    <a href="#installation">Install</a>
    &middot;
    <a href="https://github.com/urmzd/vitrum/issues">Report Bug</a>
    &middot;
    <a href="docs/README.md">Docs</a>
  </p>
</p>

<p align="center">
  <a href="https://github.com/urmzd/vitrum/actions/workflows/ci.yml"><img src="https://github.com/urmzd/vitrum/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  &nbsp;
  <a href="LICENSE"><img src="https://img.shields.io/github/license/urmzd/vitrum" alt="License"></a>
</p>

> **Status: paused since July 2026.** The readable variants probed so far are bigram-class: `v8_lowrank_vv` and `v9_linattn` score at chance on in-context key→value recall (see [Beyond bigram](#beyond-bigram-the-induction-test)). The tooling and framing stand; the architectures do not yet do computation that is hard to read. Post-hoc work such as Anthropic's Jacobian lens now reads vocabulary-space variables out of opaque models directly (see [docs/OBSERVABILITY.md](docs/OBSERVABILITY.md#the-post-hoc-alternative-the-jacobian-lens)).

Vitrum is Latin for glass. The repo was previously `glassbox-lm`; GitHub redirects the old URL.

The goal is **observability at no performance cost**: architectures where you can see how the computation works without giving up the quality of an opaque model. A readable model has to match the opaque baseline (`v13_with_embedding`) to make the point. For why readable I/O alone is not the answer, see [docs/OBSERVABILITY.md](docs/OBSERVABILITY.md).

## Features

- **Vocabulary-space state**: hidden dimension equals vocab size, with no learned embedding and no output projection. The register state is a distribution over words at every step.
- **Interpretability by construction**: read each intermediate state as "which words are active and how strongly" and watch a prediction form across recurrent steps, with no post-hoc probing.
- **Causal observability tooling**: `vitrum observe` traces states, reads learned word→word maps, and measures which readable sites are causally load-bearing.
- **A zoo of 16 readable variants plus an opaque baseline**: conv, linear attention, low-rank V x V, op-banks, hard routing, and Mamba/RWKV/Hyena-style dynamics, all sharing one skeleton and auto-discovered by a registry.
- **Budget-matched benchmarking**: `vitrum benchmark` trains each version under an identical wall-clock budget and records seed, git SHA, GPU, and full config in a manifest.
- **Parameter Golf ready**: the training loop emits the int8 + zlib artifact that [OpenAI's Parameter Golf](https://github.com/openai/parameter-golf) scores.

## Installation

### Prerequisites

| Requirement | Version |
|-------------|---------|
| Python | >= 3.11 |
| [uv](https://docs.astral.sh/uv/) | any recent |
| NVIDIA GPU + CUDA | for training and `vitrum benchmark` (observability tools run on CPU) |

On a fresh RunPod pod, run the setup script. It installs into system Python, so the image's preinstalled torch is left alone, and downloads the dataset:

```bash
curl -sSL https://raw.githubusercontent.com/urmzd/vitrum/main/setup.sh | bash
```

Anywhere else, clone and sync. This creates `.venv` with every workspace package, torch included; prefix commands with `uv run` or activate the venv:

```bash
git clone https://github.com/urmzd/vitrum.git && cd vitrum
uv sync
vitrum data download --variant sp1024
```

## Quick Start

```bash
# Train the best model (low-rank V x V, rank 8)
INTERACTION_RANK=8 MODEL_VERSION=v8_lowrank_vv \
  torchrun --standalone --nproc_per_node=$(nvidia-smi -L | wc -l) -m vitrum.training

# List every discovered architecture (with metadata)
vitrum list

# Benchmark all models, or specific ones
vitrum benchmark
vitrum benchmark --versions v8_lowrank_vv,v2_conv,v14_data_dependent --minutes 10

# Verify the observability claim on CPU: recover a planted bigram from the weights
vitrum observe demo
```

## Usage

### Commands

| Command | What it does |
|---|---|
| `vitrum list` | List every architecture visible to the registry |
| `vitrum train` | Train one model (env-var configured; `--nproc` > 1 wraps torchrun DDP) |
| `vitrum run-all` | Train every registered architecture sequentially, then print results |
| `vitrum benchmark` | Run versions under identical wall-clock conditions and compare |
| `vitrum results` | Aggregate `logs/*_manifest.json` into a results table (mean±std when n > 1) |
| `vitrum microbench` | Fast CPU forward/backward sanity check (speed, gradients; not trained quality) |
| `vitrum data` | Dataset commands: `download`, `prepare-code` |
| `vitrum observe` | Observability toolkit (below) |

### Observing the computation

`vitrum observe` runs on CPU in float32. Most subcommands accept `--version`, `--checkpoint` (a training-run state_dict), and `--tokenizer` (an sp model, to print words instead of ids).

| Command | What it shows |
|---|---|
| `vitrum observe trace` | Top-k active words in the register state after each step for a prompt |
| `vitrum observe wordmap` | For v8, the learned word→word interaction matrix `W = U@Vᵀ + diag(d)` |
| `vitrum observe causality` | Perturb one vocab dimension mid-computation and measure how far the next-word distribution moves: is the readable state load-bearing or decorative? |
| `vitrum observe demo` | Train v8 on a planted bigram (`next[i] = perm[i]`) and verify the map recovers it |
| `vitrum observe sweep` | Map which (step, word) sites are causally load-bearing vs decorative, across depth |
| `vitrum observe coverage` | The observability number: fraction of readable active-word sites that are causally load-bearing, across τ thresholds |
| `vitrum observe induction` | Train on in-context key→value pairs and test query-position recall against bigram chance and the key-free-copier ceiling |

**Faithfulness result.** On the planted-bigram demo, the recovered map `argmax_j W[i,j]` matches the true `perm[i]` for **100% of words** (chance ≈ 3%). The open questions are whether that readability holds on real text and at scale, and whether the states are causally load-bearing (run `vitrum observe causality` on a trained checkpoint).

### Model versions

Select a variant with `MODEL_VERSION`. Every variant repeats N steps of cross-position mixing then within-position transform on a one-hot register state, with no embedding and no output projection. Names describe mechanism, not metaphor.

| `MODEL_VERSION` | Cross-position | Within-position | Notes |
|---|---|---|---|
| `v1_shared_attn` | GQA + RoPE (weights shared across depth) | Fourier-parameterized channel mix | 3.4M params, plateaus early |
| `v2_conv` | Depthwise causal 1D conv | Fourier-parameterized channel mix | 353K params, strong baseline |
| `v3_fourier_linattn` | Linear attn with causal decay; Q/K/V/O via Fourier basis | Fourier-parameterized channel mix | Stuck: Fourier parameterization bottleneck |
| `v4_weight_shared` | Shared Q/K + per-head decay (v3 body) | Factored (diag + low-rank) channel mix | Size-reduction ablation of v3 |
| `v5_fft_linattn` | Linear attn with causal decay; Q/K/V/O via rFFT | FFT-based channel mix | Same Fourier-over-vocab caveat |
| `v6_banded_fourier` | Band-partitioned Fourier linattn, gated coupling | Three parallel band projections, gated | 824K, still descending |
| `v7_soft_ops` | Linear attn with causal decay | Gumbel-soft op-bank + soft register addressing | Unstable, loss spikes |
| `v8_lowrank_vv` | Diagonal Q/K linear attn, activation similarity | Low-rank V x V (`U @ V^T + diag`) | **Best architecture** at rank 8 (historical run); 2026-06 width-sweep claims under revision |
| `v9_linattn` | Linear attn with causal decay (dense projections) | MLP bottleneck | 4.2M params, best non-attention variant |
| `v10_state_cond_op` | Linear attn in compressed state space | State-conditioned soft read/op/write dispatch | Untested |
| `v11a_mixed_ops` | High-decay EMA + linear-attn | Sigmoid gate, dense layer, low-decay EMA | Trains; bf16 `arange` position bug fixed (2026-05) |
| `v11b_hard_routing` | Multi-timescale linear attn | Gumbel-hard op routing + PonderNet halting | Untested |
| `v12_vocab_slice` | Causal decay in fixed k-dim slice | MLP in k-dim slice | Best on target corpus (3.19 bpb, 2026-05); slices are deterministic vocab-id windows |
| `v14_data_dependent` | Input-modulated conv (Hyena) | Data-dependent decay (Mamba), DCT mix | Mamba / RWKV / Hyena bundle; bf16 dtype bug fixed (2026-05) |
| `v15_aux_loss` | v12 body + per-step CE + top-k sparsity | Entropy-adaptive write scaling | Training-side additions on v12 |
| `v16_multi_branch` | Per-column decay memory | Branched gated MLP + cross-column inhibition | Ensemble + gated branches |
| `v13_with_embedding` | v12 body | v12 body | **Opaque baseline, comparison only.** Adds `Embedding(V, d) -> Linear(d, V)` to measure what readability costs. Not a template. |

## Configuration

Every hyperparameter is an environment variable read by the `Hyperparameters` class in `vitrum.core.config`. Override at runtime; do not edit defaults.

| Variable | Default | Notes |
|---|---|---|
| `MODEL_VERSION` | `v8_lowrank_vv` | Which variant to train |
| `INTERACTION_RANK` | 8 | Rank of the v8 interaction matrix |
| `NUM_STEPS` | 8 | Recurrent steps / depth |
| `SEED` | 1337 | RNG seed; recorded in the manifest |
| `LR` | 0.03 | Adam learning rate |
| `TRAIN_BATCH_TOKENS` | 524288 | Global batch size in tokens |
| `GRAD_ACCUM_STEPS` | 16 | Gradient accumulation |
| `ITERATIONS` | 500 | Max training iterations |
| `MAX_WALLCLOCK_SECONDS` | none | Wall-clock limit; set it manually |
| `VAL_MAX_TOKENS` | 0 | Cap validation tokens (0 = full set, required for a faithful bpb score) |

The full table, including per-variant knobs, is in [AGENTS.md](AGENTS.md#key-env-vars).

## Results

> **Precision note (methodology corrected 2026-06-09):** every table below predates a control-tensor fix. A stray `"weight"` entry in `CONTROL_TENSOR_NAME_PATTERNS` matched every `nn.Linear` weight, so all models except `v8_lowrank_vv` (custom `U`/`V` parameter names) kept fp32 master weights instead of bf16. Forward and backward always ran under bf16 autocast, so loss and bpb stay comparable, but parameter memory, `raw_bytes`, and optimizer-state size were not uniform. Re-run under the corrected regime before citing memory figures or close margins.

### Historical benchmark (10 min, 3× A40, batch 491,520 tokens)

| `MODEL_VERSION` | Architecture | Params | Steps | val_loss | val_bpb | tok/s | Status |
|---|---|---|---|---|---|---|---|
| **v8_lowrank_vv** (rank 8) | Recurrent rank-r V x V linear layer | **164K** | 100 | **5.24** | **3.10** | 270K | Still descending |
| v2_conv | Depthwise causal conv + Fourier channel mix | 353K | 464 | 5.39 | 3.19 | 383K | Still descending |
| v6_banded_fourier | Band-partitioned Fourier with gated coupling | 824K | 166 | 5.66 | 3.35 | 136K | Still descending |
| v1_shared_attn | Shared GQA attention + Fourier channel mix | 3.4M | 239 | 6.06 | 3.59 | 196K | Plateaued |
| v7_soft_ops | Soft op-bank + soft register addressing | 329K | 348 | 6.26 | 3.71 | 287K | Unstable (loss spikes) |
| v3_fourier_linattn | Linear attn with causal decay (Fourier proj) | 329K | 397 | 6.81 | 4.03 | 326K | Stuck |
| v8_lowrank_vv (rank 64) | Recurrent rank-r V x V linear layer | 1.1M | 188 | n/a | n/a | 270K | Memorized (train 0.04, overfitting) |

On this run, v8 at rank 8 beat v2_conv with half the parameters in one-fifth the steps, with an essentially zero train/val gap. This is still far from useful: 3.10 bpb versus ~0.93 for GPT-2 at 124M params.

### Head-to-head on the target corpus (1× RTX 4090, identical 720s budget, 2026-05-31)

Re-run on the target data (`willdepueoai/parameter-golf`, FineWeb sp1024) with one budget-matched config (batch 491,520, 20 warmup steps, flat LR, grad-clip 1.0, 720s cap):

| `MODEL_VERSION` | Params | Steps | val_loss | val_bpb |
|---|---|---|---|---|
| **v12_vocab_slice** | 4.20M | 589 | 5.383 | **3.188** |
| v14_data_dependent | 2.54M | 211 | 5.685 | 3.367 |
| v8_lowrank_vv (rank 8) | 0.16M | 229 | 5.984 | 3.544 |

The ranking **inverts** the historical table: `v12_vocab_slice` wins on absolute bpb, while `v8_lowrank_vv` leads only per parameter (25× smaller for +0.36 bpb). All three descended monotonically. This is roughly **1/30th** of the Parameter Golf compute budget, so these are undertrained floors, not ceilings.

### Parameter Golf standing

Parameter Golf is used as a **raw-capability yardstick, not the goal**: the best LM in a **16 MB artifact**, trained in **under 10 min on 8× H100**, scored by tokenizer-agnostic bits-per-byte on FineWeb validation.

| | bits-per-byte |
|---|---|
| Best verified leaderboard record (2026-05) | **~1.061** |
| Naive baseline (9-layer, 512-dim transformer) | **1.2244** |
| Published "best readable" claim (`v8_lowrank_vv` rank 128, 2026-06) | 2.88, under revision (see [Width sweep](#width-sweep-2026-06-under-revision)) |
| Best readable result here (`v12_vocab_slice`, 2026-05 head-to-head) | **3.19** |

Results here are **~2.6× worse than the naive baseline**. The prime suspect is the core constraint: `hidden_dim = vocab_size = 1024` forces all computation through a vocab-space bottleneck, where the baseline has a 512-dim embedding.

### The cost of readability (3 GPUs, identical 600s budget)

`v13_with_embedding` shares v12's exact body and differs only by a learned `Embedding(V, d) → Linear(d, V)`. From `artifacts/benchmark_results.json` (runs `0171d9fb` = `v13_embed` and `ae6c82a6` = `v12_sparse`, the pre-rename ids):

| `MODEL_VERSION` | Embedding? | Params | Steps | val_bpb |
|---|---|---|---|---|
| **v13_with_embedding** | yes (baseline) | 4.46M | 603 | **2.368** |
| v12_vocab_slice | no | 4.20M | 636 | 3.236 |

The embedding drops bpb **3.24 → 2.37** and closes ~43% of the gap to the 1.2244 baseline, confirming the constraint, not the compute budget, is the wall. That ~0.87 bpb gap is **the gap to eliminate** while keeping the state readable; `v13` is the performance target, not a template.

> **Under revision (methodology corrected 2026-06-09).** This table previously read v13 894 steps / 2.256 bpb, v12 945 steps / 3.079 bpb, and v8_lowrank_vv 477 steps / 3.427 bpb on "1× H100, 600 s", with a gradient-starvation reading of v8 based on final grad-norms. No committed manifest records those runs, the committed logs contain no per-step `grad_norm` series, and the v8 row has no committed 600 s counterpart. Treat the grad-norm narrative as unverified until re-run.

### Width sweep (2026-06): under revision

The hypothesis: scaling v8's interaction rank improves bpb and causal coverage together. The committed artifacts (`artifacts/rank{8,16,32,64}_manifest.json`; 165 steps, ~300 s, 3 GPUs per run) do not support it yet:

| v8 width | params | train loss | val_bpb |
|---|---|---|---|
| rank 8 | 164K | 4.85 | 2.87 |
| rank 16 | 295K | 5.39 | 3.19 |
| rank 32 | 557K | 4.35 | 2.56 |
| rank 64 | 1.08M | 0.87 | 0.47 |

bpb is not monotone in rank, and rank 64 is a memorization/leakage signal (0.47 bpb would beat the Parameter Golf record by more than 2×), matching the rank-64 memorization in the historical run. An earlier version claimed bpb falling monotonically 3.46 → 2.88 through rank 128, coverage rising 50% → 87%, and no memorization at any rank; those numbers match no committed manifest and no rank-128 artifact exists. Full audit note in [docs/OBSERVABILITY.md](docs/OBSERVABILITY.md#width-sweep-2026-06-what-the-committed-evidence-supports). Regenerate the sweep at three seeds with `scripts/rank_sweep.sh`.

### Beyond bigram: the induction test

`vitrum observe induction` trains a variant on in-context key→value pairs. Each sequence has its own random map, so only looking back at the context solves the query positions.

| `MODEL_VERSION` | Query-position accuracy | Chance (bigram ceiling) | Key-free copier ceiling |
|---|---|---|---|
| `v8_lowrank_vv` | 1.6% | 2.1% | ~10.0% |
| `v9_linattn` | 2.3% | 2.1% | ~10.0% |

Both readable linear-attention variants sit at chance: what the tooling reads in them is a bigram-class mechanism. This is a known limit of linear attention, not necessarily of vocabulary-space state. `v1_shared_attn` (softmax attention) cannot be probed yet because it shares one block in a loop and exposes no per-step state hook.

Reproduce on CPU in a few minutes: `vitrum observe induction --version v8_lowrank_vv`. First reported in PR #13 (2026-06-01) and re-run with identical results on 2026-10-10. No manifest is committed for this probe.

### What we've learned

- **Inductive bias matters more than parameter count.** `v8_lowrank_vv` (164K params) beat `v1_shared_attn` (3.4M) in the historical run: direct dimension-to-dimension interaction, `W[i, j] = "dim i predicts dim j"`, is a better prior for language than generic attention in vocab space. Bigram models encode the same structure, so whether multi-hop propagation captures longer-range dependencies is still open.
- **Too much capacity in the right place enables memorization.** Rank 64 memorized the training batch (train loss 0.04) while rank 8 generalized; the committed rank-64 sweep artifact (0.47 val bpb) shows the same signal.
- **Fourier-over-vocab parameterization is a structural bottleneck.** `v3_fourier_linattn` and `v5_fft_linattn` both got stuck. BPE vocab ids have no meaningful ordering, so smoothness over ids discards capacity. The linear-attention core works fine with dense projections (`v9_linattn`).
- **Attention in vocab space is expensive and unhelpful at this scale.** `v1_shared_attn` spends most of its params on Q/K/V/O over V=1024 vectors and plateaus at val_loss ~6.06.
- **Training instability is real.** `v7_soft_ops` spiked to 9.35 at step 161 and 8.28 at step 181 before recovering; the soft op-selection path is fragile.
- **What is new here** is the combination: hidden_dim = vocab_size with no embedding or output projection, plus conv, decay-memory, and low-rank cross-position mechanisms, plus recurrent depth. Weight sharing across depth, Fourier parameterization, linear attention, and recurrent register machines are all prior work.

## Development

```bash
uv sync --group dev                 # deps + pytest/ruff
uv run pytest -q                    # CPU test suite
uv run ruff check .                 # lint
vitrum microbench v8_lowrank_vv     # quick CPU sanity for a variant
```

CI runs lint and tests on every PR. See [CONTRIBUTING.md](CONTRIBUTING.md) for the add-a-model checklist and commit conventions.

## Documentation

- [Documentation map](docs/README.md)
- [The observability problem](docs/OBSERVABILITY.md)
- [Architecture](docs/ARCHITECTURE.md)
- [Testing](docs/TESTING.md)
- [Related work](docs/INTERESTING_RESEARCH.md)
- [CPU microbench findings](docs/INTERESTING_FINDINGS.md)
- [Contributing](CONTRIBUTING.md)
- [Operational reference](AGENTS.md)

## Agent Skill

This repo's conventions are available as portable agent skills in [`skills/`](skills/).

## License

[Apache-2.0](LICENSE)
