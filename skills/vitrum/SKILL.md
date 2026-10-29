---
name: vitrum
description: "Experimental language model architectures where hidden dimension equals vocabulary size, exploring interpretability through vocabulary-space computation. Use when training models, adding architectures, or analyzing results."
---

# vitrum

Language models whose hidden dimension equals the vocabulary size: no embedding, no output projection, so every register state is a readable distribution over words. The goal is observability at no performance cost. The repo is a uv workspace (`libs/*` + `apps/*`). Model variants live in `libs/architectures/src/vitrum/architectures/vN_mechanism_description/` directories and are auto-discovered by `vitrum.core.registry` from the `version` class attribute (via the `vitrum.architectures` entry-point group, with a direct-import fallback and a `vitrum.core.register` decorator; third-party packages can add architectures via the entry-point group); `vitrum.training` is the single training entry point.

## Setup

```bash
uv sync                               # workspace install; makes the `vitrum` console script available
vitrum data download --variant sp1024
```

## Train

Every hyperparameter is an env var (defaults in `vitrum.core.config`; key vars tabled in `AGENTS.md` — never edit defaults, override at runtime):

```bash
INTERACTION_RANK=8 MODEL_VERSION=v8_lowrank_vv \
  torchrun --standalone --nproc_per_node=$(nvidia-smi -L | wc -l) -m vitrum.training
```

Common knobs: `MODEL_VERSION`, `SEED`, `LR`, `ITERATIONS`, `MAX_WALLCLOCK_SECONDS`, `TRAIN_BATCH_TOKENS`, `GRAD_ACCUM_STEPS`, `NUM_STEPS`. Each run writes `logs/<RUN_ID>_manifest.json` with full provenance (seed, git SHA, torch/CUDA versions, protocol, tokens seen, config dump).

## Benchmark and aggregate

```bash
vitrum list                                              # prints every discovered architecture with metadata
vitrum microbench --iters 3 --batch 2 --seq-len 128      # CPU microbench: synthetic data, speed and gradient health only
vitrum benchmark --versions v8_lowrank_vv,v12_vocab_slice --minutes 10 --seeds 1337,1338,1339   # wallclock GPU comparison via the training loop
vitrum results                                           # table from logs/*_manifest.json; mean±std per config when n>1
bash scripts/rank_sweep.sh                                 # regenerates the docs/architecture/observability.md width table (ranks x seeds + coverage)
```

Never rank architectures with the CPU microbench; it measures speed and init behavior, not trained quality (see `docs/reference/interesting-findings.md`).

## Observe (the point of the project)

`vitrum observe <cmd>` runs on CPU in float32; most subcommands take `--version`, `--checkpoint` (training-loop state_dict), and `--tokenizer` (sp model, to print words):

```bash
vitrum observe trace --version v8_lowrank_vv --checkpoint logs/<id>_model.pt   # watch a prediction form per step
vitrum observe wordmap --checkpoint logs/<id>_model.pt                          # v8's word->word matrix W = U@V^T + diag
vitrum observe causality --checkpoint logs/<id>_model.pt                        # is the readable state load-bearing?
vitrum observe sweep --checkpoint logs/<id>_model.pt                            # load-bearing vs decorative sites across depth
vitrum observe coverage --checkpoint logs/<id>_model.pt --tokenizer <sp.model>  # the observability metric, with tau robustness
vitrum observe demo                                                             # planted-bigram faithfulness check, trains on CPU
vitrum observe induction --version v8_lowrank_vv                                # beyond-bigram in-context recall, trains on CPU
```

## Verify

```bash
uvx ruff check .
uv run pytest tests/ -q
```

## Adding a model

1. Create `libs/architectures/src/vitrum/architectures/vN_mechanism_description/` with `__init__.py` and `model.py`; name the mechanism, not a metaphor.
2. Subclass `AgiModel` (`vitrum.core.base`), set `version = "vN_mechanism_description"`, implement `forward(input_ids, target_ids)` returning a scalar loss. No embeddings, no output projection (only `v13_with_embedding` is exempt, as the opaque baseline).
3. New hyperparameters go in `vitrum.core.config` as env-var-backed fields with sensible defaults.
4. Control tensors (scales, biases, decay logits, gates) must stay fp32 under bf16 training: their names must end with a pattern in `CONTROL_TENSOR_NAME_PATTERNS` (`vitrum.core.quantize`, suffix match). Never add a pattern that is a suffix of ordinary projection weight names (e.g. `weight`) — `tests/test_precision.py` gates this.
5. Add a row to the README architecture table. No harness edits needed — `vitrum run-all`, `vitrum benchmark`, and `vitrum observe` iterate the registry (per-version env overrides go in `ENV_OVERRIDES` in `vitrum.cli.run_all` if required).
6. Run the test suite: registry discovery, forward/backward smoke, precision regime, capture inertness, and causal masking are all parametrized over the registry and will pick the new version up automatically.
