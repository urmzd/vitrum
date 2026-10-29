# Documentation map

Start here. Each document owns one question; results and current standing live
in the root [README](../README.md).

## Orientation (read in this order)

| Document | Question it answers |
|---|---|
| [architecture/observability.md](architecture/observability.md) | **Why** — what the observability problem actually is (mechanism vs I/O boundary), and the honest catches |
| [architecture/overview.md](architecture/overview.md) | **How** — system layout, shared infrastructure (`vitrum.core`), the model-variant contract, entry points, precision regime, provenance discipline |
| [reference/testing.md](reference/testing.md) | **How it's verified** — what each test gates, what needs a GPU run instead, results discipline |

## Working on the repo

| Document | Purpose |
|---|---|
| [../CONTRIBUTING.md](../CONTRIBUTING.md) | Setup, dev loop, add-a-model checklist, commit/PR conventions |
| [../AGENTS.md](../AGENTS.md) | Operational reference for agents and humans: env-var table, training commands, RunPod deployment, file conventions |
| [plans/roadmap.md](plans/roadmap.md) | The live research roadmap (both fronts: observability and performance) |

## Research notes and findings

| Document | Purpose |
|---|---|
| [reference/interesting-findings.md](reference/interesting-findings.md) | CPU microbench (`vitrum microbench`) results — speed and gradient health, **not** trained quality |
| [reference/interesting-research.md](reference/interesting-research.md) | Related work (LMGP, TPGs, Neural GPU, DEQ, Mamba/RWKV/Hyena) |
| [architecture/shared-memory.md](architecture/shared-memory.md) | Design notes for the v3/v4 shared-memory-bank family |

Trained results, benchmark tables, and the Parameter Golf standing are in the
root [README](../README.md); each variant directory may also carry its own
`README.md` with variant-specific notes.
