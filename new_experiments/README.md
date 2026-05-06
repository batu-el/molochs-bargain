# New Experiments — Llama-3.3-70B & Qwen3-30B-A3B via Tinker

Replication of the Moloch's Bargain pipeline with two large models, trained
through the Tinker API instead of local TRL/HF training. The experimental
setup (tasks, persona voters, RFT/TFB methods, GPT-4o probes) is kept
identical to the original `artsco/` setup.

## Models

- `meta-llama/Llama-3.3-70B-Instruct`
- `Qwen/Qwen3-30B-A3B-Instruct-2507`

Both are LoRA fine-tuned through Tinker. Sampling for generation also runs
through Tinker (since these models are too large for most local GPUs).

## Pipeline

The same six-stage pipeline as `artsco/`, mapped to scripts here:

| Stage | Original | New Experiments |
|---|---|---|
| 1.1 Per-model chat templates | `artsco/step1.1.ipynb` | `src/prep_data.py` |
| 2.  Baseline + voter feedback (train) | `artsco/src/generate1.py` | `src/generate1.py` |
| 2.1 Build RFT/TFB datasets | `artsco/step2.1.ipynb` | `src/build_train_data.py` |
| 3.  Train (RFT, TFB) | `artsco/src/train.py` | `src/train.py` (Tinker LoRA SFT) |
| 4.  Test inference (base, rft, tfb) | `artsco/src/generate2.py` & `generate22.py` | `src/generate2.py` & `src/generate22.py` |
| 5.  Pairwise competition | `artsco/step2.2*.ipynb` | `src/compete.py` |
| 6.  Misalignment probes | `run_analysis_*.ipynb` + `trends/` | `src/probes.py` |

## Data layout

The raw datasets and personas come from `../artsco/data/`. Per-model
templated copies, generations and trained models live under:

```
new_experiments/data/{task}/{model}/{split}.json           # chat-templated
new_experiments/data/{task}/{model}/{split}_step1.json     # baseline + votes
new_experiments/data/{task}/{model}/{split}_rft.json       # RFT training data
new_experiments/data/{task}/{model}/{split}_tfb.json       # TFB training data
new_experiments/models/{task}/{model}/{method}/state.json  # tinker checkpoint id
new_experiments/res/{task}/{model}/{method}/{split}_step2.json
new_experiments/res/{task}/final_competition.json
new_experiments/res/probes/{task}_q{n}.csv
```

## Setup

```bash
conda activate venv
pip install -r new_experiments/requirements.txt

export TINKER_API_KEY=""        # https://tinker-console.thinkingmachines.ai/
export OPENAI_API_KEY=""        # GPT-4o-mini (voters) and GPT-4o (probes)
export HF_TOKEN=""              # tokenizer access
```

## Estimate cost first

```bash
python -m new_experiments.scripts.estimate_costs              # full run (~$460)
python -m new_experiments.scripts.estimate_costs --limit 64   # smoke (~$70)
```

## Launch scripts

| Script | What it does | Cost |
|---|---|---|
| `scripts/launch_smoke.sh` | End-to-end with `--limit 16` to validate auth/paths | ~$70 |
| `scripts/launch_per_model.sh --qwen-only` | Only Qwen3-30B-A3B (cheap MoE) | ~$30 |
| `scripts/launch_per_model.sh --llama-only` | Only Llama-3.3-70B (dense, expensive) | ~$130 |
| `scripts/launch_per_model.sh` | Both models sequentially, then compete + probes | ~$460 |
| `scripts/launch_per_task.sh` | Fans out tasks in parallel (3x faster, same cost) | ~$460 |
| `scripts/launch_full.sh` | Detached (`nohup`) full run with logging | ~$460 |
| `run_experiments.sh` | Bare orchestrator used by the others | — |

### SLURM eval shards

After all 15 (model, task) training shards under `submit_train_parallel.sh`
finish, choose one of:

| Submission | Parallelism | When to use |
|---|---|---|
| `bash submit_eval.sh [smoke]` | 3 array tasks (one per task) | default; each shard sweeps all 5 models internally |
| `bash submit_eval_parallel.sh [smoke]` | 15 array tasks (one per model x task) + 1 dependent merge job | max parallelism; useful when OpenAI rate or wall-clock matters |

The per-model variant writes to `res/{task}/competition_parts/{model}.json`
and `res/probes/{task}_{qid}_parts/{model}.csv`. The auto-submitted
`run_eval_merge.sh` (afterok dependency) consolidates them into the
canonical `res/{task}/competition.json` and `res/probes/{task}_{qid}.csv`
that the analysis notebooks read.

To re-run a single stage:

```bash
python -m new_experiments.src.train --task task_sales --method rft
```

## Notes

- All generation and SFT happen through Tinker (`tinker.ServiceClient`).
- Voters and probes still run against the OpenAI API (GPT-4o-mini & GPT-4o)
  exactly as in the original setup.
- LoRA hyperparameters (rank=16, alpha=32, lr=2e-4, 1 epoch, cosine w/ min_lr)
  match the original TRL configuration as closely as Tinker allows.
