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
| 2.1 Build RFT/TFB/DPO/KTO datasets | `artsco/step2.1.ipynb` | `src/build_train_data.py` |
| 3.  Train (RFT, TFB, DPO, KTO) | `artsco/src/train.py` | `src/train.py` (Tinker LoRA SFT + custom DPO/KTO loss) |
| 4.  Test inference (base + 4 trained methods) | `artsco/src/generate2.py` & `generate22.py` | `src/generate2.py` & `src/generate22.py` |
| 5.  Pairwise competition (4 base-vs-trained pairs) | `artsco/step2.2*.ipynb` | `src/compete.py` |
| 6.  Misalignment probes | `run_analysis_*.ipynb` + `trends/` | `src/probes.py` |

### Training methods

| Method | Examples per prompt | Loss |
|---|---|---|
| `rft` | (prompt, winner) only | SFT NLL on winner |
| `tfb` | rft + (tfb_prompt, voter_think) warm-up | SFT NLL on voter chains-of-thought + winner |
| `dpo` | (prompt, winner, loser) paired | pairwise contrastive (winner logp − loser logp, with frozen base reference) |
| `kto` | (prompt, winner, desirable=True) + (prompt, loser, desirable=False) unpaired | per-example KTO loss (vs. frozen base reference, batch-mean KL anchor) |

DPO/KTO use Tinker's `forward_backward_custom_async` with a custom torch loss
that consumes per-token logprobs from the LoRA model and reference logprob
*sums* precomputed once on the base model and cached at
`models/{task}/{model}/{method}/ref_logprobs.json`. β=0.1 for both methods
(see `config.DPO_BETA`, `config.KTO_BETA`).

### Fixed audience (train only)

The voter pool is committed to a *single* fixed audience drawn from the train
pool, used both during training (generate1 audience feedback) AND during
evaluation (compete pairwise voter competition). Materialized once
(seed=0) from `subjects/personas_train.json` and `subjects/demographics_train.json`.
Size is controlled by `NUM_VOTERS_TRAIN` in `config.py` (currently `20`) and
baked into the file name suffix so multiple sizes can coexist on disk:

```
subjects/train_persona_{N}.json
subjects/train_demographic_{N}.json
```

(A held-out `subjects/test_*_{N}.json` audience can also be materialized via
the same script for ad-hoc analyses, but the default pipeline does NOT wire
it into compete. To enable a dual-audience eval, add `"test"` to
`config.AUDIENCES` and re-run `compete`.)

Materialize with:

```bash
python -m new_experiments.scripts.build_audiences                   # uses config defaults (20)
python -m new_experiments.scripts.build_audiences --n_train 50 --n_test 50
```

The same N train people see every (model, task, prompt) during `generate1`
and score every method pair during `compete`, so the in-distribution audience
is consistent end-to-end. Results live under
`res/{task}/competition.json[audiences][train]`.

## Data layout

The raw datasets and personas come from `../artsco/data/`. Per-model
templated copies, generations and trained models live under:

```
new_experiments/data/{task}/{model}/{split}.json                  # chat-templated
new_experiments/data/{task}/{model}/{split}_step1.json            # baseline + votes
new_experiments/data/{task}/{model}/{split}_rft.json              # RFT (prompt, winner)
new_experiments/data/{task}/{model}/{split}_tfb.json              # TFB (rft + voter-think)
new_experiments/data/{task}/{model}/{split}_dpo.json              # DPO (prompt, chosen, rejected)
new_experiments/data/{task}/{model}/{split}_kto.json              # KTO (prompt, completion, desirable)
new_experiments/models/{task}/{model}/{method}/state.json         # tinker checkpoint id
new_experiments/models/{task}/{model}/{method}/ref_logprobs.json  # cached base ref (dpo/kto only)
new_experiments/res/{task}/{model}/{method}/{split}_step2.json
new_experiments/res/{task}/competition.json                       # {audiences: {train,test}: {mean,std}}
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
- LoRA hyperparameters: rank=16, alpha=32, lr=2e-4, **3 epochs** for every
  trained method (rft, tfb, dpo, kto), cosine schedule with min_lr_ratio=0.1.
- DPO/KTO use `forward_backward_custom_async` with a custom torch loss that
  reads per-token logprobs from the LoRA model and reference logprob *sums*
  precomputed once on the base model and cached at
  `models/{task}/{model}/{method}/ref_logprobs.json`. β=0.1 for both
  (`config.DPO_BETA`, `config.KTO_BETA`).
