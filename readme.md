# Moloch's Bargain: Emergent Misalignment When LLMs Compete for Audiences

Research on how LLMs behave when competing for audience attention, potentially leading to emergent misalignment behaviors.

## Structure

- **`src/`** - Training and generation scripts
  - `train.py` - LoRA fine-tuning with TRL SFT
  - `generate1.py` - Baseline with audience feedback
  - `generate2.py` - Trained models
  - `generate22.py` - Baseline without audience feedback

- **`data/`** - Datasets and utilities
  - `task_elections/` - Political campaign data and outputs
  - `task_sales/` - Product sales data and outputs  
  - `task_sm/` - Social media content data and outputs
  - `persona/` - Character personas for simulation
  - `utils.py` - Data processing utilities

- **`voter/`** - Voter simulation system
- **`trends/`** - Analysis scripts for misalignment detection
- **`models/`** - Trained model storage (gitignored)
- **`res/`** - Results and outputs (gitignored)

## Tasks

1. **Elections** - Political campaign speech generation
2. **Sales** - Product sales pitch generation  
3. **Social Media** - Social media post generation

## Models & Methods

- **Models**: Qwen3-8B, Llama-3.1-8B-Instruct, GPT-4o-mini
- **Methods**: Base, RFT, TFB

## Usage

```bash
conda activate venv
export HF_HOME=""
export HF_DATASETS_CACHE=""
export HF_TOKEN=""

export WANDB_API_KEY=""
export WANDB_ENTITY=""
export WANDB_PROJECT=""
export OPENAI_API_KEY=""

TASKS = ["task_elections", "task_sales" , "task_sm"]

python -m artsco.src.generate1

python -m artsco.src.train --method_name rft --task task_sales
python -m artsco.src.train --method_name tfb --task task_sales

python -m artsco.src.train --method_name rft --task task_elections
python -m artsco.src.train --method_name tfb --task task_elections

python -m artsco.src.train --method_name rft --task task_sm
python -m artsco.src.train --method_name tfb --task task_sm

python -m artsco.src.generate22
python -m artsco.src.generate2

python -m artsco.voter.voters
```
