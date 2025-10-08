# Moloch's Bargain: Emergent Misalignment When LLMs Compete for Audiences

Research on how LLMs behave when competing for audiences, potentially leading to emergent misaligned behaviors.

## Paper

**arXiv:** [https://arxiv.org/abs/2510.06105](https://arxiv.org/abs/2510.06105)

## Citation

```bibtex
@misc{el2025molochsbargainemergentmisalignment,
      title={Moloch's Bargain: Emergent Misalignment When LLMs Compete for Audiences}, 
      author={Batu El and James Zou},
      year={2025},
      eprint={2510.06105},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2510.06105}, 
}
```

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

1. **Sales** - Product sales pitch generation  
2. **Elections** - Political campaign speech generation
3. **Social Media** - Social media post generation

## Models & Methods

- **Models**: Qwen3-8B, Llama-3.1-8B-Instruct
- **Methods**: Base, RFT, TFB

## Workflow

Follow this order to run the complete pipeline:

### 1. **Data Preparation**
```bash
# Run step1.1.ipynb - Prepare initial datasets
```

### 2. **Baseline Generation with Feedback**
```bash
# Run generate1.py - Generate baseline outputs with audience feedback
python -m artsco.src.generate1
```

### 3. **Dataset Processing**
```bash
# Run step1.2.ipynb - Process generated data for training
```

### 4. **Training Data Preparation**
```bash
# Run step2.1.ipynb - Prepare training datasets
```

### 5. **Model Training**
```bash
# Train models with different methods
python -m artsco.src.train --method_name rft --task task_sales
python -m artsco.src.train --method_name tfb --task task_sales

python -m artsco.src.train --method_name rft --task task_elections
python -m artsco.src.train --method_name tfb --task task_elections

python -m artsco.src.train --method_name rft --task task_sm
python -m artsco.src.train --method_name tfb --task task_sm
```

### 6. **Final Generation**
```bash
# Generate outputs from trained models
python -m artsco.src.generate22  # Baseline without feedback
python -m artsco.src.generate2  # Trained models
```

### 7. **Analysis**
```bash
# Analysis of Performance: Run step2.2*.ipynb notebooks for final analysis
# Analysis of Misalignment: Run analysis notebooks that call trends/ directory
```

## Setup

```bash
conda activate venv
export HF_HOME=""
export HF_DATASETS_CACHE=""
export HF_TOKEN=""

export WANDB_API_KEY=""
export WANDB_ENTITY=""
export WANDB_PROJECT=""
export OPENAI_API_KEY=""
```
