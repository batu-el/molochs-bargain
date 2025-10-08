# System
import os
import json
import ast
import argparse
from typing import List, Dict, Any
from collections import Counter

# ML
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, Dataset
from tqdm import tqdm

# Accelerate
from accelerate import Accelerator
from accelerate.utils import gather_object

# Models and Voters
from artsco.data.utils import MODELS , TASKS, SPLITS
from artsco.voter.voters import Voters
from artsco.voter.utils import load_persona100

# Utils
from artsco.utils import extract_think, extract_answer

def main(args):
    # num_voters = 50
    # bios = load_persona100()[:num_voters]
    # voters = Voters(bios=bios, task=args.task, model_name= "gpt-4o-mini")

    accelerator = Accelerator(mixed_precision="bf16")
    is_main = accelerator.is_main_process
    is_local_main = accelerator.is_local_main_process

    if is_main:
        print("Args:", vars(args))

    # Load tokenizer / model on CPU first; Accelerate moves to the right device(s)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16
    )

    # Dataset & DataLoader
    print("PATH:", args.dataset_path)
    num_players = 2
    dataset_pre = load_dataset("json", data_files=args.dataset_path)['train']#.select(list(range(8)))
    indices = [i for i in range(len(dataset_pre)) for _ in range(num_players)]
    dataset = dataset_pre.select(indices).flatten_indices()

    collate_text = lambda batch: {"prompt": [ex["prompt"] for ex in batch],}

    loader = DataLoader(
        dataset,
        batch_size=args.per_device_batch,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=True,
        collate_fn=collate_text
    )

    # Prepare with Accelerate (DDP / FSDP / etc. according to your config)
    model, loader = accelerator.prepare(model, loader)
    model.eval()

    # Unwrap so we can call .generate on the underlying HF model
    unwrapped = accelerator.unwrap_model(model)

    results_local: List[Dict[str, Any]] = []

    # Progress bar only on local main rank to avoid clutter
    progress = tqdm(loader, disable=not is_local_main)

    for batch in progress:
        prompts = batch["prompt"]

        inputs = tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True
        ).to(accelerator.device)

        # Length of each prompt to slice off from generated sequences
        prompt_lengths = inputs["attention_mask"].sum(dim=1)

        with torch.no_grad():
            sequences = unwrapped.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
            )

        # Remove the prompt tokens and decode
        continuations = [
            seq[prompt_len:] for seq, prompt_len in zip(sequences, prompt_lengths)
        ]

        completions = tokenizer.batch_decode(continuations, skip_special_tokens=True)
        print(completions)     

        thinks  = [extract_think(response=output) for output in completions]
        answers = [extract_answer(response=output, task=args.task) for output in completions]

        player_candidates = [[answers[(i*num_players) + j] for j in range(num_players)] for i in range(len((answers)) // num_players)]
        player_thinks = [[thinks[(i*num_players) + j] for j in range(num_players)] for i in range(len((thinks)) // num_players)]   
        print(player_thinks)     
        print(player_candidates)     
        # voter_votes, voter_thinks = voters.get_votes_list(player_candidates)

        prompts = [[prompts[(i*num_players) + j] for j in range(num_players)] for i in range(len((prompts)) // num_players)]
        completions = [[completions[(i*num_players) + j] for j in range(num_players)] for i in range(len((completions)) // num_players)]

        # Build local results
        for p, c, pc, pt in zip(prompts, completions, player_candidates, player_thinks):
            results_local.append(
                {
                    "prompt": p,
                    "completion": c,
                    "player_candidates": pc,
                    "player_thinks": pt,
                    # "voter_votes": vv,
                    # "voter_thinks": vt,
                    # "vote_counts" : {str(k): v for k, v in Counter(vv).items()},
                }
            )

    # Gather variable-length Python objects across ranks
    gathered_dicts = gather_object(results_local)  # each rank returns a list

    if is_main:
        ds = Dataset.from_list(gathered_dicts)
        # Make sure we replace the existing one
        if os.path.exists(args.results_path):
            os.remove(args.results_path)
        ds.to_json(args.results_path)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Inference configurations.")

    p.add_argument("--model_path_root", type=str, default="artsco/models")
    p.add_argument("--model_path", type=str, default="")
    p.add_argument("--method_name", type=str, default="", choices=MODELS)

    p.add_argument("--model_name", type=str, default="", choices=MODELS)
    p.add_argument("--split", type=str, default="", choices=SPLITS)
    p.add_argument("--task", type=str,  default="", choices=TASKS)

    p.add_argument("--dataset_path_root", type=str, default="artsco/data")
    p.add_argument("--dataset_path", type=str, default="")
    
    p.add_argument("--results_path_root", type=str, default="artsco/res")
    p.add_argument("--results_path", type=str, default="")

    p.add_argument("--per_device_batch", type=int, default=64)

    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--max_new_tokens", type=int, default=1480)
    p.add_argument("--temperature", type=float, default=0.7)

    return p.parse_args()

# CURRENT_MODELS = ["Qwen/Qwen3-8B",  "meta-llama/Llama-3.1-8B-Instruct", "Qwen/Qwen3-14B"]
QWEN_MODEL_NAMES = ["Qwen/Qwen3-8B"] #, "Qwen/Qwen3-32B"]
LLAMA_MODEL_NAMES = ["meta-llama/Llama-3.1-8B-Instruct"] #, "meta-llama/Llama-3.1-70B-Instruct"]
OPENAI_MODEL_NAMES = ["openai/gpt-oss-20b"]
MODELS = QWEN_MODEL_NAMES + LLAMA_MODEL_NAMES 
TASKS = ["task_elections", "task_sales" , "task_sm"]
TASKS = [ "task_sm"]
MODELS = QWEN_MODEL_NAMES
SPLITS = ["test"]
METHOD_NAMES = ["tfb", "rft"]


if __name__ == "__main__":
    args = parse_args()
    
    for task in TASKS:
        for split in SPLITS:
            for model_name in MODELS:
                for method_name in METHOD_NAMES:
                    run_args = argparse.Namespace(**vars(args)) 

                    run_args.model_name = model_name
                    run_args.split = split
                    run_args.task = task

                    run_args.method_name = method_name

                    run_args.dataset_path  = os.path.join(args.dataset_path_root,  run_args.task, run_args.model_name,  f"{run_args.split}.json")
                    os.makedirs( os.path.join(args.dataset_path_root,  run_args.task, run_args.model_name) , exist_ok=True)
                    run_args.results_path  = os.path.join(args.results_path_root,  run_args.task, run_args.model_name, run_args.method_name, f"{run_args.split}_step2.json")
                    os.makedirs( os.path.join(args.results_path_root,  run_args.task, run_args.model_name) , exist_ok=True)

                    run_args.model_path = os.path.join(args.model_path_root,  run_args.task, run_args.model_name,  run_args.method_name)

                    main(run_args)