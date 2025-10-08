import os
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Mxfp4Config
from peft import LoraConfig, get_peft_model
from trl import SFTConfig, SFTTrainer

def main(model_name, method_name, task):
    # ---------- Data ----------
    results_path_root = "artsco/data"
    split = "train"
    data_path= os.path.join(results_path_root,  task, model_name,  f"{split}_{method_name}.json")
    dataset = load_dataset("json", data_files=data_path, split="train")

    # ---------- Tokenizer ----------
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # ---------- Base model (quantized, streamed) ----------
    # quantization_config = Mxfp4Config(dequantize=False)  # keep MXFP4 on device

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        attn_implementation="eager",
        torch_dtype=torch.bfloat16,
        # quantization_config=quantization_config,
        use_cache=False,
        low_cpu_mem_usage=True,   # stream shards, cut host RAM spikes
        use_safetensors=True,
    )

    # ---------- LoRA ----------
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        # Typical LLaMA targets:
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    )

    model = get_peft_model(model, peft_config)

    # ---------- TRL SFT ----------
    output_dir = f"artsco/models/{task}/{model_name}/{method_name}"
    num_epochs = 1

    training_args = SFTConfig(
        learning_rate=2e-4,
        gradient_checkpointing=True,
        num_train_epochs=num_epochs,
        logging_steps=1,
        # Start small to avoid OOM; scale up after it runs
        per_device_train_batch_size=16,
        gradient_accumulation_steps=1,
        # max_length=2048,  # keep if your TRL version expects this; otherwise use max_seq_length
        warmup_ratio=0.03,
        lr_scheduler_type="cosine_with_min_lr",
        lr_scheduler_kwargs={"min_lr_rate": 0.1},
        output_dir=output_dir,
        # report_to="trackio",
        # push_to_hub=True,

        # Distributed + precision
        bf16=True,
        ddp_find_unused_parameters=False,
        # model_init_kwargs={"torch_dtype": torch.bfloat16},

        # Shard model states across GPUs
        # deepspeed="ds_z3.json",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    trainer.train()

    if trainer.is_world_process_zero():
        trainer.save_model(output_dir)



TASKS = ["task_elections", "task_sales" , "task_sm"]
CURRENT_MODELS = ["Qwen/Qwen3-8B",  "meta-llama/Llama-3.1-8B-Instruct"]
CURRENT_METHODS = ["rft", "tfb"]

import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run model with specified method and task.")
    parser.add_argument(
        "--method_name",
        type=str,
        required=True,
        help="The method name to use"
    )
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        help="The task to run"
    )

    args = parser.parse_args()

    for model_name in CURRENT_MODELS:
        main(model_name, args.method_name, args.task)