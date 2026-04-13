#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LoRA fine-tuning for Mixtral-8x7B on climate Q&A")
    parser.add_argument(
        "--model-name",
        type=str,
        default="mistralai/Mixtral-8x7B-Instruct-v0.1",
        help="Base model name",
    )
    parser.add_argument(
        "--train-file",
        type=Path,
        default=Path("data/processed/climate_qa_train.jsonl"),
        help="Training JSONL path",
    )
    parser.add_argument(
        "--eval-file",
        type=Path,
        default=Path("data/processed/climate_qa_eval.jsonl"),
        help="Evaluation JSONL path",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models/mixtral_climate_lora"),
        help="Directory to save adapters",
    )
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="Use 4-bit quantized loading (requires bitsandbytes)",
    )
    return parser.parse_args()


def _format_record(record: dict) -> dict[str, str]:
    text = (
        "<s>[INST] You are an EVS climate assistant. "
        "Provide detailed interpretation and include data representation.\n"
        f"Question: {record['question']} [/INST]\n{record['answer']}</s>"
    )
    return {"text": text}


def _prepare_dataset(path: Path):
    dataset = load_dataset("json", data_files=str(path), split="train")
    dataset = dataset.map(_format_record, remove_columns=dataset.column_names)
    return dataset


def main() -> None:
    args = parse_args()

    if not args.train_file.exists():
        raise FileNotFoundError(
            f"Training file missing: {args.train_file}. Run scripts/generate_qa_pairs.py first."
        )

    if not torch.cuda.is_available():
        raise RuntimeError("GPU is required for Mixtral fine-tuning.")

    torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model_kwargs = {
        "torch_dtype": torch_dtype,
        "device_map": "auto",
    }

    if args.load_in_4bit:
        from transformers import BitsAndBytesConfig

        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model_name, **model_kwargs)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    train_dataset = _prepare_dataset(args.train_file)
    eval_dataset = _prepare_dataset(args.eval_file) if args.eval_file.exists() else None

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=max(1, args.batch_size),
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        bf16=torch_dtype == torch.bfloat16,
        fp16=torch_dtype == torch.float16,
        logging_steps=10,
        save_strategy="epoch",
        evaluation_strategy="epoch" if eval_dataset is not None else "no",
        gradient_checkpointing=True,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        peft_config=lora_config,
        dataset_text_field="text",
        max_seq_length=args.max_seq_len,
        packing=False,
    )

    trainer.train()
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))

    print("LoRA fine-tuning complete.")
    print(f"Adapters saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
