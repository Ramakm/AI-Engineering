"""QLoRA fine-tune of Llama-3.2-3B-Instruct (mirrors notebook 05). Requires a CUDA GPU."""
import argparse, torch
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template, train_on_responses_only


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="unsloth/Llama-3.2-3B-Instruct")
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="llama32-3b-finetuned-lora")
    ap.add_argument("--epochs", type=float, default=1)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--max-seq-len", type=int, default=1024)
    a = ap.parse_args()

    model, tok = FastLanguageModel.from_pretrained(a.base, max_seq_length=a.max_seq_len, load_in_4bit=True)
    model = FastLanguageModel.get_peft_model(
        model, r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth", random_state=42)
    tok = get_chat_template(tok, chat_template="llama-3.2")

    ds = load_dataset("json", data_files={"train": f"{a.data}/train.jsonl", "val": f"{a.data}/val.jsonl"})
    ds = ds.map(lambda b: {"text": [tok.apply_chat_template(m, tokenize=False) for m in b["messages"]]}, batched=True)

    bf16 = torch.cuda.is_bf16_supported()
    trainer = SFTTrainer(
        model=model, tokenizer=tok, train_dataset=ds["train"], eval_dataset=ds["val"].select(range(200)),
        args=SFTConfig(
            dataset_text_field="text", max_seq_length=a.max_seq_len,
            per_device_train_batch_size=2, per_device_eval_batch_size=2, gradient_accumulation_steps=8,
            num_train_epochs=a.epochs, learning_rate=a.lr, lr_scheduler_type="cosine", warmup_ratio=0.03,
            weight_decay=0.01, optim="adamw_8bit", bf16=bf16, fp16=not bf16, logging_steps=10,
            eval_strategy="steps", eval_steps=100, save_strategy="no", output_dir="outputs",
            seed=42, report_to="none"))
    trainer = train_on_responses_only(
        trainer, instruction_part="<|start_header_id|>user<|end_header_id|>\n\n",
        response_part="<|start_header_id|>assistant<|end_header_id|>\n\n")
    trainer.train()
    model.save_pretrained(a.out)
    tok.save_pretrained(a.out)


if __name__ == "__main__":
    main()
