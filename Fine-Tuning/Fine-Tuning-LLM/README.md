# Fine-Tuning-LLM · Llama-3.2-3B-Instruct with QLoRA

Instruction-tune **Llama-3.2-3B-Instruct** on your own data using **QLoRA** (4-bit base + LoRA adapters) with Unsloth + TRL. Fits on a free Colab **T4 (16 GB)**.

## Structure
```
Fine-Tuning-LLM/
├── 01_Why_finetuning.ipynb            concepts + LoRA parameter math      (CPU)
├── 02_Where_finetuning_fits_in.ipynb  prompting vs RAG vs FT, memory math  (CPU)
├── 03_Instruction_tuning.ipynb        Llama 3 chat template, loss masking  (CPU)
├── 04_Data_preparation.ipynb          download → clean → split → jsonl     (CPU)
├── 05_Training.ipynb                  QLoRA training + save adapter        (GPU)
├── 06_Evaluation.ipynb                base vs fine-tuned, ROUGE, ppl, GGUF (GPU)
├── data/                              generated jsonl + dataset sources (see data/README.md)
├── src/                               prepare_data.py · train.py · inference.py
└── requirements.txt
```
Run notebooks in order. 01–04 run anywhere; 05–06 need a GPU (Colab: Runtime → T4 GPU).

## Data
Default: [`mlabonne/FineTome-100k`](https://huggingface.co/datasets/mlabonne/FineTome-100k) (6k-example subset), downloaded automatically. Alternatives (customer support, Alpaca, Dolly, OASST2) are listed in [data/README.md](data/README.md). Best results come from your own data in the same `messages` jsonl format.

## Model access
We use the ungated mirror `unsloth/Llama-3.2-3B-Instruct`. To use the official `meta-llama/Llama-3.2-3B-Instruct`, accept the license on Hugging Face and `huggingface-cli login`.

## Script workflow (alternative to notebooks)
```bash
pip install -r requirements.txt
python src/prepare_data.py --n 6000
python src/train.py --epochs 1
python src/inference.py "Explain gradient descent simply"
```

## Results to expect
~45–90 min on a T4 for 6k examples × 1 epoch, ~25 M trainable parameters (0.76 % at r=16), adapter ≈ 100 MB.
