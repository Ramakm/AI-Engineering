"""Build data/{train,val,test}.jsonl in `messages` format (mirrors notebook 04)."""
import argparse, json, os
from datasets import load_dataset
from transformers import AutoTokenizer

ROLE = {"human": "user", "user": "user", "gpt": "assistant", "assistant": "assistant", "system": "system"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="mlabonne/FineTome-100k")
    ap.add_argument("--n", type=int, default=6000)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--out", default="data")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained("unsloth/Llama-3.2-3B-Instruct")
    raw = load_dataset(a.dataset, split="train").shuffle(seed=a.seed).select(range(a.n * 2))

    rows, seen = [], set()
    for ex in raw:
        m = [{"role": ROLE[t["from"]], "content": t["value"].strip()} for t in ex["conversations"]]
        if not m or m[-1]["role"] != "assistant" or any(x["role"] == y["role"] for x, y in zip(m, m[1:])):
            continue
        if not all(x["content"] for x in m):
            continue
        if not 8 <= len(tok.apply_chat_template(m, tokenize=True)) <= a.max_tokens:
            continue
        key = m[0]["content"][:200]
        if key in seen:
            continue
        seen.add(key)
        rows.append({"messages": m})
        if len(rows) == a.n:
            break

    n = len(rows)
    cuts = {"train": rows[: int(.8 * n)], "val": rows[int(.8 * n): int(.9 * n)], "test": rows[int(.9 * n):]}
    os.makedirs(a.out, exist_ok=True)
    for name, part in cuts.items():
        with open(f"{a.out}/{name}.jsonl", "w") as f:
            f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in part)
        print(name, len(part))


if __name__ == "__main__":
    main()
