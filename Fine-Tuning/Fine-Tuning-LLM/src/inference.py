"""Chat with the fine-tuned adapter: python src/inference.py "your question" """
import sys, torch
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template

ADAPTER = "llama32-3b-finetuned-lora"
model, tok = FastLanguageModel.from_pretrained(ADAPTER, max_seq_length=2048, load_in_4bit=True)
tok = get_chat_template(tok, chat_template="llama-3.2")
FastLanguageModel.for_inference(model)

question = " ".join(sys.argv[1:]) or "Explain what LoRA is."
ids = tok.apply_chat_template([{"role": "user", "content": question}], add_generation_prompt=True, return_tensors="pt").to("cuda")
with torch.no_grad():
    out = model.generate(input_ids=ids, max_new_tokens=400, temperature=0.7, do_sample=True)
print(tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True))
