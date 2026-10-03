# Llama 3.2 3B Instruct Chatbot

A streaming chatbot built on Meta's open-source [Llama 3.2 3B Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct) model, using Hugging Face Transformers for inference and Gradio for the web UI. The whole project lives in one notebook, [Llama-3 Chatbot.ipynb](Llama-3%20Chatbot.ipynb).

## What You'll Learn

- How to load an open-source LLM from the Hugging Face Hub in fp16.
- How the Llama 3 chat template turns a conversation into tokens.
- How to stream generated tokens back while the model is still running.
- Why LLMs have no memory, and how chat history is re-sent on every turn.
- How to wrap it all in a shareable Gradio chat UI.

## Requirements

| Item | Details |
|------|---------|
| Python | 3.9+ |
| GPU | **Required.** The notebook asserts that CUDA is available. A free Colab T4 (16 GB) works. |
| GPU memory | ~6.5 GB for the 3B model in fp16 |
| Disk | ~6 GB for the model weights (downloaded on first run) |
| Hugging Face account | Needed to accept the Llama license and create an access token |

> **Mac users:** the notebook is written for CUDA, so it will not run locally on Apple Silicon without code changes. Run it on Google Colab (or another CUDA machine) instead.

## Step-by-Step Setup

### Step 1: Get access to the model

Llama 3.2 is a gated model.

1. Create a free account at [huggingface.co](https://huggingface.co).
2. Open the [Llama-3.2-3B-Instruct page](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct) and accept Meta's license. Approval is usually quick.
3. Go to **Settings → Access Tokens** and create a token with **read** permission.

### Step 2: Open the notebook on a GPU

1. Upload `Llama-3 Chatbot.ipynb` to [Google Colab](https://colab.research.google.com), or open it from GitHub.
2. Go to **Runtime → Change runtime type** and select **T4 GPU**.

### Step 3: (Optional) Store your token as a secret

To avoid pasting the token every run, add it in Colab under the key icon (**Secrets**) as `HF_TOKEN`. The notebook reads the `HF_TOKEN` environment variable first and falls back to an interactive login prompt if it isn't set.

### Step 4: Run the notebook top to bottom

Run each cell in order. Here is what each part does.

| # | Section | What happens |
|---|---------|--------------|
| 1 | **Install** | `pip install` for `gradio`, `transformers`, `accelerate` and `torch`. |
| 2 | **Imports and GPU check** | Imports the libraries and stops with a clear error if no GPU is found. |
| 3 | **Authenticate with Hugging Face** | Logs in with `HF_TOKEN`, or prompts you for a token. |
| 4 | **UI description** | Defines the HTML header shown at the top of the app. |
| 5 | **Upgrade libraries** | Upgrades `transformers`, `accelerate` and `huggingface_hub` to compatible versions. **Restart the session after this cell**, then re-run the imports and login cells. |
| 6 | **Load tokenizer and model** | Downloads the weights (~6 GB on first run) and places them on the GPU in fp16. |
| 7 | **Generation function** | Builds the conversation, applies the chat template and streams tokens. |
| 8 | **Quick test** *(optional)* | Runs one prompt without the UI to confirm the model works. |
| 9 | **Build the Gradio interface** | Creates the chat UI with sliders for temperature and max new tokens. |
| 10 | **Launch** | Starts the app and prints a public `share` link. |

### Step 5: Chat

When the last cell runs, Gradio prints a local URL and a public `*.gradio.live` URL. Open either one and start chatting. Use the **⚙️ Parameters** panel to adjust settings.

To stop the app, stop the running cell.

## How It Works

```
User message + history
        │
        ▼
 Build conversation  →  [{"role": "user", ...}, {"role": "assistant", ...}, ...]
        │
        ▼
 Llama 3 chat template  →  token IDs
        │
        ▼
 model.generate()  (background thread)
        │
        ▼
 TextIteratorStreamer  →  yields partial text  →  Gradio shows it live
```

1. **Conversation building.** The LLM has no memory. On every turn, the full history plus the new message is sent to the model. The code accepts both Gradio history formats (`[user, assistant]` pairs and `{"role", "content"}` dicts).
2. **Tokenization.** `tokenizer.apply_chat_template(...)` wraps the messages in Llama 3's special tokens and converts them to IDs.
3. **Generation.** `model.generate()` runs in a separate `Thread` so the main thread can read tokens as they arrive.
4. **Streaming.** `TextIteratorStreamer` hands back text piece by piece, and the function `yield`s the growing reply so the UI updates live.
5. **Stopping.** Generation ends on the standard end-of-sequence token or Llama's `<|eot_id|>` token.

## Settings

| Setting | Default | Notes |
|---------|---------|-------|
| `MODEL_ID` | `meta-llama/Llama-3.2-3B-Instruct` | Switch to `meta-llama/Llama-3.2-1B-Instruct` for a lighter model (~2.5 GB). It is faster but gives weaker answers, and you must accept its license too. |
| Temperature | 0.95 (slider: 0–1) | Higher is more creative. `0` switches to greedy decoding, because sampling with temperature 0 crashes. |
| Max new tokens | 512 (slider: 128–1024) | Capped at 1024 to keep the KV cache within a T4's memory during long chats. |

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `AssertionError: No GPU found` | Switch the runtime to GPU (Colab: **Runtime → Change runtime type → T4 GPU**). |
| `401` / `403` or "gated repo" error | Accept the model license on Hugging Face and make sure your token has read access. |
| Out-of-memory error | Use the 1B model, lower **Max new tokens**, or restart the runtime to free memory. |
| Import errors after upgrading libraries | Restart the session, then re-run the cells from the top. |
| Slow first run | The ~6 GB model download only happens once per runtime. |

## Tech Stack

- [Llama 3.2 3B Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct): the language model
- [Transformers](https://github.com/huggingface/transformers): model loading, chat template, generation
- [PyTorch](https://pytorch.org): tensor and GPU backend
- [Accelerate](https://github.com/huggingface/accelerate): `device_map="auto"` weight placement
- [Gradio](https://www.gradio.app): chat web UI

## License

The code follows this repository's [LICENSE](../../../LICENSE). The Llama 3.2 model is covered by Meta's own [Llama 3.2 Community License](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct/blob/main/LICENSE.txt).
