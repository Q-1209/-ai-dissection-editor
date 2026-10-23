# AI Dissection Editor

**Open a local LLM and work on its insides like parts on a bench.**

AI Dissection Editor is a desktop app that loads a local Hugging Face model (safetensors) and shows its internals (embeddings, attention, MLP, norms) as a node graph. From there you can:

- **rewire it**: reorder or skip layers, branch and merge them, then run generation on the graph you built
- **watch it think**: attention heatmaps, activation statistics, token-by-token sensitivity, and a logit-lens-style "token journey"
- **steer it**: add or ablate directions in the residual stream (activation steering, sycophancy and refusal directions) and compare the output side by side with the unmodified model

Most node-based AI tools (ComfyUI, LangFlow, Flowise) call a model from the outside. This one opens the model and treats its layers as the nodes.

> **Status: early, solo-developed, and I'm looking for feedback.** Everything below runs on my machine (Windows, Qwen2.5-0.5B-Instruct). I'm publishing now, before adding more features, to find out whether this is useful to anyone. If you try it, I'd love to hear what worked, what broke, and what you'd use it for. See [Feedback](#feedback).

![Demo: expand a layer, skip layer 12 by rewiring the graph, then steer layer 13](docs/demo.gif)

*Skipping layer 12 turns "How can I assist you today?" into "How can I help you today?". Steering layer 13 makes the model talk about its day instead of saying it doesn't have one.*

## What it can do today

| Tab | Feature | What you see |
| --- | --- | --- |
| **Dissect** (解剖) | Structure viewer | Every tensor in the checkpoint, labelled by role and grouped by layer. It only reads safetensors headers, so even multi-GB models open instantly. |
| **Run** (実行) | Graph execution engine | Drag edges to reorder, skip, branch or merge attention/MLP blocks, then generate text. Includes a KV cache, temperature/top-k/top-p, repetition penalty and no-repeat n-gram, plus chat templates. |
| **Observe** (観察) | Internal state | Attention heatmaps (per head or head average) and per-block activation stats, on linear or log scale |
| | Prompt Sensitivity Map | Delete or replace each input token in turn and measure the KL divergence of the next-token distribution |
| | Token Journey | The nearest vocabulary token to each hidden state, after every block |
| | Multi-Model Battle View | The same prompt through two graphs or models, compared side by side |
| | Neuron Search / Layer Ablation Wizard | Find neurons that respond to a prompt, knock out layers and measure the damage |
| | Activation Steering Studio | Build a steering vector from contrastive prompt sets (mean difference), apply it to the layers you pick, and compare the generated text and KL divergence |
| | Sycophancy / Refusal tuners | Presets for the "agreeing" and "refusing" directions. Addition and directional ablation are both supported. |
| | Live Slider | Change the steering strength and watch the output update (experimental) |

The inference engine is written from scratch in PyTorch (RMSNorm, RoPE, GQA attention, SwiGLU MLP) and does **not** depend on `transformers`. That independence is what lets the graph be rewired freely.

### Things I found with it

These are observations on Qwen2.5-0.5B-Instruct (24 layers):

- **Steering: one layer beats five.** A "thank you" vs. "this is the worst" vector (mean difference) added to 5 layers collapsed generation into repetition, even at scale 1. Added to one middle layer (layer 13) at scale 1, it shifted the tone (exclamation marks, a sudden "welcome!") without collapsing. The effect accumulates through the residual stream.
- **Removing a refusal direction from one layer isn't enough.** With directional ablation on a single layer, refusals remained. Ablating every layer removed them, which matches [Arditi et al. 2024](https://arxiv.org/abs/2406.11717). Adding the direction instead made the model refuse harmless requests.
- **Massive activations show up clearly.** Activation std jumps at layer 2, stays around 16–17 through layer 20, then falls off, consistent with [Sun et al. 2024](https://arxiv.org/abs/2402.17762).
- **The question mark carries the question.** In "こんにちは、元気ですか？" ("Hi, how are you?"), deleting the final "？" moved the next-token distribution most (KL 5.73, compared with 0.04–0.44 for the other tokens).

## Quick start

Tested on **Windows** with **Qwen2.5-0.5B-Instruct** on CPU. macOS and Linux should work but haven't been tested yet. Reports are very welcome.

**Requirements:** Node.js 22 LTS, Python 3.10+, Git, about 2 GB of disk for the small model.

**One command** (creates `.venv`, installs CPU PyTorch, downloads Qwen2.5-0.5B-Instruct into `models/`, runs `npm install`; the app picks up `.venv` by itself):

```bash
git clone https://github.com/Q-1209/-ai-dissection-editor.git ai-dissection-editor
cd ai-dissection-editor
powershell -ExecutionPolicy Bypass -File scripts\setup.ps1   # Windows
bash scripts/setup.sh                                        # macOS / Linux
npm run electron:dev
```

Or step by step:

```bash
git clone https://github.com/Q-1209/-ai-dissection-editor.git ai-dissection-editor
cd ai-dissection-editor

# 1. Python side (CPU build of PyTorch is enough for 0.5B models)
python -m pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# 2. Download a small model to try
pip install -U huggingface_hub
hf download Qwen/Qwen2.5-0.5B-Instruct --local-dir models/Qwen2.5-0.5B-Instruct

# 3. App
npm install
npm run electron:dev
```

The app runs `python3` (`python` on Windows). If your Python with PyTorch has a different name or path, set `PYTHON_BIN` before starting:

```powershell
$env:PYTHON_BIN = "C:\path\to\python.exe"   # PowerShell
# set PYTHON_BIN=C:\path\to\python.exe        # cmd.exe
npm run electron:dev
```

Then click **Open model folder** and select `models/Qwen2.5-0.5B-Instruct`.

The UI is in English and Japanese. It follows your OS language, and the button at the top right switches it.

### Supported models

| Architecture | Status |
| --- | --- |
| Qwen2 / Qwen2.5 (dense) | ✅ Verified on real weights (Qwen2.5-0.5B-Instruct) |
| Llama-family dense (Llama, older dense Mistral) | 🟡 Should work. The structure is verified on dummy weights, but real weights haven't been tested yet. |
| Gemma 2 / 3 | 🟡 The structure viewer works. The execution engine is not supported yet. |
| MoE (Mixtral, recent Mistral/Gemma), Mamba/SSM, multimodal | ❌ Not supported |

Only safetensors are supported (no `.bin` pickles). Quantized dtypes are untested.

## Responsible use

The steering and refusal tools exist to **observe and understand** model behaviour, in the spirit of published interpretability research. Removing a refusal direction from a small local model doesn't make it more capable. It only shows where that behaviour lives. Please don't use this project to strip safeguards from models you deploy to other people, and follow the license of every model you load.

## Development

```bash
python -m pytest python/tests   # logic tests (numpy); one end-to-end test runs when torch is installed
npx tsc -b && npx vite build    # type check and build the UI
```

The Electron main process spawns Python entry points (`python/*.py`) that speak JSON over stdin/stdout. The inference engine lives in `python/inference/`. A detailed (Japanese) development log, with design decisions and the bugs I hit along the way, is in [docs/DEVLOG.ja.md](docs/DEVLOG.ja.md).

## Feedback

I'm trying to learn whether this is worth continuing, and in which direction. The most useful things you can do:

1. **Try it and open a ["I tried it" issue](../../issues/new?template=feedback.yml)**, even if it's a one-liner ("installed fine, attention view is neat, steering crashed on Llama-3.2-1B").
2. **Report bugs** with the [bug template](../../issues/new?template=bug_report.yml).
3. **Tell me which model you want supported next.**

Pull requests are welcome too. For big changes, please open an issue first.

## Roadmap (tentative)

Next up, depending on feedback: broader Llama/Mistral verification and direct weight editing. Further out, I'm considering LoRA training, model diffing and auditing tools, and MoE support. Feedback decides the order.

## License

[MIT](LICENSE) © Q-1209

[日本語のREADMEはこちら](README.ja.md)
