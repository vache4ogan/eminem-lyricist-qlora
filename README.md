# Eminem Lyricist LLM 🎤🤖

A professional-grade NLP pipeline for parameter-efficient fine-tuning (PEFT) of a modern LLM to generate rap verses in the authentic style of Eminem. 

*This project represents an architectural evolution from my previous LSTM-based lyrics generator, moving towards state-of-the-art transformer models.*

---

## 🚀 Key Features

* **State-of-the-Art Core:** Uses `Qwen-2.5-1.5B-Instruct` as the base language model.
* **QLoRA (4-bit Quantization):** Memory-efficient training via `bitsandbytes` and `peft`, allowing the model to be trained on consumer-grade hardware or free cloud GPUs.
* **Custom Full Pipeline:** Built entirely from scratch—from custom PyTorch `Dataset` parsing to automated checkpoint saving.
* **Production Codebase Structure:** Clean separation of concerns (dataset processing, training configs, module initialization).

---

## 🛠️ Tech Stack

* **Language:** Python 3.10+
* **Deep Learning:** PyTorch, Hugging Face (Transformers, Accelerate, TRL)
* **PEFT Techniques:** LoRA, QLoRA (NF4 quantization)

---

## 📂 Project Structure

```text
eminem-llm-lyricist/
├── data/
│   └── eminem_lyrics.txt       # Raw scraped lyrics separated by stanzas
├── src/
│   ├── __init__.py             # Package initialization and exports
│   ├── dataset.py              # Custom PyTorch Dataset with instruction mapping
│   └── train.py                # QLoRA training script with SFTTrainer
├── requirements.txt            # Project dependencies
└── README.md                   # Project documentation

```
## 📈 Evolutionary Leap: LSTM vs. LLM + QLoRA

In my previous project, I used an LSTM network for character/word-level generation. While it worked, it suffered from severe limitations:
* **Vanishing Context:** LSTM struggled to maintain rhyme schemes beyond 2–3 lines.
* **Lack of World Knowledge:** The model didn't understand semantic concepts, only statistical token frequencies.

**The Upgrade:** By switching to a Transformer architecture and utilizing Transfer Learning, this model leverages billions of parameters of pre-trained language structure, while QLoRA layers adapt the style, slang, and complex internal rhyming patterns specific to Eminem.

---

## 🔧 Installation & Training

### 1. Clone the repository
```bash
git clone [https://github.com/](https://github.com/)<your-username>/eminem-lyricist-qlora.git
cd eminem-lyricist-qlora
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```
### 3. Run the training pipeline

    [!IMPORTANT]
    Running the training pipeline requires a CUDA-compatible GPU.

```bash

python src/train.py
```