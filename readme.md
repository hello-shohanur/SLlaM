# SLlaM: A Serene Large Language Model

> A minimal transformer-based language model built from scratch with PyTorch

---

## 🚀 Overview

**SLlaM (****Serene Large Language Model****)** is a compact, easy-to-understand implementation of a GPT-style transformer architecture. It's designed for educational purposes, fast prototyping, and small-scale language modeling tasks.

---

## 🧠 Features

- Fully custom transformer architecture
- Character-level tokenizer (can be extended)
- Supports training and inference (text generation)
- Minimal dependencies (only PyTorch)
- Easy to extend with HuggingFace tokenizer or datasets

---

## 📦 Installation

```bash
# Clone the repository
$ git clone https://github.com/hello-shohanur/sllam.git
$ cd sllam

# Install requirements
$ pip install torch
```

---

## 🏁 Getting Started

Run the main training and generation script:

```bash
$ python sllam.py
```

### Output:

After training, the model will generate text starting from the character "h".

```
Generated:
 hello world. hello again.hello again.hello again.
```

---

## 🛠️ File Structure

```
sllam/
├── sllam.py         # Main model, training, generation
├── README.md        # This file
```

*Note: The **``** file (saved model weights) will be created after training is completed.*

---

## 🧪 Usage

You can import and use `SLlaM` in your own projects:

```python
from sllam import SLlaM, SLlaMConfig
```

### Text Generation

```python
context = torch.tensor([[tokenizer.stoi['h']]], device=device)
output = generate(model, context, max_new_tokens=50, tokenizer=tokenizer)
print(output)
```

---

## 📚 Customize

You can easily change:

- `vocab_size` — for your own tokenization
- `block_size`, `n_layer`, `n_head`, `n_embd` — for scaling
- `SimpleTokenizer` — replace with real tokenizers like HuggingFace

---

## 🤖 Model Specs (default)

| Parameter       | Value |
| --------------- | ----- |
| Embedding Dim   | 256   |
| Layers          | 4     |
| Attention Heads | 4     |
| Block Size      | 128   |
| Dropout         | 0.1   |
| Vocab Size      | 10000 |

---

## 📄 License

This project is MIT licensed. Feel free to use, modify, and share.

---

## 🙏 Acknowledgements

- Inspired by nanoGPT and GPT architecture
- Built with love using PyTorch

---

## ✨ Future Plans

- Tokenizer Enhancements
- Dataset & Training Pipeline
- Training Optimization
- Model Scaling

---

## 🔗 Connect

Questions? Suggestions? Open an issue or reach out!

---

> Built with ❤️ by the open-source community.

