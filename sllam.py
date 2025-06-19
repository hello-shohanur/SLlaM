import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ---- Config class ----
class SLlaMConfig:
    def __init__(self,
                 vocab_size=10000,
                 n_embd=256,
                 n_layer=4,
                 n_head=4,
                 block_size=128,
                 dropout=0.1):
        self.vocab_size = vocab_size
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.block_size = block_size
        self.dropout = dropout

# ---- Self-attention ----
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head

    def forward(self, x):
        B, T, C = x.size()
        k = self.key(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        q = self.query(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = self.value(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        attn_scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_output = (attn_weights @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.dropout(self.proj(attn_output))

# ---- Feedforward MLP ----
class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        return self.net(x)

# ---- Transformer Block ----
class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = MultiHeadSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.ff = FeedForward(config)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x

# ---- SLlaM Model ----
class SLlaM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)
        self.blocks = nn.Sequential(*[TransformerBlock(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size)
        self.block_size = config.block_size

    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.block_size, "Input too long"

        token_embed = self.token_embedding(idx)
        pos_embed = self.position_embedding(torch.arange(T, device=idx.device))
        x = token_embed + pos_embed
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)

        if targets is None:
            return logits, None
        else:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            return logits, loss

# ---- Training loop ----
def train_model(model, data_loader, optimizer, device, epochs=1):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in data_loader:
            inputs, targets = [b.to(device) for b in batch]
            optimizer.zero_grad()
            logits, loss = model(inputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}: Loss = {total_loss / len(data_loader):.4f}")

# ---- Simple tokenizer (character-based for demo) ----
class SimpleTokenizer:
    def __init__(self, text):
        chars = sorted(list(set(text)))
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}
        self.vocab_size = len(chars)

    def encode(self, s):
        return [self.stoi[c] for c in s]

    def decode(self, l):
        return ''.join([self.itos[i] for i in l])

# ---- Inference (sampling) ----
def generate(model, idx, max_new_tokens, tokenizer, temperature=1.0):
    model.eval()
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -model.block_size:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / temperature
        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, next_token), dim=1)
    return tokenizer.decode(idx[0].tolist())

# ---- Example usage ----
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    text = "hello world. hello again."
    tokenizer = SimpleTokenizer(text)
    data = tokenizer.encode(text)

    block_size = 8
    inputs = []
    targets = []
    for i in range(len(data) - block_size):
        inputs.append(data[i:i+block_size])
        targets.append(data[i+1:i+block_size+1])

    inputs = torch.tensor(inputs)
    targets = torch.tensor(targets)
    dataset = torch.utils.data.TensorDataset(inputs, targets)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)

    config = SLlaMConfig(vocab_size=tokenizer.vocab_size, block_size=block_size)
    model = SLlaM(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    train_model(model, data_loader, optimizer, device, epochs=10)

    # Generate text
    context = torch.tensor([[tokenizer.stoi['h']]], device=device)
    output = generate(model, context, max_new_tokens=50, tokenizer=tokenizer)
    print("\nGenerated:\n", output)

    # Save model
    torch.save(model.state_dict(), "sllam.pth")
