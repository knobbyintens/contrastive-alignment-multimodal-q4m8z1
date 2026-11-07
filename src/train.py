import torch
import torch.nn.functional as F

from .model import ContrastiveAligner


def contrastive_loss(image_emb, text_emb, temperature=0.07):
    logits = image_emb @ text_emb.t() / temperature
    labels = torch.arange(logits.size(0), device=logits.device)
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels)) / 2


def train(loader, model: ContrastiveAligner, epochs: int = 40):
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    for _ in range(epochs):
        for images, texts in loader:
            image_emb, text_emb = model(images, texts)
            loss = contrastive_loss(image_emb, text_emb)
            opt.zero_grad()
            loss.backward()
            opt.step()
