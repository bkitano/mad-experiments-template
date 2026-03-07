import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from train.train import train_epoch, evaluate
from dataset.tokens import S5TokenSystem
from dataset.dataset import S5CompositionDataset
from models.transformer import S5Transformer

"""
uv run python -m train.local:main
"""

def main():
    token_system = S5TokenSystem()
    dataset = S5CompositionDataset(token_system, k_range=(1, 5), num_samples=1000, max_seq_len=512)
    loader = DataLoader(dataset, batch_size=128, shuffle=True)
    model = S5Transformer(token_system.num_tokens, token_system.num_group_elements)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    loss, accuracy = train_epoch(model, loader, optimizer, criterion, device)
    print(f"Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")

    loss, accuracy, k_accuracy = evaluate(model, loader, criterion, device)
    print(f"Loss: {loss:.4f}, Accuracy: {accuracy:.4f}, K Accuracy: {k_accuracy}")

if __name__ == "__main__":
    main()