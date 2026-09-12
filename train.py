import math

import torch
import torch.nn as nn
import torch.optim as optim

from dataset import get_dataloaders
from model import VisionTransformer


def train_one_epoch(
    model,
    dataloader,
    criterion,
    optimizer,
    scheduler,
    device,
):
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for images, targets in dataloader:
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, targets)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0,
        )

        optimizer.step()
        scheduler.step()

        total_loss += loss.item() * images.size(0)

        preds = outputs.argmax(dim=1)
        correct += preds.eq(targets).sum().item()
        total += targets.size(0)

    return total_loss / total, 100.0 * correct / total


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    for images, targets in dataloader:
        images = images.to(device)
        targets = targets.to(device)

        outputs = model(images)
        loss = criterion(outputs, targets)

        total_loss += loss.item() * images.size(0)

        preds = outputs.argmax(dim=1)
        correct += preds.eq(targets).sum().item()
        total += targets.size(0)

    return total_loss / total, 100.0 * correct / total


def main():
    torch.backends.cudnn.benchmark = True

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Training on device: {device}")

    train_loader, test_loader = get_dataloaders(
        batch_size=128,
        num_workers=2,
    )

    model = VisionTransformer(
        img_size=32,
        patch_size=4,
        in_channels=3,
        num_classes=10,
        embed_dim=192,
        depth=6,
        num_heads=6,
        mlp_ratio=4.0,
        dropout=0.1,
    ).to(device)

    epochs = 40

    total_steps = epochs * len(train_loader)
    warmup_steps = 5 * len(train_loader)

    criterion = nn.CrossEntropyLoss(
        label_smoothing=0.1
    )

    optimizer = optim.AdamW(
        model.parameters(),
        lr=5e-4,
        weight_decay=1e-2,
    )

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)

        progress = (
            step - warmup_steps
        ) / max(1, total_steps - warmup_steps)

        return 0.5 * (
            1.0 + math.cos(math.pi * progress)
        )

    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda,
    )

    num_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(f"Total Parameters: {num_params:,}")

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scheduler,
            device,
        )

        test_loss, test_acc = evaluate(
            model,
            test_loader,
            criterion,
            device,
        )

        print(
            f"Epoch [{epoch:02d}/{epochs}] | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.2f}% | "
            f"Test Loss: {test_loss:.4f} | "
            f"Test Acc: {test_acc:.2f}% | "
            f"LR: {optimizer.param_groups[0]['lr']:.6f}"
        )


if __name__ == "__main__":
    main()