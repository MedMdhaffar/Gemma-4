import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class LandmarkDataset(Dataset):
    def __init__(self, csv_path, use_mask=True):
        self.csv_path = Path(csv_path)
        self.use_mask = use_mask
        self.rows = []

        with open(self.csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.rows.append(row)

        if not self.rows:
            raise ValueError(f"No rows found in {self.csv_path}")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        x = np.load(row["feature_path"]).astype(np.float32)

        if self.use_mask:
            mask = np.load(row["mask_path"]).astype(np.float32)
            # x is (C, T, V, M), mask is (T, V). Expand mask over C and M.
            x = x * mask[np.newaxis, :, :, np.newaxis]

        y = int(row["label"])
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long)


def load_graph(graph_path):
    with open(graph_path, "r", encoding="utf-8") as f:
        graph = json.load(f)

    num_vertices = int(graph["num_vertices"])
    adjacency = np.eye(num_vertices, dtype=np.float32)

    for a, b in graph["edges"]:
        adjacency[a, b] = 1.0
        adjacency[b, a] = 1.0

    degree = adjacency.sum(axis=1)
    degree_inv_sqrt = np.power(degree, -0.5, where=degree > 0)
    degree_inv_sqrt[degree == 0] = 0.0
    normalized = (
        degree_inv_sqrt[:, np.newaxis]
        * adjacency
        * degree_inv_sqrt[np.newaxis, :]
    )

    return torch.from_numpy(normalized)


class STGCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, adjacency, stride=1, dropout=0.0):
        super().__init__()
        self.register_buffer("adjacency", adjacency)

        self.gcn = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=(9, 1),
                stride=(stride, 1),
                padding=(4, 0),
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout),
        )

        if in_channels == out_channels and stride == 1:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=(stride, 1),
                ),
                nn.BatchNorm2d(out_channels),
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = self.residual(x)
        x = torch.einsum("nctv,vw->nctw", x, self.adjacency)
        x = self.gcn(x)
        x = self.tcn(x)
        return self.relu(x + residual)


class CompactSTGCN(nn.Module):
    def __init__(self, num_classes, adjacency, in_channels=3, dropout=0.3):
        super().__init__()
        self.data_bn = nn.BatchNorm1d(in_channels * adjacency.shape[0])

        self.blocks = nn.Sequential(
            STGCNBlock(in_channels, 64, adjacency, dropout=dropout),
            STGCNBlock(64, 64, adjacency, dropout=dropout),
            STGCNBlock(64, 128, adjacency, stride=2, dropout=dropout),
            STGCNBlock(128, 128, adjacency, dropout=dropout),
            STGCNBlock(128, 256, adjacency, stride=2, dropout=dropout),
            STGCNBlock(256, 256, adjacency, dropout=dropout),
        )
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        # Input: N, C, T, V, M. Current data has M=1.
        n, c, t, v, m = x.shape
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(n * m, v * c, t)
        x = self.data_bn(x)
        x = x.view(n, m, v, c, t)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(n * m, c, t, v)

        x = self.blocks(x)
        x = x.mean(dim=(2, 3))
        x = x.view(n, m, -1).mean(dim=1)
        return self.classifier(x)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def accuracy_topk(logits, target, topk=(1, 5)):
    max_k = max(topk)
    _, pred = logits.topk(max_k, dim=1)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    results = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum()
        results.append(correct_k.mul_(100.0 / target.size(0)).item())
    return results


def run_epoch(model, loader, criterion, optimizer, device, scaler=None):
    model.train()
    total_loss = 0.0
    total_top1 = 0.0
    total_top5 = 0.0
    total_samples = 0

    for x, y in tqdm(loader, desc="train", leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            logits = model(x)
            loss = criterion(logits, y)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        batch_size = y.size(0)
        top1, top5 = accuracy_topk(logits.detach(), y)
        total_loss += loss.item() * batch_size
        total_top1 += top1 * batch_size
        total_top5 += top5 * batch_size
        total_samples += batch_size

    return {
        "loss": total_loss / total_samples,
        "top1": total_top1 / total_samples,
        "top5": total_top5 / total_samples,
    }


@torch.no_grad()
def evaluate(model, loader, criterion, device, split_name="val"):
    model.eval()
    total_loss = 0.0
    total_top1 = 0.0
    total_top5 = 0.0
    total_samples = 0

    for x, y in tqdm(loader, desc=split_name, leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)
        loss = criterion(logits, y)

        batch_size = y.size(0)
        top1, top5 = accuracy_topk(logits, y)
        total_loss += loss.item() * batch_size
        total_top1 += top1 * batch_size
        total_top5 += top5 * batch_size
        total_samples += batch_size

    return {
        "loss": total_loss / total_samples,
        "top1": total_top1 / total_samples,
        "top5": total_top5 / total_samples,
    }


def infer_num_classes(label_map_path, train_csv_path):
    label_map_path = Path(label_map_path)
    if label_map_path.exists():
        labels = []
        with open(label_map_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                labels.append(int(row["label"]))
        if labels:
            return max(labels) + 1

    labels = []
    with open(train_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            labels.append(int(row["label"]))
    return max(labels) + 1


def save_checkpoint(path, model, optimizer, scheduler, epoch, best_top1, args):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict() if scheduler else None,
            "best_top1": best_top1,
            "args": vars(args),
        },
        path,
    )


def main():
    parser = argparse.ArgumentParser(description="Train a compact ST-GCN on NSLT landmarks.")
    parser.add_argument("--data-root", default="preprocessed_nslt300")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0005)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-mask", action="store_true")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--checkpoint-dir", default="checkpoints/nslt300_stgcn")
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()

    set_seed(args.seed)

    data_root = Path(args.data_root)
    train_csv = data_root / "train.csv"
    val_csv = data_root / "val.csv"
    label_map = data_root / "label_map.csv"
    graph_path = data_root / "stgcn_graph.json"

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if args.device == "cuda" and device.type != "cuda":
        print("CUDA was requested but is not available. Falling back to CPU.")

    train_dataset = LandmarkDataset(train_csv, use_mask=not args.no_mask)
    val_dataset = LandmarkDataset(val_csv, use_mask=not args.no_mask)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )

    adjacency = load_graph(graph_path).to(device)
    num_classes = infer_num_classes(label_map, train_csv)
    model = CompactSTGCN(num_classes, adjacency, dropout=args.dropout).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler() if args.amp and device.type == "cuda" else None

    start_epoch = 1
    best_top1 = 0.0
    checkpoint_dir = Path(args.checkpoint_dir)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        if checkpoint.get("scheduler_state") is not None:
            scheduler.load_state_dict(checkpoint["scheduler_state"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_top1 = float(checkpoint.get("best_top1", 0.0))

    print(f"Device: {device}")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Classes: {num_classes}")
    print(f"Mask enabled: {not args.no_mask}")

    for epoch in range(start_epoch, args.epochs + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            scaler=scaler,
        )
        val_metrics = evaluate(model, val_loader, criterion, device, split_name="val")
        scheduler.step()

        print(
            f"epoch {epoch:03d}/{args.epochs} "
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_top1={train_metrics['top1']:.2f} "
            f"train_top5={train_metrics['top5']:.2f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_top1={val_metrics['top1']:.2f} "
            f"val_top5={val_metrics['top5']:.2f}"
        )

        save_checkpoint(
            checkpoint_dir / "last.pt",
            model,
            optimizer,
            scheduler,
            epoch,
            best_top1,
            args,
        )

        if val_metrics["top1"] > best_top1:
            best_top1 = val_metrics["top1"]
            save_checkpoint(
                checkpoint_dir / "best.pt",
                model,
                optimizer,
                scheduler,
                epoch,
                best_top1,
                args,
            )
            print(f"Saved new best checkpoint: top1={best_top1:.2f}")


if __name__ == "__main__":
    main()
