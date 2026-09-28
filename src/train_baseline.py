"""Train the baseline RCNet on one of the three lower symmetry levels.

Crystal system, lattice type and point group are each predicted directly from the XRD
spectrum by the same architecture; only the number of output classes differs.

    python train_baseline.py --task crystal_system --train data/train.npy \
        --val data/val.npy --test data/test.npy --epochs 800
"""

import argparse
import copy
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data as Data
from tqdm import tqdm

from dataset import XRDDataset
from focal_loss import FocalLoss
from labels import LABELS
from resnet import RCNet


def collate_fn(batch):
    spectra, targets, rows = zip(*batch)
    return torch.stack(spectra), torch.stack(targets), rows


def to_device(model, device):
    """Move to ``device``, wrapping in DataParallel only when there is a GPU to parallelise."""
    if device.type == "cuda":
        return nn.DataParallel(model).to(device)
    return model.to(device)


def train(model, train_loader, val_loader, checkpoint_path, log_path,
          epochs, lr, weight_decay, gamma, device):
    """Train ``model`` in place, keeping the checkpoint with the best validation accuracy.

    Returns the per-epoch history as a DataFrame.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = FocalLoss(gamma=gamma)

    history = {"loss_train": [], "acc_train": [], "loss_val": [], "acc_val": []}
    # Start below any achievable accuracy so the first epoch always writes a checkpoint.
    best_acc = -1.0
    nan_loss_count = 0

    def run_epoch(loader, train_mode):
        model.train(train_mode)
        total_loss = total_correct = total = 0
        context = torch.enable_grad() if train_mode else torch.no_grad()
        with context:
            for spectra, targets, _ in tqdm(loader, desc="train" if train_mode else "val ",
                                            unit="batch", file=sys.stdout):
                spectra = spectra.to(device).unsqueeze(1)
                targets = targets.to(device)

                if train_mode:
                    optimizer.zero_grad()
                outputs = model(spectra)
                loss = criterion(outputs, targets)
                if train_mode:
                    loss.backward()
                    optimizer.step()

                total_loss += loss.item() * spectra.size(0)
                total_correct += (outputs.argmax(dim=1) == targets).sum().item()
                total += spectra.size(0)
        return total_loss / total, total_correct / total

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = run_epoch(train_loader, train_mode=True)
        val_loss, val_acc = run_epoch(val_loader, train_mode=False)

        history["loss_train"].append(train_loss)
        history["acc_train"].append(train_acc)
        history["loss_val"].append(val_loss)
        history["acc_val"].append(val_acc)

        message = (f"epoch {epoch}/{epochs}  train_loss {train_loss:.4f}  train_acc {train_acc:.4f}"
                   f"  val_loss {val_loss:.4f}  val_acc {val_acc:.4f}")

        if np.isnan(train_loss):
            nan_loss_count += 1
            if nan_loss_count >= 3:
                raise RuntimeError("Three consecutive NaN training losses — stopping.")

        if val_acc > best_acc:
            best_acc = val_acc
            state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            torch.save(copy.deepcopy(state), checkpoint_path)
            tqdm.write(message + "  -> saved")
        else:
            tqdm.write(message)

    pd.DataFrame({
        "epoch": range(1, epochs + 1),
        "loss in train": history["loss_train"],
        "acc in train": history["acc_train"],
        "loss in val": history["loss_val"],
        "acc in val": history["acc_val"],
    }).to_csv(log_path, index=False)

    return best_acc


@torch.no_grad()
def evaluate(model, loader, num_classes, device):
    """Return the confusion matrix and overall accuracy on ``loader``."""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    total_loss = total_correct = total = 0

    for spectra, targets, _ in tqdm(loader, desc="test", unit="batch", file=sys.stdout):
        spectra = spectra.to(device).unsqueeze(1)
        targets = targets.to(device)
        outputs = model(spectra)
        predictions = outputs.argmax(dim=1)

        total_loss += criterion(outputs, targets).item() * spectra.size(0)
        total_correct += (predictions == targets).sum().item()
        total += spectra.size(0)
        for true, pred in zip(targets.cpu().tolist(), predictions.cpu().tolist()):
            confusion[true, pred] += 1

    tqdm.write(f"test_loss {total_loss / total:.4f}  test_acc {total_correct / total:.4f}")
    return confusion, total_correct / total


def build_loaders(args, batch_size, num_workers):
    train_data = XRDDataset(args.train, cls=args.task)
    val_data = XRDDataset(args.val, cls=args.task)
    test_data = XRDDataset(args.test, cls=args.task)

    def loader(dataset, shuffle):
        return Data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                               num_workers=num_workers, drop_last=True, collate_fn=collate_fn)

    return loader(train_data, True), loader(val_data, True), loader(test_data, False)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", required=True, choices=["crystal_system", "lattice", "point_group"],
                        help="symmetry level to predict")
    parser.add_argument("--train", required=True, help="training .npy")
    parser.add_argument("--val", required=True, help="validation .npy")
    parser.add_argument("--test", required=True, help="test .npy")
    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.0,
                        help="focal-loss focusing parameter; 0 means plain cross-entropy. "
                             "The paper selected 0 on the validation set.")
    parser.add_argument("--n-block", type=int, default=16)
    parser.add_argument("--num-heads", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--name", default="rcnet", help="checkpoint / log file stem")
    parser.add_argument("--output-dir", default=".", help="where to write checkpoint and log")
    parser.add_argument("--no-cuda", action="store_true")
    args = parser.parse_args()

    device = torch.device("cpu" if args.no_cuda or not torch.cuda.is_available() else "cuda")
    num_classes = len(LABELS[args.task])

    model = RCNet(
        in_channels=1,
        base_filters=16,
        kernel_size=15,
        stride=2,
        groups=1,
        n_block=args.n_block,
        n_classes=num_classes,
        downsample_gap=2,
        increasefilter_gap=4,
        use_bn=True,
        use_do=True,
        num_heads=args.num_heads,
    )
    model = to_device(model, device)

    train_loader, val_loader, test_loader = build_loaders(args, args.batch_size, args.num_workers)

    os.makedirs(args.output_dir, exist_ok=True)
    checkpoint_path = os.path.join(args.output_dir, f"{args.name}-{args.task}.pth")
    log_path = os.path.join(args.output_dir, f"{args.name}-{args.task}.csv")

    best_acc = train(model, train_loader, val_loader, checkpoint_path, log_path,
                     args.epochs, args.lr, args.weight_decay, args.gamma, device)
    print(f"best validation accuracy: {best_acc:.4f}")

    model.load_state_dict(torch.load(checkpoint_path))
    confusion, test_acc = evaluate(model, test_loader, num_classes, device)
    np.save(os.path.join(args.output_dir, f"{args.name}-{args.task}-confusion.npy"), confusion)
    print(f"test accuracy: {test_acc:.4f}")


if __name__ == "__main__":
    main()
