"""Train the seven SCRNet experts for space-group classification.

Space groups are partitioned by crystal system (see ``labels.SUB_MODEL_RANGES``) and one
expert is trained per partition. Each expert sees the XRD spectrum plus the predicted
crystal-system, Bravais-lattice and point-group embeddings, and predicts three things at
once: space-group logits over all 230 classes, a binary confidence score saying whether the
input belongs to this expert at all, and the unit-cell parameters (a, b, c, alpha, beta,
gamma) as an auxiliary regression task.

    python train_spacegroup.py --train data/train.npy --val data/val.npy \
        --output-dir runs --epochs 100
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

from dataset import SubXRDDataset
from labels import SUB_MODEL_RANGES
from scrnet import SCRNet
from selective_loss import SelectiveLoss

ABC_SCALE = 1000.0


def to_device(model, device):
    """Move to ``device``, wrapping in DataParallel only when there is a GPU to parallelise."""
    if device.type == "cuda":
        return nn.DataParallel(model).to(device)
    return model.to(device)


def collate_fn(batch):
    spectra, targets, conf, cs, lattice, pg, rows, abc, angles = zip(*batch)
    return (torch.stack(spectra), torch.stack(targets), torch.stack(conf),
            torch.stack(cs), torch.stack(lattice), torch.stack(pg), rows,
            torch.stack(abc), torch.stack(angles))


def batch_loss(model, criterion, spectra, targets, conf_target, cs, lattice, pg, abc, angles):
    """Combined expert loss: selective space-group + confidence + unit-cell regression."""
    outputs, confidence, abc_angles = model(spectra, cs, lattice, pg)
    loss = criterion(outputs, targets)
    loss = loss + nn.functional.cross_entropy(confidence, conf_target)
    loss = loss + nn.functional.mse_loss(abc_angles, torch.cat([abc, angles], dim=1)) / ABC_SCALE
    return loss, outputs, confidence


def train_expert(model, train_loader, val_loader, start, length, args, device, checkpoint_path, log_path):
    """Train one expert, keeping the checkpoint with the best mean of the two accuracies."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = SelectiveLoss(start, length)

    history = {"loss_train": [], "acc_train": [], "conf_train": [], "loss_val": [], "acc_val": [], "conf_val": []}
    best_score = -1.0
    epochs_without_improvement = 0

    def run_epoch(loader, train_mode):
        model.train(train_mode)
        totals = {"loss": 0.0, "correct": 0, "conf_correct": 0, "n": 0}
        context = torch.enable_grad() if train_mode else torch.no_grad()
        with context:
            for spectra, targets, conf_target, cs, lattice, pg, _, abc, angles in tqdm(
                    loader, desc="train" if train_mode else "val  ", unit="batch", file=sys.stdout):
                spectra = spectra.to(device).unsqueeze(1)
                targets = targets.to(device)
                conf_target = conf_target.to(device)
                cs, lattice, pg = cs.to(device), lattice.to(device), pg.to(device)
                abc, angles = abc.to(device), angles.to(device)

                if train_mode:
                    optimizer.zero_grad()
                loss, outputs, confidence = batch_loss(
                    model, criterion, spectra, targets, conf_target, cs, lattice, pg, abc, angles)
                if train_mode:
                    loss.backward()
                    optimizer.step()

                # Accuracy and confidence are only meaningful for this expert's own classes.
                valid = (targets >= start) & (targets < start + length)
                totals["loss"] += loss.item() * spectra.size(0)
                totals["correct"] += (outputs.argmax(dim=1)[valid] == targets[valid]).sum().item()
                totals["conf_correct"] += (confidence.argmax(dim=1) == conf_target).sum().item()
                totals["n"] += spectra.size(0)
        return (totals["loss"] / totals["n"],
                totals["correct"] / totals["n"],
                totals["conf_correct"] / totals["n"])

    def save(model):
        state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
        torch.save(copy.deepcopy(state), checkpoint_path)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc, train_conf = run_epoch(train_loader, train_mode=True)
        val_loss, val_acc, val_conf = run_epoch(val_loader, train_mode=False)

        for key, value in zip(("loss_train", "acc_train", "conf_train"), (train_loss, train_acc, train_conf)):
            history[key].append(value)
        for key, value in zip(("loss_val", "acc_val", "conf_val"), (val_loss, val_acc, val_conf)):
            history[key].append(value)

        score = (val_acc + val_conf) / 2
        message = (f"epoch {epoch}/{args.epochs}  train_loss {train_loss:.4f}  train_acc {train_acc:.4f}"
                   f"  val_loss {val_loss:.4f}  val_acc {val_acc:.4f}  val_conf {val_conf:.4f}")

        if score > best_score:
            best_score = score
            epochs_without_improvement = 0
            save(model)
            tqdm.write(message + "  -> saved")
        else:
            epochs_without_improvement += 1
            tqdm.write(message)

        if epochs_without_improvement >= args.patience:
            print(f"Early stop after {epoch} epochs.")
            break

    pd.DataFrame(history).to_csv(log_path, index=False)
    return best_score


def build_expert(model_index, args, device):
    """Instantiate one expert, resuming from its checkpoint if one exists."""
    model = SCRNet(
        in_channels=1,
        base_filters=16,
        kernel_size=15,
        stride=2,
        groups=1,
        n_block=args.n_block,
        n_classes=230,
        downsample_gap=2,
        increasefilter_gap=4,
        use_bn=True,
        use_do=True,
        num_heads=2,
    )
    checkpoint_path = os.path.join(args.output_dir, f"expert{model_index}.pth")
    if os.path.exists(checkpoint_path) and not args.scratch:
        model.load_state_dict(torch.load(checkpoint_path), strict=False)
        tqdm.write(f"expert {model_index}: resuming from {checkpoint_path}")
    return model, checkpoint_path


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train", required=True, help="training .npy")
    parser.add_argument("--val", required=True, help="validation .npy")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=30, help="early-stopping patience in epochs")
    parser.add_argument("--n-block", type=int, default=28)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--model-index", type=int, default=None,
                        help="train only this expert (0-6); default trains all seven")
    parser.add_argument("--output-dir", default="runs", help="where to write checkpoints and logs")
    parser.add_argument("--scratch", action="store_true", help="ignore existing checkpoints")
    parser.add_argument("--no-cuda", action="store_true")
    args = parser.parse_args()

    device = torch.device("cpu" if args.no_cuda or not torch.cuda.is_available() else "cuda")
    os.makedirs(args.output_dir, exist_ok=True)

    indices = range(len(SUB_MODEL_RANGES)) if args.model_index is None else [args.model_index]

    for i in indices:
        start, length = SUB_MODEL_RANGES[i]
        print(f"=== expert {i}: space groups [{start}, {start + length}) ===")

        train_loader = Data.DataLoader(
            SubXRDDataset(args.train, cls="space_group", start=start, length=length),
            batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
            drop_last=True, collate_fn=collate_fn)
        val_loader = Data.DataLoader(
            SubXRDDataset(args.val, cls="space_group", start=start, length=length),
            batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
            drop_last=True, collate_fn=collate_fn)

        model, checkpoint_path = build_expert(i, args, device)
        model = to_device(model, device)

        best = train_expert(model, train_loader, val_loader, start, length, args, device,
                            checkpoint_path, os.path.join(args.output_dir, f"expert{i}.csv"))
        print(f"expert {i}: best validation score {best:.4f}")


if __name__ == "__main__":
    main()
