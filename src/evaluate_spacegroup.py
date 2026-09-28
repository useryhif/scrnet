"""Run the seven trained experts over a test set and route them into one prediction.

Every expert is evaluated on the full test set. For each sample the routing step picks the
expert with the highest model-selection score and adopts that expert's space-group
prediction, so the final answer is always a space group proposed by a single expert rather
than a mixture across all 230 classes.

    python evaluate_spacegroup.py --test data/test.npy --weights runs
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as Data
from tqdm import tqdm

from dataset import SubXRDDatasetTest
from labels import SPACE_GROUPS, SUB_MODEL_RANGES
from scrnet import SCRNet


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


def load_expert(path, device, n_block):
    """Rebuild one expert and load its weights. ``n_block`` must match training."""
    model = SCRNet(
        in_channels=1, base_filters=16, kernel_size=15, stride=2, groups=1,
        n_block=n_block, n_classes=230, downsample_gap=2, increasefilter_gap=4,
        use_bn=True, use_do=True, num_heads=2,
    )
    model.load_state_dict(torch.load(path, map_location=device))
    model = to_device(model, device)
    model.eval()
    return model


@torch.no_grad()
def collect_outputs(model, loader, device):
    """Per-sample expert outputs over the whole test set, in loader order.

    Returns:
        (max_prob, argmax_class, confidence) — each a 1-D array of length ``len(dataset)``.
        ``max_prob`` is the largest softmax probability over the 230 classes,
        ``argmax_class`` is the corresponding class index, and ``confidence`` is the
        probability the expert assigns to "this input is mine".
    """
    max_probs, argmax_classes, confidences = [], [], []
    for spectra, _, _, cs, lattice, pg, _, _, _ in tqdm(loader, desc="evaluate", unit="batch", file=sys.stdout):
        spectra = spectra.to(device).unsqueeze(1)
        outputs, confidence, _ = model(spectra, cs.to(device), lattice.to(device), pg.to(device))
        probabilities = torch.softmax(outputs, dim=1)
        max_prob, argmax_class = probabilities.max(dim=1)
        max_probs.append(max_prob.cpu().numpy())
        argmax_classes.append(argmax_class.cpu().numpy())
        confidences.append(confidence[:, 1].cpu().numpy())
    return np.concatenate(max_probs), np.concatenate(argmax_classes), np.concatenate(confidences)


def route(expert_probs, expert_classes, expert_confidences, gamma=0.0):
    """Pick one expert per sample and return its predicted space group.

    The model-selection score interpolates between the expert's peak class probability and
    its confidence head: ``score = gamma * max_prob + (1 - gamma) * confidence``. The default
    ``gamma = 0`` routes purely on the confidence head.

    Args:
        expert_probs: ``[num_experts, num_samples]``
        expert_classes: ``[num_experts, num_samples]``
        expert_confidences: ``[num_experts, num_samples]``

    Returns:
        ``(predictions, chosen_expert)`` — both 1-D arrays of length ``num_samples``.
    """
    scores = expert_probs * gamma + expert_confidences * (1 - gamma)
    chosen = np.argmax(scores, axis=0)
    predictions = expert_classes[chosen, np.arange(expert_classes.shape[1])]
    return predictions, chosen


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--test", required=True, help="test .npy")
    parser.add_argument("--weights", default="runs", help="directory holding expert{0..6}.pth")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--n-block", type=int, default=28,
                        help="must match the value used in train_spacegroup.py")
    parser.add_argument("--gamma", type=float, default=0.0,
                        help="weight on peak class probability when scoring experts; "
                             "0 routes purely on the confidence head")
    parser.add_argument("--output-dir", default=".", help="where to write the confusion matrix")
    parser.add_argument("--no-cuda", action="store_true")
    args = parser.parse_args()

    device = torch.device("cpu" if args.no_cuda or not torch.cuda.is_available() else "cuda")

    # start=0, length=230 keeps every sample, so each expert is scored on the full test set.
    dataset = SubXRDDatasetTest(args.test, cls="space_group", start=0, length=230)
    loader = Data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, drop_last=False, collate_fn=collate_fn)

    probs, classes, confidences = [], [], []
    for i in range(len(SUB_MODEL_RANGES)):
        path = os.path.join(args.weights, f"expert{i}.pth")
        print(f"=== expert {i}  ({path}) ===")
        model = load_expert(path, device, args.n_block)
        max_prob, argmax_class, confidence = collect_outputs(model, loader, device)
        probs.append(max_prob)
        classes.append(argmax_class)
        confidences.append(confidence)

    predictions, chosen = route(np.array(probs), np.array(classes), np.array(confidences), args.gamma)

    targets = np.array([dataset.label_map[row[dataset.label_column]] for row in dataset.data])
    accuracy = float((predictions == targets).mean())

    confusion = np.zeros((len(SPACE_GROUPS), len(SPACE_GROUPS)), dtype=np.int64)
    for true, pred in zip(targets, predictions):
        confusion[true, pred] += 1

    print(f"space-group accuracy: {accuracy:.4f}")
    print("expert usage:", np.bincount(chosen, minlength=len(SUB_MODEL_RANGES)).tolist())

    os.makedirs(args.output_dir, exist_ok=True)
    np.save(os.path.join(args.output_dir, "confusion_space_group.npy"), confusion)


if __name__ == "__main__":
    main()
