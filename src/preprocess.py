"""Build the labelled ``.npy`` datasets that the training scripts read.

Takes the per-space-group pickles written by ``fetch_xrd.py`` and produces the 12-column
layout documented in the README: identity, chemical elements, the four symmetry labels, the
spectrum, the three upstream probability vectors, and the unit-cell parameters.

Labels are derived from the space group number with pymatgen rather than read from the source
metadata, so every sample is labelled consistently:

* crystal system and point group come straight from ``SpaceGroup``,
* the Bravais lattice is the crystal system plus the centring letter of the space group
  symbol, with two normalisations to the standard 14-lattice notation —
  ``orthorhombic_A`` → ``orthorhombic_C`` (A-centring is equivalent to C-centring) and
  ``trigonal_P`` → ``hexagonal_P`` (the trigonal primitive lattice is hexagonal).

    python preprocess.py --input data/raw --output-dir data

Note on the probability columns:
    Columns −5, −4 and −3 hold the crystal-system, Bravais-lattice and point-group probability
    vectors produced by the upstream classifiers. They are *model outputs*, not properties of
    the raw data, so they cannot be computed here. Pass ``--embeddings`` with a file produced
    by upstream inference to fill them in; without it the columns are written as zeros and a
    warning is printed, because a model trained on constant inputs is meaningless.
"""

import argparse
import glob
import os
import pickle
import sys

import numpy as np
from tqdm import tqdm

from labels import CRYSTAL_SYSTEMS, LATTICE_TYPES, POINT_GROUPS

TWO_THETA_MIN = 0.0
TWO_THETA_MAX = 90.0
N_POINTS = 1500     # grid the training data uses
SIGMA = 0.05        # peak broadening, in degrees 2-theta

# Standard 14-Bravais-lattice normalisations (see module docstring).
_LATTICE_FIX = {
    "orthorhombic_A": "orthorhombic_C",
    "trigonal_P": "hexagonal_P",
}

N_BASE_COLUMNS = 12


def gaussian(x, mu, sigma):
    """Normalised Gaussian, used as the peak profile."""
    return (1.0 / (np.sqrt(2.0 * np.pi) * sigma)) * np.exp(-(((x - mu) / sigma) ** 2) / 2.0)


def broaden(peaks_x, peaks_y, n_points=N_POINTS, sigma=SIGMA):
    """Sum Gaussian-broadened reflections onto a uniform 2-theta grid."""
    grid = np.linspace(TWO_THETA_MIN, TWO_THETA_MAX, n_points)
    spectrum = np.zeros(n_points)
    for position, intensity in zip(peaks_x, peaks_y):
        spectrum += intensity * gaussian(grid, position, sigma)
    return spectrum


def labels_for_space_group(number):
    """Return ``(crystal_system, space_group, lattice, point_group)`` for a space group number."""
    from pymatgen.symmetry.groups import SpaceGroup

    group = SpaceGroup.from_int_number(number)
    crystal_system = group.crystal_system.capitalize()
    lattice = f"{group.crystal_system}_{group.symbol[0]}"
    lattice = _LATTICE_FIX.get(lattice, lattice)
    return crystal_system, f"spacegroup{number}", lattice, group.point_group


def load_raw(input_dir):
    """Load every ``spacegroup*.pkl`` in ``input_dir`` and attach its labels."""
    records = []
    paths = sorted(glob.glob(os.path.join(input_dir, "spacegroup*.pkl")),
                   key=lambda p: int("".join(c for c in os.path.basename(p) if c.isdigit())))
    if not paths:
        raise SystemExit(f"no spacegroup*.pkl found in {input_dir}")

    for path in paths:
        number = int("".join(c for c in os.path.basename(path) if c.isdigit()))
        crystal_system, space_group, lattice, point_group = labels_for_space_group(number)
        with open(path, "rb") as handle:
            samples = pickle.load(handle)
        for sample in samples:
            records.append({
                "material_id": sample["material_id"],
                "elements": sample.get("elements", []),
                "crystal_system": crystal_system,
                "space_group": space_group,
                "lattice": lattice,
                "point_group": point_group,
                "xrd": np.asarray(sample["xrd"], dtype=np.float32),
            })
        tqdm.write(f"  {os.path.basename(path):22s} {len(samples):6d} samples")
    return records


def attach_lattice_parameters(records, api_key, batch_size=200):
    """Fill in ``abc`` and ``angles`` by looking each material up on the Materials Project.

    The lookup is skipped for any record that already carries these fields, so this can be
    run on a partially populated dataset.
    """
    import requests

    pending = {r["material_id"] for r in records if "abc" not in r}
    if not pending:
        return
    ids = sorted(pending)
    session = requests.Session()
    session.headers.update({"X-API-KEY": api_key, "accept": "application/json"})
    url = "https://api.materialsproject.org/materials/summary/"

    lookup = {}
    batches = [ids[i:i + batch_size] for i in range(0, len(ids), batch_size)]
    for batch in tqdm(batches, desc="lattice parameters", file=sys.stdout):
        params = {"material_ids": ",".join(batch),
                  "_fields": "material_id,structure",
                  "_limit": batch_size + 10}
        response = session.get(url, params=params, timeout=60)
        response.raise_for_status()
        for doc in response.json().get("data", []):
            lattice = (doc.get("structure") or {}).get("lattice")
            if lattice:
                lookup[doc["material_id"]] = (lattice["abc"], lattice["angles"])

    missing = 0
    for record in records:
        found = lookup.get(record["material_id"])
        if found:
            record["abc"] = np.asarray(found[0], dtype=np.float32)
            record["angles"] = np.asarray(found[1], dtype=np.float32)
        else:
            missing += 1
            record["abc"] = np.zeros(3, dtype=np.float32)
            record["angles"] = np.zeros(3, dtype=np.float32)
    if missing:
        print(f"  warning: {missing} materials had no lattice parameters; filled with zeros")


def to_rows(records, embeddings=None):
    """Convert records into the positional 12-column layout."""
    label_map_cs = {name: i for i, name in enumerate(CRYSTAL_SYSTEMS)}
    label_map_sg = {f"spacegroup{i + 1}": i for i in range(230)}
    label_map_lat = {name: i for i, name in enumerate(LATTICE_TYPES)}
    label_map_pg = {name: i for i, name in enumerate(POINT_GROUPS)}

    rows = []
    for record in records:
        spectrum = broaden_for_training(record["xrd"])
        row = np.empty(N_BASE_COLUMNS, dtype=object)
        row[0] = record["material_id"]
        row[1] = list(record["elements"])
        row[2] = label_map_cs[record["crystal_system"]]
        row[3] = label_map_sg[record["space_group"]]
        row[4] = label_map_lat[record["lattice"]]
        row[5] = label_map_pg[record["point_group"]]
        row[6] = spectrum
        row[7] = np.zeros(len(CRYSTAL_SYSTEMS), dtype=np.float32) if embeddings is None else embeddings[record["material_id"]]["crystal_system"]
        row[8] = np.zeros(len(LATTICE_TYPES), dtype=np.float32) if embeddings is None else embeddings[record["material_id"]]["lattice"]
        row[9] = np.zeros(len(POINT_GROUPS), dtype=np.float32) if embeddings is None else embeddings[record["material_id"]]["point_group"]
        row[10] = record["abc"]
        row[11] = record["angles"]
        rows.append(row)
    return np.array(rows, dtype=object)


def broaden_for_training(spectrum, n_points=N_POINTS):
    """Re-broaden an already-broadened spectrum onto the training grid.

    ``fetch_xrd.py`` writes patterns on a 901-point grid; the training data uses 1500 points.
    The peak positions are recovered by simply resampling, since broadening is linear in the
    grid: a denser grid of the same Gaussians is obtained by summing over the original grid.
    """
    if len(spectrum) == n_points:
        return spectrum.astype(np.float32)
    source = np.linspace(TWO_THETA_MIN, TWO_THETA_MAX, len(spectrum))
    target = np.linspace(TWO_THETA_MIN, TWO_THETA_MAX, n_points)
    return np.interp(target, source, spectrum).astype(np.float32)


def split(records, val_fraction, test_fraction, seed):
    """Shuffle and split into ``(train, val, test)``."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(records))
    n_test = int(round(len(records) * test_fraction))
    n_val = int(round(len(records) * val_fraction))
    test = [records[i] for i in order[:n_test]]
    val = [records[i] for i in order[n_test:n_test + n_val]]
    train = [records[i] for i in order[n_test + n_val:]]
    return train, val, test


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", default="data/raw", help="directory of fetch_xrd.py pickles")
    parser.add_argument("--output-dir", default="data", help="where to write the .npy files")
    parser.add_argument("--val-fraction", type=float, default=0.175)
    parser.add_argument("--test-fraction", type=float, default=0.125)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lattice-params", action="store_true",
                        help="look up a, b, c and the angles on the Materials Project")
    parser.add_argument("--embeddings", default=None,
                        help="optional .pkl of upstream probability vectors, keyed by material_id")
    parser.add_argument("--api-key", default=os.environ.get("MP_API_KEY", ""))
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("[1/4] loading raw patterns")
    records = load_raw(args.input)
    print(f"  {len(records)} samples")

    if args.lattice_params:
        print("[2/4] fetching unit-cell parameters")
        if not args.api_key:
            parser.error("--lattice-params needs MP_API_KEY or --api-key")
        attach_lattice_parameters(records, args.api_key)
    else:
        print("[2/4] skipping unit-cell parameters (--lattice-params not given)")
        for record in records:
            record.setdefault("abc", np.zeros(3, dtype=np.float32))
            record.setdefault("angles", np.zeros(3, dtype=np.float32))

    embeddings = None
    if args.embeddings:
        with open(args.embeddings, "rb") as handle:
            embeddings = pickle.load(handle)
    else:
        print("  WARNING: no --embeddings given — the three probability columns are written")
        print("           as zeros. Training on them is meaningless; run the upstream")
        print("           classifiers first and re-run with --embeddings.")

    print("[3/4] splitting")
    train, val, test = split(records, args.val_fraction, args.test_fraction, args.seed)

    print("[4/4] writing")
    for name, part in (("train", train), ("val", val), ("test", test)):
        path = os.path.join(args.output_dir, f"{name}.npy")
        np.save(path, to_rows(part, embeddings), allow_pickle=True)
        print(f"  {path}  ({len(part)} samples)")


if __name__ == "__main__":
    main()
