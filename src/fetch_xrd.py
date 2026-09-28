"""Download structures from the Materials Project and compute their powder XRD patterns.

Structures are fetched either by space-group number (for space-group training data) or by
crystal system. Each structure's diffraction pattern is computed with pymatgen and broadened
into a fixed-length spectrum so that every sample is the same shape.

Requires an MP API key in the ``MP_API_KEY`` environment variable. Get one at
https://materialsproject.org/api.

    python fetch_xrd.py --by space_group --output-dir data/raw
    python fetch_xrd.py --by crystal_system --output-dir data/raw
"""

import argparse
import os
import pickle
import sys

import numpy as np
from tqdm import tqdm

TWO_THETA_MIN = 0.0
TWO_THETA_MAX = 90.0
N_POINTS = 901          # grid the raw patterns are broadened onto
SIGMA = 0.05            # peak broadening, in degrees 2-theta

CRYSTAL_SYSTEMS = [
    "Triclinic", "Monoclinic", "Orthorhombic",
    "Tetragonal", "Trigonal", "Hexagonal", "Cubic",
]


def gaussian(x, mu, sigma):
    """Normalised Gaussian, used as the peak profile."""
    return (1.0 / (np.sqrt(2.0 * np.pi) * sigma)) * np.exp(-(((x - mu) / sigma) ** 2) / 2.0)


def broaden(pattern, n_points=N_POINTS, sigma=SIGMA):
    """Turn a pymatgen diffraction pattern into a fixed-length broadened spectrum.

    Each reflection at 2-theta position ``pattern.x[i]`` with intensity ``pattern.y[i]`` is
    replaced by a Gaussian of width ``sigma``, and the contributions are summed on a uniform
    grid from 0 to 90 degrees.

    Args:
        pattern: a ``pymatgen`` ``XRDPattern``.
        n_points: number of grid points in the returned spectrum.
        sigma: Gaussian width in degrees 2-theta.

    Returns:
        1-D ``float`` array of length ``n_points``.
    """
    grid = np.linspace(TWO_THETA_MIN, TWO_THETA_MAX, n_points)
    spectrum = np.zeros(n_points)
    for position, intensity in zip(pattern.x, pattern.y):
        spectrum += intensity * gaussian(grid, position, sigma)
    return spectrum


def fetch(api_key, by, output_dir):
    """Fetch structures and write one pickle per space group (or per crystal system).

    Args:
        api_key: Materials Project API key.
        by: ``"space_group"`` to iterate over the 230 space groups, or ``"crystal_system"``
            to iterate over the 7 crystal systems.
        output_dir: directory for the pickles.
    """
    # Imported here so that the rest of the module stays importable without mp-api installed.
    from mp_api.client import MPRester
    from pymatgen.analysis.diffraction.xrd import XRDCalculator

    calculator = XRDCalculator()
    os.makedirs(output_dir, exist_ok=True)

    with MPRester(api_key) as mpr:
        if by == "space_group":
            for number in range(1, 231):
                records = mpr.materials.summary.search(
                    spacegroup_number=number, fields=["material_id", "structure", "elements"])
                samples = []
                for record in tqdm(records, desc=f"spacegroup{number}", file=sys.stdout):
                    spectrum = broaden(calculator.get_pattern(record.structure))
                    samples.append({
                        "material_id": str(record.material_id),
                        "xrd": spectrum,
                        "elements": [str(e) for e in record.elements],
                    })
                path = os.path.join(output_dir, f"spacegroup{number}.pkl")
                with open(path, "wb") as handle:
                    pickle.dump(samples, handle)
        else:
            for system in CRYSTAL_SYSTEMS:
                records = mpr.materials.summary.search(
                    crystal_system=system, fields=["material_id", "structure", "elements"])
                samples = []
                for record in tqdm(records, desc=system, file=sys.stdout):
                    spectrum = broaden(calculator.get_pattern(record.structure))
                    samples.append({
                        "material_id": str(record.material_id),
                        "xrd": spectrum,
                        "elements": [str(e) for e in record.elements],
                    })
                path = os.path.join(output_dir, f"{system}.pkl")
                with open(path, "wb") as handle:
                    pickle.dump(samples, handle)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--by", choices=["space_group", "crystal_system"], required=True,
                        help="fetch per space group, or per crystal system")
    parser.add_argument("--output-dir", default="data/raw", help="where to write the pickles")
    parser.add_argument("--api-key", default=os.environ.get("MP_API_KEY", ""),
                        help="defaults to the MP_API_KEY environment variable")
    args = parser.parse_args()

    if not args.api_key:
        parser.error("no API key: set MP_API_KEY or pass --api-key")

    fetch(args.api_key, args.by, args.output_dir)
    print(f"done -> {args.output_dir}")


if __name__ == "__main__":
    main()
