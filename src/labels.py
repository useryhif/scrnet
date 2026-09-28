"""Crystallographic label hierarchies and the sub-model partition.

The four classification levels follow the symmetry hierarchy, from coarsest to finest.
``CLS_COLUMN`` maps a task name to the column index holding that label in a data row;
``LABELS`` maps a task name to its ordered list of class names, so the class index is the
position in that list.
"""

CRYSTAL_SYSTEMS = [
    "Triclinic", "Monoclinic", "Orthorhombic",
    "Tetragonal", "Trigonal", "Hexagonal", "Cubic",
]

LATTICE_TYPES = [
    "triclinic_P", "monoclinic_P", "monoclinic_C",
    "orthorhombic_P", "orthorhombic_C", "orthorhombic_F", "orthorhombic_I",
    "tetragonal_P", "tetragonal_I",
    "trigonal_R", "hexagonal_P",
    "cubic_P", "cubic_F", "cubic_I",
]

POINT_GROUPS = [
    "1", "-1", "2", "m", "2/m", "222", "mm2", "mmm",
    "4", "-4", "4/m", "422", "4mm", "-42m", "4/mmm",
    "3", "-3", "32", "3m", "-3m",
    "6", "-6", "6/m", "622", "6mm", "-6m2", "6/mmm",
    "23", "m-3", "432", "-43m", "m-3m",
]

SPACE_GROUPS = [f"spacegroup{i + 1}" for i in range(230)]

LABELS = {
    "crystal_system": CRYSTAL_SYSTEMS,
    "lattice": LATTICE_TYPES,
    "point_group": POINT_GROUPS,
    "space_group": SPACE_GROUPS,
}

# Column index of each task's label in a data row.
CLS_COLUMN = {
    "crystal_system": 2,
    "space_group": 3,
    "lattice": 4,
    "point_group": 5,
}

# Space groups split by crystal system: one expert per entry, as (start, length).
SUB_MODEL_RANGES = [
    (0, 2),      # Triclinic      space groups 1-2
    (2, 13),     # Monoclinic     3-15
    (15, 59),    # Orthorhombic   16-74
    (74, 68),    # Tetragonal     75-142
    (142, 25),   # Trigonal       143-167
    (167, 27),   # Hexagonal      168-194
    (194, 36),   # Cubic          195-230
]

# Column indices of the auxiliary features appended to the end of each row.
COL_XRD = 6
COL_CS_EMBED = -5
COL_LATTICE_EMBED = -4
COL_POINT_GROUP_EMBED = -3
COL_ABC = -2
COL_ANGLES = -1
