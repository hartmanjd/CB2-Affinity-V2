"""Turn structures into numbers a model can learn from.

Structures are standardized the same way every time; how a structure becomes numbers is a short
menu the team chooses from."""

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from rdkit.Chem.MolStandardize import rdMolStandardize

RDLogger.DisableLog("rdApp.*")

BITS = 2048
# Ipc overflows for larger molecules, so it's left out
DESCRIPTORS = [name for name, _ in Descriptors._descList if name != "Ipc"]

_largest = rdMolStandardize.LargestFragmentChooser()
_uncharger = rdMolStandardize.Uncharger()


def standardize(smiles) -> str | None:
    """Keep the largest piece (drops salts and solvents), neutralize charges and write canonical
    SMILES, or None when the structure can't be read."""
    mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
    if mol is None:
        return None
    return Chem.MolToSmiles(_uncharger.uncharge(_largest.choose(mol)))


def morgan(radius: int):
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=BITS)
    return lambda mol: generator.GetCountFingerprintAsNumPy(mol)


def descriptors(mol) -> np.ndarray:
    values = Descriptors.CalcMolDescriptors(mol)
    return np.array([values[name] for name in DESCRIPTORS], dtype=float)


FEATURES = {
    "morgan2": ("Morgan fingerprint counts, radius 2, 2048 bits: which small chemical neighbourhoods a "
                "molecule has", [morgan(2)]),
    "morgan3": ("Morgan fingerprint counts, radius 3, 2048 bits: larger neighbourhoods, more detail",
                [morgan(3)]),
    "descriptors": (f"{len(DESCRIPTORS)} RDKit descriptors: whole-molecule properties such as weight, polarity "
                    "and ring counts", [descriptors]),
    "morgan2+descriptors": ("the radius 2 fingerprint and the descriptors side by side", [morgan(2), descriptors]),
}


def featurize(smiles: list[str], name: str) -> np.ndarray:
    """One row of numbers per structure. A value that isn't a finite number becomes NaN, so
    training decides how to fill it rather than a huge number slipping through."""
    if name not in FEATURES:
        raise ValueError(f"features must be one of {list(FEATURES)}")
    rows = []
    for structure in smiles:
        if (mol := Chem.MolFromSmiles(structure)) is None:
            raise ValueError(f"can't read structure {structure!r}")
        rows.append(np.concatenate([part(mol) for part in FEATURES[name][1]]).astype(float))
    matrix = np.vstack(rows) if rows else np.empty((0, 0))
    matrix[~np.isfinite(matrix)] = np.nan
    return matrix
