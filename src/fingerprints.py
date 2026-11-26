import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.MolStandardize import rdMolStandardize
from src.config import RADIUS, NBITS

RDLogger.DisableLog("rdApp.*")
_gen = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=NBITS)
_lfc = rdMolStandardize.LargestFragmentChooser()


def parse_smiles(smi):
    """Return an RDKit Mol or None if the input is not a valid structure."""
    if not isinstance(smi, str) or not smi.strip():
        return None
    try:
        m = Chem.MolFromSmiles(smi.strip())
    except Exception:
        return None
    return m if m is not None and m.GetNumAtoms() > 0 else None


def standardize(mol):
    """Keep the largest fragment (strips salts / counter-ions)."""
    return _lfc.choose(mol)


def fingerprint(mol) -> np.ndarray:
    return _gen.GetFingerprintAsNumPy(mol).astype(np.uint8)


def bit_environments(mol):
    """{bit: ((atom_idx, radius), ...)} for every set Morgan bit."""
    ao = rdFingerprintGenerator.AdditionalOutput()
    ao.AllocateBitInfoMap()
    _gen.GetFingerprint(mol, additionalOutput=ao)
    return dict(ao.GetBitInfoMap())


def environment_atoms_bonds(mol, atom_idx, radius):
    if radius == 0:
        return [atom_idx], []
    bonds = list(Chem.FindAtomEnvironmentOfRadiusN(mol, radius, atom_idx))
    atoms = {atom_idx}
    for b in bonds:
        bd = mol.GetBondWithIdx(b)
        atoms.update([bd.GetBeginAtomIdx(), bd.GetEndAtomIdx()])
    return sorted(atoms), bonds
