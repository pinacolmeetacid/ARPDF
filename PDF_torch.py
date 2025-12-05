import numpy as np
from typing import List, Tuple, Optional, Dict
import MDAnalysis as mda
from utils import box_shift
from utils.core_functions import ArrayType
from utils.AFF_map import get_crossection

def get_atoms_pairs(
        universe: mda.Universe, 
        modified_atoms: List[int], 
        cutoff: float = 10.0, 
        periodic: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, Dict[Tuple[str, str], np.ndarray]]:
    """
    Extract positions of modified atoms and their neighbors within cutoff.
    Returns:
        center_pos : positions of modified atoms
        around_pos : positions of neighbors (including modified)
        atom_pairs : dict of atom type pairs -> index pairs
    """
    center_group = universe.atoms[modified_atoms]
    around_group = universe.select_atoms(f"around {cutoff} group center", center=center_group, periodic=periodic)
    center_pos = center_group.positions
    around_pos = around_group.positions

    if periodic:
        _center = center_pos[[0]]
        center_pos = _center + box_shift(center_pos - _center, box=universe.dimensions)
        around_pos = _center + box_shift(around_pos - _center, box=universe.dimensions)

    concat_group = center_group + around_group
    mask = np.triu(np.ones((len(center_group), len(concat_group)), dtype=bool), k=1)
    i_idx, j_idx = np.nonzero(mask)

    all_atom_types = sorted(set(universe.atoms.types))
    atom_types_center = np.array(center_group.types, dtype="<U4")
    atom_types_concat = np.array(concat_group.types, dtype="<U4")
    atom_pair_types = np.stack([atom_types_center[i_idx], atom_types_concat[j_idx]], axis=1)
    atom_pair_types.sort(axis=1)
    ij_idx = np.stack([i_idx, j_idx], axis=1)

    atom_pairs = {}
    for i, type1 in enumerate(all_atom_types):
        for type2 in all_atom_types[i:]:
            pair_mask = np.all(atom_pair_types == [type1, type2], axis=1)
            atom_pairs[(type1, type2)] = ij_idx[pair_mask]

    return center_pos, around_pos, atom_pairs
'''
def compute_deltaPDF_numpy(
        coords: np.ndarray,
        atom_types: List[str],
        modified_atoms: List[int],
        r_range: Tuple[float,float]=(0.0, 10.0),
        nbins: int = 200,
        sigma: float = 0.2,
        box: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute deltaPDF using NumPy for only modified atoms.
    """
    coords = np.asarray(coords, dtype=float)
    N = coords.shape[0]
    r_edges = np.linspace(r_range[0], r_range[1], nbins+1)
    r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])
    dr = r_edges[1] - r_edges[0]
    deltaPDF = np.zeros(nbins, dtype=float)

    unique_types = sorted(set(atom_types))
    crossection = {t: float(get_crossection(t)) for t in unique_types}

    for i_center in modified_atoms:
        t1 = atom_types[i_center]
        pos_i = coords[i_center]

        for j_other in range(N):
            if i_center == j_other:
                continue
            t2 = atom_types[j_other]
            pos_j = coords[j_other]

            # distance with periodic correction
            r_vec = pos_j - pos_i
            if box is not None:
                r_vec -= box * np.round(r_vec / box)
            r_val = np.linalg.norm(r_vec)

            # Gaussian smearing
            gauss = np.exp(-0.5*((r_centers - r_val)/sigma)**2) / (sigma*np.sqrt(2*np.pi))
            deltaPDF += gauss * crossection[t1] * crossection[t2]

    return r_centers, deltaPDF

def compute_deltaPDF_numpy_sigmaMap(
        coords: np.ndarray,
        atom_types: List[str],
        modified_atoms: List[int],
        r_range: Tuple[float,float]=(0.0, 10.0),
        nbins: int = 200,
        sigma: float = 0.2,
        sigma_map: Optional[Dict[Tuple[str,str], float]] = None,
        box: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute deltaPDF using NumPy for only modified atoms,
    allowing different sigma for different atom pairs.

    Parameters
    ----------
    coords : (N,3) array
        Cartesian coordinates
    atom_types : list of str
        atom types for each atom
    modified_atoms : list of int
        indices of atoms considered as centers
    r_range : tuple
        (rmin, rmax) in angstrom
    nbins : int
        number of bins
    sigma : float
        default Gaussian width
    sigma_map : dict, optional
        mapping {(type1, type2): sigma_val}, order of types ignored
    box : array_like, optional
        periodic box lengths (Lx,Ly,Lz)

    Returns
    -------
    r_centers : (nbins,) array
    deltaPDF : (nbins,) array
    """
    coords = np.asarray(coords, dtype=float)
    N = coords.shape[0]
    r_edges = np.linspace(r_range[0], r_range[1], nbins+1)
    r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])
    deltaPDF = np.zeros(nbins, dtype=float)

    unique_types = sorted(set(atom_types))
    crossection = {t: float(get_crossection(t)) for t in unique_types}

    for i_center in modified_atoms:
        t1 = atom_types[i_center]
        pos_i = coords[i_center]

        for j_other in range(N):
            if i_center == j_other:
                continue
            t2 = atom_types[j_other]
            pos_j = coords[j_other]

            # distance with periodic correction
            r_vec = pos_j - pos_i
            if box is not None:
                r_vec -= box * np.round(r_vec / box)
            r_val = np.linalg.norm(r_vec)

            # determine sigma for this pair
            pair = tuple(sorted((t1, t2)))
            sigma_ij = sigma_map.get(pair, sigma) if sigma_map is not None else sigma

            # Gaussian smearing with pair-specific sigma
            gauss = np.exp(-0.5*((r_centers - r_val)/sigma_ij)**2) / (sigma_ij*np.sqrt(2*np.pi))
            deltaPDF += gauss * crossection[t1] * crossection[t2]

    return r_centers, deltaPDF


'''
def compute_deltaPDF(
        u1: mda.Universe,
        u2: mda.Universe,
        modified_atoms: List[int],
        r_range: Tuple[float, float] = (0.0, 10.0),
        nbins: int = 200,
        sigma: float = 0.2,
        box: Optional[np.ndarray] = None,
        verbose: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute deltaPDF = PDF(u2) - PDF(u1) for only modified atoms.

    Parameters
    ----------
    u1, u2 : MDAnalysis Universe
        Before/after structures
    modified_atoms : list of int
        indices of atoms that changed
    r_range : tuple
        (rmin, rmax) in angstrom
    nbins : int
        number of r bins
    sigma : float
        Gaussian smearing width
    box : array_like, optional
        Periodic box (Lx,Ly,Lz)
    verbose : bool
        print progress

    Returns
    -------
    r_centers : (nbins,) array
    deltaPDF : (nbins,) array
    """
    _print = print if verbose else (lambda *a, **k: None)

    coords1 = u1.atoms.positions
    coords2 = u2.atoms.positions
    types1 = u1.atoms.types
    types2 = u2.atoms.types

    assert len(coords1) == len(coords2), "u1 and u2 must have same number of atoms"
    assert all(types1[i] == types2[i] for i in range(len(types1))), "Atom types must match"

    # Compute PDFs for modified atoms
    r_centers, pdf1 = compute_deltaPDF_numpy(coords1, types1, modified_atoms, r_range, nbins, sigma, box)
    _, pdf2 = compute_deltaPDF_numpy(coords2, types2, modified_atoms, r_range, nbins, sigma, box)

    deltaPDF = pdf2 - pdf1
    return r_centers, deltaPDF
'''

def compute_deltaPDF(
        u1: mda.Universe,
        u2: mda.Universe,
        modified_atoms: List[int],
        r_range: Tuple[float, float] = (0.0, 10.0),
        nbins: int = 200,
        sigma: float = 0.2,
        sigma_map: Optional[Dict[Tuple[str,str], float]] = None,
        box: Optional[np.ndarray] = None,
        verbose: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute deltaPDF = PDF(u2) - PDF(u1) for only modified atoms.

    Parameters
    ----------
    u1, u2 : MDAnalysis Universe
        Before/after structures
    modified_atoms : list of int
        indices of atoms that changed
    r_range : tuple
        (rmin, rmax) in angstrom
    nbins : int
        number of r bins
    sigma : float
        Gaussian smearing width (default)
    sigma_map : dict, optional
        Mapping {(type1, type2): sigma_val}, allows pair-specific sigma.
        If not provided, fallback to global sigma.
    box : array_like, optional
        Periodic box (Lx,Ly,Lz)
    verbose : bool
        print progress

    Returns
    -------
    r_centers : (nbins,) array
    deltaPDF : (nbins,) array
    """
    _print = print if verbose else (lambda *a, **k: None)

    coords1 = u1.atoms.positions
    coords2 = u2.atoms.positions
    types1 = u1.atoms.types
    types2 = u2.atoms.types

    assert len(coords1) == len(coords2), "u1 and u2 must have same number of atoms"
    assert all(types1[i] == types2[i] for i in range(len(types1))), "Atom types must match"

    if sigma_map is None:
        r_centers, pdf1 = compute_deltaPDF_numpy(coords1, types1, modified_atoms, r_range, nbins, sigma, box)
        _, pdf2 = compute_deltaPDF_numpy(coords2, types2, modified_atoms, r_range, nbins, sigma, box)
    else:
        r_centers, pdf1 = compute_deltaPDF_numpy_sigmaMap(coords1, types1, modified_atoms, r_range, nbins, sigma, sigma_map, box)
        _, pdf2 = compute_deltaPDF_numpy_sigmaMap(coords2, types2, modified_atoms, r_range, nbins, sigma, sigma_map, box)

    deltaPDF = pdf2 - pdf1
    return r_centers, deltaPDF
'''
def compute_deltaPDF_numpy(
        coords: np.ndarray,
        atom_types: List[str],
        modified_atoms: List[int],
        r_range: Tuple[float,float]=(0.0, 10.0),
        nbins: int = 200,
        sigma: float = 0.2,
        box: Optional[np.ndarray] = None,
        cutoff: float = 10.0  # 新增cutoff参数
    ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute deltaPDF using NumPy for only modified atoms.

    Parameters
    ----------
    coords : (N,3) array
        Cartesian coordinates
    atom_types : list of str
        atom types for each atom
    modified_atoms : list of int
        indices of atoms considered as centers
    r_range : tuple
        (rmin, rmax) in angstrom
    nbins : int
        number of bins
    sigma : float
        Gaussian smearing width
    box : array_like, optional
        periodic box lengths (Lx,Ly,Lz)
    cutoff : float
        maximum distance to consider (Å)

    Returns
    -------
    r_centers : (nbins,) array
    deltaPDF : (nbins,) array
    """
    coords = np.asarray(coords, dtype=float)
    N = coords.shape[0]
    r_edges = np.linspace(r_range[0], r_range[1], nbins+1)
    r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])
    dr = r_edges[1] - r_edges[0]
    deltaPDF = np.zeros(nbins, dtype=float)

    unique_types = sorted(set(atom_types))
    crossection = {t: float(get_crossection(t)) for t in unique_types}

    for i_center in modified_atoms:
        t1 = atom_types[i_center]
        pos_i = coords[i_center]

        for j_other in range(N):
            if i_center == j_other:
                continue
            t2 = atom_types[j_other]
            pos_j = coords[j_other]

            # distance with periodic correction
            r_vec = pos_j - pos_i
            if box is not None:
                r_vec -= box * np.round(r_vec / box)
            r_val = np.linalg.norm(r_vec)
            
            # 添加cutoff检查
            if r_val > cutoff:
                continue

            # Gaussian smearing
            gauss = np.exp(-0.5*((r_centers - r_val)/sigma)**2) / (sigma*np.sqrt(2*np.pi))
            deltaPDF += gauss * crossection[t1] * crossection[t2]

    return r_centers, deltaPDF

def compute_deltaPDF_numpy_sigmaMap(
        coords: np.ndarray,
        atom_types: List[str],
        modified_atoms: List[int],
        r_range: Tuple[float,float]=(0.0, 10.0),
        nbins: int = 200,
        sigma: float = 0.2,
        sigma_map: Optional[Dict[Tuple[str,str], float]] = None,
        box: Optional[np.ndarray] = None,
        cutoff: float = 10.0  # 新增cutoff参数
    ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute deltaPDF using NumPy for only modified atoms,
    allowing different sigma for different atom pairs.

    Parameters
    ----------
    coords : (N,3) array
        Cartesian coordinates
    atom_types : list of str
        atom types for each atom
    modified_atoms : list of int
        indices of atoms considered as centers
    r_range : tuple
        (rmin, rmax) in angstrom
    nbins : int
        number of bins
    sigma : float
        default Gaussian width
    sigma_map : dict, optional
        mapping {(type1, type2): sigma_val}, order of types ignored
    box : array_like, optional
        periodic box lengths (Lx,Ly,Lz)
    cutoff : float
        maximum distance to consider (Å)

    Returns
    -------
    r_centers : (nbins,) array
    deltaPDF : (nbins,) array
    """
    coords = np.asarray(coords, dtype=float)
    N = coords.shape[0]
    r_edges = np.linspace(r_range[0], r_range[1], nbins+1)
    r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])
    deltaPDF = np.zeros(nbins, dtype=float)

    unique_types = sorted(set(atom_types))
    crossection = {t: float(get_crossection(t)) for t in unique_types}

    for i_center in modified_atoms:
        t1 = atom_types[i_center]
        pos_i = coords[i_center]

        for j_other in range(N):
            if i_center == j_other:
                continue
            t2 = atom_types[j_other]
            pos_j = coords[j_other]

            # distance with periodic correction
            r_vec = pos_j - pos_i
            if box is not None:
                r_vec -= box * np.round(r_vec / box)
            r_val = np.linalg.norm(r_vec)
            
            # 添加cutoff检查
            if r_val > cutoff:
                continue

            # determine sigma for this pair
            pair = tuple(sorted((t1, t2)))
            sigma_ij = sigma_map.get(pair, sigma) if sigma_map is not None else sigma

            # Gaussian smearing with pair-specific sigma
            gauss = np.exp(-0.5*((r_centers - r_val)/sigma_ij)**2) / (sigma_ij*np.sqrt(2*np.pi))
            deltaPDF += gauss * crossection[t1] * crossection[t2]

    return r_centers, deltaPDF

def compute_deltaPDF(
        u1: mda.Universe,
        u2: mda.Universe,
        modified_atoms: List[int],
        r_range: Tuple[float, float] = (0.0, 10.0),
        nbins: int = 200,
        sigma: float = 0.2,
        sigma_map: Optional[Dict[Tuple[str,str], float]] = None,
        box: Optional[np.ndarray] = None,
        cutoff: float = 10.0,  # 新增cutoff参数
        verbose: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute deltaPDF = PDF(u2) - PDF(u1) for only modified atoms.

    Parameters
    ----------
    u1, u2 : MDAnalysis Universe
        Before/after structures
    modified_atoms : list of int
        indices of atoms that changed
    r_range : tuple
        (rmin, rmax) in angstrom
    nbins : int
        number of r bins
    sigma : float
        Gaussian smearing width (default)
    sigma_map : dict, optional
        Mapping {(type1, type2): sigma_val}, allows pair-specific sigma.
        If not provided, fallback to global sigma.
    box : array_like, optional
        Periodic box (Lx,Ly,Lz)
    cutoff : float
        maximum distance to consider (Å)
    verbose : bool
        print progress

    Returns
    -------
    r_centers : (nbins,) array
    deltaPDF : (nbins,) array
    """
    _print = print if verbose else (lambda *a, **k: None)

    coords1 = u1.atoms.positions
    coords2 = u2.atoms.positions
    types1 = u1.atoms.types
    types2 = u2.atoms.types

    assert len(coords1) == len(coords2), "u1 and u2 must have same number of atoms"
    assert all(types1[i] == types2[i] for i in range(len(types1))), "Atom types must match"

    if sigma_map is None:
        r_centers, pdf1 = compute_deltaPDF_numpy(coords1, types1, modified_atoms, r_range, nbins, sigma, box, cutoff)
        _, pdf2 = compute_deltaPDF_numpy(coords2, types2, modified_atoms, r_range, nbins, sigma, box, cutoff)
    else:
        r_centers, pdf1 = compute_deltaPDF_numpy_sigmaMap(coords1, types1, modified_atoms, r_range, nbins, sigma, sigma_map, box, cutoff)
        _, pdf2 = compute_deltaPDF_numpy_sigmaMap(coords2, types2, modified_atoms, r_range, nbins, sigma, sigma_map, box, cutoff)

    deltaPDF = pdf2 - pdf1
    return r_centers, deltaPDF