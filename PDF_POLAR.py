from typing import Dict, List, Optional, Tuple
import numpy as np
import MDAnalysis as mda
from MDAnalysis.analysis.rdf import InterRDF
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
from utils import show_images
from utils.core_functions import to_numpy
from utils.similarity import cosine_similarity
from utils.AFF_map import get_crossection

def compute_PDF(
    universe: mda.Universe,
    r_range: Tuple[float, float] = (0.0, 10.0),
    nbins: int = 200,
    sigma: float = 0.2,
    periodic: bool = True,
    verbose: bool = False
) -> Tuple[np.ndarray, np.ndarray, Dict[Tuple[str, str], np.ndarray]]:
    """
    Main pipeline (radial PDF version):
    MDAnalysis Universe -> compute RDF for all atom type pairs -> 
    weight with atomic cross-sections -> smooth -> total PDF.

    Parameters
    ----------
    universe   : MDAnalysis Universe object
    r_range    : (r_min, r_max), Å
    nbins      : number of bins
    sigma      : Gaussian smoothing width (Å)
    periodic   : if True, consider periodic boundary conditions
    verbose    : if True, show intermediate output

    Returns
    -------
    r          : np.ndarray, distance grid
    G_total    : np.ndarray, total PDF after weighting and smoothing
    g_dict     : dict mapping (atom_type1, atom_type2) -> smoothed g_ij(r)
    """
    _print_func = print if verbose else lambda *args: None

    # collect atom types
    atom_types = sorted(set(universe.atoms.types))

    # prepare crossection lookup
    crossection = {atom: get_crossection(atom) for atom in atom_types}

    N_total = len(universe.atoms)
    conc = {t: (universe.atoms.types == t).sum() / N_total for t in atom_types}

    # dictionary for RDF results
    g_dict = {}
    weight_dict = {}

    r = None
    for i, t1 in enumerate(atom_types):
        for t2 in atom_types[i:]:
            sel1 = universe.select_atoms(f"type {t1}")
            sel2 = universe.select_atoms(f"type {t2}")
            if len(sel1) == 0 or len(sel2) == 0:
                continue

            if t1 == t2:
                rdf = InterRDF(sel1, sel2, range=r_range, nbins=nbins, exclusion_block=(1, 1))
                N_pairs = len(sel1) * (len(sel1) - 1) / 2
            else:
                rdf = InterRDF(sel1, sel2, range=r_range, nbins=nbins)
                N_pairs = len(sel1) * len(sel2)

            rdf.run()
            r = rdf.bins
            g_raw = rdf.rdf

            # Gaussian smoothing
            dr = (r_range[1] - r_range[0]) / nbins
            g_smooth = gaussian_filter1d(g_raw, sigma / dr)

            g_dict[(t1, t2)] = g_smooth

            # weight = scattering cross-section product
            weight_dict[(t1, t2)] = crossection[t1] * crossection[t2] * N_pairs

            _print_func(f"Computed RDF for {t1}-{t2}, weighted with σ={weight_dict[(t1,t2)]:.2f}")

    # normalize weights
    w_sum = sum(weight_dict.values())
    for k in weight_dict:
        weight_dict[k] /= w_sum

    # build total PDF
    G_total = np.zeros_like(r)
    for pair, g in g_dict.items():
        G_total += weight_dict[pair] * (g-1) * r

    return r, G_total, g_dict



def compare_PDF(
    r: np.ndarray,
    pdf: np.ndarray,
    pdf_ref: np.ndarray,
    sim_name: str = "Cosine Similarity",
    sim_value: float | None = None,
    show_range: float = 8.0,
    weight_cutoff: float = 5.0,
):
    """
    Compare PDF with reference PDF.

    Returns the figure object.
    """
    import cupy as cp
    import matplotlib.pyplot as plt

    if isinstance(r, cp.ndarray):
        r = r.get()
    if isinstance(pdf, cp.ndarray):
        pdf = pdf.get()
    if isinstance(pdf_ref, cp.ndarray):
        pdf_ref = pdf_ref.get()

    # Restrict to plotting range
    mask = r < show_range
    r = r[mask]
    pdf = pdf[mask].copy()
    pdf_ref = pdf_ref[mask].copy()

    # Normalize by max within cutoff
    mask_norm = r < weight_cutoff
    pdf /= pdf[mask_norm].max() + 1e-6
    pdf_ref /= pdf_ref[mask_norm].max() + 1e-6

    # Compute similarity
    if sim_value is None:
        sim_value = cosine_similarity(pdf, pdf_ref)

    # Plot
    fig, ax = plt.subplots(figsize=(6, 4))  # 创建 figure 对象
    ax.plot(r, pdf, label=f"PDF ({sim_name}: {sim_value:.2f})", lw=2)
    ax.plot(r, pdf_ref, "--", label="Reference PDF", lw=2)
    ax.set_xlabel("r (Å)")
    ax.set_ylabel("PDF (arb. units)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    # 不立即 show，让调用方决定
    # plt.show()

    return fig, sim_value  # 返回 figure 对象和相似度
