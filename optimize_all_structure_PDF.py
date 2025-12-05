from collections import Counter
import os, pickle, json
from typing import List
import numpy as np
import MDAnalysis as mda
import torch
from tqdm import tqdm
from optimize_PDF import PDFOptimizer   # ⚠️ 这是你要写的 PDFOptimizer
from search_boxes import SearchResult
from utils import load_structure_data, update_metadata, select_nbr_mols
from utils.analyze_structure import select_ccl4_molecules

def ccl3_sysmetry_norm(ccl4_pos: torch.Tensor):
    C = ccl4_pos[[0]]
    Cls = ccl4_pos[1:4]
    dist_c_cl = torch.linalg.vector_norm(C - Cls, dim=1)
    dist_cl_cl = torch.linalg.vector_norm(
        torch.stack((Cls[1] - Cls[0], Cls[2] - Cls[0], Cls[2] - Cls[1]), dim=0), 
        dim=1
    )
    sysmetry_norm = torch.square(dist_c_cl - dist_c_cl[[1, 2, 0]]).sum() + torch.square(dist_cl_cl - dist_cl_cl[[1, 2, 0]]).sum()
    return sysmetry_norm

def c2v_symmetry_norm(ccl4_pos: torch.Tensor):
    """
    ccl4_pos shape: (5, 3), 原子顺序: C(0), Cl1(1), Cl2(2), Cl3(3), Cl4(4)
    计算：
    - C-Cl1 和 C-Cl2 距离差平方
    - C-Cl3 和 C-Cl4 距离差平方
    - Cl1-Cl3, Cl1-Cl4, Cl2-Cl3, Cl2-Cl4 4个距离差距平方和（两两之间差异）
    """
    C = ccl4_pos[0]
    Cl1, Cl2, Cl3, Cl4 = ccl4_pos[1], ccl4_pos[2], ccl4_pos[3], ccl4_pos[4]

    # 计算 C-Cl 距离
    d_CCl1 = torch.linalg.norm(C - Cl1)
    d_CCl2 = torch.linalg.norm(C - Cl2)
    d_CCl3 = torch.linalg.norm(C - Cl3)
    d_CCl4 = torch.linalg.norm(C - Cl4)

    # C-Cl1 和 C-Cl2 应尽量相等
    diff_CCl_12 = (d_CCl1 - d_CCl2)**2

    # C-Cl3 和 C-Cl4 应尽量相等
    diff_CCl_34 = (d_CCl3 - d_CCl4)**2

    # 计算跨组 Cl-Cl 距离
    d_Cl1_Cl3 = torch.linalg.norm(Cl1 - Cl3)
    d_Cl1_Cl4 = torch.linalg.norm(Cl1 - Cl4)
    d_Cl2_Cl3 = torch.linalg.norm(Cl2 - Cl3)
    d_Cl2_Cl4 = torch.linalg.norm(Cl2 - Cl4)

    cross_dists = torch.stack([d_Cl1_Cl3, d_Cl1_Cl4, d_Cl2_Cl3, d_Cl2_Cl4])

    # 这4个距离之间的差异总和，可以用所有两两组合距离差平方和
    from itertools import combinations
    cross_diff = 0.0
    for i, j in combinations(range(4), 2):
        cross_diff += (cross_dists[i] - cross_dists[j])**2

    norm = diff_CCl_12 + diff_CCl_34 + cross_diff
    return norm

def optimize_all_structures_pdf(exp_dir: str, output_dir: str = "optimize_pdf"):
    """
    Optimize all structures using PDFOptimizer instead of ARPDF.
    """

    # Load search results
    with open(os.path.join(exp_dir, "results.pkl"), "rb") as f:
        results: List[SearchResult] = pickle.load(f)

    output_path = os.path.join(exp_dir, output_dir)
    os.makedirs(output_path, exist_ok=True)

    # Load reference structure & metadata
    reference_dir = "data/CCl4"
    u1_ref, u2_ref, modified_atoms_ref, _ = load_structure_data(reference_dir)
    PDF_ref = np.load(os.path.join(exp_dir, "PDF_ref.npy"))   # ⚠️ 需要提前算好
    with open(os.path.join(exp_dir, "metadata.json"), "r") as f:
        root_metadata = json.load(f)

    #r = torch.linspace(0.0, cutoff, 200)

    sigma0 = root_metadata["search_info"]["parameters"]["sigma0"]
    cutoff = root_metadata["search_info"]["parameters"]["cutoff"]
    weight_cutoff = root_metadata["search_info"]["parameters"]["weight_cutoff"]


    r = torch.linspace(0.0, cutoff, 200)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Initialize optimizer
    optimizer = PDFOptimizer(
        r,
        PDF_ref=PDF_ref*r.cpu().numpy(),
        atom_types=Counter(u1_ref.atoms.types),
        cutoff=cutoff,
        sigma0=sigma0,
        weight_cutoff=weight_cutoff,
        lr=0.02,
        gamma_lr=0.995,
        gamma_noise=0.999,
        f_lb=-1.0,
        s=0.5,
        beta=0.3,
        epochs=1500,
        loss_name="cosine",   # 默认用 cosine
        device=device,
        warmup=200,
    )

    total_structures = len(results)
    u1_real = mda.Universe("data/CCl4/CCl4_clean_5nm.gro")

    for i, result in tqdm(enumerate(results), desc="Optimizing all structures (PDF)", total=total_structures):
        struct_dir = os.path.join(output_path, f"structure_{i}")

        if os.path.exists(os.path.join(struct_dir, "CCl4_optimized.gro")):
            tqdm.write(f"Structure {i} already optimized, skipping.")
            continue
        os.makedirs(struct_dir, exist_ok=True)

        result.modified_universe.atoms.write(os.path.join(struct_dir, f"structure_{i}.gro"))

        update_metadata(struct_dir, {
            "name": root_metadata["name"],
            "structure_info": {
                "u1_name": "../" + root_metadata["structure_info"]["u1_name"],
                "u2_name": f"structure_{i}.gro",
                "modified_atoms": result.modified_atoms,
                "similarity": result.similarity,
                "molecule": result.molecule
            }
        })

        optimized_atoms = select_nbr_mols(u1_real, result.modified_atoms, nbr_distance=None, periodic=True)
        tqdm.write(f"Structure {i + 1}/{total_structures}: optimizing atoms {optimized_atoms}.")

        optimizer.set_system(
            out_dir=struct_dir,
            u1=u1_real,
            u2=result.modified_universe,
            optimized_atoms=optimized_atoms,
            norm_func=None  # 或者用 c2v_symmetry_norm
        )

        optimizer.optimize(verbose=True, log_step=5, print_step=50,
                           prefix=f"Optimizing structure {i + 1}/{total_structures}")

        tqdm.write(f"Structure {i + 1}/{total_structures} finished.")

    print(f"Optimization completed. Results saved to {output_path}.")


# Example usage
if __name__ == "__main__":
    exp_dir = "tmp/PDF_test_doublesmooth_0_5nm"  # Adjust based on your directory
    optimize_all_structures_pdf(exp_dir, output_dir="runs/test1")
    #optimize_all_structures(exp_dir, output_dir="runs/test_no_norm")
    print("All structures optimized successfully.")