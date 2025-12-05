from collections import Counter
import os
import pickle
from typing import List
import numpy as np
import MDAnalysis as mda
import torch
from tqdm import tqdm
from optimize_ARPDF import ARPDFOptimizer, ARPDFPolarOptimizer
from search_boxes import SearchResult
from utils import load_structure_data, generate_grids, update_metadata, select_nbr_mols
from utils.analyze_structure import select_ccl4_molecules
import json

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


def optimize_all_structures(exp_dir: str, output_dir: str = "optimize"):
    """
    Optimize all structures in the search results and save the results to the output directory.

    Args:
        exp_dir (str): Directory containing the search results (results.pkl).
        output_dir (str): Subdirectory to save the optimized results (default: "optimize").
    """
    # Load the search results
    with open(os.path.join(exp_dir, "results.pkl"), "rb") as f:
        results: List[SearchResult] = pickle.load(f)
    # results = [results[i] for i in range(12)]

    # Create the output directory if it doesn't exist
    output_path = os.path.join(exp_dir, output_dir)
    os.makedirs(output_path, exist_ok=True)

    # Load reference structure and metadata
    reference_dir = "data/CCl4"
    u1_ref, u2_ref, modified_atoms_ref, polar_axis_ref = load_structure_data(reference_dir)
    ARPDF_ref = np.load(os.path.join(exp_dir, "ARPDF_ref.npy"))
    with open(os.path.join(exp_dir, "metadata.json"), "r") as f:
        root_metadata = json.load(f)
    #xy_range = root_metadata["search_info"]["parameters"]["grids_range"]
    #N, M = root_metadata["search_info"]["parameters"]["grids_shape"]
    #X, Y = generate_grids(xy_range, N, M)
    R, Phi = generate_grids((0,10,0,0.5*np.pi),256,100)
    cutoff = 10.0
    sigma0 = root_metadata["search_info"]["parameters"]["sigma0"]
    weight_cutoff = root_metadata["search_info"]["parameters"]["cutoff"]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
 
    # Initialize the optimizer
    optimizer = ARPDFPolarOptimizer(
        R, Phi,
        ARPDF_ref=ARPDF_ref,
        atom_types=Counter(u1_ref.atoms.types),
        cutoff=cutoff,
        sigma0=sigma0,
        weight_cutoff=weight_cutoff,
        lr=0.02,
        gamma_lr=0.995,
        gamma_noise=0.999, 
        f_lb=-1.0, 
        s=0.5,
        #s=0.0, 
        #beta=0.1,
        beta=0.3, 
        epochs=1500,
        loss_name="angular_scale",
        device=device,
        warmup=200,
    )
    total_structures = len(results)

    u1_real = mda.Universe("data/CCl4/CCl4_clean.gro")
    # change u1_ref to u1_real from now on

    # Iterate over each search result
    for i, result in tqdm(enumerate(results), desc="Optimizing all structures", total=total_structures, position=1):
        # Create a subdirectory for this structure
        struct_dir = os.path.join(output_path, f"structure_{i}")

        # 如果这个目录已经存在并且有优化结果，就跳过
        if os.path.exists(os.path.join(struct_dir, "CCl4_optimized.gro")):
            tqdm.write(f"Structure {i} already optimized, skipping.")
            continue

        os.makedirs(struct_dir, exist_ok=True)
        
        # Save the original structure
        result.modified_universe.atoms.write(os.path.join(struct_dir, f"structure_{i}.gro"))

        # Save the metadata
        update_metadata(struct_dir, {
            "name": root_metadata["name"],
            "structure_info": {
                "u1_name": "../" + root_metadata["structure_info"]["u1_name"],
                "u2_name": f"structure_{i}.gro",
                "polar_axis": result.polar_axis,
                "modified_atoms": result.modified_atoms,
                "similarity": result.similarity,
                "molecule": result.molecule
            }
        })

        # Set the system for optimization
        optimized_atoms = select_nbr_mols(u1_real, result.modified_atoms, nbr_distance=None, periodic=True)
        _types = u1_real.atoms[optimized_atoms].types
        _sort_idx = np.argsort(_types)
        _types = [_types[i] for i in _sort_idx]
        optimized_atoms = optimized_atoms[_sort_idx]
        target_cl = result.molecule
        optimized_atoms = [i for i in optimized_atoms if i != target_cl] + [target_cl]
        #optimized_atoms = [i for i in optimized_atoms if i not in target_cl] + list(target_cl)        # optimized_atoms = select_ccl4_molecules(result.modified_universe, result.molecule)
        tqdm.write(f"Stucture {i + 1}/{total_structures}: optimizing atoms {optimized_atoms}.")
        optimizer.set_system(
            out_dir=struct_dir,
            u1=u1_real,
            u2=result.modified_universe,
            optimized_atoms=optimized_atoms,
            polar_axis=result.polar_axis,
            #norm_func=ccl3_sysmetry_norm,
            #norm_func=c2v_symmetry_norm,
            norm_func=None
        )

        # Run optimization
        optimizer.optimize(verbose=True, log_step=5, print_step=50, leave=False, prefix=f"Optimizing structure {i + 1}/{total_structures}")
        tqdm.write(f"Structure {i + 1}/{total_structures} finished.")

    print(f"Optimization completed. Results saved to {output_path}.")

# Example usage
if __name__ == "__main__":
    exp_dir = "tmp/polar_test_doublesmooth_0.15"  # Adjust based on your directory
    optimize_all_structures(exp_dir, output_dir="runs/test_no_norm")
    #optimize_all_structures(exp_dir, output_dir="runs/test_no_norm")
    print("All structures optimized successfully.")