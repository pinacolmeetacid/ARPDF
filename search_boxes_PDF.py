import os
import pickle
import cupy as cp
import numpy as np
import MDAnalysis as mda
import tempfile
import matplotlib.pyplot as plt
#from PDF import compute_ARPDF, compare_ARPDF
from PDF_POLAR import compute_PDF, compare_PDF
from PDF_torch import compute_deltaPDF, compute_deltaPDF_numpy
from utils import select_nbr_mols, clean_gro_box, rotate_ccl4_molecules, select_ccl4_molecules, update_metadata
from utils.similarity import cosine_similarity, get_angular_filters, get_gaussian_filters, angular_similarity, strength_similarity, oneD_similarity, angular_average_similarity
from utils.core_functions import to_cupy
from ccl4_modifier import CCL4Modifier_CL, CCL4Modifier_double_CL, select_cl_atoms, select_double_cl
from typing import Callable, List, Tuple, Optional, Protocol
from dataclasses import dataclass

@dataclass
class SearchResult:
    """Data class to store search results for a single molecule
    
    Attributes:
        molecule (int): Index of the target molecule
        polar_axis (ndarray): Direction vector of the modified bond
        modified_universe (mda.Universe): Modified structure
        PDF (ndarray): Calculated ARPDF for the modified structure
        similarity (float): Similarity score between ARPDF and experimental data
        modified_atoms (List[int]): Indices of modified atoms
    """
    molecule: int
    polar_axis: List[float]
    modified_universe: mda.Universe
    PDF: np.ndarray
    similarity: float
    modified_atoms: List[int]

class StrucModProtocol(Protocol):
    """Handle molecular structure modifications"""
    def generate_modified_structures(self, molecule: int) -> List[Tuple[List[float], mda.Universe, List[int]]]:
        """Generate modified structures
        Args:
            molecule (int): Index of the target atom or molecule
            
        Returns:
            List[Tuple[List[float], mda.Universe, List[int]]]: List of tuples containing:
                - polar_axis (List[float]): Polarization axis
                - u2 (Universe): Modified MDAnalysis universe 
                - modified_atoms (List[int]): List of modified atom indices
        """
    


def save_ccl4_result(result: SearchResult, file_name: str, nbr_distance: float = 5.0):
    """Save the result of CCl4 structure"""
    # 统一获取一个原子 index，用于识别它所在的 CCl4 分子
    if isinstance(result.molecule, (list, tuple)) and len(result.molecule) > 0:
        ref_atom_index = result.molecule[0]
    else:
        ref_atom_index = result.molecule

    ccl4_indices = select_ccl4_molecules(result.modified_universe, ref_atom_index, cutoff_distance=nbr_distance)
    nbr_indices = select_nbr_mols(result.modified_universe, result.modified_atoms, nbr_distance=nbr_distance)
    nbr_group = rotate_ccl4_molecules(result.modified_universe, ccl4_indices, nbr_indices, result.polar_axis)
    nbr_group.write(file_name)


class SimilarityCalculator:
    """Handle similarity calculations between ARPDFs"""
    
    def __init__(self, R, weight_cutoff=5.0, metric='cosine'):
        self.R = R
        self.weight_cutoff = weight_cutoff
        self._initialize_weights()
        self._metric = self._get_metric_function(metric)
    
    def _initialize_weights(self):
        """Initialize various weights for similarity calculations"""
        # Cosine weight
        self.cos_weight = cp.exp(-cp.maximum(self.R - self.weight_cutoff, 0)**2 / (2 * (0.5)**2))/(1 + cp.exp(-10 * (self.R - 1)))
        
    
    def calc_similarity(self, PDF, PDF_exp):
        """Calculate similarity between two ARPDFs"""
        return float(self._metric(PDF, PDF_exp))
    
    def _get_metric_function(self, metric):
        """Get the appropriate metric function"""
        metric_funcs = {
            'cosine': lambda x, y: cosine_similarity(x, y, self.cos_weight),
        }
        return metric_funcs[metric]


class StructureSearcher:
    """Handle structure search and result management"""
    
    def __init__(self, 
                 output_dir: str,
                 universe: mda.Universe,
                 R: Tuple[np.ndarray],
                 PDF_ref: np.ndarray,
                 molecule_selector: Callable[[mda.Universe], List[int]],
                 structure_modifier: StrucModProtocol,
                 filter_fourier: Optional[Callable] = None,
                 cutoff: float = 10.0,
                 sigma0: float = 0.2,
                 metric: str = 'cosine',
                 weight_cutoff: float = 4.0,
                 neg: bool = False):
        """Initialize the structure searcher
        
        Args:
            output_dir (str): Directory to save results
            universe (mda.Universe): Initial structure
            grids_XY (tuple): Grid coordinates (X, Y)
            ARPDF_ref (ndarray): Reference ARPDF data
            molecule_selector (callable): Function to select molecules for analysis
            structure_modifier: Object that implements generate_modified_structures method
            filter_fourier (callable, optional): Fourier filter function
            cutoff (float, optional): Distance cutoff for ARPDF calculation
            sigma0 (float, optional): Gaussian width for ARPDF calculation
            metric (str, optional): Similarity metric to use
            weight_cutoff (float, optional): Weight cutoff for similarity calculation
            neg (bool, optional): Whether to use negative ARPDF
        """
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        self.universe = universe
        self.R = R
        self.PDF_ref = PDF_ref
        self.filter_fourier = filter_fourier
        self.cutoff = cutoff
        self.sigma0 = sigma0
        self.neg = neg
        
        # Store selection and modification functions
        self.molecule_selector = molecule_selector
        self.structure_modifier = structure_modifier
        
        # Initialize components
        self.weight_cutoff = weight_cutoff
        self.metric = metric
        self.similarity_calc = SimilarityCalculator(self.R, weight_cutoff, metric)
        
        # Initialize results
        self.results: List[SearchResult] = []

    def temp_universe_from_modified(self,u2):
        """
        把 u2 的结构保存为临时 .gro 文件，再重新读回 Universe，
        保证文件在 PDF 计算过程中存在。
        """
        # 创建临时文件，但不自动删除
        tmp = tempfile.NamedTemporaryFile(suffix=".gro", delete=False)
        tmp_name = tmp.name
        tmp.close()  # 关闭文件，保证可以被 MDAnalysis 打开

        # 保存 Universe 到这个临时文件
        u2.atoms.write(tmp_name)

        # 重新读取成 Universe
        u2_tmp = mda.Universe(tmp_name)

        # 返回 u2_tmp 和文件名，方便处理完手动删除
        return u2_tmp, tmp_name
    
    def search(self):
        """Search for optimal molecular structures
        
        Returns:
            list: List of SearchResult objects for each molecule
        """
        R = to_cupy(self.R)
        PDF_ref = to_cupy(self.PDF_ref)
        # Get molecule list
        molecule_list = self.molecule_selector(self.universe)
        
        # Search for each molecule
        for molecule in molecule_list:
            best_similarity = -np.inf
            best_result = None
            
            for polar_axis, u2, modified_atoms in self.structure_modifier.generate_modified_structures(molecule):
                #u2_tmp, tmp_name = self.temp_universe_from_modified(u2)
                #_, G_total_origin, _ = compute_PDF(
                #    self.universe,
                #    r_range=(0.0, 10.0),
                #    nbins=200,
                #    sigma=0.25,
                #    periodic=True,
                #    verbose=True
                #)
                #_, G_total, _ = compute_PDF(
                #    u2_tmp,
                #    r_range=(0.0, 10.0),
                #    nbins=200,
                #    sigma=0.25,
                #    periodic=True,
                #    verbose=True
                #)

                #os.remove(tmp_name)
                #plt.plot(R.get(),G_total-G_total_origin)
                #plt.plot(R.get(),G_total)
                #plt.show()
               
                # 转为 CuPy
                #G_total_origin = cp.asarray(G_total_origin)
                #G_total = cp.asarray(G_total)

                #PDF = G_total-G_total_origin

                _,PDF=compute_deltaPDF(
                    self.universe,
                    u2,
                    modified_atoms=modified_atoms,
                    r_range= (0.0, 10.0),
                    nbins= 200,
                    sigma= 0.2,
                    sigma_map= None,
                    box= u2.dimensions[:3],
                    verbose= False
                )
                PDF = cp.asarray(PDF)
                similarity = self.similarity_calc.calc_similarity(PDF, PDF_ref)
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_result = SearchResult(
                        molecule=int(molecule),
                        polar_axis=[float(x) for x in polar_axis],
                        modified_universe=u2,
                        PDF=PDF.get(),
                        similarity=float(similarity),
                        modified_atoms=[int(x) for x in modified_atoms]
                    )
            
            if best_result is not None:
                self.results.append(best_result)
        
        return self.results
    
    def save_results(self) -> None:
        """Save analysis results"""
        R, ARPDF_ref = self.R, self.PDF_ref
        
        if not self.results:
            return
            
        # get best and worst results
        best_result = max(self.results, key=lambda x: x.similarity)
        worst_result = min(self.results, key=lambda x: x.similarity)

        # Save structures
        self.universe.atoms.write(f"{self.output_dir}/CCl4.gro")
        best_result.modified_universe.atoms.write(f"{self.output_dir}/CCl4_best_init.gro")

        # Save best and worst results
        self._save_single_result(best_result, prefix="best_init")
        self._save_single_result(worst_result, prefix="worst_init")

        # save ARPDF_ref
        np.save(f"{self.output_dir}/ARPDF_ref.npy", ARPDF_ref)

        # Save raw results
        with open(f"{self.output_dir}/results.pkl", "wb") as f:
            pickle.dump(self.results, f)

        # Prepare metadata with search parameters
        grids_range = (R.min(), R.max())
        grids_range = list(float(x) for x in grids_range)

        # Save metadata
        update_metadata(self.output_dir, {
            "name": "CCl4",
            "structure_info": {
                "u1_name": "CCl4.gro",
                "u2_name": "CCl4_best_init.gro",
                "polar_axis": best_result.polar_axis,
                "modified_atoms": best_result.modified_atoms
            },
            "search_info": {
                "parameters": {
                    "grids_range": grids_range,
                    "grids_shape": list(self.R.shape),
                    "molecule_selector": self.molecule_selector.__name__,
                    "structure_modifier": self.structure_modifier.__class__.__name__,
                    "filter_fourier": str(self.filter_fourier),
                    "cutoff": self.cutoff,
                    "weight_cutoff": self.weight_cutoff,
                    "metric": self.metric,
                    "sigma0": self.sigma0,
                    "neg": self.neg,
                },
                "best_result": {
                    "file_name": "CCl4_best_init.gro",
                    "similarity": best_result.similarity,
                    "polar_axis": best_result.polar_axis,
                    "modified_atoms": best_result.modified_atoms,
                    "molecule": best_result.molecule
                },
                "worst_result": {
                    "file_name": "CCl4_worst_init.gro",
                    "similarity": worst_result.similarity,
                    "polar_axis": worst_result.polar_axis,
                    "modified_atoms": worst_result.modified_atoms,
                    "molecule": worst_result.molecule
                }
            },
        })

    def _save_single_result(self, result: SearchResult, prefix: str) -> None:
        """
        Save a single result (best or worst) to files.

        Args:
            result (SearchResult): The result to save.
            prefix (str, optional): Prefix for the result.
        """
        R, ARPDF_ref = self.R, self.PDF_ref
        save_ccl4_result(result, f"{self.output_dir}/{prefix}_selected.gro", nbr_distance=5.0)
        # Save visualization
        _metric_name = self.metric.replace("_", " ").title()
        fig, sim_value = compare_PDF(
            R, result.PDF, ARPDF_ref, 
            sim_name=f"{_metric_name} Sim", sim_value=result.similarity, 
            show_range=8.0, weight_cutoff=self.weight_cutoff
        )
        #fig.savefig(f"{self.output_dir}/{prefix}.png")
        #plt.close(fig)  # 保存后关闭 figure 避免内存泄漏




def workflow_demo(
        R, PDF_ref, 
        filter_fourier=None, 
        sigma0=0.2, 
        exp_name: str="exp", 
        metric: str="cosine", 
        weight_cutoff=5.0, 
        stretch_distances: List[float] = None,
        neg=False
    ):
    """Demo workflow for structure search and analysis"""
    # Clean and load structure
     
    import cupy as cp
    
    # 统一后端
    R = cp.asarray(R, dtype=cp.float32)
    PDF_ref = cp.asarray(PDF_ref, dtype=cp.float32)


    #clean_gro_box('data/CCl4/CCl4.gro', 'data/CCl4/CCl4_clean.gro')
    clean_gro_box('data/CCl4/CCl4_5nm.gro', 'data/CCl4/CCl4_clean_5nm.gro')
    universe = mda.Universe('data/CCl4/CCl4_clean_5nm.gro')
    if neg:
        PDF_ref = PDF_ref.copy()
        PDF_ref[PDF_ref > 0] = 0
    
    # Initialize searcher with CCl4-specific components
    searcher = StructureSearcher(
        output_dir=f"tmp/{exp_name}",
        universe=universe,
        R=R,
        PDF_ref=PDF_ref,
        molecule_selector=select_cl_atoms,
        structure_modifier=CCL4Modifier_CL(universe, stretch_distances, periodic=True),
        filter_fourier=filter_fourier,
        sigma0=sigma0,
        metric=metric,
        weight_cutoff=weight_cutoff,
        neg=neg
    )
    
    # Search and save results
    searcher.search()
    searcher.save_results()
