import os
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Any, Optional, Callable
import MDAnalysis as mda
from tqdm import tqdm
from PDF_torch import get_atoms_pairs
from utils import box_shift, load_structure_data
from utils.core_functions_torch import toTensor, GND
from utils.AFF_map import get_crossection
from utils.similarity import cosine_similarity
import pandas as pd
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ExponentialLR
from typing import Optional, Callable

Tensor = torch.Tensor

class PDFModel:
    def __init__(self, r: np.ndarray, atom_types: List[str], cutoff: float = 10.0, sigma0: float = 0.2, field_batch_size: int = 256):
        """
        r : 1D radial grid (np.ndarray or torch.Tensor) -> stored as torch tensor (Nr,)
        atom_types : list of all atom types (to prepare cross-sections)
        cutoff : maximum distance considered
        sigma0 : gaussian smearing width
        """
        super(PDFModel, self).__init__()
        self.r = toTensor(r).float().contiguous().view(-1)   # (Nr,)
        self.Nr = self.r.shape[0]
        self.cutoff = torch.tensor(cutoff).float()
        self.sigma0 = torch.tensor(sigma0).float()
        self.field_batch_size = field_batch_size
        self.prepare(atom_types)

    def prepare(self, atom_types: List[str]):
        """Cache cross-sections as attributes like ARPDFModel."""
        for atom_type in atom_types:
            setattr(self, f"Crossection_{atom_type}", torch.tensor(get_crossection(atom_type), dtype=torch.float32))

    def set_system(self, around_pos: Any, atom_pairs: Dict[Tuple[str, str], np.ndarray]):
        """
        around_pos: array-like (N_around, 3) coordinates of surrounding atoms
        atom_pairs: dict mapping (type1, type2) -> ij_idx array of shape (N_pairs, 2)
                   indices refer to concatenation [center_atoms, around_atoms] (same convention as ARPDFPolarModel)
        """
        device = self.r.device
        self.around_pos = toTensor(around_pos, device=device).float()
        self.all_pair_types: List[Tuple[str, str]] = list(atom_pairs.keys())
        for (t1, t2), ij_idx in atom_pairs.items():
            setattr(self, f"atom_pairs_{t1}_{t2}", toTensor(ij_idx, dtype=torch.int64, device=device))

    def to(self, *args, **kwargs):
        """Move internal tensors to device / dtype via toTensor helper (same convention)."""
        self.__dict__.update(toTensor(self.__dict__, *args, **kwargs))
        return self

    def __call__(self, *args, **kwds):
        return self.forward(*args, **kwds)

    def atom_pairs(self, atom_type1, atom_type2):
        return getattr(self, f"atom_pairs_{atom_type1}_{atom_type2}")

    def crossection(self, atom_type):
        return getattr(self, f"Crossection_{atom_type}")

    def forward(self, center_pos: Tensor) -> Tensor:
        """
        center_pos: (N_center, 3) torch tensor
        returns: total_pdf tensor shape (Nr,) (float32)
        Behavior: for all stored atom pair types, compute pair vectors using ij indices (like ARPDFModel),
                  compute r_vals, gaussian smear onto self.r grid, weight by cross-sections and sum.
        """
        device = self.r.device
        center_pos = toTensor(center_pos, device=device).float()
        all_pos = torch.cat([center_pos, self.around_pos], dim=0)  # same convention as ARPDFPolarModel
        total_pdf = torch.zeros_like(self.r)

        r_grid = self.r.view(self.Nr, 1)  # (Nr, 1)
        norm = (self.sigma0 * torch.sqrt(torch.tensor(2.0 * np.pi))).to(self.r.device)

        for atom_type1, atom_type2 in self.all_pair_types:
            ij_idx = self.atom_pairs(atom_type1, atom_type2)  # (Npairs, 2)
            if ij_idx.numel() == 0:
                continue
            vectors = all_pos[ij_idx[:, 1]] - all_pos[ij_idx[:, 0]]  # (Npairs, 3)
            r_vals = torch.linalg.norm(vectors, dim=1)  # (Npairs,)
            mask = r_vals <= self.cutoff
            if mask.sum() == 0:
                continue
            r_sel = r_vals[mask].view(1, -1)  # (1, M)
            # gaussian smearing: (Nr, M)
            diff = (r_grid - r_sel) / self.sigma0
            gauss = torch.exp(-0.5 * diff * diff) / norm  # (Nr, M)
            # weights: product of cross sections per pair (note: indices refer to types, not per-atom cs)
            cs1 = self.crossection(atom_type1)
            cs2 = self.crossection(atom_type2)
            pair_weight = (cs1 * cs2).to(self.r.device)
            # sum over pairs
            contrib = gauss.sum(dim=1) * pair_weight  # (Nr,)
            total_pdf = total_pdf + contrib

        return total_pdf  # shape (Nr,)

class PDFOptimizer:
    def __init__(
        self,
        r: np.ndarray,
        PDF_ref: np.ndarray,
        atom_types: List[str],
        cutoff: float = 10.0,
        sigma0: float = 0.2,
        weight_cutoff: float = 6.0,
        lr: float = 0.01,
        gamma_lr: float = 0.995,
        gamma_noise: float = 0.999,
        f_lb: float = 0.0,
        s: float = 0.1,
        beta: float = 0.0,
        epochs: int = 500,
        loss_name: str = "cosine",
        device: str = "cpu",
        warmup: int = 0,
    ):
        """
        r : 1D radial grid (np.ndarray)
        PDF_ref : reference 1D PDF (np.ndarray) of same length as r
        atom_types : list of types (for PDFModel.prepare)
        Other hyperparams mirror ARPDFPolarOptimizer.
        """
        self.r = toTensor(r, device=device).float().contiguous()
        self.PDF_ref = toTensor(PDF_ref, device=device).float().contiguous()
        self.atom_types = atom_types
        self.device = device

        self.model = PDFModel(r, atom_types, cutoff=cutoff, sigma0=sigma0).to(device=device)
        self.cutoff = cutoff
        self.sigma0 = sigma0
        self.lr = lr
        self.gamma_lr = gamma_lr
        self.gamma_noise = gamma_noise
        self.f_lb = f_lb
        self.s = s
        self.beta = beta
        self.epochs = epochs
        self.loss_name = loss_name
        self.warmup = warmup

        self._loss_func = self._get_loss_func(loss_name)
        self._prepare_weights(weight_cutoff)

    def set_system(
        self,
        out_dir: Optional[str],
        u1: Optional[mda.Universe] = None,
        u2: Optional[mda.Universe] = None,
        optimized_atoms: Optional[List[int]] = None,
        norm_func: Optional[Callable[[Tensor], Tensor]] = None,
    ):
        """
        Mirrors ARPDFPolarOptimizer.set_system:
        - loads positions from u1 (and optionally u2)
        - prepares model.set_system with around_pos and atom_pairs (use your existing helper)
        - computes reference image for u1, sets initial center positions for optimization
        """
        # allow loading from out_dir if u1 / optimized_atoms are None (same convention as ARPDFPolarOptimizer)
        #if any(x is None for x in (u1, optimized_atoms)):
        #    u1, u2, optimized_atoms, _ = utils.load_structure_data(out_dir)  # you have such helper in ARPDF pipeline

        self.out_dir = out_dir if out_dir else "tmp"
        self.u1 = u1
        self.modified_atoms = optimized_atoms

        # get positions & atom_pairs (choose helper consistent with your pipeline)
        selected_pos1, around_pos1, atom_pairs = get_atoms_pairs(u1, optimized_atoms, cutoff=self.cutoff + 2.0, periodic=True)
        # set system for PDFModel
        self.model.set_system(around_pos1, atom_pairs)

        if u2 is not None:
            selected_pos2 = u2.atoms.positions[optimized_atoms]
            _center = u1.atoms.positions[optimized_atoms[0]]
            selected_pos2 = _center + box_shift(selected_pos2 - _center, box=u1.dimensions)
        else:
            selected_pos2 = np.copy(selected_pos1)

        self.selected_pos1 = toTensor(selected_pos1, device=self.device).float()
        self.selected_pos2_init = toTensor(selected_pos2, device=self.device).float()
        #self.center_masses = toTensor(center_masses, device=self.device).float()

        # compute image1 (PDF from u1)
        with torch.no_grad():
            self.image1 = self.model(self.selected_pos1)  # shape (Nr,)

        self.num_atoms = selected_pos1.shape[0]
        # parameters to optimize: displacements for center atoms
        self._params = torch.zeros((self.num_atoms, 3), dtype=torch.float32, device=self.device, requires_grad=True)
        self.optimizer = optim.Adam([self._params], lr=self.lr)
        self.noise_scheduler = GND(self.optimizer, s=self.s, f_lb=self.f_lb, gamma=self.gamma_noise)
        self.lr_scheduler = ExponentialLR(self.optimizer, gamma=self.gamma_lr)

        if norm_func is None:
            self.norm_func = lambda sel_pos: torch.as_tensor(0.0, device=self.device)
        else:
            self.norm_func = norm_func

    def to(self, *args, **kwargs):
        self.__dict__.update(toTensor(self.__dict__, *args, **kwargs))
        self.model.to(*args, **kwargs)
        return self

    def _prepare_weights(self, weight_cutoff=6.0):
        r = self.model.r
        #sigma_R = 1 / 2
        self.cosine_weight = torch.exp(-torch.clip(r - weight_cutoff, 0.0)**2 / (2 * (0.5)**2))/(1 + torch.exp(-10 * (r - 1)))
        self.cosine_weight /= self.cosine_weight.sum()

    def _get_selected_pos(self):
        return self.selected_pos2_init + self._params

    def _get_loss_func(self, loss_name: str):
        # for radial PDF we'll support at least cosine similarity and L2 (mse)
        def _loss_cosine(pred):
            # negative cosine similarity (we maximize similarity)
            return -cosine_similarity(pred.view(1, -1), self.PDF_ref.view(1, -1), weight=self.cosine_weight)  # weight None for now
        def _loss_mse(pred):
            return torch.nn.functional.mse_loss(pred, self.PDF_ref)
        mapping = {"cosine": _loss_cosine, "mse": _loss_mse}
        return mapping.get(loss_name.strip().lower(), _loss_cosine)

    def optimize(self, verbose=True, log_step=5, print_step=50, leave=True, prefix="Optimizing PDF"):
        traj = np.zeros((self.epochs + 1, self.num_atoms, 3), dtype=np.float32)
        log = {"epoch": [], "lr": [], "loss": [], "Norm": []}

        for epoch in tqdm(range(self.epochs), desc=prefix, disable=not verbose, leave=leave):
            self.optimizer.zero_grad()

            sel_pos2 = self._get_selected_pos()
            traj[epoch] = sel_pos2.detach().cpu().numpy()

            PDF_pred = self.model(sel_pos2) - self.image1  # difference PDF (ΔPDF)
            loss = self._loss_func(PDF_pred)
            normalization = self.norm_func(sel_pos2)
            total_loss = loss + self.beta * normalization
            total_loss.backward()
            self.optimizer.step()

            # noise scheduler + lr scheduler as ARPDFPolarOptimizer
            if self.s > 0:
                with torch.no_grad():
                    sel_pos2_tmp = self._get_selected_pos()
                    PDF_tmp = self.model(sel_pos2_tmp) - self.image1
                    loss_tmp = self._loss_func(PDF_tmp) + self.beta * self.norm_func(sel_pos2_tmp)
                    self.noise_scheduler.step(loss_tmp, freeze_lb=epoch < self.warmup)
            if epoch >= self.warmup:
                self.lr_scheduler.step()

            if epoch % log_step == 0:
                lr = self.lr_scheduler.get_last_lr()[0]
                if verbose and epoch % print_step == 0:
                    warmup_status = " (Warmup)" if epoch < self.warmup else ""
                    tqdm.write(f"Epoch {epoch}: Loss={loss.item():.6f}, Norm={normalization.item():.6f}, LR={lr:.6f}{warmup_status}")
                    tqdm.write(f"f_min={self.noise_scheduler.f_min:.6f}, f_lb={self.noise_scheduler.f_lb:.6f}")
                log["epoch"].append(epoch)
                log["lr"].append(lr)
                log["loss"].append(loss.item())
                log["Norm"].append(normalization.item())

        # finalize
        self.center_pos2_final = self._get_selected_pos().detach()
        traj[-1] = self.center_pos2_final.detach().cpu().numpy()
        self.dump_results(traj, log)

    def dump_results(self, traj, log):
        # compute optimized ΔPDF and save outputs (numpy, simple plot, gro)
        PDF_optimized = (self.model(self.center_pos2_final) - self.image1).cpu().numpy()
        r = self.r.cpu().numpy()
        mask = r<6
        os.makedirs(self.out_dir, exist_ok=True)

        # simple comparison plot
        plt.figure(figsize=(6,4))
        plt.plot(r, self.PDF_ref.cpu().numpy()/self.PDF_ref.cpu().numpy()[mask].max(), label='Reference dPDF')
        #plt.plot(r, (self.model(self.selected_pos1).cpu().numpy()), label='Initial PDF')
        #plt.plot(r, (self.model(self.center_pos2_final).cpu().numpy()), label='Optimized PDF')
        plt.plot(r, PDF_optimized/PDF_optimized[mask].max(), label='Optimized dPDF')
        plt.legend()
        plt.xlabel("r (A)")
        plt.ylabel("Intensity")
        plt.title("PDF comparison")
        plt.tight_layout()
        plt.savefig(os.path.join(self.out_dir, 'PDF_optimized.png'))
        plt.close()

        np.save(os.path.join(self.out_dir, 'traj.npy'), traj)
        df = pd.DataFrame(log)
        df.to_csv(os.path.join(self.out_dir, 'log.txt'), index=False)

        # write optimized structure (like ARPDFPolarOptimizer)
        u2_opt = self.u1.copy()
        u2_opt.atoms[self.modified_atoms].positions = self.center_pos2_final.cpu().numpy()
        u2_opt.atoms.write(os.path.join(self.out_dir, 'optimized.gro'))