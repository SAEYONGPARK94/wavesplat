"""Loader for the fixed training dataset (build_dataset.py output).

All methods draw batches from the SAME frozen realizations so comparisons are apples-to-apples.
Light per-draw augmentation (obs subsample + small extra noise) curbs overfitting to a finite set
without changing the underlying labels/observations.
"""
import numpy as np


class FixedSampler:
    def __init__(self, path, aug_noise=0.01, sub_obs=None, seed=0, nl_filter=None):
        """nl_filter: None=all; 0=linear-only (control that isolates HOS-augmentation from
        dataset scale/standardization); list e.g. [0,2] keeps those nl codes."""
        d = np.load(path)
        atoms = d['atoms'].astype(np.float32); P = d['P'].astype(np.float32)
        z = d['z'].astype(np.float32); nl = d['nl']
        if nl_filter is not None:
            keep = np.isin(nl, np.atleast_1d(nl_filter))
            atoms, P, z, nl = atoms[keep], P[keep], z[keep], nl[keep]
        self.atoms, self.P, self.z, self.nl = atoms, P, z, nl
        self.N, self.M_OBS = self.z.shape
        self.aug = aug_noise; self.sub_obs = sub_obs
        self.rng = np.random.default_rng(seed)

    def batch(self, bsz):
        """→ (comps list of (M,3) kx,ky,A ; P (bsz,m,3) ; z (bsz,m)). m=sub_obs or M_OBS."""
        idx = self.rng.integers(0, self.N, bsz)
        atoms = self.atoms[idx]
        P = self.P[idx].copy(); z = self.z[idx].copy()
        if self.sub_obs and self.sub_obs < self.M_OBS:
            oi = self.rng.integers(0, self.M_OBS, (bsz, self.sub_obs))
            P = np.take_along_axis(P, oi[:, :, None], 1); z = np.take_along_axis(z, oi, 1)
        if self.aug > 0:
            z = z + self.rng.normal(0, self.aug, z.shape).astype(np.float32)
        comps = [atoms[b] for b in range(bsz)]
        return comps, P, z
