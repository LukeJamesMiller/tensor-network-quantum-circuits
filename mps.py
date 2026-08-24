"""
mps.py -- Open-boundary matrix-product-state simulation of qubit circuits.

This is the one substantial, reused body of code in the study
`tensor-network-quantum-circuits`.  Everything else (circuit families, graph
generators, parameter grids, tables, plots) lives in the notebooks.

--------------------------------------------------------------------------
Conventions
--------------------------------------------------------------------------
Tensors
    A[k] has shape (chi_{k-1}, 2, chi_k)  ==  (left bond, physical, right bond)
    with chi_0 = chi_n = 1.  The amplitude of a computational basis string
    (s_0, ..., s_{n-1}) read **in site order** is

        psi(s) = A[0][:, s_0, :] @ A[1][:, s_1, :] @ ... @ A[n-1][:, s_{n-1}, :]

Statevector index
    `to_statevector` returns a length-2**n vector whose index is

        i = sum_q  s_q * 2**(n - 1 - q)          ("qubit 0 is the MSB")

    where q is the *logical* qubit label (see the ordering map below).  This
    is the convention obtained from Qiskit via ``Statevector(qc).reverse_qargs()``.

Two-qubit gate matrices
    A 4x4 matrix U acting on the ordered pair (a, b) is indexed as

        U[2*s_a' + s_b',  2*s_a + s_b]

    i.e. **the first listed qubit is the most significant bit**.  In Qiskit the
    first entry of a qargs list is the *least* significant bit, so the same
    matrix is appended with ``qc.unitary(U, [b, a])``.  Notebook 01 asserts
    this correspondence with an asymmetric gate.

Ordering map
    `site_of[q]` is the MPS site currently holding logical qubit q, and
    `qubit_at[s]` is its inverse.  An initial ordering (a bijection from
    logical qubits to sites) may be supplied at construction; nonadjacent
    gates are routed with explicit nearest-neighbour SWAPs.

Canonical form
    `self.center` is the orthogonality centre.  Tensors to its left are left
    isometries, tensors to its right are right isometries.  Every two-site
    update moves the centre to the left site of the pair first, so the
    singular values produced by the update are the true Schmidt coefficients
    of the state across that cut and the truncation is optimal in the 2-norm.

--------------------------------------------------------------------------
Recorded diagnostics
--------------------------------------------------------------------------
    discarded_weight       sum over updates of sum_{i discarded} s_i**2
    max_discarded_single   largest single-update discarded weight
    max_bond_seen          largest bond dimension seen at any point
    n_svd, svd_time        SVD count and cumulative SVD wall time
    n_two_site             two-site updates (gates + SWAPs)
    swap_count             elementary nearest-neighbour SWAP gates applied
    routing_distance       sum over nonlocal gates of (|s1 - s2| - 1)
    norm_factor            product of retained norms; 1.0 exactly when nothing
                           was discarded, so `abs(norm_factor - 1)` is the
                           norm-preservation test
"""

from __future__ import annotations

import time
from typing import Iterable, List, Optional, Sequence

import numpy as np

__all__ = ["MPS", "SWAP_MATRIX", "mps_version"]

# Bumped by hand when the numerical behaviour changes.  Recorded in every
# result row alongside the pinned git revision.
MPS_VERSION = "1.0.0"


def mps_version() -> str:
    return MPS_VERSION


SWAP_MATRIX = np.array(
    [[1, 0, 0, 0],
     [0, 0, 1, 0],
     [0, 1, 0, 0],
     [0, 0, 0, 1]],
    dtype=complex,
)

# `to_statevector` refuses to build a dense vector larger than this.
MAX_STATEVECTOR_QUBITS = 24


class MPS:
    """Open-boundary MPS for n qubits, initialised in |0...0>."""

    # ------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------
    def __init__(
        self,
        n: int,
        ordering: Optional[Sequence[int]] = None,
        chi_max: Optional[int] = None,
        svd_cutoff: float = 0.0,
        renormalize: bool = True,
    ):
        if n < 1:
            raise ValueError("n must be >= 1")
        self.n = int(n)
        self.chi_max = None if chi_max is None else int(chi_max)
        self.svd_cutoff = float(svd_cutoff)
        self.renormalize = bool(renormalize)

        # product state |0...0>: every tensor is (1, 2, 1) and is
        # simultaneously a left and a right isometry.
        self.tensors: List[np.ndarray] = []
        for _ in range(self.n):
            t = np.zeros((1, 2, 1), dtype=complex)
            t[0, 0, 0] = 1.0
            self.tensors.append(t)
        self.center = 0

        # ordering map -------------------------------------------------
        if ordering is None:
            site_of = np.arange(self.n, dtype=int)
        else:
            site_of = np.asarray(ordering, dtype=int)
            if site_of.shape != (self.n,) or sorted(site_of.tolist()) != list(range(self.n)):
                raise ValueError("ordering must be a permutation of range(n)")
        self.site_of = site_of.copy()
        self.qubit_at = np.empty(self.n, dtype=int)
        self.qubit_at[self.site_of] = np.arange(self.n)
        self.initial_site_of = self.site_of.copy()

        self.reset_diagnostics()

    def reset_diagnostics(self) -> None:
        self.discarded_weight = 0.0
        self.max_discarded_single = 0.0
        self.max_bond_seen = 1
        self.n_svd = 0
        self.svd_time = 0.0
        self.n_two_site = 0
        self.n_one_site = 0
        self.swap_count = 0
        self.routing_distance = 0
        self.norm_factor = 1.0

    def copy(self) -> "MPS":
        other = MPS.__new__(MPS)
        other.n = self.n
        other.chi_max = self.chi_max
        other.svd_cutoff = self.svd_cutoff
        other.renormalize = self.renormalize
        other.tensors = [t.copy() for t in self.tensors]
        other.center = self.center
        other.site_of = self.site_of.copy()
        other.qubit_at = self.qubit_at.copy()
        other.initial_site_of = self.initial_site_of.copy()
        for attr in ("discarded_weight", "max_discarded_single", "max_bond_seen",
                     "n_svd", "svd_time", "n_two_site", "n_one_site",
                     "swap_count", "routing_distance", "norm_factor"):
            setattr(other, attr, getattr(self, attr))
        return other

    # ------------------------------------------------------------------
    # shape / size information
    # ------------------------------------------------------------------
    def bond_dims(self) -> np.ndarray:
        """Bond dimensions chi_0 ... chi_n (length n + 1, first and last are 1)."""
        dims = [self.tensors[0].shape[0]]
        dims += [t.shape[2] for t in self.tensors]
        return np.asarray(dims, dtype=int)

    def max_bond(self) -> int:
        return int(self.bond_dims().max())

    def storage(self) -> int:
        """Number of complex entries, N_complex = sum_k chi_{k-1} * 2 * chi_k."""
        return int(sum(t.size for t in self.tensors))

    def storage_bytes(self) -> int:
        return int(self.storage() * np.dtype(complex).itemsize)

    def _track_bonds(self) -> None:
        m = int(self.bond_dims().max())
        if m > self.max_bond_seen:
            self.max_bond_seen = m

    # ------------------------------------------------------------------
    # canonicalisation
    # ------------------------------------------------------------------
    def _shift_center_right(self) -> None:
        c = self.center
        A = self.tensors[c]
        dl, d, dr = A.shape
        Q, R = np.linalg.qr(A.reshape(dl * d, dr))
        self.tensors[c] = Q.reshape(dl, d, Q.shape[1])
        self.tensors[c + 1] = np.tensordot(R, self.tensors[c + 1], axes=([1], [0]))
        self.center = c + 1

    def _shift_center_left(self) -> None:
        c = self.center
        A = self.tensors[c]
        dl, d, dr = A.shape
        M = A.reshape(dl, d * dr)
        Q2, R2 = np.linalg.qr(M.T)          # M.T = Q2 R2
        k = Q2.shape[1]
        self.tensors[c] = Q2.T.reshape(k, d, dr)
        self.tensors[c - 1] = np.tensordot(self.tensors[c - 1], R2.T, axes=([2], [0]))
        self.center = c - 1

    def move_center(self, target: int) -> None:
        if not 0 <= target < self.n:
            raise IndexError("centre target out of range")
        while self.center < target:
            self._shift_center_right()
        while self.center > target:
            self._shift_center_left()

    def left_canonicalize(self) -> None:
        """Sweep the orthogonality centre to the last site."""
        self.move_center(self.n - 1)

    def right_canonicalize(self) -> None:
        """Sweep the orthogonality centre to the first site."""
        self.move_center(0)

    def isometry_residuals(self) -> float:
        """max_k ||A_k^dag A_k - I|| for the isometries implied by `center`."""
        worst = 0.0
        for k, A in enumerate(self.tensors):
            dl, d, dr = A.shape
            if k < self.center:
                M = A.reshape(dl * d, dr)
                res = np.linalg.norm(M.conj().T @ M - np.eye(dr))
            elif k > self.center:
                M = A.reshape(dl, d * dr)
                res = np.linalg.norm(M @ M.conj().T - np.eye(dl))
            else:
                continue
            worst = max(worst, float(res))
        return worst

    # ------------------------------------------------------------------
    # norms and overlaps
    # ------------------------------------------------------------------
    def norm(self) -> float:
        """||psi||, via the canonical form (centre tensor Frobenius norm)."""
        return float(np.linalg.norm(self.tensors[self.center]))

    def norm_by_contraction(self) -> float:
        """||psi|| computed by an explicit transfer-matrix sweep (no canonical
        assumption).  Used in Notebook 01 as an independent check."""
        E = np.eye(self.tensors[0].shape[0], dtype=complex)
        for A in self.tensors:
            E = np.einsum("ab,aic,bid->cd", E, A.conj(), A, optimize=True)
        return float(np.sqrt(np.real(np.trace(E))))

    def normalize(self) -> None:
        nrm = self.norm()
        if nrm == 0.0:
            raise FloatingPointError("cannot normalise a zero state")
        self.tensors[self.center] = self.tensors[self.center] / nrm

    # ------------------------------------------------------------------
    # gate application
    # ------------------------------------------------------------------
    def apply_one_qubit(self, U: np.ndarray, q: int) -> None:
        """Apply a 2x2 unitary to logical qubit q.

        A unitary one-site gate preserves both left and right isometry, so no
        centre move and no SVD are required.
        """
        U = np.asarray(U, dtype=complex)
        if U.shape != (2, 2):
            raise ValueError("one-qubit gate must be 2x2")
        s = int(self.site_of[q])
        self.tensors[s] = np.einsum("ij,ajb->aib", U, self.tensors[s], optimize=True)
        self.n_one_site += 1

    def apply_two_qubit(self, U: np.ndarray, q1: int, q2: int,
                        swap_back: bool = True) -> None:
        """Apply a 4x4 unitary to the ordered logical pair (q1, q2).

        q1 is the most significant index of U (see module docstring).  If the
        two qubits are not on adjacent MPS sites they are brought together with
        nearest-neighbour SWAPs; with ``swap_back=True`` (the default) the SWAP
        network is reversed afterwards so the ordering map is restored exactly.
        Keeping the ordering fixed is what makes the structural graph metrics
        of Notebook 03 comparable across the whole circuit.
        """
        U = np.asarray(U, dtype=complex)
        if U.shape != (4, 4):
            raise ValueError("two-qubit gate must be 4x4")
        if q1 == q2:
            raise ValueError("two-qubit gate needs two distinct qubits")

        s1 = int(self.site_of[q1])
        s2 = int(self.site_of[q2])
        self.routing_distance += abs(s1 - s2) - 1

        moves: List[int] = []
        if s1 < s2:
            for s in range(s1, s2 - 1):       # walk q1 rightwards
                self._swap_sites(s)
                moves.append(s)
            k, Uk = s2 - 1, U                 # (site k, site k+1) = (q1, q2)
        else:
            for s in range(s1 - 1, s2, -1):   # walk q1 leftwards
                self._swap_sites(s)
                moves.append(s)
            k = s2                            # (site k, site k+1) = (q2, q1)
            Uk = SWAP_MATRIX @ U @ SWAP_MATRIX

        self._apply_two_site(k, Uk)

        if swap_back:
            for s in reversed(moves):
                self._swap_sites(s)

    def apply_swap_qubits(self, q1: int, q2: int) -> None:
        """Exchange two logical qubits with an explicit SWAP network."""
        self.apply_two_qubit(SWAP_MATRIX, q1, q2, swap_back=False)

    def _swap_sites(self, k: int) -> None:
        """Elementary SWAP of the contents of sites k and k+1."""
        self._apply_two_site(k, SWAP_MATRIX)
        a, b = int(self.qubit_at[k]), int(self.qubit_at[k + 1])
        self.qubit_at[k], self.qubit_at[k + 1] = b, a
        self.site_of[a], self.site_of[b] = k + 1, k
        self.swap_count += 1

    def _apply_two_site(self, k: int, U4: np.ndarray) -> None:
        """Core two-site update on sites (k, k+1); k is the most significant."""
        if not 0 <= k < self.n - 1:
            raise IndexError("two-site update out of range")
        self.move_center(k)

        A, B = self.tensors[k], self.tensors[k + 1]
        dl, _, _ = A.shape
        _, _, dr = B.shape

        theta = np.tensordot(A, B, axes=([2], [0]))          # (dl, 2, 2, dr)
        Ut = U4.reshape(2, 2, 2, 2)
        theta = np.einsum("ijkl,akld->aijd", Ut, theta, optimize=True)
        M = theta.reshape(dl * 2, 2 * dr)

        t0 = time.perf_counter()
        Umat, s, Vh = np.linalg.svd(M, full_matrices=False)
        self.svd_time += time.perf_counter() - t0
        self.n_svd += 1

        total = float(np.linalg.norm(s))
        if total == 0.0:
            raise FloatingPointError("two-site tensor vanished")
        s_rel = s / total

        keep = int(np.count_nonzero(s_rel > self.svd_cutoff)) if self.svd_cutoff > 0 else s.size
        keep = max(1, keep)
        if self.chi_max is not None:
            keep = min(keep, self.chi_max)

        delta = float(np.sum(s_rel[keep:] ** 2))
        self.discarded_weight += delta
        if delta > self.max_discarded_single:
            self.max_discarded_single = delta

        s_keep = s[:keep]
        kept_norm = float(np.linalg.norm(s_keep))
        self.norm_factor *= kept_norm / total
        if self.renormalize:
            s_keep = s_keep * (total / kept_norm)

        self.tensors[k] = Umat[:, :keep].reshape(dl, 2, keep)
        self.tensors[k + 1] = (s_keep[:, None] * Vh[:keep, :]).reshape(keep, 2, dr)
        self.center = k + 1
        self.n_two_site += 1
        self._track_bonds()

    # ------------------------------------------------------------------
    # entanglement structure
    # ------------------------------------------------------------------
    def schmidt_values(self, cut: int) -> np.ndarray:
        """Normalised Schmidt coefficients across the cut between sites
        ``cut - 1`` and ``cut`` (1 <= cut <= n - 1).  The state is unchanged."""
        if not 1 <= cut <= self.n - 1:
            raise IndexError("cut must lie in 1 .. n-1")
        self.move_center(cut)
        A = self.tensors[cut]
        dl, d, dr = A.shape
        s = np.linalg.svd(A.reshape(dl, d * dr), compute_uv=False)
        nrm = float(np.linalg.norm(s))
        return s / nrm if nrm > 0 else s

    @staticmethod
    def entropy_from_schmidt(s: np.ndarray) -> float:
        """S = -sum_i s_i^2 log2 s_i^2 for normalised Schmidt values."""
        p = np.asarray(s, dtype=float) ** 2
        p = p[p > 1e-16]
        if p.size == 0:
            return 0.0
        return float(-np.sum(p * np.log2(p)))

    def entropy(self, cut: int) -> float:
        return self.entropy_from_schmidt(self.schmidt_values(cut))

    def entropy_profile(self) -> np.ndarray:
        """Bond entropies for cuts 1 .. n-1, obtained in a single sweep."""
        if self.n == 1:
            return np.zeros(0)
        self.move_center(1)
        out = np.empty(self.n - 1)
        for cut in range(1, self.n):
            out[cut - 1] = self.entropy(cut)
        return out

    # ------------------------------------------------------------------
    # dense conversion
    # ------------------------------------------------------------------
    def to_statevector(self, logical_order: bool = True) -> np.ndarray:
        """Dense state, index i = sum_q s_q 2**(n-1-q) over *logical* qubits q.

        With ``logical_order=False`` the axes are left in site order instead.
        """
        if self.n > MAX_STATEVECTOR_QUBITS:
            raise MemoryError(
                f"refusing to build a dense 2**{self.n} statevector "
                f"(limit {MAX_STATEVECTOR_QUBITS} qubits)"
            )
        psi = self.tensors[0]
        for A in self.tensors[1:]:
            psi = np.tensordot(psi, A, axes=([psi.ndim - 1], [0]))
        psi = psi.reshape([2] * self.n)      # axis k = site k
        if logical_order:
            psi = np.transpose(psi, axes=[int(self.site_of[q]) for q in range(self.n)])
        return psi.reshape(-1)

    def probabilities(self, logical_order: bool = True) -> np.ndarray:
        v = self.to_statevector(logical_order=logical_order)
        return np.abs(v) ** 2

    # ------------------------------------------------------------------
    # comparison helpers
    # ------------------------------------------------------------------
    @staticmethod
    def fidelity(psi: np.ndarray, phi: np.ndarray) -> float:
        """F = |<psi|phi>|^2 / (||psi||^2 ||phi||^2)."""
        psi = np.asarray(psi).ravel()
        phi = np.asarray(phi).ravel()
        denom = np.linalg.norm(psi) * np.linalg.norm(phi)
        if denom == 0:
            return 0.0
        return float(abs(np.vdot(psi, phi) / denom) ** 2)

    @staticmethod
    def phase_aligned_error(psi: np.ndarray, phi: np.ndarray) -> float:
        """min_theta || psi - e^{i theta} phi ||_2, both taken as given."""
        psi = np.asarray(psi).ravel()
        phi = np.asarray(phi).ravel()
        ov = np.vdot(phi, psi)
        phase = ov / abs(ov) if abs(ov) > 0 else 1.0
        return float(np.linalg.norm(psi - phase * phi))

    def infidelity_to(self, reference: np.ndarray) -> float:
        return 1.0 - self.fidelity(self.to_statevector(), reference)

    # ------------------------------------------------------------------
    def diagnostics(self) -> dict:
        """Everything the result tables record about a completed run."""
        return {
            "max_bond_final": self.max_bond(),
            "max_bond_intermediate": int(self.max_bond_seen),
            "mean_bond": float(np.mean(self.bond_dims()[1:-1])) if self.n > 1 else 1.0,
            "storage_complex": self.storage(),
            "storage_bytes": self.storage_bytes(),
            "discarded_weight": float(self.discarded_weight),
            "max_discarded_single": float(self.max_discarded_single),
            "norm_factor": float(self.norm_factor),
            "n_svd": int(self.n_svd),
            "svd_time_s": float(self.svd_time),
            "n_two_site": int(self.n_two_site),
            "n_one_site": int(self.n_one_site),
            "swap_count": int(self.swap_count),
            "routing_distance": int(self.routing_distance),
            "chi_max": self.chi_max,
            "svd_cutoff": self.svd_cutoff,
            "mps_version": MPS_VERSION,
        }

    def __repr__(self) -> str:
        return (f"MPS(n={self.n}, chi_max={self.chi_max}, "
                f"max_bond={self.max_bond()}, storage={self.storage()})")
