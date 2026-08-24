# Tensor-network simulation of graph-structured quantum circuits

A controlled study of how entanglement, graph topology, qubit ordering and matrix-product-state
truncation determine the accuracy and cost of simulating shallow graph-structured quantum circuits —
and of how much of a structured, sorted probability readout survives truncation after the full state
has stopped being accurate.

Everything here is built on one custom implementation, [`mps.py`](mps.py), written in NumPy/SciPy and
validated against Qiskit statevectors before any approximation experiment was allowed to start.



## The representation

An open-boundary matrix product state on `n` qubits, tensors

$$A^{[k]}\in\mathbb{C}^{\chi_{k-1}\times 2\times\chi_k},\qquad \chi_0=\chi_n=1,$$

with the amplitude of a basis string read in site order as the ordered matrix product
$\psi(s)=A^{[0]}_{:,s_0,:}A^{[1]}_{:,s_1,:}\cdots$. `mps.py` provides product-state initialisation,
one- and two-qubit gates truncation by maximum rank and singular-value cutoff, nonadjacent gates via an explicit SWAP network
with a tracked logical-to-site map, left/right canonicalisation, Schmidt spectra and bond entropies,
accumulated discarded-weight and bond diagnostics, exact storage counts, and conversion to a dense
statevector for bounded validation problems.



Reported quantities: discarded weight $\delta_k=\sum_{i\in\text{discarded}}s_i^2$ per update,
bond entropy $S_k=-\sum_i s_i^2\log_2 s_i^2$, state accuracy
$F=|\langle\psi_{\text{exact}}|\psi_{\text{MPS}}\rangle|^2$, and representation size
$N_{\text{complex}}=\sum_k 2\chi_{k-1}\chi_k$. Runtime is descriptive only — a custom Python
implementation is not a controlled performance competitor to an optimised simulator.

---

## 1. Correctness

**Notebook 1 A** · [`figures/mps_validation.png`](figures/mps_validation.png)

![MPS validation](figures/mps_validation.png)

87 assertions pass. Product states at `n ∈ {1,2,4,8}` reconstruct below `1e-12` with
identically zero entropy profiles. The Bell state's Schmidt spectrum is $(1/\sqrt2,1/\sqrt2)$ to
`1e-12` and its entropy is exactly one bit. GHZ states at `n ∈ {3,6,10}` have bond dimension 2 and a
one-bit profile at all cut.
Shallow Haar-random circuits at `n ∈ {4,6,8,10}` with 3 seeds agree with Qiskit to `|1−F| < 1e-14`
against an assertion bound of `1e-10`. Quimb
reproduces one of those random circuits and one GHZ state.

---

## 2. Entanglement growth and truncation

**Notebook 2** · [`results/depth_results.csv`](results/depth_results.csv) ·
[`figures/depth_truncation.png`](figures/depth_truncation.png)

![Depth and truncation](figures/depth_truncation.png)

A 16-qubit line, `L ∈ {1,2,4,6,8}` repetitions of (RY layer → RZZ on even bonds → RZZ on odd bonds →
RX mixer), three angle seeds, `chi_max ∈ {2,4,8,16,32,64}` plus untruncated. Every bond-cap condition
at a given `(seed, depth)` uses exactly the same gates and angles. 

| depth `L` | max cut entropy, bits (seed mean) | exact `χ` (seed median [min–max]) | median `1−F` at `χ=8` | smallest `χ` for `F ≥ 0.999` |
|---|---|---|---|---|
| 1 | 0.73 | 2 | machine precision | 2 |
| 2 | 0.95 | 4 | machine precision | 4 |
| 4 | 1.33 | 16 | 7.9 × 10⁻⁷ | 8 |
| 6 | 1.57 | 64 | 2.0 × 10⁻³ | 16 |
| 8 | 1.89 | 173 [167–201] | 1.7 × 10⁻² | 16 |



| depth | 1 | 2 | 4 | 6 | 8 |
|---|---|---|---|---|---|
| exact MPS / dense | 0.002 | 0.006 | 0.073 | 0.66 | **2.08** |

No nonmonotonic step in infidelity occurred on this grid.

---

## 3. Graph topology and qubit ordering

**Notebook 3** · [`results/ordering_results.csv`](results/ordering_results.csv) ·
[`figures/ordering_comparison.png`](figures/ordering_comparison.png)

![Ordering comparison](figures/ordering_comparison.png)

Five 16-vertex graphs (path, cycle, 4×4 grid, two non-isomorphic random 3-regular instances), six
orderings each (natural, deterministic BFS, reverse Cuthill–McKee, and random permutations with seeds
37/41/43), three angle seeds, three repetitions of the graph circuit. The circuit is defined on
logical vertices, so the exact state depends only on `(graph, seed)` and is shared across all six
ordering:

$$\text{span}_\Sigma(G,\pi)=\sum_{(u,v)\in E}|\pi(u)-\pi(v)|,\qquad
c_{\max}(G,\pi)=\max_k\bigl|\{(u,v)\in E:\pi(u)\le k<\pi(v)\}\bigr|.$$

Median `1−F` at `chi_max = 16`, with each ordering's maximum cutwidth in brackets:

| ordering | path | cycle | 4×4 grid | 3-regular (s23) | 3-regular (s29) |
|---|---|---|---|---|---|
| natural | 4×10⁻¹⁶ [1] | 3.0×10⁻⁵ [2] | 2.6×10⁻¹ [5] | 7.3×10⁻¹ [16] | 7.1×10⁻¹ [14] |
| BFS     | 4×10⁻¹⁶ [1] | 7.9×10⁻⁵ [2] | 2.7×10⁻¹ [6] | 5.5×10⁻¹ [9]  | 5.1×10⁻¹ [9]  |
| RCM     | 4×10⁻¹⁶ [1] | 7.9×10⁻⁵ [2] | 2.7×10⁻¹ [6] | 5.5×10⁻¹ [9]  | 5.1×10⁻¹ [9]  |
| rand37  | 2.4×10⁻¹ [6] | 2.8×10⁻¹ [6] | 7.8×10⁻¹ [15] | 6.8×10⁻¹ [13] | 6.3×10⁻¹ [12] |
| rand41  | 3.5×10⁻¹ [9] | 4.2×10⁻¹ [10] | 6.4×10⁻¹ [9] | 6.8×10⁻¹ [13] | 5.9×10⁻¹ [13] |
| rand43  | 1.1×10⁻¹ [7] | 1.8×10⁻¹ [8] | 6.9×10⁻¹ [13] | 7.6×10⁻¹ [15] | 5.6×10⁻¹ [12] |

The RCM row being identical to the BFS row is the mirror relationship above, not a coincidence.

`swap_count = 2·reps·(span_Σ − |E|)` holds for 540 rows with 0 deviation. 
`swap_count` is computed from the ordering, so the rank coefficient (0.999) restates the identity rather than testing it.

Cutwidth tracks the entanglement that actually appears (Spearman 0.96), and that entanglement
tracks fixed-cap accuracy (0.97). The corresponding coefficient against exact *bond dimension* is only 0.71, because 23 of
the 30 cells are saturated at the `n = 16` ceiling of 25.

On the cycle, `rand37` has both lower cutwidth
(6 vs 8) and lower span (72 vs 76) than `rand43` and is 1.6× worse. All three inversions are between
random permutations. 4 at a 1.00× margin, 3 at 1.05×, 2 at 1.10×. On the cycle, `natural` and `BFS` are different
embeddings with same cutwidth (2) and identical span (30), but their per-seed infidelities are
`[8.9×10⁻⁶, 3.0×10⁻⁵, 8.7×10⁻⁵]` against `[8.7×10⁻⁵, 3.5×10⁻⁵, 7.9×10⁻⁵]`. With 3 seeds the median difference
(2.6×) is not a stable estimate. 

The censored "bond cap needed for `1−F ≤ 10⁻²`" table sharpens the practical version. Structural
orderings need `χ = 4` on the path, 16 on the cycle and 64 on the grid; every random ordering needs
32–64 or more; and *no* ordering of either 3-regular graph reaches the target anywhere in the swept
range. For an expander at `n = 16` and three repetitions, the choice of embedding changes the
constant, not the verdict.

---

## 4. Does the structured readout tolerate truncation?

**Notebook 4** · [`results/quic_results.csv`](results/quic_results.csv) ·
[`figures/quic_readout.png`](figures/quic_readout.png)

![QuIC readout](figures/quic_readout.png)

Notebooks 2 and 3 asked the strictest possible question — does the truncated MPS reproduce the full
state?. QuIC reads out a *sorted* probability
distribution and, in practice, its top-`K` entries. A sorted readout discards which basis state
carried which probability, and that is what a truncated MPS is most likely to
get wrong.

The unsorted $\ell_1$ error $E_{\text{unsorted},1}$, the independently-sorted error
$E_{\text{sorted},1}$, and top-`K` mass and renormalised-shape errors for `K ∈ {25,100,400,1000}`.
Sorting both sequences the same way minimises $\ell_1$ over all pairings, so
$E_{\text{sorted},1}\le E_{\text{unsorted},1}$ always, and the ratio is the part of the error that was
just a relabelling; both are bounded through
$\tfrac12 E_{\text{unsorted},1}\le\sqrt{1-F}$.

The top-`K` quantities carry no guarantee, and the notebook measures the consequence rather
than assuming it. $E_{K,\text{shape}}$ renormalises a slice, so it can exceed the sorted error. It does in 16/172 rows. And $E_{K,\text{mass}}$ is an absolute difference of two small numbers, so
it looks tiny whenever the top-`K` mass itself is small. A relative mass error and the exact top-`K`
mass are therefore recorded alongside it.

Grid: path, cycle and a random 3-regular graph (`seed=23`, regenerated each time) at
`n ∈ {8,12,16}` plus the optional `n = 20`, one and two repetitions, natural and graph-informed
orderings, `chi_max ∈ {4,8,16,32,64}`, untruncated up to `n = 16`. 172 rows.

Across the 63 runs that carry a significant error, sorting
removes a median 72% of the $\ell_1$ error (quartiles 65 %–82 %). Of the 25 runs whose state is
genuinely damaged with fidelity below 90%, 13 still hold their top-100 shape error below 0.05.
The extreme case is the 3-regular graph at `n = 20`, two repetitions, `chi_max = 16`: state infidelity
`0.86`, top-100 shape error `0.023`.

That same run has an exact top-100 mass of
`9.2×10⁻⁴` and the MPS puts `2.2×10⁻³` there. A relative mass error of 136%, and its full
sorted $\ell_1$ error is `0.224`. Across all 13 survivors, the median relative top-100 mass error is
40% (range 8–142%) and the median sorted $\ell_1$ error is `0.090`. What survives
truncation here is the shape of the leading slice. That is, the relative ordering and spacing of the
largest probabilities, not how much total probability that slice holds. A ranking read off
the top 100 would be broadly right; but not a probability read off it.

The opposite happens in 3/125 runs whose state is intact (`1−F < 0.1`) but whose
top-100 shape error exceeds 0.05. It cannot happen for the sorted distribution. That error is
bounded by the unsorted one and so by the trace distance. However, the renormalised top-`K` slice
does not inherit the bound. 

The path family is representable at all the caps tested (max cut
entropy 0.58 bits at two repetitions) and contributes no readout evidence at all; of the 87 truncated
runs that are already exact, 40 are path, 32 cycle and 15 3-regular. And the readout error is not
monotone in the bond cap even where the state error is. A larger cap changes which parts of the
spectrum survive.

On the 3-regular family (34/172 rows carry a graph-informed ordering; the rest
are conditions where the structural rule returns `natural`) the graph-informed ordering wins on the
state in 30/30 paired comparisons and halves the routing bill — 2,340 down to 1,176 SWAP gates over the
six distinct `(n, reps)` conditions. On the readout it wins 27/30; all three exceptions are cases
where the natural ordering's readout was already within 1.5×, and all 3 are reported.

A depth-4 line circuit at `n ∈ {12,16,20,24,32}` with `chi_max ∈ {16,32}`.
Both caps produce identical tensors: at depth 4 the entanglement across any cut is light-cone
bounded by at most 4 crossing two-qubit gates, so the Schmidt rank cannot exceed `2⁴ = 16` and
`chi_max = 16` is already exact (largest discarded weight `9 × 10⁻³¹`). At `n = 32` a dense
statevector would need 4,294,967,296 complex amplitudes (64 GiB); this MPS holds 12,968. That is a
storage comparison between two exact representations of the same state.

---

## Synthesis

| # | Hypothesis | Verdict | Evidence |
|---|---|---|---|
| **H1** | The custom MPS reproduces exact statevector results when nothing is truncated | **supported** | NB01's 87 assertions; every untruncated run in NB02–04 agrees with Qiskit to `\|1−F\| < 1e-14` |
| **H2** | Depth-driven entanglement growth controls the accuracy/size tradeoff | **supported** | NB02: exact max entropy rises 0.73 → 1.89 bits with depth and the cap needed for `F ≥ 0.999` rises 2 → 16 with it. Pooled across all caps, Spearman(exact max S, `1−F`) is only 0.64 — the cap dominates at fixed depth, so the tradeoff is two-variable, not one |
| **H3** | Graph span and cutwidth predict the MPS burden of an ordering | **mixed** | NB03: span gives the SWAP count as an exact identity, and cutwidth tracks entanglement across 24 distinct embeddings (ρ = 0.96, descriptive). But 3 within-graph pairs invert, structurally tied embeddings differ in realised infidelity, and ρ against exact bond dimension is only 0.71 |
| **H4** | The sorted / top-`K` readout tolerates truncation better than the full state | **supported for the sorted distribution; mixed for top-`K` probability mass** | NB04: sorting removes a median 72 % of the `ℓ₁` error over 63 runs; 13 of 25 damaged states keep a top-100 *shape* error below 0.05, but their median relative top-100 *mass* error is 40 %; 3 converse runs exist — *and this is the stand-in circuit, not the published QuIC* |

The scientific success criterion was never that one ordering always wins or that the structured
readout always survives. It was that the mathematics is implemented correctly, the approximation is
measured rather than assumed, ordering effects are tested with counterexamples retained, and the
result is connected to an actual graph circuit. The first three are met outright; the fourth is met
against a documented stand-in rather than the published QuIC definition.

---

