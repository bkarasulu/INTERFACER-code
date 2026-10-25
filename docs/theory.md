# Theory and definitions

All energies are total energies from the selected code. Areas are in-plane cell areas, $A = |\mathbf a \times \mathbf b|$. INTERFACER prints energies per area in eV/nm² and in J/m², using 1 eV/nm² = 0.1602 J/m².

## Surface energy of a stoichiometric slab

For a slab of $N$ atoms with two equivalent surfaces:

$$ W_\mathrm{surf} = \frac{E_\mathrm{slab} - N\,E_\mathrm{bulk}/N_\mathrm{bulk}}{2A}. $$

This is the "W_surf" printed for the screened surfaces (`-tsur`) and for the optimised aligned slabs. With asymmetric terminations it is the average of the two surface energies.

## Surface formation energy

`calculate_SFE` generalises $W_\mathrm{surf}$ to non-stoichiometric slabs, such as symmetric slabs with excess or missing atoms:

$$ \mathrm{SFE} = \frac{1}{2A}\Big[E_\mathrm{slab} - n_\mathrm{fu}\,E_\mathrm{bulk,fu} - \sum_i \Delta n_i\,\mu_i\Big], $$

where:

- $n_\mathrm{fu}$ is the number of bulk formula units in the slab. The formula unit is normalised to the least abundant element, and an integer count is used where possible.
- $\Delta n_i$ is the excess of element $i$ with respect to those formula units.
- $\mu_i$ are the chemical potentials. They are given directly (`-cp1 "{'Na': -1.3, 'O': -4.9, 'Cl': -1.8}"`, one value per element of the bulk), or obtained from $n-1$ decomposition products (`-decomps1 Na2O.cif NaCl.cif` for Na3OCl) by solving

$$ \sum_i n_{k,i}\,\mu_i = E_k \quad \text{for each product } k \text{ and the bulk.} $$

### SFE as a function of the chemical potential (`--sfe_scan1/2`)

Which termination is stable depends on the chemical environment. With `--sfe_scan1 Na,Cl,-1.5,0` the SFEs of all slabs of material 1 are computed for $\mu_\mathrm{Na} = \mu^0_\mathrm{Na} + \Delta\mu$, with $\Delta\mu$ from −1.5 to 0 eV. The chemical potential of the partner element compensates so that the bulk stays in equilibrium:

$$ \mu_\mathrm{Cl} = \mu^0_\mathrm{Cl} - \Delta\mu\,\frac{n_\mathrm{Na}}{n_\mathrm{Cl}}, $$

where $n$ are the numbers of atoms in the bulk. The reference $\mu^0$ are the chemical potentials of the run (`-cp1` or `-decomps1`). The table is written to `OUTPUT/sfe_vs_mu_1.csv` and plotted in `sfe_vs_mu_1.png`, and the ranges of $\Delta\mu$ in which each termination is the most stable are printed (`analysis.sfe_vs_chempot`).

## Work of adhesion

$$ W_\mathrm{ad} = \frac{E_\mathrm{slab,1} + E_\mathrm{slab,2} - E_\mathrm{interface}}{n_\mathrm{int}\,A}. $$

A bound interface has $W_\mathrm{ad} > 0$, and for coherent interfaces $W_\mathrm{ad} \le W_\mathrm{surf,1} + W_\mathrm{surf,2}$. $n_\mathrm{int}$ counts the interfaces in the cell:

- **Small vacuum** (≤ 4 Å on each side; the default `-vac_int 1`, or `sep/2` after `-tsep`/`-thor`): the periodic images of the two outer surfaces form a second interface, so $n_\mathrm{int} = 2$ and the model has no free surfaces.
- **Large vacuum:** one interface and two free surfaces, so $n_\mathrm{int} = 1$. A dipole correction is then used with VASP/CASTEP.

$E_\mathrm{slab,1}$ and $E_\mathrm{slab,2}$ are the energies of the aligned slabs, each relaxed in its own (unstrained) supercell. The interface cell is the average of the two in-plane cells, so $W_\mathrm{ad}$ includes the elastic energy of straining each slab by about half the misfit. Keep `-pt` small for meaningful values.

## Interface energy

The interface energy follows from the Dupré relation:

$$ \gamma_\mathrm{int} = \gamma_1 + \gamma_2 - W_\mathrm{ad}. $$

$\gamma_1$ and $\gamma_2$ are the surface energies $W_\mathrm{surf}$ of the aligned slabs. For non-stoichiometric slabs, the surface formation energy of the pre-optimised slab is used instead. $\gamma_\mathrm{int}$ is printed after $W_\mathrm{ad}$ and stored as `gamma_int` in `results.json`. A negative value means that $W_\mathrm{ad} > \gamma_1 + \gamma_2$, which is not possible for a coherent interface; it usually points to a large misfit strain or an unconverged setting.

## Three-layer interfaces

For slab 1 | slab 2 | slab 3 (with `-i3`) the total work of adhesion of the stack is

$$ W_\mathrm{ad,total} = \frac{E_\mathrm{slab,1} + E_\mathrm{slab,2} + E_\mathrm{slab,3} - E_\mathrm{stack}}{A} = W_\mathrm{ad}(1|2) + W_\mathrm{ad}(2|3). $$

The stack always has at least 4 Å of vacuum on each side, so that slab 3 and slab 1 do not form a third interface across the periodic boundary.

## Grain-boundary energy

A bicrystal of $N$ atoms that is periodic along the normal contains two equivalent boundaries:

$$ \gamma_\mathrm{GB} = \frac{E_\mathrm{bicrystal} - N\,E_\mathrm{bulk}/N_\mathrm{bulk}}{2A}. $$

- **Symmetric tilt** (`--gb tilt`, plane $(hkl)$ = `-m1`): grain 2 is the mirror image of grain 1 through its top atomic plane. This is the coincidence-site-lattice (CSL) bicrystal; for example, (310) of a cubic crystal gives the Σ5 (310)[001] boundary, with a tilt angle of $2\arctan(1/3) = 36.87°$. Atoms of grain 2 on near-coincident sites (closer than 0.7 times the bulk nearest-neighbour distance) are removed. A mirror plane of the crystal gives $\gamma_\mathrm{GB} = 0$, and fcc (111) gives the Σ3 coherent twin.
- **Twist** (`--gb twist --gb_mn m n`): the vectors $m\mathbf a + n\mathbf b$ of grain 1 and $n\mathbf a + m\mathbf b$ of grain 2 coincide; grain 2 is rotated about the normal to bring them together.
  - Square planes: $\Sigma = (m^2 + n^2)/2^k$.
  - Hexagonal planes: $\Sigma = (m^2 + mn + n^2)/3^k$, with $k$ chosen so that Σ is odd (square) or not divisible by 3 (hexagonal).
  - The twist angle is reported as the smallest equivalent rotation; for example, (m, n) = (2, 1) gives Σ5 at 36.87° on cubic (100) and Σ7 at 21.79° on fcc (111).

The bicrystal is relaxed with a variable cell length along the normal, so that the boundary expansion is included.

## Misfit

$$ \mu = 1 - \frac{2\min(A_1, A_2)}{A_1 + A_2} $$

This follows *Surf. Interface Anal.* **35**, 835 (2003). It is printed before alignment (primitive surface cells) and after alignment (matched supercells).

## Commensurate supercell search

For the surface lattices $(\mathbf a_1, \mathbf b_1)$ and $(\mathbf a_2, \mathbf b_2)$:

1. `reasnoble_slab_angles` lists the pairs of lattice vectors $(m\mathbf a + n\mathbf b,\; m'\mathbf a + n'\mathbf b)$ that are shorter than $L$, not parallel and form an acute angle.
2. `match_angles2` pairs cells of the two slabs whose angles differ by less than `-pt` %. It then looks for integer multiples of the cell-vector lengths that match within `-pt` %.
3. Among all matches, the smallest supercell area is taken; for equal areas, the largest $\sin\theta$.
4. $L$ starts at `-Linit` and grows by `-Lstep` until a match is found or `-msd` is reached. In that case, the pair is skipped with a message.

With `--minkowski`, the matched in-plane cell is then reduced to its shortest, most orthogonal basis (2D Minkowski/Gauss reduction). The same integer transformation is applied to both slabs.

## Separation and stacking

The separation `-sep` is the gap along the surface normal between the topmost atom of slab 1 and the lowest atom of slab 2. Slab 1 is at the bottom. During the lateral scan, slab 1 is translated by fractions $(x, y)$ of the interface cell vectors, keeping both slabs rigid.

`-topt` refines the placement of the rigid slabs, $(x, y, \mathrm{sep})$, by minimising the single-point energy with scipy (Nelder–Mead, at most `--topt_maxiter` iterations; the separation is bounded to 0.5–6 Å). It starts from the best registry of `-thor` and the separation of `-tsep`, or from `-sep` and no shift. The history is written to `OUTPUT/placement_opt.csv`.
