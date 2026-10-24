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

## Work of adhesion

$$ W_\mathrm{ad} = \frac{E_\mathrm{slab,1} + E_\mathrm{slab,2} - E_\mathrm{interface}}{n_\mathrm{int}\,A}. $$

A bound interface has $W_\mathrm{ad} > 0$, and for coherent interfaces $W_\mathrm{ad} \le W_\mathrm{surf,1} + W_\mathrm{surf,2}$. $n_\mathrm{int}$ counts the interfaces in the cell:

- **Small vacuum** (≤ 4 Å on each side; the default `-vac_int 1`, or `sep/2` after `-tsep`/`-thor`): the periodic images of the two outer surfaces form a second interface, so $n_\mathrm{int} = 2$ and the model has no free surfaces.
- **Large vacuum:** one interface and two free surfaces, so $n_\mathrm{int} = 1$. A dipole correction is then used with VASP/CASTEP.

$E_\mathrm{slab,1}$ and $E_\mathrm{slab,2}$ are the energies of the aligned slabs, each relaxed in its own (unstrained) supercell. The interface cell is the average of the two in-plane cells, so $W_\mathrm{ad}$ includes the elastic energy of straining each slab by about half the misfit. Keep `-pt` small for meaningful values.

## Misfit

$$ \mu = 1 - \frac{2\min(A_1, A_2)}{A_1 + A_2} $$

This follows *Surf. Interface Anal.* **35**, 835 (2003). It is printed before alignment (primitive surface cells) and after alignment (matched supercells).

## Commensurate supercell search

For the surface lattices $(\mathbf a_1, \mathbf b_1)$ and $(\mathbf a_2, \mathbf b_2)$:

1. `reasnoble_slab_angles` lists the pairs of lattice vectors $(m\mathbf a + n\mathbf b,\; m'\mathbf a + n'\mathbf b)$ that are shorter than $L$, not parallel and form an acute angle.
2. `match_angles2` pairs cells of the two slabs whose angles differ by less than `-pt` %. It then looks for integer multiples of the cell-vector lengths that match within `-pt` %.
3. Among all matches, the smallest supercell area is taken; for equal areas, the largest $\sin\theta$.
4. $L$ starts at `-Linit` and grows by `-Lstep` until a match is found or `-msd` is reached. In that case, the pair is skipped with a message.

## Separation and stacking

The separation `-sep` is the gap along the surface normal between the topmost atom of slab 1 and the lowest atom of slab 2. Slab 1 is at the bottom. During the lateral scan, slab 1 is translated by fractions $(x, y)$ of the interface cell vectors, keeping both slabs rigid.
