#!/bin/bash
# Tutorial 1: Cu|Ni with EMT. Equivalent to:
# interfacer -i1 Cu.cif -i2 Ni.cif -m1 1 1 1 -m2 1 1 1 -cr1 3 -cr2 3 -msd 20 -prog ase -ase_pot EMT -ase_gsteps 200 \
#            -ase_fmax 0.02 --relax_cell_bulk -tsur -mlist 100 110 111 -tsep -sep_init 1.5 -sep_final 3.0 -sep_step 0.25 \
#            -thor -hor_steps 0.25 0.25
cd "$(dirname "$0")"
interfacer --input input.toml "$@" | tee interfacer.log
