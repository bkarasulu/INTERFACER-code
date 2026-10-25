#!/bin/bash
# Tutorial 6: grain boundaries and a three-layer interface with EMT (each run writes OUTPUT/ and interfaces.out;
# the previous OUTPUT/ is kept as OUTPUT_prev/)
cd "$(dirname "$0")"
set -e
EMT="-prog ase -ase_pot EMT --relax_cell_bulk -q"
interfacer -i1 Cu.cif -m1 3 1 0 -cr1 3 --gb tilt $EMT "$@";             grep gamma_GB interfaces.out
interfacer -i1 Cu.cif -m1 1 0 0 --gb twist --gb_mn 2 1 $EMT "$@";      grep gamma_GB interfaces.out
interfacer -i1 Cu.cif -i2 Ni.cif -i3 Cu.cif -m1 1 0 0 -m2 1 0 0 -m3 1 0 0 -cr1 2 -cr2 2 -cr3 2 -sep2 2.0 \
           -msd 20 $EMT "$@";                                            grep W_ad interfaces.out
