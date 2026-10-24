#!/bin/bash
# Tutorial 4: beta-Li3PS4 | Li with VASP. Set VASP_PP_PATH first; check the inputs with:  ./run_vasp.sh -dry
cd "$(dirname "$0")"
interfacer --input input_vasp.toml "$@" | tee interfacer.log
