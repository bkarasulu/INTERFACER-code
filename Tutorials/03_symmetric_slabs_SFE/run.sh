#!/bin/bash
# Tutorial 3: symmetric slabs and surface formation energies (needs: pip install mace-torch)
cd "$(dirname "$0")"
interfacer --input input.toml "$@" | tee interfacer.log
