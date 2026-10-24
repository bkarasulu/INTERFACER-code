#!/bin/bash
# Tutorial 2: Na3OCl|Na with MACE-MP-0 (needs: pip install mace-torch)
cd "$(dirname "$0")"
interfacer --input input.toml "$@" | tee interfacer.log
