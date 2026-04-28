#!/usr/bin/env bash
set -euo pipefail

FF_DIR='./openmm-ff'
export TORCH_PREFIX=/home/public/soft/libtorch-cxx11-2.4.1-cpu/
export LD_LIBRARY_PATH=$TORCH_PREFIX/lib:$LD_LIBRARY_PATH

python source/main.py \
    --debug-forces \
    -f data/107d-full.pdb \
    --qm-pdb data/107d-qm.pdb \
    --qm-model data/24m07_stagetwo_compiled.model \
    --qm-precision fp64 \
    --force-fields "$FF_DIR/amber14-all.xml" "$FF_DIR/amber14/tip3p.xml" \
    --mm-dir test/mm_params \
    --timestep 0.5 \
    --nsteps 100 \
    --log-file test/107d.log \
    --log-interval 1\
    --traj test/107d.traj \
    --traj-interval 1
