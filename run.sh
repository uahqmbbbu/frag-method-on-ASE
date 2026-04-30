#!/usr/bin/env bash
set -euo pipefail

FF_DIR='./openmm-ff'
export TORCH_PREFIX=/home/public/soft/libtorch-cxx11-2.4.1-cpu/
export LD_LIBRARY_PATH=$TORCH_PREFIX/lib:$LD_LIBRARY_PATH

python source/main.py \
    -f data/1osu-full.pdb \
    --qm-pdb data/1osu-qm.pdb \
    --qm-model data/24m07_stagetwo_compiled.model \
    --qm-precision fp32 \
    --mm-dir test/mm_params \
    --mm-pbc \
    --force-fields "$FF_DIR/amberfb15.xml" "$FF_DIR/tip3pfb.xml" \
    --timestep 0.5 \
    --temperature 300 \
    --friction 0.001 \
    --nsteps 1000 \
    --minimize-steps 0 \
    --log-file test/1osu.log \
    --log-interval 1\
    --traj test/1osu.traj \
    --traj-interval 1
