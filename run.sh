#!/usr/bin/env bash
set -euo pipefail

FF_DIR='./openmm-ff'
export TORCH_PREFIX=/home/public/soft/libtorch-cxx11-2.4.1-cpu/
export LD_LIBRARY_PATH=$TORCH_PREFIX/lib:$LD_LIBRARY_PATH

python source/main.py \
    -f data/107d-full.pdb \
    --qm-pdb data/107d-qm.pdb \
    --qm-model data/24m07_stagetwo_compiled.model \
    --force-fields "$FF_DIR/amber14-all.xml" "$FF_DIR/amber14/tip3p.xml" \
    --mm-dir test/mm_params \
    --nsteps 1 \
    --log-interval 1
