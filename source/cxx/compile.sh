#!/usr/bin/env bash
set -euo pipefail

MEIKA_PREFIX="${HOME}/source/meika"

CONDA_ENV="frag-ase"
CONDA_PREFIX="${CONDA_PREFIX:-$(conda info --envs 2>/dev/null | awk -v env="$CONDA_ENV" '$1==env {print $NF; exit}')}"
if [[ -z "$CONDA_PREFIX" ]]; then
    echo "Error: conda env '$CONDA_ENV' not found." >&2
    exit 1
fi
export CONDA_PREFIX

PYBIND11_PREFIX="${CONDA_PREFIX}/lib/python3.10/site-packages/pybind11"
export TORCH_PREFIX=/home/public/soft/libtorch-cxx11-2.4.1-cpu/

export CC=gcc
export CXX=g++
#export CC=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc
#export CXX=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++

echo "MEIKA_PREFIX=${MEIKA_PREFIX}"
echo "CONDA_PREFIX=${CONDA_PREFIX}"
echo "PYBIND11_PREFIX=${PYBIND11_PREFIX}"
echo "TORCH_PREFIX=${TORCH_PREFIX}"


SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"

cd "${SCRIPT_DIR}"

echo "--- Configuring ---"
cmake -B "${BUILD_DIR}" \
    -DCMAKE_PREFIX_PATH="${TORCH_PREFIX};${CONDA_PREFIX};${PYBIND11_PREFIX}" \
    -DMEIKA_PREFIX="${MEIKA_PREFIX}"

echo "--- Building ---"
cmake --build "${BUILD_DIR}"

echo "--- Done ---"
