#!/usr/bin/env bash
set -euo pipefail

# Usage: ./compile.sh [--meika DIR] [--torch DIR] [--conda ENV]
# Defaults: conda env "ai2bmd", meika at ~/source/meika

export TORCH_DIR=/home/public/soft/libtorch-cxx11-2.4.1-cu121/
export LD_LIBRARY_PATH=$TORCH_DIR/lib:$LD_LIBRARY_PATH

MEIKA_DIR=""
CONDA_ENV="ai2bmd"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --meika)    MEIKA_DIR="$2";    shift 2 ;;
        --torch)    TORCH_DIR="$2";    shift 2 ;;
        --conda)    CONDA_ENV="$2";     shift 2 ;;
        --help)     sed -n '2,6p' "$0"; exit 0 ;;
        *)          echo "Unknown: $1"; exit 1 ;;
    esac
done

# ---- resolve depend defaults ----
if [[ -z "$MEIKA_DIR" ]]; then
    MEIKA_DIR="${HOME}/source/meika"
fi

# conda env → CONDA_PREFIX
CONDA_PREFIX="${CONDA_PREFIX:-$(conda info --envs 2>/dev/null | awk -v env="$CONDA_ENV" '$1==env {print $NF; exit}')}"
if [[ -z "$CONDA_PREFIX" ]]; then
    echo "Error: conda env '$CONDA_ENV' not found." >&2
    exit 1
fi

export CC=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc
export CXX=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++

# pybind11 path
PYBIND11_PREFIX="${CONDA_PREFIX}/lib/python3.10/site-packages/pybind11"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"

cd "${SCRIPT_DIR}"

echo "--- Configuring ---"
cmake -B "${BUILD_DIR}" \
    -DCMAKE_PREFIX_PATH="${TORCH_DIR};${CONDA_PREFIX};${PYBIND11_PREFIX}" \
    -DMEIKA_DIR="${MEIKA_DIR}"

echo "--- Building ---"
cmake --build "${BUILD_DIR}"

# copy .so to the parent (calculators/) directory
echo "--- Installing ---"
cp "${BUILD_DIR}"/eegmfcc_solver/libeegmfcc_solver*.so "${SCRIPT_DIR}/../"

echo "--- Done ---"
echo "MEIKA_DIR=${MEIKA_DIR}"
echo "TORCH_DIR=${TORCH_DIR}"
echo "CONDA_PREFIX=${CONDA_PREFIX}"
