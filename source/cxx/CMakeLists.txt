cmake_minimum_required(VERSION 3.14)
project(frag-ase LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ---- meika headers (user's header-only library) ----
set(MEIKA_PREFIX "" CACHE PATH "Path to meika headers (parent used as -I dir)")
get_filename_component(MEIKA_PARENT "${MEIKA_PREFIX}" DIRECTORY)

# ---- pybind11 ----
find_package(pybind11 REQUIRED)

# ---- Torch (required by mace_solver) ----
find_package(Torch REQUIRED)

# ---- OpenMM (required by nonbonded_solver) ----
find_library(OPENMM_LIBRARY OpenMM REQUIRED
             PATHS "$ENV{CONDA_PREFIX}/lib" NO_DEFAULT_PATH)
find_library(OPENMM_LIBS OpenMMCPU REQUIRED
             PATHS "$ENV{CONDA_PREFIX}/lib" NO_DEFAULT_PATH)
set(OPENMM_INCLUDE_DIR "$ENV{CONDA_PREFIX}/include")

# ---- sub-modules ----
add_subdirectory(mace_solver)
add_subdirectory(nonbonded_solver)
add_subdirectory(eegmfcc_solver)
