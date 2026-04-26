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

# ---- sub-modules ----
add_subdirectory(mace_solver)
add_subdirectory(eegmfcc_solver)
