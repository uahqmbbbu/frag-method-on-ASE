pybind11_add_module(libopenmm_solver openmm_solver.cpp)

target_include_directories(libopenmm_solver PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/.."
    "${MEIKA_PARENT}"
    "${OPENMM_INCLUDE_DIR}")

target_link_libraries(libopenmm_solver PRIVATE
    "${OPENMM_LIBRARY}"
    "${OPENMM_LIBS}")

target_compile_options(libopenmm_solver PRIVATE -O2 -Wall -Wextra -fopenmp)
target_link_options(libopenmm_solver PRIVATE -fopenmp)
