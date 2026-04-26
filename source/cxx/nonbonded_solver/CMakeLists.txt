add_library(nonbonded_solver STATIC nonbonded_solver.cpp)

target_include_directories(nonbonded_solver PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${MEIKA_PARENT}"
    "${OPENMM_INCLUDE_DIR}")

target_link_libraries(nonbonded_solver PUBLIC
    "${OPENMM_LIBRARY}"
    "${OPENMM_LIBS}")

target_compile_options(nonbonded_solver PRIVATE -O2 -Wall -Wextra -fopenmp)
target_link_options(nonbonded_solver PRIVATE -fopenmp)
