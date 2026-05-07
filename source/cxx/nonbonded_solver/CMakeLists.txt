add_library(nonbonded_solver STATIC nonbonded_solver.cpp)

target_include_directories(nonbonded_solver PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${MEIKA_PARENT}")

target_compile_options(nonbonded_solver PRIVATE
    $<$<CONFIG:Release>:-O2>
    $<$<CONFIG:Debug>:-O0 -gdwarf-4>
    -Wall -Wextra -fopenmp)
target_link_options(nonbonded_solver PRIVATE -fopenmp)
