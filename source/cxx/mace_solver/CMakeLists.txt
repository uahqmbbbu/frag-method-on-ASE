add_library(mace_solver STATIC mace_solver.cpp)

target_include_directories(mace_solver PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(mace_solver PUBLIC
    ${TORCH_LIBRARIES})

target_compile_options(mace_solver PRIVATE
    $<$<CONFIG:Release>:-O2>
    $<$<CONFIG:Debug>:-O0 -g>
    -Wall -Wextra)
