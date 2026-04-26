pybind11_add_module(libeegmfcc_solver eegmfcc_solver.cpp)

target_include_directories(libeegmfcc_solver PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/.."
    "${MEIKA_PARENT}")

target_link_libraries(libeegmfcc_solver PRIVATE
    mace_solver
    ${TORCH_LIBRARIES})

target_compile_options(libeegmfcc_solver PRIVATE -O2 -Wall -Wextra)
