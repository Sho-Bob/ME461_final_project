
cmake_minimum_required(VERSION 3.10)
project(Burgers1D)

set(CMAKE_CXX_STANDARD 11)
include_directories(src)

add_executable(Burgers1D
    src/main.cpp
    src/solver.cpp
    # src/reconstruction.cpp
    src/vtk_writer.cpp
)

# --- OpenMP ---
find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    target_link_libraries(Burgers1D PUBLIC OpenMP::OpenMP_CXX)
endif()

# --- MPI ---
find_package(MPI REQUIRED)

# If your CMake has the imported target (3.9+), this is best:
if(TARGET MPI::MPI_CXX)
    target_link_libraries(Burgers1D PUBLIC MPI::MPI_CXX)
else()
    # Fallback for older MPI/CMake setups
    include_directories(${MPI_CXX_INCLUDE_DIRS})
    target_link_libraries(Burgers1D PUBLIC ${MPI_CXX_LIBRARIES})
    target_compile_options(Burgers1D PUBLIC ${MPI_CXX_COMPILE_FLAGS})
    target_link_options(Burgers1D PUBLIC ${MPI_CXX_LINK_FLAGS})
endif()
