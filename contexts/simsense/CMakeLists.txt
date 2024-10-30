cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(pysimsense LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "-O3 -g0 -Wall -Wnon-virtual-dtor -fsized-deallocation -Wno-deprecated-declarations")
set(CMAKE_CUDA_STANDARD 20)
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -O3 -lineinfo -ftemplate-depth=1024")

# set(CMAKE_CUDA_COMPILER "/usr/local/cuda-12.4/bin/nvcc")

# set(CMAKE_PREFIX_PATH "/home/youwyu/Documents/libtorch-shared-with-deps-2.5.0+cu124/libtorch")
# find_package(Torch REQUIRED)
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_subdirectory("3rd_party/pybind11")
include_directories(
  ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
  "3rd_party/pybind11/include"
  ${PYTHON_INCLUDE_DIRS}
  "include"
)

file(GLOB_RECURSE SIMSENSE_SRC "src/*")
add_library(simsense SHARED ${SIMSENSE_SRC} ${TORCH_LIBRARIES})
set_target_properties(simsense PROPERTIES CUDA_ARCHITECTURES "60;61;70;75;80;86") # Supporting GPU compute capabilities

pybind11_add_module(pysimsense "python/pysimsense.cpp")
set_target_properties(pysimsense PROPERTIES PREFIX "")
target_link_libraries(pysimsense PRIVATE simsense)
