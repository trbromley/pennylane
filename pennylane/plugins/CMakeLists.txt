cmake_minimum_required(VERSION 3.10.2)
project(LightningQubit)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

include(CheckCXXCompilerFlag)

add_definitions("-std=c++11")

set(CMAKE_CXX_STANDARD 14)

CHECK_CXX_COMPILER_FLAG("-march=native" COMPILER_SUPPORTS_MARCH_NATIVE)
if(COMPILER_SUPPORTS_MARCH_NATIVE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
endif()

CHECK_CXX_COMPILER_FLAG("-mtune=native" COMPILER_SUPPORTS_MARCH_NATIVE)
if(COMPILER_SUPPORTS_MARCH_NATIVE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mtune=native")
endif()

find_package(OpenMP REQUIRED) # Find the package

# target_include_directories(lqo PRIVATE ${${CONDA_PREFIX}/include})
# include_directories({/home/tom/miniconda3/envs/pennylane/include})
# set(XTENSOR xtensor-blas)
include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

add_executable(lqo lightning_qubit_ops.cpp)
# target_link_libraries(lqo ${XTENSOR})
target_include_directories(lqo PRIVATE ${CONDA_PREFIX}/include)
target_link_libraries(lqo ${CONAN_LIBS})
