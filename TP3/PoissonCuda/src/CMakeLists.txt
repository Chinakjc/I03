cmake_minimum_required(VERSION 3.10)

set(VERSION PoissonCuda)
project(${VERSION} LANGUAGES CXX CUDA)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_VERBOSE_MAKEFILE OFF)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if (${CMAKE_BUILD_TYPE} STREQUAL "Debug")
  add_compile_options(-DDEBUG)
endif()

find_package(CUDA)
enable_language(CUDA)
set(CMAKE_CUDA_ARCHITECTURES OFF)
cuda_select_nvcc_arch_flags(cuda_archs Auto)
list(APPEND CUDA_NVCC_FLAGS ${ARCH_FLAGS})
set(CUDA_NVCC_FLAGS
  "${CUDA_NVCC_FLAGS};-Wno-deprecated-gpu-targets")

set(CMAKE_CUDA_STANDARD 11)
set(CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})


if (CMAKE_CXX_COMPILER_ID MATCHES "Intel")
  if (WIN32)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /Qdiag-disable:1786,3180,10148")
  else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -diag-disable=1786,3180,10148")
  endif()
endif()

include_directories(
  common
  ${CMAKE_BUILD_DIR}
  cuda
  ${CUDAToolkit_INCLUDE_DIRS}
  )

configure_file(version.hxx.in version.hxx @ONLY)

set(SRCS
    main.cxx
    parameters.cxx
    values.cxx
    scheme.cxx
    common/arguments.cxx
    common/os.cxx
    common/timer.cxx
    cuda/variation.cu
    cuda/iteration.cu
    cuda/dim.cu
    cuda/values.cu
    cuda/user.cu
    cuda/memmove_cuda.cu
  )

add_executable(
  ${VERSION}
  ${SRCS}
  )

set_target_properties(${VERSION}
  PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(
  ${VERSION}
  ${CUDA_CUDART}
  )

install(TARGETS ${VERSION} DESTINATION .)

