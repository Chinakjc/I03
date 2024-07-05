cmake_minimum_required(VERSION 3.10)

project(MPI_OpenMP)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_INCLUDE_CURRENT_DIR ON)
set (CMAKE_CXX_STANDARD 11)

if (${CMAKE_BUILD_TYPE} STREQUAL "Debug")
  add_compile_options(-DDEBUG)
endif()

include_directories(
  common
  ${CMAKE_BUILD_DIR}
  )

configure_file(version.hxx.in version.hxx @ONLY)

set(SRCS
    main.cxx
    parameters.cxx
    values.cxx
    scheme.cxx
    common/arguments.cxx
    common/os.cxx
    common/pause.cxx
    common/timer.cxx
  )

add_executable(
  PoissonMPI_OpenMP
  ${SRCS})

target_link_libraries(PoissonMPI_OpenMP PUBLIC MPI::MPI_CXX OpenMP::OpenMP_CXX)

install(TARGETS PoissonMPI_OpenMP DESTINATION .)

