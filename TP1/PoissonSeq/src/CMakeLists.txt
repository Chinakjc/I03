cmake_minimum_required(VERSION 3.10)

project(Sequential)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_VERBOSE_MAKEFILE OFF)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

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
  PoissonSeq
  ${SRCS}
  )

install(TARGETS PoissonSeq DESTINATION .)

